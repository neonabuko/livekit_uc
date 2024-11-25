import asyncio
from typing import AsyncGenerator
from uuid import uuid4
import aiofiles
from gtts import gTTS
from livekit.agents.tts import TTS, ChunkedStream, TTSCapabilities, SynthesizedAudio
from livekit.agents import utils
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import aiohttp
import re


SAMPLE_RATE = 24000
NUM_CHANNELS = 1

class GTTSChunkedStream(ChunkedStream):
    def __init__(self, tts: TTS, text: str) -> None:
        super().__init__(tts, text)

    async def _stream_audio_chunks(self, file_path: str) -> AsyncGenerator[bytes, None]:
        decoder = utils.codecs.Mp3StreamDecoder()
        async with aiofiles.open(file_path, "rb") as stream:
            while data := await stream.read(1024):  
                for frame in decoder.decode_chunk(data):
                    yield frame.data.tobytes()        

    async def convert_sentence(self, sentence: str, tts_id: int) -> str:
        loop = asyncio.get_event_loop()
        clean_sentence = self.remove_emojis(sentence)
        with ThreadPoolExecutor() as pool:
            tts = await loop.run_in_executor(pool, partial(gTTS, text=clean_sentence, lang="en"))
            await loop.run_in_executor(pool, tts.save, f"stt/tts_{tts_id}.mp3")

        return f"stt/tts_{tts_id}.mp3"


    async def convert_text_to_speech(self, text: str) -> AsyncGenerator[str, None]:
        async with aiohttp.ClientSession():
            tasks = []
            for i, sentence in enumerate(text.split(". ")):
                if sentence.strip():
                    task = asyncio.create_task(self.convert_sentence(sentence, i))
                    tasks.append(task)

            for task in tasks:
                try:
                    result = await task
                    yield result
                except Exception as e:
                    print(f"Error converting sentence: {e}")
                    continue
    
    def remove_emojis(self, text: str) -> str:
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text)

    async def _main_task(self) -> None:
        processed_text = self._input_text.replace('*', '').replace('\n', ' ')
        self._input_text = processed_text
        
        audio_bstream = utils.audio.AudioByteStream(sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
        async for chunk_path in self.convert_text_to_speech(self._input_text):
            async for chunk_path in self._stream_audio_chunks(chunk_path):
                for frame in audio_bstream.write(chunk_path):
                    await self._event_ch.send(
                        SynthesizedAudio(
                            frame=frame,
                            request_id=str(uuid4()),
                        )
                    )
            for frame in audio_bstream.flush():
                self._event_ch.send_nowait(
                    SynthesizedAudio(
                        frame=frame,
                        request_id=str(uuid4()),
                    )
                )


class LocalTTS(TTS):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

    def synthesize(self, text: str) -> GTTSChunkedStream:
        return GTTSChunkedStream(self, text)
