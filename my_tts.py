import asyncio
import os
import subprocess
from typing import AsyncGenerator
from uuid import uuid4
import aiofiles
from gtts import gTTS
from livekit.agents.tts import TTS, ChunkedStream, TTSCapabilities, SynthesizedAudio
from livekit.agents import utils


SAMPLE_RATE = 24000
NUM_CHANNELS = 1

class GTTSChunkedStream(ChunkedStream):
    def __init__(self, tts: TTS, text: str) -> None:
        super().__init__(tts, text)

    async def _stream_audio_chunks(self, file_path: str) -> AsyncGenerator[bytes, None]:
        """Async generator that yields audio frames from an MP3 file."""
        decoder = utils.codecs.Mp3StreamDecoder()
        async with aiofiles.open(file_path, "rb") as stream:
            while data := await stream.read(1024):  
                for frame in decoder.decode_chunk(data):
                    yield frame.data.tobytes()        

    async def _main_task(self) -> None:
        os.system('pkill flite')
        tts_filename = "tts.mp3"

        with open(tts_filename, "wb"):
            pass

        processed_text = self._input_text.replace('*', '').replace('\n', ' ')
        self._input_text = processed_text
        
        process = subprocess.Popen(
            ["flite", "-t", self._input_text], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        await asyncio.to_thread(process.communicate)

        # tts = gTTS(text=self._input_text, lang="en")
        # tts.save(tts_filename)

        audio_bstream = utils.audio.AudioByteStream(sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
        async for chunk in self._stream_audio_chunks(tts_filename):
            for frame in audio_bstream.write(chunk):
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
