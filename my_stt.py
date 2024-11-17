import asyncio
import io
import subprocess
import time
from typing import Optional
import wave
from livekit.agents.stt import STT, SpeechEvent, SpeechEventType, STTCapabilities, SpeechData
from livekit.agents.utils import AudioBuffer, merge_frames
from livekit.agents.metrics import STTMetrics
from uuid import uuid4
from icecream import ic


class LocalSTT(STT):
    def __init__(self) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=False)
        )

    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language: Optional[str] = None
    ) -> SpeechEvent:
        
        buffer = merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        audio_file_path = "stt/audio.wav"
        with open(audio_file_path, "wb") as f:
            f.write(io_buffer.getvalue())
        ic("Running STT model...")

        command = [
            "models/main",
            "--no-prints",
            "--no-timestamps",
            "-m",
            "models/ggml-tiny.en.bin",
            "-f",
            audio_file_path,
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout, stderr = await asyncio.to_thread(process.communicate)

        if stderr:
            raise Exception(f"STT failed: {stderr}")

        transcribed_text = stdout.strip() or ""
        speech_data = SpeechData(language="en", text=transcribed_text)
        ic(transcribed_text)

        event = SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid4()),
            alternatives=[speech_data]
        )

        stt_metrics = STTMetrics(
            request_id=event.request_id,
            timestamp=time.time(),
            duration=time.perf_counter(),
            label=self._label,
            audio_duration=0.0,
            streamed=False,
            error=None,
        )

        self.emit("metrics_collected", stt_metrics)

        return event
