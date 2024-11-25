import asyncio
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import silero
from langchain_google_genai import ChatGoogleGenerativeAI
from my_stt import LocalSTT
from my_tts import LocalTTS
from my_llm import LocalLLM, Assistant
from icecream import ic
# from addons import WebcamStream


load_dotenv()

webcam_stream = None

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        ic(f"VAD model loading failed: {e}")
        raise RuntimeError("Failed to load VAD model.")
    ic("Applied VAD prewarm.")


async def entry_point(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    ass = Assistant(model=gemini)

    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=LocalSTT(),
        llm=LocalLLM(assistant=ass),
        tts=LocalTTS(),
    )

    assistant.start(ctx.room)
    await asyncio.sleep(1)

    if webcam_stream is not None:
        ctx.room.on("participant_disconnected", webcam_stream.stop())


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point, prewarm_fnc=prewarm))
