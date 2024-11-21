import asyncio
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import silero
from langchain_google_genai import ChatGoogleGenerativeAI
from my_stt import LocalSTT
from my_tts import LocalTTS
from my_llm import LocalLLM, Assistant
from icecream import ic

load_dotenv()

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        ic(f"VAD model loading failed: {e}")
        raise RuntimeError("Failed to load VAD model.")
    ic("Applied VAD prewarm.")


async def entry_point(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(role="system", text=("You are a helpful assistant."))
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    gemini_assistant = Assistant(
        ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    )

    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=LocalSTT(),
        llm=LocalLLM(assistant=gemini_assistant),
        tts=LocalTTS(),
        chat_ctx=initial_ctx,
    )

    assistant.start(ctx.room)
    await asyncio.sleep(1)
    await assistant.say("Welcome back, Matt!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point, prewarm_fnc=prewarm))
