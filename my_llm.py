from uuid import uuid4
import aiohttp
import json
from chatgpt_client import stream_chat
from livekit.agents.llm import LLM
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.function_context import FunctionContext
from livekit.agents.llm.llm import (
    LLMStream,
    ChatChunk,
    Choice,
    ChoiceDelta,
)
from icecream import ic


class MyLLMStream(LLMStream):
    def __init__(
        self, llm: LLM, *, chat_ctx: ChatContext, fnc_ctx: FunctionContext | None = None
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)

    async def _main_task(self) -> None:
        try:
            last_user_prompt = self.chat_ctx.messages[-1].content

            ic("Requesting ChatGPT response...")
            async for data in stream_chat(last_user_prompt):
                choice = Choice(ChoiceDelta(role="assistant", content=data))
                chat_chunk = self._parse_choice(choice=choice)
                self._event_ch.send_nowait(chat_chunk)

        except Exception as e:
            raise RuntimeError(f"Local model error: {e}") from e

    def _parse_choice(self, choice: Choice) -> ChatChunk:
        delta = choice.delta
        return ChatChunk(
            request_id=str(uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(content=delta.content, role="assistant"),
                    index=choice.index,
                )
            ],
        )


class LocalLLM(LLM):
    def __init__(self) -> None:
        super().__init__()

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        fnc_ctx: FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> MyLLMStream:
        return MyLLMStream(self, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
