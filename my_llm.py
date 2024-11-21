from typing import Any, AsyncIterator, Generator, AsyncGenerator
from uuid import uuid4
from api import check_time_query, check_update_query
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

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessageChunk, HumanMessage, BaseMessage
from langchain_core.messages.ai import AIMessage


class Assistant:
    def __init__(self, model):
        self.message_history = ChatMessageHistory()
        self.model = model

    def answer_to_image(self, prompt, image):
        if not prompt:
            return

        try:
            base64_str = f"data:image/jpeg;base64,{image.decode()}"
            message = HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": base64_str}},
                    prompt,
                ]
            )
            self.message_history.add_message(message)
            response: BaseMessage = self.model.invoke(
                input=[*self.message_history.messages]
            )
            self.message_history.add_message(response)

            return response
        except Exception as e:
            ic("Error:", str(e))

    async def answer_to_text(self, prompt: str):
        if not prompt:
            return
            
        message = HumanMessage(content=prompt)
        self.message_history.add_message(message)
        response: AsyncIterator[BaseMessageChunk] = self.model.astream(
            input=[*self.message_history.messages],
            generation_config={
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            }                
        )

        full_response = ""
        async for chunk in response:
            if chunk.content:
                assert isinstance(chunk.content, str)
                yield chunk
                full_response += chunk.content
        
        final_message = AIMessage(content=full_response)
        self.message_history.add_message(final_message)


class MyLLMStream(LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        fnc_ctx: FunctionContext | None = None,
        assistant: Assistant,
    ) -> None:
        self.assistant = assistant
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)

    async def _send_response(self, data: str) -> None:
        choice = Choice(ChoiceDelta(role="assistant", content=data))
        chat_chunk = self._parse_choice(choice=choice)
        self._event_ch.send_nowait(chat_chunk)

    async def _main_task(self) -> None:
        last_user_prompt = self.chat_ctx.messages[-1].content
        assert isinstance(last_user_prompt, str)

        ic("Requesting LLM response...")

        async for chunk in self.assistant.answer_to_text(last_user_prompt):
            if isinstance(chunk, AIMessage) and chunk.content:
                assert isinstance(chunk.content, str)
                await self._send_response(chunk.content)

        # time_query = await check_time_query(last_user_prompt)
        # update_query = await check_update_query(last_user_prompt)

        # if not time_query and not update_query:
        #     async for data in stream_chat(last_user_prompt):
        #         await self._send_response(data)
        # else:
        #     response = time_query or update_query
        #     assert isinstance(response, str)
        #     await self._send_response(response)

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
    def __init__(self, assistant: Any = None) -> None:
        self.assistant = assistant
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
        return MyLLMStream(
            self, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, assistant=self.assistant
        )
