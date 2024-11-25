from typing import Any
from uuid import uuid4
from langchain_core.prompts import ChatPromptTemplate
from livekit.agents.llm import LLM, FunctionInfo
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.function_context import FunctionContext
from livekit.agents.llm.llm import (
    LLMStream,
    ChatChunk,
    Choice,
    ChoiceDelta,
)
from icecream import ic
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent

from addons import WebcamStream, get_current_time


class Assistant:
    def __init__(self, model):
        self.message_history = ChatMessageHistory()
        self.model = model
        self.initial_prompt = (
            "You are a chill, relaxed assistant that uses slangs. You're honest even if it means you have to tell hard truths."
            "You have functions you can call if you can't provide the information user requested by yourself."
            "User can ask you to see him, which means to look at the image you received."
        )
        self.tools = [get_current_time]
        
        messages = [
            ("system", self.initial_prompt),
            ("placeholder", "{agent_scratchpad}")
        ]
        prompt_template = ChatPromptTemplate.from_messages(messages)
        agent = create_tool_calling_agent(
            tools=self.tools,
            llm=self.model,
            prompt=prompt_template,
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)

    def _update_agent(self):
        messages = [
            ("system", self.initial_prompt),
            *[msg for msg in self.message_history.messages],
            ("placeholder", "{agent_scratchpad}")
        ]
        prompt_template = ChatPromptTemplate.from_messages(messages)
        agent = create_tool_calling_agent(
            tools=self.tools,
            llm=self.model,
            prompt=prompt_template,
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)

    async def answer(self, prompt: str, image):
        if not prompt:
            return

        content_type = {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{image.decode()}"
                }
            } if image is not None else ""
        message = HumanMessage(
            content=[
                prompt,
                content_type,
            ]
        )
        self.message_history.add_message(message)
        self._update_agent()

        full_response = ""
        async for chunk in self.agent_executor.astream(
            {"agent_scratchpad": ""}
        ):
            if "output" in chunk:
                content = chunk["output"]
                full_response += content
                yield AIMessage(content=content)

        ai_response = AIMessage(content=full_response)
        self.message_history.add_message(ai_response)


class MyLLMStream(LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        webcam_stream: WebcamStream | None = None,
        fnc_ctx: FunctionContext | None = None,
        assistant: Assistant
    ) -> None:
        self.assistant = assistant
        self.webcam_stream = webcam_stream
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)

    async def _send_response(self, data: str) -> None:
        choice = Choice(ChoiceDelta(role="assistant", content=data))
        chat_chunk = self._parse_choice(choice=choice)
        self._event_ch.send_nowait(chat_chunk)

    async def _main_task(self) -> None:
        last_user_prompt = self.chat_ctx.messages[-1].content
        assert isinstance(last_user_prompt, str)

        ic("Requesting LLM response...")
        frame = None
        if self.webcam_stream:
            frame = await self.webcam_stream.read(encode=True)
        async for chunk in self.assistant.answer(prompt=last_user_prompt, image=frame):
            if isinstance(chunk, AIMessage):
                if chunk.content:
                    assert isinstance(chunk.content, str)
                    await self._send_response(chunk.content)


    def _parse_choice(self, choice: Choice) -> ChatChunk:
            delta = choice.delta
            
            return ChatChunk(
                request_id=str(uuid4()),
                choices=[
                    Choice(
                        delta=ChoiceDelta(
                            content=delta.content,
                            role="assistant",
                        ),
                        index=choice.index,
                    )
                ],
            )


class LocalLLM(LLM):
    def __init__(self, webcam_stream: WebcamStream | None = None, assistant: Any = None) -> None:
        self.assistant = assistant
        self.webcam_stream = webcam_stream
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
            self,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            assistant=self.assistant,
            webcam_stream=self.webcam_stream
        )
