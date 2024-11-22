from typing import Any, AsyncIterator
from uuid import uuid4
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import AddableDict
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
from langchain_core.messages import BaseMessageChunk, HumanMessage, BaseMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent

from test import get_current_time


class Assistant:
    def __init__(self, model):
        self.message_history = ChatMessageHistory()
        self.model = model
        self.initial_prompt = (
            "You are a chill, relaxed assistant that uses slangs and emojis. "
            "When asked about time, use the get_current_time function."
        )
        self.tool = Tool(
            name="get_current_time",
            description="Get the current time.",
            func=get_current_time,
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.initial_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )        
        agent = create_tool_calling_agent(
            tools=[self.tool],
            llm=self.model,
            prompt=prompt_template,
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=[self.tool], verbose=True)        


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
                input=[
                    SystemMessage(content=self.initial_prompt),                    
                    *self.message_history.messages
                ]
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

        full_response = ""
        async for chunk in self.agent_executor.astream(
            {"input": prompt, "agent_scratchpad": ""}
        ):
            if "output" in chunk:
                content = chunk["output"]
                full_response += content
                yield AIMessage(content=content)

        final_message = AIMessage(content=full_response)
        self.message_history.add_message(final_message)


    async def initialize(self, initial_prompt: str) -> None:
        system_message = AIMessage(content=initial_prompt)
        self.message_history.add_message(system_message)


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
            if isinstance(chunk, AIMessage):
                if chunk.content:
                    assert isinstance(chunk.content, str)
                    await self._send_response(chunk.content)
                if chunk.additional_kwargs.get('tool_calls'):
                    tool_calls = chunk.additional_kwargs['tool_calls']
                    for tool_call in tool_calls:
                        if tool_call['type'] == 'function' and self.fnc_ctx:
                            function_call = tool_call['function']
                            name = function_call['name']
                            result = await self.fnc_ctx.ai_callable(name=name)()
                            await self._send_response(str(result))


    def _parse_choice(self, choice: Choice) -> ChatChunk:
            delta = choice.delta
            tool_calls = []
            if delta.tool_calls:
                logger.info(f"Processing tool calls: {len(delta.tool_calls)} calls found")
                for tool_call in delta.tool_calls:
                    tool_call_function = getattr(tool_call, 'function', None)
                    if tool_call_function is None:
                        logger.warning("Tool call function attribute is None, skipping")
                        continue
                    
                    name = getattr(tool_call_function, 'name', '')
                    arguments = getattr(tool_call_function, 'arguments', '')
                    logger.info(f"Processing function call: {name} with arguments: {arguments}")
                    
                    tool_calls.append({
                        'id': getattr(tool_call, 'id', str(uuid4())),
                        'type': 'function',
                        'function': {
                            'name': name,
                            'arguments': arguments
                        }
                    })
            
            return ChatChunk(
                request_id=str(uuid4()),
                choices=[
                    Choice(
                        delta=ChoiceDelta(
                            content=delta.content,
                            role="assistant",
                            tool_calls=tool_calls
                        ),
                        index=choice.index,
                    )
                ],
            )


class LocalLLM(LLM):
    def __init__(self, assistant: Any = None) -> None:
        self.assistant = assistant
        super().__init__()

    def _build_function_description(self, fnc_info: FunctionInfo) -> dict:
        """Convert function definition to a format compatible with the LLM."""
        return {
            "name": fnc_info.name,
            "description": fnc_info.description,
            "callable": fnc_info.callable,
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        fnc_ctx: FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> MyLLMStream:
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        
        if fnc_ctx and len(fnc_ctx.ai_functions) > 0:
            fnc_desc = []
            for func in fnc_ctx.ai_functions.values():
                fnc_desc.append(self._build_function_description(func))
            
            generation_config["tools"] = fnc_desc
            if parallel_tool_calls is not None:
                generation_config["parallel_tool_calls"] = parallel_tool_calls

        return MyLLMStream(
            self,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            assistant=self.assistant,
        )
