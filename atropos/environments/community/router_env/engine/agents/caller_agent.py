import logging
import os
import random
from typing import List, Optional

from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger("caller-agent")


class CallerAgent(Agent):
    def __init__(
        self, chat_ctx: ChatContext, tools: Optional[List[function_tool]] = None
    ) -> None:

        final_instructions = (
            "You are a Caller specialist. Your primary function is to initiate phone calls. "
            + "If the user asks to call someone, use the 'make_phone_call' tool. "
            + "Currently, you can only call a predefined contact (Sam at +16467085301). "
            + "Confirm with the user if they want to call this specific contact. "
            + "If your task is complete or the user asks for something outside your calling "
            + "capabilities (e.g., math, web search), you MUST use the 'delegate_to_router_agent' "
            + "tool to return to the main assistant."
        )

        agent_tools = [self.make_phone_call]
        all_tools = agent_tools + (tools if tools is not None else [])

        super().__init__(
            instructions=final_instructions,
            chat_ctx=chat_ctx,
            allow_interruptions=True,
            tools=all_tools,
        )
        self.lkapi = api.LiveKitAPI()

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def make_phone_call(self, context: RunContext, phone_number: str):
        """
        Call this function to make a phone call to a user number.
        Args:
            phone_number: The phone number to call.
        """
        await self.lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="my-telephony-agent",
                room=f"outbound-{''.join(str(random.randint(0, 9)) for _ in range(10))}",
                metadata='{"phone_number": "+16467085301"}',  # HARDCODED
            )
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="ash"),
        turn_detection=MultilingualModel(),
    )

    await session.start(agent=CallerAgent(chat_ctx=session._chat_ctx), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="mcp-agent"))
