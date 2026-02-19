import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    mcp,
)
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


logger = logging.getLogger("calendar-agent")


class CalendarAgent(Agent):
    def __init__(
        self, chat_ctx: ChatContext, tools: Optional[List[function_tool]] = None
    ) -> None:

        final_instructions = (
            "You are a Calendar specialist. You can help with scheduling, creating, modifying, or "
            "querying calendar events, appointments, and meetings. "
            "Use tools like 'create_calendar_event', 'get_calendar_events', etc., when available. "
            "If your task is complete or the user asks for something outside your calendar capabilities "
            "(e.g., math, web search), you MUST use the 'delegate_to_router_agent' tool to return to "
            "the main assistant."
        )

        all_tools = tools if tools is not None else []

        mcp_servers_list = []
        gumloop_mcp_url = os.getenv("GUMLOOP_CALENDAR_MCP_URL")
        if gumloop_mcp_url:
            mcp_servers_list.append(
                mcp.MCPServerHTTP(
                    url=gumloop_mcp_url,
                    timeout=5,
                    client_session_timeout_seconds=5,
                )
            )

        super().__init__(
            instructions=final_instructions,
            chat_ctx=chat_ctx,
            allow_interruptions=True,
            tools=all_tools,
            mcp_servers=mcp_servers_list,
        )

    async def on_enter(self):
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="ash"),
        turn_detection=MultilingualModel(),
    )

    await session.start(agent=CalendarAgent(chat_ctx=session._chat_ctx), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
