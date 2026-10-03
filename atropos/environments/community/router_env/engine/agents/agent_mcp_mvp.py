import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, mcp
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


logger = logging.getLogger("mcp-agent")

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You can retrieve data via the MCP server. The interface is voice-based: "
                "accept spoken user queries and respond with synthesized speech."
            ),
            vad=silero.VAD.load(),
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="ash"),
            turn_detection=MultilingualModel(),
            mcp_servers=[
                mcp.MCPServerHTTP(
                    url=(
                        "https://mcp.gumloop.com/gcalendar/"
                        "cY3bcaFS1qNdeVBnj0XIhnP4FEp2%3Aae99858e75594251bea9e05f32bb99b3"
                    ),
                    timeout=5,
                    client_session_timeout_seconds=5,
                ),
            ],
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

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="mcp-agent"))
