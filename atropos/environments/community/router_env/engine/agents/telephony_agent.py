import logging
import os
from pathlib import Path

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
    get_job_context,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel


# Add this function definition anywhere
async def hangup_call():
    ctx = get_job_context()
    if ctx is None:
        # Not running in a job context
        return

    await ctx.api.room.delete_room(
        api.DeleteRoomRequest(
            room=ctx.room.name,
        )
    )


load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


logger = logging.getLogger("mcp-agent")

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class MyAgent(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            instructions=(
                "You can have phone calls. The interface is voice-based: "
                "accept spoken user queries and respond with synthesized speech."
            ),
            chat_ctx=chat_ctx,
        )

    @function_tool
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        # let the agent finish speaking
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

        await hangup_call()

    @function_tool
    async def end_call_finished_by_you(self, ctx: RunContext):
        """Called when you have accomplished your task and can end the call safely"""
        # let the agent finish speaking
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

        await hangup_call()

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

    await session.start(agent=MyAgent(chat_ctx=session._chat_ctx), room=ctx.room)

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, agent_name="my-telephony-agent")
    )
