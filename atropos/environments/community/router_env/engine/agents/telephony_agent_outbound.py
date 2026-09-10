import json
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
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

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

    async def on_enter(self):
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # If a phone number was provided, then place an outbound call
    # By having a condition like this, you can use the same agent for inbound/outbound telephony
    # as well as web/mobile/etc.
    dial_info = json.loads(ctx.job.metadata)
    phone_number = dial_info["phone_number"]

    # The participant's identity can be anything you want, but this example uses the phone number itself
    sip_participant_identity = phone_number
    if phone_number is not None:
        # The outbound call will be placed after this method is executed
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    # This ensures the participant joins the correct room
                    room_name=ctx.room.name,
                    # This is the outbound trunk ID to use (i.e. which phone number the call will come from)
                    # You can get this from LiveKit CLI with `lk sip outbound list`
                    sip_trunk_id=os.environ.get("TWILIO_SIP_TRUNK_ID"),
                    # The outbound phone number to dial and identity to use
                    sip_call_to=phone_number,
                    participant_identity=sip_participant_identity,
                    # This will wait until the call is answered before returning
                    wait_until_answered=True,
                )
            )

            print("call picked up successfully")
        except api.TwirpError as e:
            print(
                f"error creating SIP participant: {e.message}, "
                f"SIP status: {e.metadata.get('sip_status_code')} "
                f"{e.metadata.get('sip_status')}"
            )
            ctx.shutdown()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="ash"),
        turn_detection=MultilingualModel(),
    )

    await session.start(agent=MyAgent(chat_ctx=session._chat_ctx), room=ctx.room)

    if phone_number is None:
        await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, agent_name="my-telephony-agent")
    )
