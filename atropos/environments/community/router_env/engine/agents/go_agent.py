import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, function_tool, mcp
from livekit.agents.llm import ChatChunk, ChatContext
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("go-agent-livekit")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

if not OPENAI_API_KEY:
    logger.critical("🔴 CRITICAL: OPENAI_API_KEY not found. OpenAI plugins will fail.")

if not GOOGLE_MAPS_API_KEY:
    logger.critical(
        "🔴 CRITICAL: GOOGLE_MAPS_API_KEY not found. Google Maps MCP server will fail."
    )

if not DEEPGRAM_API_KEY:
    logger.warning(
        "⚠️ WARNING: DEEPGRAM_API_KEY not found. Deepgram STT plugin may have issues."
    )


mcp_script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "tools",
        "mcp",
        "google-maps",
        "dist",
        "index.js",
    )
)

if not os.path.exists(mcp_script_path):
    logger.critical(
        f"CRITICAL: Google Maps MCP script not found at {mcp_script_path}. Agent cannot start tools."
    )


class GoAgent(Agent):
    """A LiveKit agent specialized in location-based queries using Google Maps via MCP."""

    def __init__(
        self, chat_ctx: ChatContext, tools: Optional[List[function_tool]] = None
    ):

        final_instructions = (
            "You are the Go Agent, specialized in providing location-based information using Google Maps. "
            "You MUST use the available tools to fulfill user queries about locations, directions, "
            "distances, and places.\n\n"
            "RULE FOR LOCATION REQUESTS: When a user asks about finding a location, getting directions, "
            "calculating distances, or information about a place, you MUST use the appropriate Google Maps tool.\n\n"
            "Key tools available to you (provided by Google Maps MCP):\n"
            "- maps_geocode: Convert an address to coordinates "
            '(e.g., maps_geocode address="1600 Amphitheatre Parkway, Mountain View, CA")\n'
            "- maps_reverse_geocode: Convert coordinates to an address "
            "(e.g., maps_reverse_geocode latitude=37.422 longitude=-122.084)\n"
            '- maps_search_places: Search for places (e.g., maps_search_places query="restaurants in London")\n'
            "- maps_place_details: Get details for a place_id "
            '(e.g., maps_place_details place_id="ChIJN1t_tDeuEmsRUsoyG83frY4")\n'
            "- maps_directions: Get directions "
            '(e.g., maps_directions origin="San Francisco" destination="Los Angeles" mode="driving")\n'
            "- maps_distance_matrix: Calculate distances "
            '(e.g., maps_distance_matrix origins="New York,Washington D.C." '
            'destinations="Boston,Philadelphia" mode="...")\n\n'
            "RULE FOR TOOL RESULTS: After you receive results from a tool, you MUST analyze the data and "
            "provide a clear, helpful response. Format addresses and directions in a readable way, "
            "extract key information from place details, and always provide context for coordinates and distances.\n\n"
            "If a tool call fails or returns no relevant information, explain clearly to the user and "
            "suggest alternatives. "
            "If your task is complete or the user asks for something outside your location/maps capabilities "
            "(e.g., math, calendar), you MUST use the 'delegate_to_router_agent' tool to return to the main assistant."
        )

        all_tools = tools if tools is not None else []

        mcp_servers_list = []
        if GOOGLE_MAPS_API_KEY and os.path.exists(mcp_script_path):
            mcp_servers_list.append(
                mcp.MCPServerStdio(
                    command="node",
                    args=[mcp_script_path],
                    env={"GOOGLE_MAPS_API_KEY": GOOGLE_MAPS_API_KEY},
                )
            )
        else:
            logger.warning(
                "Google Maps MCP server not configured due to missing API key or script path."
            )

        super().__init__(
            instructions=final_instructions,
            allow_interruptions=True,
            chat_ctx=chat_ctx,
            mcp_servers=mcp_servers_list,
            tools=all_tools,
        )
        if not self.llm:
            logger.error(
                "GoAgentLivekit initialized, but LLM might be missing if API key was not "
                "provided to plugin."
            )

    async def llm_node(self, chat_ctx: ChatContext, tools: list, model_settings: dict):
        """Override the llm_node to log tool calls or add custom behavior."""
        tool_call_detected_this_turn = False
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            if (
                isinstance(chunk, ChatChunk)
                and chunk.delta
                and chunk.delta.tool_calls
                and not tool_call_detected_this_turn
            ):
                tool_call_detected_this_turn = True
                logger.info(
                    "GoAgentLivekit: LLM is attempting to call a tool. Informing user."
                )
                if hasattr(self, "session") and self.session is not None:
                    self.session.say("Okay, let me check that for you.")
                else:
                    logger.warning(
                        "Agent has no session to 'say' through during tool call detection."
                    )
            yield chunk

    async def on_enter(self):
        # when the agent is added to the session, we'll initiate the conversation by
        # using the LLM to generate a reply
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit Go Agent application."""
    logger.info(
        f"Go Agent LiveKit starting entrypoint for Job ID: {getattr(ctx.job, 'id', 'unknown')}"
    )

    await ctx.connect()
    logger.info(
        f"Successfully connected to LiveKit room: {ctx.room.name if ctx.room else 'N/A'}"
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(
            model="nova-2", language="en-US", api_key=os.environ.get("DEEPGRAM_API_KEY")
        ),
        llm=openai.LLM(model="gpt-4o", api_key=OPENAI_API_KEY),
        tts=openai.TTS(voice="alloy", api_key=OPENAI_API_KEY),
        turn_detection=MultilingualModel(),
    )
    logger.info("AgentSession configured with Google Maps MCP server.")

    agent = GoAgent(chat_ctx=session._chat_ctx)
    logger.info("GoAgentLivekit instantiated.")

    logger.info(
        f"Starting AgentSession with agent for room: {ctx.room.name if ctx.room else 'N/A'}"
    )
    await session.start(agent=agent, room=ctx.room)
    logger.info("AgentSession started. GoAgentLivekit is now running.")


if __name__ == "__main__":
    logger.info("Starting Go Agent LiveKit application via cli.run_app.")
    if not os.environ.get("DEEPGRAM_API_KEY"):
        logger.warning(
            "DEEPGRAM_API_KEY not found in environment. STT plugin may fail."
        )

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
