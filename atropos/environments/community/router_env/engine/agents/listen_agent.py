import logging
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    ChatContext,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    mcp,
    tts,
)
from livekit.agents.llm import ChatChunk
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pydantic import BaseModel, Field

logger = logging.getLogger("agent-spotify-official")

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

_this_file_dir = os.path.dirname(os.path.abspath(__file__))
_stone_ui_dir = os.path.abspath(os.path.join(_this_file_dir, "..", ".."))
if _stone_ui_dir not in sys.path:
    sys.path.insert(0, _stone_ui_dir)

# Removed ANTHROPIC_API_KEY check as it seems unrelated to this OpenAI-based agent.

# --- Spotify Tool Input Models (Based on spotify-mcp-server README) ---


class PlayMusicInput(BaseModel):
    uri: Optional[str] = Field(
        None,
        description="Spotify URI of the item to play (e.g., spotify:track:...). Overrides type and id.",
    )
    type: Optional[str] = Field(
        None, description="Type of item to play (track, album, artist, playlist)"
    )
    id: Optional[str] = Field(None, description="Spotify ID of the item to play")
    deviceId: Optional[str] = Field(
        None, description="ID of the device to play on (optional)"
    )


# Add other input models here as needed (e.g., SearchSpotifyInput, PlaylistInput etc.)

# --- Configure Spotify MCP Server ---
spotify_mcp_server = None

# Define the path to the BUILT MCP server script
# IMPORTANT: Ensure the MCP server is built (npm run build) and authenticated (npm run auth)
mcp_script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "tools", "mcp", "spotify", "build", "index.js"
    )
)
spotify_config_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "tools",
        "mcp",
        "spotify",
        "spotify-config.json",
    )
)

if not os.path.exists(mcp_script_path):
    logger.error(
        f"❌ Spotify MCP script not found at {mcp_script_path}. Make sure you've run "
        "'npm install && npm run build' in the server directory."
    )
    logger.warning("⚠️ Spotify tools will be unavailable.")
elif not os.path.exists(spotify_config_path):
    logger.error(
        f"❌ Spotify config file not found at {spotify_config_path}. Make sure you've run "
        "'npm run auth' after setting credentials."
    )
    logger.warning("⚠️ Spotify tools will likely be unavailable due to missing auth.")
else:
    # Check if config contains tokens (basic check)
    try:
        with open(spotify_config_path, "r") as f:
            config_content = f.read()
            if (
                "accessToken" not in config_content
                or "refreshToken" not in config_content
                or "run-npm auth" in config_content
            ):
                logger.warning(
                    f"⚠️ Spotify config file at {spotify_config_path} seems incomplete or "
                    "unauthenticated. Run 'npm run auth'."
                )
                # We still configure the server, but it might fail at runtime
            else:
                logger.info("✅ Spotify config file seems authenticated.")
    except Exception as e:
        logger.error(f"Error reading Spotify config {spotify_config_path}: {e}")

    logger.info(f"📂 Configuring Spotify MCP server with script: {mcp_script_path}")
    spotify_mcp_server = mcp.MCPServerStdio(
        "node",  # Command to run the server
        args=[mcp_script_path],  # Argument is the script path
        # No specific env vars needed here, reads from spotify-config.json
        env={},
        client_session_timeout_seconds=5 * 60,
    )
    logger.info("✅ Spotify MCP Server configured (runtime auth check still needed).")


class ListenAgent(Agent):
    """A LiveKit agent that uses MCP tools from one or more MCP servers."""

    def __init__(
        self,
        chat_ctx: ChatContext,
        instructions: Optional[str] = None,
        tts: Optional[tts.TTS] = NOT_GIVEN,
        tools: Optional[List[function_tool]] = None,
    ):

        final_instructions = (
            instructions
            if instructions is not None
            else (
                "You are the Listen Agent, specialized in controlling Spotify music playback. "
                + "You MUST use the available tools to fulfill user requests related to Spotify. "
                + "Available tools include 'playMusic', and potentially others like 'searchSpotify', "
                "'pausePlayback', etc.\n\n"
                + "RULE FOR MUSIC REQUESTS: When a user asks to play music, search for music, "
                "control playback (pause, skip, etc.), manage playlists, or ask what's playing, "
                "you MUST use the appropriate Spotify tool (like 'playMusic'). "
                + "Be precise with parameters like 'uri' or 'type' and 'id'. "
                "Infer parameters from the user query. If essential info is missing "
                "(like what to play), ask the user.\n\n"
                + "RULE FOR TOOL RESULTS: After a tool is successfully executed, you MUST confirm "
                "the action to the user "
                "(e.g., 'Okay, playing 'Bohemian Rhapsody' now.'). "
                + "If a tool fails or returns an error, inform the user clearly. "
                + "If your task is complete or the user asks for something outside your Spotify capabilities "
                "(e.g., math, calendar), you MUST use the 'delegate_to_router_agent' tool to return to the "
                "main assistant."
            )
        )

        all_tools = tools if tools is not None else []

        active_mcp_servers = []
        if spotify_mcp_server is not None:
            active_mcp_servers.append(spotify_mcp_server)

        super().__init__(
            instructions=final_instructions,
            chat_ctx=chat_ctx,
            allow_interruptions=True,
            mcp_servers=active_mcp_servers,  # MODIFIED: Pass filtered list
            tools=all_tools,  # Pass the tools to the parent Agent class
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Override the llm_node to say a message when a tool call is detected."""
        tool_call_detected = False

        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            if (
                isinstance(chunk, ChatChunk)
                and chunk.delta
                and chunk.delta.tool_calls
                and not tool_call_detected
            ):
                tool_call_detected = True
                # Use self.session.say() to make the agent speak, only if TTS is configured
                if self.tts:  # Check if the agent has a TTS instance
                    self.session.say("Sure, I'll check that for you.")
            yield chunk

    async def on_enter(self):
        # when the agent is added to the session, we'll initiate the conversation by
        # using the LLM to generate a reply
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent application."""
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-2", language="en-US"),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(voice="alloy"),
        turn_detection=MultilingualModel(),
    )

    agent = ListenAgent(chat_ctx=session._chat_ctx)
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
