import logging  # Added logging
import os
from typing import List, Optional  # Add Optional & List import

from livekit.agents import mcp  # Corrected import for mcp
from livekit.agents import tts  # Corrected import for tts module
from livekit.agents import (  # Changed import; Add ChatContext import
    ChatContext,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import (  # Added function_tool for delegate_to_router_agent if it were defined here
    ChatChunk,
    function_tool,
)
from livekit.agents.types import NOT_GIVEN  # Corrected import for NOT_GIVEN
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, silero

# Removed: from mcp_client import MCPServerStdio
# Removed: from mcp_client.agent_tools import MCPToolsIntegration
from livekit.plugins.turn_detector.multilingual import (  # Added from official example
    MultilingualModel,
)

logger = logging.getLogger("agent-math-official")  # Added logger

mcp_script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "tools", "mcp", "calc", "calc_server.py"
    )
)


class CalculatorAgent(Agent):
    """A LiveKit agent that uses MCP tools from one or more MCP servers."""

    def __init__(
        self,
        chat_ctx: ChatContext,
        instructions: Optional[str] = None,
        mcp_servers: Optional[list[mcp.MCPServer]] = None,
        tts: Optional[tts.TTS] = NOT_GIVEN,
        tools: Optional[List[function_tool]] = None,
    ):  # Added tools parameter

        final_instructions = (
            instructions
            if instructions is not None
            else """
                You are a specialist Math assistant. Your expertise is in solving mathematical problems,
                performing calculations, arithmetic, and answering questions about numbers.
                You have two calculation tools: 'multiply' and 'add'.
                When your current math task is complete, or if the user asks for something not related to math,
                you MUST use the 'delegate_to_router_agent' tool to return to the main assistant.
            """
        )

        # Combine passed tools with any class-defined tools if necessary (none here for now)
        all_tools = tools if tools is not None else []

        super().__init__(
            instructions=final_instructions,
            chat_ctx=chat_ctx,
            allow_interruptions=True,
            mcp_servers=[
                mcp.MCPServerStdio(
                    command="python",
                    args=[mcp_script_path],
                )
                # MODIFIED: Removed chat_ctx=chat_ctx argument
            ],
            tools=all_tools,  # Pass the tools to the parent Agent class
        )
        # MCP tools are automatically integrated by AgentSession if mcp_servers is configured.
        # No need for MCPToolsIntegration or manually adding tools here.

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
                # Example: if self.tts: self.session.say("Working on the math problem.")
                # Currently, Math agent does not say anything here.
            yield chunk

    async def on_enter(self):
        # when the agent is added to the session, we'll initiate the conversation by
        # using the LLM to generate a reply
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent application."""
    await ctx.connect()  # Connect earlier as in official example

    # Directly configure AgentSession with mcp_servers
    session = AgentSession(
        vad=silero.VAD.load(),  # Redundant if agent has it, but official example does this
        stt=deepgram.STT(model="nova-2", language="en-US"),  # Consistent with agent
        llm=openai.LLM(model="gpt-4o"),  # Consistent with agent
        tts=openai.TTS(voice="alloy"),  # Consistent with agent
        turn_detection=MultilingualModel(),  # Consistent with agent
    )

    # Instantiate the agent
    agent = CalculatorAgent(chat_ctx=session._chat_ctx)

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
