import logging
import os
import sys  # Import sys for sys.exit
from pathlib import Path

# Import the original FunctionAgents from the official agent files
# These files should be in the same directory as router_agent.py
from ask_agent import AskAgent
from calc_agent import CalculatorAgent
from calendar_agent import CalendarAgent
from caller_agent import CallerAgent
from contact_agent import ContactAgent
from dotenv import load_dotenv
from gmail_agent import GmailAgent
from go_agent import GoAgent
from listen_agent import ListenAgent
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
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# from mem_agent import MemoryAgent

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = logging.getLogger("router-agent")

# Determine the absolute path for server scripts relative to this file
_current_dir = os.path.dirname(os.path.abspath(__file__))


@function_tool
async def delegate_to_router_agent(
    context: RunContext,
    original_query: str = "User wants to talk about something else.",
):
    """
    Call this function to delegate the conversation back to the main RouterAgent.
    This is used when your current task is complete, or the user asks for functionality
    that you (the specialist agent) do not provide.
    Args:
        original_query: A brief description of why the delegation is happening or the user's last relevant query.
    """
    logger.info(
        f"Specialist Agent: Delegating back to RouterAgent. Reason/Query: '{original_query}'"
    )
    # Try to access _chat_ctx via context.session, as context.agent was problematic
    if not hasattr(context, "session") or context.session is None:
        logger.error(
            "delegate_to_router_agent: RunContext does not have a valid 'session' attribute."
        )
        # This is a critical failure for context propagation.
        # Depending on desired behavior, could raise an error or attempt a recovery (though recovery is hard here).
        # For now, we'll let it fail if it tries to access _chat_ctx on a None session,
        # or re-raise a more specific error.
        raise AttributeError(
            "RunContext is missing the session attribute, cannot retrieve ChatContext."
        )

    return (
        RouterAgent(chat_ctx=context.session._chat_ctx),
        "Okay, let me switch you back to the main assistant.",
    )


class RouterAgent(Agent):
    """Routes user queries to specialized agents."""

    def __init__(self, chat_ctx: ChatContext):
        super().__init__(
            instructions="""
                You are a router agent. Your primary responsibility is to understand the user's voice query
                and delegate it to the most appropriate specialist agent.
                - If the query is primarily about mathematics, calculations, arithmetic, or numbers,
                  you MUST use the 'delegate_to_math_agent' tool.
                - For general knowledge questions, facts, explanations, requests to 'search the web',
                'make a web search', or any other type of query not strictly mathematical, not about
                specific addresses/locations, and not covered by other specialists,
                  you MUST use the 'delegate_to_perplexity_agent' tool.
                - If the query involves calendar events, scheduling, creating appointments, or asking
                about your schedule, you MUST use the 'delegate_to_calendar_agent' tool.
                - If the user explicitly asks to make a phone call,
                  you MUST use the 'delegate_to_caller_agent' tool.
                - If the query is about finding contact information (like phone numbers or email
                addresses of people), you MUST use the 'delegate_to_contact_agent' tool.
                - For tasks related to managing emails (reading, sending, searching Gmail),
                  you MUST use the 'delegate_to_gmail_agent' tool.
                - If the query is about locations, finding places, getting directions, looking up
                addresses, or anything map-related, you MUST use the 'delegate_to_go_agent' tool.
                - If the user wants to play music, control music playback, or anything related to Spotify,
                  you MUST use the 'delegate_to_listen_agent' tool.
                Listen carefully to the user's query and make a clear decision.
                Do not attempt to answer the question yourself. Your sole job is to route.
                If uncertain, you can ask one clarifying question to determine the correct agent,
                but prefer to route directly if possible.
            """,
            allow_interruptions=True,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        """Called when the RouterAgent starts. It will wait for user input."""
        logger.info("RouterAgent entered. Waiting for user query.")
        self.session.generate_reply()

    @function_tool
    async def delegate_to_math_agent(self, query: str):
        """
        Call this function to delegate a math-related query to the MathSpecialistAgent.
        Args:
            query: The user's original voice query that is mathematical in nature.
        """
        logger.info(
            f"RouterAgent: Delegating to MathSpecialistAgent for query: '{query}'"
        )
        # Pass the delegate_to_router_agent tool to the CalculatorAgent
        math_agent = CalculatorAgent(
            chat_ctx=self.session._chat_ctx,
            tools=[delegate_to_router_agent],  # Pass the tool
        )
        return math_agent, "Okay, I'll connect you with my math specialist for that."

    @function_tool
    async def delegate_to_perplexity_agent(self, query: str):
        """
        Call this function to delegate a query that needs to perform a web search to the Perplexity Agent.
        Args:
            query: The user's original voice query.
        """
        logger.info(
            f"RouterAgent: Delegating to AskAgent (for perplexity tasks) for query: '{query}'"
        )
        try:
            perplexity_agent = AskAgent(
                chat_ctx=self.session._chat_ctx,
                tools=[delegate_to_router_agent],  # Pass the tool
            )
            return (
                perplexity_agent,
                "Alright, let me get my knowledge expert to help with that question.",
            )
        except AttributeError as e:
            logger.error(f"Unexpected AttributeError: {e}")
            raise

    @function_tool
    async def delegate_to_calendar_agent(self, query: str):
        """
        Call this function to delegate a query about calendar events, scheduling, or appointments to the CalendarAgent.
        Args:
            query: The user's original voice query related to calendar.
        """
        logger.info(f"RouterAgent: Delegating to CalendarAgent for query: '{query}'")
        calendar_agent = CalendarAgent(
            chat_ctx=self.session._chat_ctx,
            tools=[delegate_to_router_agent],  # Pass the tool
        )
        return calendar_agent, "Okay, let me check your calendar."

    @function_tool
    async def delegate_to_caller_agent(self, query: str):
        """
        Call this function to delegate a request to make a phone call to the CallerAgent.
        Args:
            query: The user's original voice query about making a call.
        """
        logger.info(f"RouterAgent: Delegating to CallerAgent for query: '{query}'")
        caller_agent = CallerAgent(
            chat_ctx=self.session._chat_ctx,
            tools=[delegate_to_router_agent],  # Pass the tool
        )
        return caller_agent, "Sure, I can try to make that call for you."

    @function_tool
    async def delegate_to_contact_agent(self, query: str):
        """
        Call this function to delegate a query about finding or managing contact information to the ContactAgent.
        Args:
            query: The user's original voice query related to contacts.
        """
        logger.info(f"RouterAgent: Delegating to ContactAgent for query: '{query}'")
        contact_agent = ContactAgent(
            chat_ctx=self.session._chat_ctx,
            tools=[delegate_to_router_agent],  # Pass the tool
        )
        return contact_agent, "Let me look up that contact information for you."

    @function_tool
    async def delegate_to_gmail_agent(self, query: str):
        """
        Call this function to delegate an email-related query (reading, sending, managing emails) to the GmailAgent.
        Args:
            query: The user's original voice query related to Gmail.
        """
        logger.info(f"RouterAgent: Delegating to GmailAgent for query: '{query}'")
        gmail_agent = GmailAgent(
            chat_ctx=self.session._chat_ctx,
            tools=[delegate_to_router_agent],  # Pass the tool
        )
        return gmail_agent, "Okay, I'll check your emails."

    @function_tool
    async def delegate_to_go_agent(self, query: str):
        """
        Call this function to delegate a query about locations, directions, maps, or places to the GoAgent.
        Args:
            query: The user's original voice query related to maps or navigation.
        """
        logger.info(f"RouterAgent: Delegating to GoAgent for query: '{query}'")
        go_agent = GoAgent(
            chat_ctx=self.session._chat_ctx,
            tools=[delegate_to_router_agent],  # Pass the tool
        )
        return go_agent, "Let me get my navigation expert for that."

    @function_tool
    async def delegate_to_listen_agent(self, query: str):
        """
        Call this function to delegate a request to play or control music (Spotify) to the ListenAgent.
        Args:
            query: The user's original voice query related to music or Spotify.
        """
        logger.info(f"RouterAgent: Delegating to ListenAgent for query: '{query}'")
        listen_agent = ListenAgent(
            chat_ctx=self.session._chat_ctx,
            tools=[delegate_to_router_agent],  # Pass the tool
        )
        return listen_agent, "Okay, let's get some music playing."


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the multi-agent LiveKit application."""
    await ctx.connect()
    logger.info("Router agent connected to LiveKit.")

    session = AgentSession[None](
        vad=silero.VAD.load(),
        stt=openai.STT(model="gpt-4o-mini-transcribe", detect_language=True),
        tts=openai.TTS(voice="alloy", model="tts-1-hd"),
        llm=openai.LLM(model="gpt-4o"),
        turn_detection=MultilingualModel(),
    )
    logger.info(
        "AgentSession configured. MCP servers will be managed by individual specialist agents."
    )

    initial_agent = RouterAgent(chat_ctx=session._chat_ctx)
    await session.start(agent=initial_agent, room=ctx.room)
    logger.info("RouterAgent session started.")


if __name__ == "__main__":
    # Setup basic logging if running directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    except SystemExit:  # Allow sys.exit() to pass through without logging as critical
        raise
    except Exception as e:
        logger.critical(f"Unhandled exception at top level: {e}", exc_info=True)
        sys.exit(1)  # Ensure exit with error code
