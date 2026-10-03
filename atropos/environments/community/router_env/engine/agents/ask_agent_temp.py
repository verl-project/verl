import asyncio
import logging
import os

from dotenv import load_dotenv
from livekit.agents import mcp
from livekit.agents.llm import LLM, ChatContext  # Removed ChatRole as using strings
from livekit.plugins import openai

logger = logging.getLogger("text-perplexity-agent")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


# --- Configure Perplexity MCP Server (as a function to allow async context management) ---
def get_perplexity_mcp_server():
    if os.environ.get("PERPLEXITY_API_KEY"):
        mcp_script_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "tools",
                "mcp",
                "perplexity",
                "perplexity-ask",
                "dist",
                "index.js",
            )
        )

        if not os.path.exists(mcp_script_path):
            logger.error(
                f"❌ MCP script not found at {mcp_script_path}. Make sure you've run "
                "'npm install && npm run build' in the server directory."
            )
            logger.warning("⚠️ Perplexity tools will be unavailable.")
            return None
        else:
            logger.info(
                f"📂 Configuring Perplexity MCP server with script: {mcp_script_path}"
            )
            return mcp.MCPServerStdio(
                name="PerplexityStdioServer",
                params={
                    "command": "node",
                    "args": [mcp_script_path],
                    "cwd": os.path.dirname(mcp_script_path),
                    "env": {
                        "PERPLEXITY_API_KEY": os.environ.get("PERPLEXITY_API_KEY") or ""
                    },
                    "client_session_timeout_seconds": 30,
                },
                client_session_timeout_seconds=30,
            )
    else:
        logger.warning(
            "⚠️ PERPLEXITY_API_KEY not set. Perplexity tools will be unavailable."
        )
        return None


async def run_chat_loop(
    llm_instance: LLM,
    p_mcp_server: mcp.MCPServerStdio | None,
    initial_question: str = None,
):
    """Runs a text-based chat loop with the LLM and Perplexity tool."""

    chat_context = ChatContext()

    system_prompt = """
            You are a specialized assistant for answering general knowledge questions, providing explanations,
            and performing web searches using the 'perplexity_ask' tool.
            When the user asks for information, facts, or to 'search the web', you are the designated expert.
            When calling the 'perplexity_ask' tool, ensure the 'messages' argument is an array containing
            a single object with 'role': 'user' and 'content' set to the user's question.
            For example: {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
            You do not have other tools. Do not try to delegate.
        """
    chat_context.add_message(role="system", content=system_prompt)

    async def process_question(question: str):
        logger.info(f"You: {question}")
        chat_context.add_message(role="user", content=question)

        full_response = ""
        logger.info("Agent:")

        mcp_servers_to_use = []
        if p_mcp_server:
            # MCPServerStdio is managed by async with in main, so it should be running
            mcp_servers_to_use.append(p_mcp_server)
            logger.info("Perplexity MCP Server is available for this query.")

        try:
            logger.info(f"DEBUG: Type of chat_context: {type(chat_context)}")
            logger.info(f"DEBUG: Attributes of chat_context: {dir(chat_context)}")
            # Pass messages from ChatContext and the list of mcp_servers
            async for chunk in llm_instance.chat(
                messages=chat_context.messages, mcp_servers=mcp_servers_to_use
            ):
                if chunk.delta.content:
                    print(chunk.delta.content, end="", flush=True)
                    full_response += chunk.delta.content
                if chunk.delta.tool_calls:
                    logger.info(f"\n[Tool call detected: {chunk.delta.tool_calls}]")
        except Exception as e:
            logger.error(f"Error during LLM chat: {e}")
            print(f"Sorry, I encountered an error: {e}")
            return

        print()
        chat_context.add_message(role="assistant", content=full_response)

    if initial_question:
        await process_question(initial_question)

    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Exiting chat.")
                break
            if not user_input.strip():
                continue
            await process_question(user_input)
        except KeyboardInterrupt:
            logger.info("\nExiting chat due to interrupt.")
            break
        except EOFError:
            logger.info("\nExiting chat due to EOF.")
            break


async def main():
    """Main entrypoint for the text-based Perplexity agent."""
    logger.info("Starting Text-based Perplexity Agent...")

    llm_instance = openai.LLM(model="gpt-4o")

    p_mcp_server_instance = get_perplexity_mcp_server()

    test_question = "What is the capital of France?"

    if p_mcp_server_instance:
        try:
            # await p_mcp_server_instance.connect() # Connect to MCP server -> Removed
            logger.info(
                "Perplexity MCP Server instance created. Will be used by LLM if needed."
            )
            await run_chat_loop(
                llm_instance, p_mcp_server_instance, initial_question=test_question
            )
        finally:
            logger.info(
                "Closing Perplexity MCP server resources."
            )  # Changed log message
            await p_mcp_server_instance.aclose()  # Close MCP server connection
    else:
        logger.warning("Running chat loop without Perplexity MCP server.")
        await run_chat_loop(llm_instance, None, initial_question=test_question)

    logger.info("Text-based Perplexity Agent finished.")


if __name__ == "__main__":
    if not os.environ.get("PERPLEXITY_API_KEY"):
        logger.error("🔴 PERPLEXITY_API_KEY is not set in the environment.")
        logger.error(
            "🔴 Please set it in your .env file for the agent to function correctly with Perplexity."
        )

    if os.environ.get("PERPLEXITY_API_KEY"):
        mcp_script_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "tools",
                "mcp",
                "perplexity",
                "perplexity-ask",
                "dist",
                "index.js",
            )
        )
        if not os.path.exists(mcp_script_path):
            logger.error(f"❌ Critical: MCP script not found at {mcp_script_path}.")
            logger.error(
                "❌ The agent cannot use Perplexity tools. Please build the MCP server "
                "('npm install && npm run build' in its directory)."
            )
            exit(1)

    asyncio.run(main())
