import logging
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import deepgram, openai, silero
from mem0 import AsyncMemoryClient

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# Configure logging
logger = logging.getLogger("memory-assistant")
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Define a global user ID for simplicity
USER_ID = "voice_user"

# Initialize Mem0 memory client
mem0 = AsyncMemoryClient()


async def _enrich_with_memory(
    last_user_msg: llm.ChatMessage, chat_ctx_to_modify: llm.ChatContext
):
    """Add memories and Augment chat context with relevant memories"""
    if (
        not last_user_msg
        or not last_user_msg.text_content
        or not last_user_msg.text_content.strip()
    ):
        logger.info("No valid last user message content to process for memory.")
        return

    try:
        # Ensure last_user_msg.text_content is a string for mem0
        content_str = last_user_msg.text_content
        if not content_str or not content_str.strip():
            logger.info("User message content is empty after getting text_content.")
            return

        logger.info(
            f"[Mem0] Attempting to add memory for USER_ID '{USER_ID}': '{content_str}'"
        )
        try:
            add_response = await mem0.add(
                [{"role": "user", "content": content_str}], user_id=USER_ID
            )
            logger.info(f"[Mem0] Successfully added memory. Response: {add_response}")
        except Exception as e:
            logger.error(f"[Mem0] Error adding memory: {e}", exc_info=True)
            # Decide if we should return or continue to search with potentially stale memory
            # For now, we'll continue to search.

        logger.info(
            f"[Mem0] Attempting to search memories for USER_ID '{USER_ID}' with query: '{content_str}'"
        )
        results = []
        try:
            results = await mem0.search(
                content_str,
                user_id=USER_ID,
            )
            logger.info(
                f"[Mem0] Search complete. Found {len(results)} results: {results}"
            )
        except Exception as e:
            logger.error(f"[Mem0] Error searching memory: {e}", exc_info=True)

        if results:
            memories_text = " ".join(
                [result["memory"] for result in results if result.get("memory")]
            )
            if memories_text.strip():
                logger.info(f"Enriching with memory: {memories_text}")

                # Create the RAG message. Ensure content is a list of ChatContent (string is fine).
                rag_msg_content = (
                    f"Relevant Memory from past interactions: {memories_text}\\n"
                    "User's current query is below."
                )
                rag_msg = llm.ChatMessage(role="system", content=[rag_msg_content])

                # Insert RAG message before the last user message in the context's items list
                inserted = False
                # Access items via the .items property
                target_items_list = chat_ctx_to_modify.items
                for i in range(len(target_items_list) - 1, -1, -1):
                    if target_items_list[i] is last_user_msg:  # Check object identity
                        target_items_list.insert(i, rag_msg)
                        inserted = True
                        logger.info(f"Inserted RAG message at index {i} in .items list")
                        break

                if not inserted:
                    logger.warning(
                        "Could not find last user message by identity in .items list. "
                        "Appending RAG message."
                    )
                    if target_items_list and target_items_list[-1] is last_user_msg:
                        target_items_list.insert(len(target_items_list) - 1, rag_msg)
                    else:
                        target_items_list.append(rag_msg)

    except Exception as e:
        logger.error(f"Error during memory enrichment: {e}", exc_info=True)


class MemoryAgent(Agent):
    def __init__(self, chat_ctx: llm.ChatContext):
        super().__init__(
            chat_ctx=chat_ctx,
            instructions="You are a helpful voice assistant that can remember past interactions.",
        )
        # System prompt is now managed by the chat_ctx passed to super().__init__

    async def on_enter(self):
        logger.info("MemoryAgent entered room.")
        try:
            # Say initial greeting
            await self.session.say(
                "Hello! I'm George. Can I help you plan an upcoming trip? ",
                allow_interruptions=True,
            )
            # Start the main interaction loop
            self.session.generate_reply()
            logger.info("MemoryAgent started generate_reply loop.")
        except Exception as e:
            logger.error(f"Error in MemoryAgent.on_enter: {e}", exc_info=True)

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ):
        logger.info(
            f"MemoryAgent.on_user_turn_completed called with new_message: '{new_message.text_content}'"
        )

        if (
            not new_message
            or not new_message.content
            or not new_message.text_content.strip()
        ):
            logger.info("No valid new_message content for memory enrichment.")
            return

        # The turn_ctx provided by the hook is the context *before* the new_message.
        # We need to add the new_message to it before enrichment,
        # so _enrich_with_memory can potentially place the RAG message *before* it.
        # The AgentActivity will use this modified turn_ctx for the LLM call.
        # It will also separately add the new_message to the agent's main context.

        # Let's make a working copy if direct modification isn't intended for the passed turn_ctx,
        # though the name temp_mutable_chat_ctx in AgentActivity suggests it's okay.
        # For safety and clarity in _enrich_with_memory, we'll operate on turn_ctx.

        # Add the new user message to the context that will be enriched
        turn_ctx.items.append(
            new_message
        )  # new_message is already part of the main context by AgentActivity
        # but for _enrich_with_memory to find it (as last_user_msg)
        # and insert RAG before it in *this specific context copy*, it needs to be here.
        # AgentActivity also adds this new_message to the agent's _chat_ctx separately.

        logger.info(
            f"Context before enrichment (with new_message added): {turn_ctx.items}"
        )

        # Enrich the context (which now includes new_message) with memories
        # _enrich_with_memory will find new_message as the last user message
        # and insert the RAG system message just before it in turn_ctx.items
        await _enrich_with_memory(new_message, turn_ctx)

        logger.info(f"Context after enrichment: {turn_ctx.items}")
        # No need to call self.update_chat_ctx() here.
        # The AgentActivity will use the modified turn_ctx for the LLM.


def prewarm_process(proc: JobProcess):
    logger.info("Prewarming VAD model.")
    # Preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model prewarmed.")


async def entrypoint(ctx: JobContext):
    logger.info("Agent entrypoint started.")
    try:
        await ctx.connect()
        logger.info("Connected to LiveKit room.")

        # Define initial system context for the LLM
        initial_ctx = llm.ChatContext()
        system_prompt_text = """
            You are a helpful voice assistant.
            You are a travel guide named George and will help the user to plan a travel trip of their dreams.
            You should help the user plan for various adventures like work retreats, family vacations or
            solo backpacking trips.
            You should be careful to not suggest anything that would be dangerous, illegal or inappropriate.
            You can remember past interactions and use them to inform your answers.
            Use semantic memory retrieval to provide contextually relevant responses.
            When relevant memory is provided, use it to enhance your response.
            """
        initial_ctx.add_message(role="system", content=system_prompt_text)
        logger.info("Initial system context defined.")

        # VAD model loading logic remains the same
        vad_model = ctx.proc.userdata.get("vad")
        if not vad_model:
            logger.info("VAD not prewarmed or not found in userdata, loading now.")
            vad_model = silero.VAD.load()
        else:
            logger.info("Using prewarmed VAD model.")

        custom_agent = MemoryAgent(chat_ctx=initial_ctx)

        # AgentSession constructor does NOT take 'agent'
        session = AgentSession(
            vad=vad_model,
            stt=deepgram.STT(model="nova-2", language="en"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="alloy"),
        )
        logger.info("AgentSession created.")

        # Agent is passed to session.start()
        await session.start(agent=custom_agent, room=ctx.room)
        logger.info("Agent session started with MemoryAgent.")

    except Exception as e:
        logger.error(f"Error in agent entrypoint: {e}", exc_info=True)


# Run the application
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
            agent_name="mem0-voice-agent",
        )
    )  # Consistent agent name
