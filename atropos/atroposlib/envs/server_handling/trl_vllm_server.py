"""
This is a server that interfaces with trl's vLLM server.

Developed with much help from @winglian when they worked on integrating Atropos into Axolotl.
"""

import time
import uuid

import aiohttp
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from transformers import AutoTokenizer

from atroposlib.envs.server_handling.server_baseline import APIServer, APIServerConfig


class TrlVllmServer(APIServer):
    """
    A server that interfaces with trl's vLLM server.
    """

    def __init__(self, config: APIServerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        super().__init__(config)

    async def check_server_status_task(self, chat_completion: bool = True):
        """
        TODO: Implement server health check for trl's vLLM server
        """
        self.server_healthy = True

    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for the chat completion using the trl's vLLM server.
        """
        url = f"{self.config.base_url}/generate/"
        prompt = kwargs.get("messages", [])
        prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={
                    "prompts": [prompt],
                    "n": kwargs.get("n", 1),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                    "temperature": kwargs.get("temperature", 1.0),
                    "top_p": kwargs.get("top_p", 1.0),
                    "top_k": kwargs.get("top_k", -1),
                    "min_p": kwargs.get("min_p", 0.0),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                },
            ) as response:
                completions = await response.json()
        completions = ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=[
                Choice(
                    finish_reason=(
                        "stop"
                        if self.tokenizer.eos_token_id in completion
                        else "length"
                    ),
                    index=i,
                    message=ChatCompletionMessage(
                        content=self.tokenizer.decode(completion),
                        role="assistant",
                    ),
                )
                for i, completion in enumerate(completions["completion_ids"])
            ],
        )
        return completions

    async def _completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for the completion using the trl's vLLM server.
        """
        url = f"{self.config.base_url}/generate/"
        prompt = kwargs.get("prompt", "")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={
                    "prompts": [prompt],
                    "n": kwargs.get("n", 1),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                    "temperature": kwargs.get("temperature", 1.0),
                    "top_p": kwargs.get("top_p", 1.0),
                    "top_k": kwargs.get("top_k", -1),
                    "min_p": kwargs.get("min_p", 0.0),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                },
            ) as response:
                completions = await response.json()
        completions = ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=[
                Choice(
                    finish_reason=(
                        "stop"
                        if self.tokenizer.eos_token_id in completion
                        else "length"
                    ),
                    index=i,
                    message=ChatCompletionMessage(
                        content=self.tokenizer.decode(completion),
                        role="assistant",
                    ),
                )
                for i, completion in enumerate(completions["completion_ids"])
            ],
        )
        return completions
