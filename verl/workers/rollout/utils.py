# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import math

import numpy as np
import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


def get_minimum_bucket_size_mb(hf_config, current_bucket_size_mb: int) -> int:
    """
    Calculate the minimum required bucket size in MB based on the embedding weight size.

    The embedding weight (embed_tokens) is typically the largest single weight tensor
    in a model. The bucket size must be larger than any single weight tensor to avoid
    AssertionError during weight transfer.

    For multimodal models (e.g. Qwen3-VL), vocab_size and hidden_size may be nested
    under text_config instead of the top-level config.

    Args:
        hf_config: HuggingFace model config object.
        current_bucket_size_mb: Current bucket size in MB.

    Returns:
        Adjusted bucket size in MB, guaranteed to fit the embedding weight.
    """
    # For multimodal models, vocab_size/hidden_size may be in text_config
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        vocab_size = getattr(text_config, "vocab_size", 0)
        hidden_size = getattr(text_config, "hidden_size", 0)
    else:
        vocab_size = getattr(hf_config, "vocab_size", 0)
        hidden_size = getattr(hf_config, "hidden_size", 0)

    if not (vocab_size and hidden_size):
        return current_bucket_size_mb

    # embed_tokens: [vocab_size, hidden_size] in float32 = 4 bytes
    embed_size_mb = math.ceil(vocab_size * hidden_size * 4 / 1024 / 1024)

    if embed_size_mb <= current_bucket_size_mb:
        return current_bucket_size_mb

    # round up to next 512MB boundary
    recommended_mb = (embed_size_mb // 512 + 1) * 512
    logger.warning(
        f"Embedding weight size ({embed_size_mb} MB) exceeds "
        f"update_weights_bucket_megabytes ({current_bucket_size_mb} MB), "
        f"automatically increasing to {recommended_mb} MB."
    )
    return recommended_mb


class _UvicornServerAutoPort(uvicorn.Server):
    """Uvicorn Server that reports the system-assigned port when port=0."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.actual_port: int | None = None
        self._startup_done: asyncio.Event = asyncio.Event()

    async def startup(self, sockets=None) -> None:
        try:
            await super().startup(sockets=sockets)
            if self.servers and self.config.port == 0:
                sock = self.servers[0].sockets[0]
                self.actual_port = sock.getsockname()[1]
            else:
                self.actual_port = self.config.port
        finally:
            self._startup_done.set()

    async def get_port(self) -> int | None:
        await self._startup_done.wait()
        return self.actual_port


async def run_uvicorn(app: FastAPI, server_args, server_address) -> tuple[int, asyncio.Task]:
    app.server_args = server_args
    config = uvicorn.Config(app, host=server_address, port=0, log_level="warning")
    server = _UvicornServerAutoPort(config)
    server_task = asyncio.create_task(server.serve())
    server_port = await server.get_port()
    if server_port is None:
        # server.startup() failed. await the task to re-raise exception from server.serve()
        await server_task

        # Fails on unexpected situation.
        raise RuntimeError("Unexpected: HTTP server started without reporting listened port")
    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


def qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> and <|video_pad|> token by image_data.
    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if (
        processor is not None
        and hasattr(processor, "image_processor")
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    ):
        prompt_ids = np.array(prompt_ids)
        mask = np.ones(len(prompt_ids), dtype=bool)
        is_value = (prompt_ids == processor.image_token_id) | (prompt_ids == processor.video_token_id)
        mask[1:] &= ~(is_value[1:] & is_value[:-1])
        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
