# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import logging
from unittest.mock import patch

from packaging import version

logger = logging.getLogger(__name__)

_PATCHES = []
_READER_BUFFER_TOKENS = None


def _kv_cache_slot_tokens(kv_cache_config) -> int | None:
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    num_blocks = getattr(kv_cache_config, "num_blocks", None)
    if not groups or num_blocks is None:
        return None

    block_size = None
    for group in groups:
        spec = getattr(group, "kv_cache_spec", None)
        if type(spec).__name__ in ("AttentionSpec", "FullAttentionSpec"):
            block_size = getattr(spec, "block_size", None)
            break
    if block_size is None:
        block_size = max(group.kv_cache_spec.block_size for group in groups)

    return int(num_blocks) * int(block_size)


def apply_vllm_routed_experts_buffer_patch() -> None:
    """Backport vLLM's slot-sized routed-expert buffer for 0.20.x.

    vLLM 0.20.x stores routed experts by physical KV slot but sizes the shared
    buffer from request token capacity. Long generations can receive physical
    slot IDs above that request-sized buffer. vLLM 0.21 replaces this with a
    scheduler-side slot buffer; for 0.20.x we keep the existing API and enlarge
    the shared buffer consistently for both writer and reader processes.
    """

    if _PATCHES:
        return

    import vllm

    vllm_ver = version.parse(vllm.__version__)
    if vllm_ver < version.parse("0.20.0") or vllm_ver >= version.parse("0.21.0"):
        return

    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
        RoutedExpertsReader,
    )

    original_attach_buffer = RoutedExpertsReader.attach_buffer

    def attach_buffer(self, max_num_kv_tokens, vllm_config):
        buffer_tokens = max_num_kv_tokens
        if _READER_BUFFER_TOKENS is not None:
            buffer_tokens = max(buffer_tokens, _READER_BUFFER_TOKENS)
        return original_attach_buffer(
            self,
            buffer_tokens,
            vllm_config,
        )

    from vllm.v1.core.sched.scheduler import Scheduler

    original_scheduler_init = Scheduler.__init__

    def scheduler_init(self, vllm_config, kv_cache_config, *args, **kwargs):
        global _READER_BUFFER_TOKENS
        previous = _READER_BUFFER_TOKENS
        if vllm_config.model_config.enable_return_routed_experts:
            _READER_BUFFER_TOKENS = _kv_cache_slot_tokens(kv_cache_config)
        try:
            return original_scheduler_init(self, vllm_config, kv_cache_config, *args, **kwargs)
        finally:
            _READER_BUFFER_TOKENS = previous

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        routed_experts_capturer = RoutedExpertsCapturer.create()
        self.routed_experts_attn_gid = self._get_attention_kv_cache_gid()
        slot_tokens = _kv_cache_slot_tokens(self.kv_cache_config)
        if slot_tokens is None:
            slot_tokens = getattr(self, "max_num_kv_tokens", self.scheduler_config.max_num_batched_tokens)
        self.max_num_kv_tokens = slot_tokens

        routed_experts_capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            vllm_config=self.vllm_config,
        )
        self._bind_routed_experts_capturer(routed_experts_capturer)
        self.routed_experts_initialized = True

    patcher1 = patch(
        "vllm.model_executor.layers.fused_moe.routed_experts_capturer.RoutedExpertsReader.attach_buffer",
        attach_buffer,
    )
    patcher2 = patch(
        "vllm.v1.core.sched.scheduler.Scheduler.__init__",
        scheduler_init,
    )
    patcher3 = patch(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner.init_routed_experts_capturer",
        init_routed_experts_capturer,
    )
    patcher1.start()
    patcher2.start()
    patcher3.start()
    _PATCHES.extend([patcher1, patcher2, patcher3])
    logger.info("Applied vLLM routed-experts buffer patch for vLLM %s", vllm.__version__)
