# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Backend-neutral dispatcher for the preprocessed-multimodal-input contract.

Public agent-loop code (agent_loop.py / single_turn_agent_loop.py / tool_agent_loop.py)
calls into this module instead of importing a backend-specific module directly. Today
only the vLLM backend has a builder; sglang / TensorRT-LLM / etc. would grow their own
branches here when they expose an equivalent direct-engine-input contract.
"""

from __future__ import annotations

from typing import Any


def build_preprocessed_multimodal_input(
    *,
    rollout_name: str | None,
    rollout_config: Any,
    processor: Any,
    prompt_ids: list[int],
    model_inputs: Any,
    images: Any = None,
    videos: Any = None,
) -> Any | None:
    """Build a backend-specific preprocessed multimodal payload, or None.

    Returns None when the active rollout backend doesn't support a direct preprocessed
    contract or when the per-backend feature flag is off, so callers can fall back to the
    raw-media path without branching on backend identity.
    """
    if rollout_name == "vllm":
        if not rollout_config.get("use_preprocessed_multimodal_input", False):
            return None
        from verl.experimental.agent_loop.preprocessed_multimodal import (
            build_vllm_preprocessed_multimodal_input,
        )

        return build_vllm_preprocessed_multimodal_input(
            prompt_ids=prompt_ids,
            processor=processor,
            model_inputs=model_inputs,
            images=images,
            videos=videos,
        )
    # Add sglang / trt branches here when those backends grow a direct-engine-input
    # contract. Until then, callers receive None and use the raw-media path.
    return None


def refresh_preprocessed_multimodal_prompt_ids(
    preprocessed_multimodal_input: Any | None,
    *,
    rollout_name: str | None,
    prompt_ids: list[int],
) -> Any | None:
    """Refresh prompt_token_ids in the previously-built payload between turns.

    No-op when there is no payload or when the backend has no refresh hook.
    """
    if preprocessed_multimodal_input is None:
        return None
    if rollout_name == "vllm":
        from verl.experimental.agent_loop.preprocessed_multimodal import (
            refresh_vllm_preprocessed_multimodal_prompt_ids,
        )

        return refresh_vllm_preprocessed_multimodal_prompt_ids(preprocessed_multimodal_input, prompt_ids=prompt_ids)
    return preprocessed_multimodal_input


def attach_preprocessed_multimodal_input_to_kwargs(
    generate_kwargs: dict[str, Any],
    *,
    rollout_name: str | None,
    preprocessed_multimodal_input: Any | None,
) -> None:
    """Inject the backend-specific kwarg name into generate_kwargs.

    The kwarg name is part of each backend's rollout-server protocol (vLLM uses
    ``preprocessed_multimodal_input``); the dispatcher owns the mapping so call sites
    don't have to. No-op for None payload or unsupported backend.
    """
    if preprocessed_multimodal_input is None:
        return
    if rollout_name == "vllm":
        generate_kwargs["preprocessed_multimodal_input"] = preprocessed_multimodal_input
