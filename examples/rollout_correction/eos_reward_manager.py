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
"""Custom NaiveRewardManager enforcing native-EOS completion for 32K math RL recipes.

Loaded via ``reward.reward_manager.source=importlib`` alongside ``reward.preset=math_dapo``:
- Resolves all effective native EOS token IDs from both ``tokenizer.eos_token_id`` and the
  model's ``generation_config.json`` (e.g., Qwen3's ``<|endoftext|>`` and ``<|im_end|>``).
- Inspects ``valid_response_ids[-1]`` directly in ``run_single`` *before* ``skip_special_tokens=True``
  detokenization strips special tokens, catching ``max_tokens`` truncation, ``max_model_len``
  truncation, and agent-loop length slicing in O(1) without decoding truncated 32K responses.
- Assigns ``reward.reward_kwargs.missing_eos_reward`` (default ``-1.0`` for ``[-1, +1]``
  ``math_dapo``; set ``0.0`` for binary verifiers or ``null`` to disable) to unfinished rollouts.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.utils.fs import copy_to_local
from verl.utils.model import get_generation_config


def _collect_int_ids(raw: Any, out: set[int]) -> None:
    """Append valid non-boolean integer token IDs from an int or sequence into ``out``."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        out.add(raw)
    elif isinstance(raw, list | tuple | set):
        for item in raw:
            if isinstance(item, int) and not isinstance(item, bool):
                out.add(item)


def resolve_eos_token_ids(config: Any, tokenizer: Any, *, fetch_remote: bool = True) -> frozenset[int]:
    """Collect effective native EOS token IDs from ``tokenizer`` and ``generation_config``."""
    eos_ids: set[int] = set()
    _collect_int_ids(getattr(tokenizer, "eos_token_id", None), eos_ids)

    candidates: list[str] = []
    for raw_path in (
        getattr(tokenizer, "name_or_path", None),
        OmegaConf.select(config, "actor_rollout_ref.model.path", default=None) if config is not None else None,
    ):
        if isinstance(raw_path, str) and raw_path and raw_path not in candidates:
            candidates.append(raw_path)

    trust_remote_code = bool(
        OmegaConf.select(config, "actor_rollout_ref.model.trust_remote_code", default=False)
        if config is not None
        else False
    )
    for path in candidates:
        if os.path.isdir(path):
            for filename in ("generation_config.json", "config.json"):
                cfg_file = os.path.join(path, filename)
                if os.path.isfile(cfg_file):
                    try:
                        with open(cfg_file, encoding="utf-8") as f:
                            _collect_int_ids(json.load(f).get("eos_token_id"), eos_ids)
                    except (OSError, ValueError, AttributeError):
                        pass
        elif fetch_remote:
            try:
                gen_cfg = get_generation_config(copy_to_local(path), trust_remote_code=trust_remote_code)
                if gen_cfg is not None:
                    _collect_int_ids(getattr(gen_cfg, "eos_token_id", None), eos_ids)
            except Exception:
                pass
    return frozenset(eos_ids)


class EOSNaiveRewardManager(NaiveRewardManager):
    """``NaiveRewardManager`` subclass that penalizes responses not terminating on a native EOS token."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            compute_score=compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )
        reward_kwargs = (
            config.reward.get("reward_kwargs") if config is not None and "reward" in config else None
        ) or {}
        raw_missing_eos = reward_kwargs.get("missing_eos_reward", -1.0)
        self.missing_eos_reward: float | None = None if raw_missing_eos is None else float(raw_missing_eos)

        ignore_eos = bool(
            OmegaConf.select(config, "actor_rollout_ref.rollout.ignore_eos", default=False)
            if config is not None
            else False
        )
        if self.missing_eos_reward is not None and ignore_eos:
            raise ValueError("missing_eos_reward requires actor_rollout_ref.rollout.ignore_eos=False")

        self.eos_token_ids = resolve_eos_token_ids(config, tokenizer, fetch_remote=self.missing_eos_reward is not None)
        if self.missing_eos_reward is not None and not self.eos_token_ids:
            raise ValueError(
                "EOSNaiveRewardManager could not resolve any native eos_token_id from tokenizer or generation_config."
            )

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        """Score one trajectory, short-circuiting unfinished (non-EOS) responses when configured."""
        last = data[-1:]
        data_item = last[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = (
            int(data_item.batch["attention_mask"][-response_length:].sum()) if response_length else 0
        )
        valid_response_ids = response_ids[:valid_response_length]

        ended_with_eos = bool(valid_response_ids.numel() > 0 and int(valid_response_ids[-1]) in self.eos_token_ids)
        if self.missing_eos_reward is not None and not ended_with_eos:
            score = float(self.missing_eos_reward)
            return {
                "reward_score": score,
                "reward_extra_info": {
                    "score": score,
                    "acc": False,
                    "pred": "[INVALID]",
                },
            }

        # Isolate extra_info per sample before delegating so concurrent rollouts never share mutations.
        isolated_extra = dict(data_item.non_tensor_batch.get("extra_info") or {})
        isolated_extra["ended_with_eos"] = ended_with_eos
        extra_arr = np.empty(1, dtype=object)
        extra_arr[0] = isolated_extra
        isolated = DataProto(
            batch=last.batch,
            non_tensor_batch={**last.non_tensor_batch, "extra_info": extra_arr},
            meta_info=last.meta_info,
        )
        result = await super().run_single(isolated)
        if self.missing_eos_reward is not None:
            extra_out = result["reward_extra_info"]
            extra_out.setdefault("score", float(result["reward_score"]))
            extra_out.setdefault("acc", bool(result["reward_score"] > 0))
            extra_out.setdefault("pred", "")
        return result
