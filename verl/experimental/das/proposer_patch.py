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
"""In-place extension of vLLM's suffix-decoding proposer, hosting DAS.

vLLM's ``SpeculativeMethod`` is a closed set, so DAS rides the stock
``method="suffix"`` proposer by rebinding ``__init__`` / ``propose`` on the
``SuffixDecodingProposer`` class object. Stock bookkeeping is kept as-is for
non-DAS requests; per-problem trees compete with the stock cache for DAS
requests and the higher-scoring draft wins.

The patch is installed only inside the GPU worker processes of engines whose
rollout config has ``das.enable=true`` (see ``worker_extension``), is
idempotent, and any structural error (vLLM drift) permanently reverts that
worker to the stock path: DAS degrades, decoding never breaks.
"""

from __future__ import annotations

import logging
import os

from verl.experimental.das.drafter_state import DASDrafterState, register_active_state
from verl.experimental.das.problem_id import parse_request_id
from verl.workers.config.das import DASConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_CONFIG: DASConfig | None = None


def install_proposer_patch(cfg: DASConfig) -> bool:
    """Rebind ``SuffixDecodingProposer.__init__`` / ``.propose``. Idempotent; never raises.

    Returns True when the patch is (already) installed, False when vLLM or its suffix
    proposer is unavailable, in which case the engine runs stock decoding.
    """
    global _CONFIG
    _CONFIG = cfg
    try:
        from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
    except ImportError as e:
        logger.error("DAS: vLLM suffix-decoding proposer not importable (%s); running without DAS", e)
        return False

    if getattr(SuffixDecodingProposer, "_verl_das_patched", False):
        return True

    orig_init = SuffixDecodingProposer.__init__
    orig_propose = SuffixDecodingProposer.propose

    def das_init(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        self._das_state = None
        self._das_failed = False
        if _CONFIG is None or not _CONFIG.enable:
            return
        try:
            self._das_state = DASDrafterState(_CONFIG, max_tree_depth=int(self.max_tree_depth))
            register_active_state(self._das_state)
            logger.info("DAS: suffix proposer initialized with per-problem drafter state")
        except Exception:
            logger.exception("DAS: drafter state init failed; running stock suffix decoding")
            self._das_state = None

    def das_propose(self, *args, **kwargs):
        state = getattr(self, "_das_state", None)
        if state is None:
            return orig_propose(self, *args, **kwargs)
        try:
            input_batch, sampled_token_ids = _propose_args(args, kwargs)
            return _propose_impl(self, state, input_batch, sampled_token_ids)
        except Exception:
            if not getattr(self, "_das_failed", False):
                logger.exception(
                    "DAS: propose failed (likely vLLM drift); permanently reverting this worker to stock suffix "
                    "decoding"
                )
            self._das_failed = True
            self._das_state = None
            register_active_state(None)
            return orig_propose(self, *args, **kwargs)

    SuffixDecodingProposer.__init__ = das_init
    SuffixDecodingProposer.propose = das_propose
    SuffixDecodingProposer._verl_das_patched = True
    logger.info("DAS: SuffixDecodingProposer patched in place")
    return True


def _propose_args(args: tuple, kwargs: dict) -> tuple:
    """``(input_batch, sampled_token_ids)`` from ``propose(input_batch, sampled_token_ids, slot_mappings=None)``.

    Tolerates a leading ``num_speculative_tokens`` int, which some vLLM lines pass first.
    """
    positional = list(args)
    if positional and isinstance(positional[0], int):
        positional.pop(0)
    input_batch = kwargs["input_batch"] if "input_batch" in kwargs else positional.pop(0)
    sampled = kwargs["sampled_token_ids"] if "sampled_token_ids" in kwargs else positional.pop(0)
    return input_batch, sampled


def _propose_impl(self, state: DASDrafterState, input_batch, sampled_token_ids):
    """Mirror of the stock propose loop with per-problem trees and budgets."""
    draft_token_ids: list[list[int]] = []
    for i, sampled_ids in enumerate(sampled_token_ids):
        if not sampled_ids:
            # Partial prefill: no sampled tokens yet.
            draft_token_ids.append([])
            continue

        req_id = input_batch.req_ids[i]
        row = _batch_index(input_batch, req_id, i)
        if req_id not in self.suffix_cache.active_requests:
            if req_id in self.suffix_cache.cached_requests:
                self.suffix_cache.evict_cached_response(req_id)
            num_prompt_tokens = int(input_batch.num_prompt_tokens[row])
            prompt_token_ids = _as_int32(input_batch.token_ids_cpu[row, :num_prompt_tokens])
            self.suffix_cache.start_request(req_id, prompt_token_ids)
        self.suffix_cache.add_active_response(req_id, sampled_ids)

        phash = parse_request_id(req_id)
        if phash is not None:
            # Transient self-insert first, so same-batch siblings see it this round.
            state.on_request_tokens(phash, req_id, sampled_ids)

        num_tokens = int(input_batch.num_tokens_no_spec[row])
        if num_tokens >= self.max_model_len:
            draft_token_ids.append([])
            continue

        if phash is None:
            budget = self.num_speculative_tokens
        else:
            observed = num_tokens - int(input_batch.num_prompt_tokens[row])
            budget = state.budget_for(phash, observed)
        budget = min(budget, self.num_speculative_tokens, self.max_model_len - num_tokens - 1)
        if budget <= 0:
            # Short class: skip drafting (the DAS "skip speculation" rule); collection above still happened.
            draft_token_ids.append([])
            continue

        start = max(0, num_tokens - self.max_tree_depth)
        pattern = _as_int32(input_batch.token_ids_cpu[row, start:num_tokens])
        best = self.suffix_cache.speculate(
            req_id,
            pattern,
            max_spec_tokens=budget,
            max_spec_factor=self.max_spec_factor,
            min_token_prob=self.min_token_prob,
        )
        if phash is not None:
            problem_draft = state.speculate(
                phash,
                [int(t) for t in pattern],
                budget,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )
            if problem_draft is not None and (
                best is None or problem_draft.score >= float(getattr(best, "score", 0.0))
            ):
                best = problem_draft
        if best is not None and getattr(best, "token_ids", None):
            state.drafts_emitted += 1
            draft_token_ids.append(list(best.token_ids))
        else:
            draft_token_ids.append([])

    # Requests that left the batch: stock cache cleanup + transient drop.
    active_ids = set(getattr(input_batch, "req_id_to_index", {}) or {})
    if not active_ids:
        active_ids = {rid for rid in input_batch.req_ids if rid is not None}
    for req_id in self.suffix_cache.active_requests - active_ids:
        self.suffix_cache.stop_request(req_id)
        state.drop_request(req_id)
    return draft_token_ids


def _as_int32(arr):
    """*arr* as an int32 ndarray: arctic-inference >= 0.3 rejects other dtypes.

    vLLM's ``token_ids_cpu`` is already int32, so this is zero-copy there.
    """
    if str(getattr(arr, "dtype", None)) == "int32":
        return arr
    import numpy as np

    return np.asarray(arr, dtype=np.int32)


def _batch_index(input_batch, req_id, fallback: int) -> int:
    """Batch row for *req_id* via ``req_id_to_index``, else the sampled position."""
    mapping = getattr(input_batch, "req_id_to_index", None)
    if not mapping:
        return fallback
    try:
        return int(mapping[req_id])
    except (KeyError, TypeError, ValueError):
        return fallback
