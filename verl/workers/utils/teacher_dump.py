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
"""Dump a forward's top-k next-token distribution, for use as a teacher.

On-policy self-distillation reads the teacher under a privileged context. When the
teacher is the training model itself, the distribution has to be exported once and
replayed during training, and verl reaches the logits through a single callable
that plays two roles:

* MegatronEngine's forward calls it as a **logits processor**
  (``student_logits=``, ``data=``, ``data_format=``) and merges the returned dict
  into the model output;
* ``MegatronEngineWithLMHead.postprocess_micro_batch_func`` calls it as a **loss**
  (``model_output=``, ``data=``, ``dp_group=``) and expects ``(loss, metrics)``.

``TeacherTopKDump`` satisfies both: the processor call computes the top-k, the loss
call returns a zero loss so the dump run does not need a real objective. The engine
keys the processor call off the batch's non-tensor ``needs_vocab_logits`` flag.
"""

from __future__ import annotations

from typing import Any

import torch

from verl.utils.megatron.vocab_parallel_topk import topk_logprobs_from_logits

DUMP_IDS_KEY = "teacher_topk_ids"
DUMP_LOGPROBS_KEY = "teacher_topk_logprobs"


def dump_consistency_stats(model_output, input_ids) -> dict:
    """Compare an exported top-k against the engine's own per-token log-probs.

    For every position whose true next token falls inside the exported top-k, the
    exported log-prob must equal ``log_probs`` exactly — the two come from the same
    logits. Coverage is the fraction of positions where that comparison was
    possible at all; a low coverage means the dump is not being checked, not that
    it is correct.
    """
    ids = model_output[DUMP_IDS_KEY]
    lps = model_output[DUMP_LOGPROBS_KEY]
    log_probs = model_output.get("log_probs")
    if log_probs is None:
        return {"coverage": 0.0, "max_abs_diff": float("inf"), "note": "no log_probs to check against"}

    def flat(t):
        return t.values() if getattr(t, "is_nested", False) else t.reshape(-1, *t.shape[2:])

    ids_v, lps_v = flat(ids), flat(lps)
    # The batch's input_ids can arrive on a CPU host while the model outputs are on
    # device (TensorDict keeps the sampled tensors where the loader left them), so
    # normalise every operand onto the dump's device before comparing.
    dev = ids_v.device
    lp_v = flat(log_probs).reshape(-1).to(dev)
    in_v = flat(input_ids).reshape(-1).to(dev)
    if not (ids_v.shape[0] == lps_v.shape[0] == lp_v.shape[0] == in_v.shape[0]):
        raise ValueError(
            "flattened row counts disagree: "
            f"ids={ids_v.shape[0]} logps={lps_v.shape[0]} log_probs={lp_v.shape[0]} input_ids={in_v.shape[0]}"
        )

    # Next-token targets; the last position of every sequence predicts nothing.
    true_next = torch.roll(in_v, shifts=-1)
    if getattr(input_ids, "is_nested", False):
        bounds = (input_ids.offsets()[1:] - 1).to(in_v.device)
    else:
        seq = input_ids.shape[1]
        bounds = torch.arange(seq - 1, in_v.numel(), seq, device=in_v.device)
    valid = torch.ones(in_v.numel(), dtype=torch.bool, device=in_v.device)
    if bounds.numel():
        valid[bounds] = False

    match = ids_v == true_next.unsqueeze(-1)
    hit = match.any(dim=-1)
    picked = lps_v.gather(-1, match.int().argmax(dim=-1, keepdim=True)).squeeze(-1)
    checkable = hit & valid
    if not checkable.any():
        return {"coverage": 0.0, "max_abs_diff": float("inf"), "note": "no position had its target in the top-k"}
    diff = (picked[checkable] - lp_v[checkable]).abs()
    return {
        "coverage": float(hit[valid].float().mean()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "checked_positions": int(checkable.sum()),
        "topk": int(ids_v.shape[-1]),
    }


def capture_dump_tensors(model_output, input_ids) -> dict:
    """Detach one step's dump into CPU tensors, keeping the sequence offsets.

    The dump is only useful if it can be replayed against the same tokens later, so
    the payload carries ``input_ids`` (and the nested offsets when the batch is
    THD/packed) beside the distribution. Everything is moved to CPU here because
    the caller accumulates across steps and must not pin device memory.
    """
    if DUMP_IDS_KEY not in model_output or DUMP_LOGPROBS_KEY not in model_output:
        raise KeyError(
            f"capture_dump_tensors needs {DUMP_IDS_KEY!r} and {DUMP_LOGPROBS_KEY!r}; "
            "the batch must carry needs_vocab_logits=True (distillation_use_topk also requires a teacher payload)"
        )
    nested = getattr(input_ids, "is_nested", False)
    # Save a nested tensor as values + offsets rather than the nested object itself:
    # torch.save/torch.load round-trips of jagged tensors write a storage record whose
    # size disagrees with the reader's expectation ("record size (3456 bytes) does not
    # match expected size (2432 ...) for dtype Byte"), and the payload cannot be read
    # back at all. Plain tensors plus the offsets reconstruct the layout exactly.
    ids = model_output[DUMP_IDS_KEY]
    logps = model_output[DUMP_LOGPROBS_KEY]
    # The engine hands the dump back wrapped in the batch's layout, so the ids and
    # log-probs are nested too — not just input_ids. Every nested operand is stored
    # as its values: torch.save writes a jagged storage record that torch.load then
    # refuses ("record size (3456 bytes) does not match expected size (2432 ...)").
    return {
        "ids": (ids.values() if getattr(ids, "is_nested", False) else ids).detach().to("cpu"),
        "logps": (logps.values() if getattr(logps, "is_nested", False) else logps).detach().to("cpu"),
        "input_ids": (input_ids.values() if nested else input_ids).detach().to("cpu"),
        "offsets": input_ids.offsets().to("cpu") if nested else None,
        "nested": nested,
    }


class TeacherTopKDump:
    """Dual-role callable: top-k exporter when used as a logits processor."""

    def __init__(self, topk: int, process_group=None):
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        self.topk = topk
        self.process_group = process_group
        self.calls = 0

    def __call__(self, **kwargs: Any):
        if "student_logits" in kwargs:
            return self._as_processor(kwargs["student_logits"])
        if "model_output" in kwargs:
            return self._as_loss(kwargs["model_output"])
        raise TypeError(
            "TeacherTopKDump is called either as a logits processor (student_logits=) or "
            f"as a loss (model_output=), got keys {sorted(kwargs)}"
        )

    def _as_processor(self, student_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        logprobs, ids = topk_logprobs_from_logits(student_logits, self.topk, process_group=self.process_group)
        self.calls += 1
        # The loss path expects exactly these two names in the batch; keep the same
        # spelling on the way out so a dump and a training batch are interchangeable.
        return {DUMP_IDS_KEY: ids.to(torch.int64), DUMP_LOGPROBS_KEY: logprobs.to(torch.float32)}

    def _as_loss(self, model_output) -> tuple[torch.Tensor, dict]:
        # A dump run has no objective; the engine still wants a loss and metrics.
        for key in (DUMP_IDS_KEY, DUMP_LOGPROBS_KEY):
            if key not in model_output:
                raise KeyError(
                    f"teacher dump expected {key!r} in the model output — the batch must carry "
                    "the non-tensor flag needs_vocab_logits=True "
                    "(distillation_use_topk also requires a teacher payload)"
                )
        device = model_output[DUMP_LOGPROBS_KEY].device
        return torch.zeros((), device=device), {"teacher_topk_dumped": float(self.calls)}
