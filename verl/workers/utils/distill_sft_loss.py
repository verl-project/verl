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
"""Top-k distillation as an SFT loss, over the engine's dual call contract.

MegatronEngine reaches this object twice per micro-batch:

* as a logits processor (``student_logits=``, ``data=``, ``data_format=``),
  where the engine expects a dict merged into the model output;
* as the loss (``model_output=``, ``data=``, ``dp_group=``), where the engine
  expects ``(loss, metrics)``.

The per-position divergence is computed in the first call (that is where the
logits live); the second call only reduces it with the same mask the CE loss uses,
so a distillation run and a CE run supervise exactly the same tokens.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.utils.losses import masked_sum

DISTILLATION_LOSSES_KEY = "distillation_losses"

# Module-level holder for the teacher payload.
#
# Why not an argument or an attribute: the batch cannot carry a ragged per-token
# payload (TensorDict requires a leading batch dimension, and the non-tensor channel
# does not survive the engine's micro-batch split), and the loss *instance* the engine
# calls is not the one the trainer constructed — anything attached to it is lost on the
# way over. A module global is shared by every rank in the process, which is exactly
# the scope the payload needs.
# The env var is the part that survives a worker spawn: verl's engine may run in a
# separate process from the trainer, where a module global set by the trainer is
# invisible. Exports are inherited, so the path travels either way.
_TEACHER_PATH: str | None = os.environ.get("DISTILL_TEACHER_PATH")
_TEACHER_LOOKUP: dict | None = None


def set_teacher_data(path: str) -> None:
    """Point the distillation loss at the precomputed teacher payload."""
    global _TEACHER_PATH, _TEACHER_LOOKUP
    _TEACHER_PATH = str(path)
    _TEACHER_LOOKUP = None


_TEACHER_BY_SAMPLE: dict | None = None
# Samples whose teacher payload was missing (validation data is not in the corpus).
# Counted rather than raised: the same loss object serves the training batches (which
# must hit) and the validation batches (which cannot), and a silent fallback would hide
# a real training miss.
_MISSING_PAYLOAD = 0


def missing_payload_count() -> int:
    return _MISSING_PAYLOAD


def _teacher_lookup_by_sample() -> dict | None:
    """Second index: {sample id: (ids, logps)}, used when the token hash misses."""
    global _TEACHER_BY_SAMPLE
    if _TEACHER_BY_SAMPLE is None and _TEACHER_PATH is not None:
        side = _TEACHER_PATH.replace(".pt", "_by_sample.pt")
        try:
            _TEACHER_BY_SAMPLE = torch.load(side, map_location="cpu")
        except FileNotFoundError:
            _TEACHER_BY_SAMPLE = {}
    return _TEACHER_BY_SAMPLE


def _teacher_lookup() -> dict | None:
    global _TEACHER_LOOKUP
    if _TEACHER_LOOKUP is None and _TEACHER_PATH is not None:
        raw = torch.load(_TEACHER_PATH, map_location="cpu")
        _TEACHER_LOOKUP = {
            k: (v["ids"], v["logps"]) if isinstance(v, dict) else (v[0], v[1]) for k, v in raw.items()
        }
    return _TEACHER_LOOKUP


def _flatten_positions(t: torch.Tensor) -> torch.Tensor:
    """Bring a per-token tensor to ``[num_tokens, ...]`` regardless of its layout.

    Nested (jagged) batches keep their values in token order already; padded batches
    are row-major, so a plain reshape lands on the same order as long as the caller
    flattens both operands the same way.
    """
    if getattr(t, "is_nested", False):
        return t.values()
    return t.reshape(-1, *t.shape[2:])


class SFTDistillationLoss:
    """Forward-KL against a precomputed teacher top-k, as an SFT loss."""

    def __init__(self, distillation_config, teacher_lookup=None, teacher_data_path=None):
        if distillation_config is None:
            raise ValueError("SFTDistillationLoss needs a DistillationConfig")
        self.config = distillation_config
        # {sha1(student token ids): (ids [tokens, k], logps [tokens, k])}
        #
        # The batch cannot carry a ragged per-token teacher payload: every tensor key in
        # the TensorDict must match the batch dimension, and the non-tensor channel does
        # not survive the engine's micro-batch split. The student's own input_ids do
        # travel correctly (they are how the engine knows the tokens), so the payload is
        # looked up by the token sequence itself.
        self.teacher_lookup = teacher_lookup
        # A path survives being handed to the engine's workers; a large dict of tensors
        # does not necessarily, and a silently empty lookup then fails at the first
        # batch with "no teacher payload".
        self.teacher_data_path = teacher_data_path
        self._lazy_loaded = False
        # The datum currently being trained: (ids [tokens, k], logps [tokens, k]).
        # Set by the trainer before each step; this is the mint datum model — the
        # payload rides beside the batch instead of inside it.
        self.current = None
        # compute_topk_loss dispatches on the actor strategy; in the SFT trainer the
        # engine is megatron, so the shim only has to carry that one field.
        self._actor_shim = SimpleNamespace(strategy="megatron")
        self.loss_calls = 0

    def __call__(self, **kwargs):
        if "student_logits" in kwargs:
            return self._as_processor(kwargs["student_logits"], kwargs["data"], kwargs["data_format"])
        if "model_output" in kwargs:
            return self._as_loss(kwargs["model_output"], kwargs["data"])
        raise TypeError(
            "SFTDistillationLoss is called either as a logits processor (student_logits=) "
            f"or as a loss (model_output=), got keys {sorted(kwargs)}"
        )

    def _as_processor(self, student_logits, data, data_format: str) -> dict:
        """Per-position forward KL, computed here rather than by the engine's branch.

        The engine's ``compute_forward_kl_topk`` requires THD/rmpad nested tensors and
        reshapes them through its own preprocessors, which do not accept the nested
        layout the SFT collator produces. The maths does not need that layout: both the
        logits and the teacher rows just have to be brought to the same flat token
        order, which is what ``_flatten_positions`` does.

        The formula is the engine's, cross-checked against the mint engine's
        ``forward_kl``: full-vocabulary log-softmax normaliser, the teacher's log-probs
        clamped at ``log_prob_min_clamp`` (which is what keeps ``0 * (-inf)`` out of the
        result), and the sum over the teacher's top-k of ``p_T * (log p_T - log p_S)``.
        """
        from verl.utils import tensordict_utils as tu

        lookup = _teacher_lookup()
        if lookup is None:
            lookup = self.teacher_lookup
        if lookup is not None:
            # Match by the student token sequence itself. The dataset's `sample` column
            # did not identify the row actually in the batch (a 314-token batch was
            # paired with a 309-token payload), while input_ids is exactly what the
            # engine is about to embed.
            import hashlib

            student = _flatten_positions(data["input_ids"])
            key = hashlib.sha1(bytes(str(student.tolist()), "utf8")).hexdigest()
            hit = lookup.get(key)
            if hit is None:
                from verl.utils import tensordict_utils as _tu

                raw_ids = _tu.get_non_tensor_data(data, key="sample", default=None)
                for accessor in ("data", "values"):
                    if hasattr(raw_ids, accessor) and not callable(getattr(raw_ids, accessor)):
                        raw_ids = getattr(raw_ids, accessor)
                cand = raw_ids[0] if isinstance(raw_ids, list | tuple) and raw_ids else raw_ids
                by_sample = _teacher_lookup_by_sample() or {}
                hit = by_sample.get(int(cand)) if cand is not None else None
                if hit is None:
                    global _MISSING_PAYLOAD
                    _MISSING_PAYLOAD += 1
            teacher_ids, teacher_logprobs = hit if hit is not None else (None, None)
        else:
            if self.current is None:
                raise RuntimeError(
                    "SFTDistillationLoss has neither a lookup nor a current payload"
                )
            teacher_ids, teacher_logprobs = self.current
        if teacher_ids is None:
            return self._cross_entropy(student_logits, data)

        # Under context parallelism each rank holds one contiguous interval of the packed
        # token buffer (the layout dsv4_hybrid asks for), while the payload covers every
        # position. Shard it the same way, otherwise the counts disagree by exactly the
        # CP factor.
        cp_size, cp_rank = 1, 0
        try:
            import megatron.core.parallel_state as mpu

            cp_size = mpu.get_context_parallel_world_size()
            cp_rank = mpu.get_context_parallel_rank()
        except Exception:  # noqa: BLE001 - no megatron (unit tests) means no sharding
            pass
        if cp_size > 1:
            total = teacher_ids.shape[0]
            chunk = total // cp_size
            lo = cp_rank * chunk
            teacher_ids = teacher_ids[lo : lo + chunk]
            teacher_logprobs = teacher_logprobs[lo : lo + chunk]

        # The payload lives on the host; the gather below runs against device logits.
        teacher_ids = teacher_ids.to(student_logits.device)
        teacher_logprobs = teacher_logprobs.to(student_logits.device)

        # The engine asserts the returned per-position tensors against the batch's
        # attention_mask ([b, s] or [1, s]), so remember the lead shape and restore it
        # after the flat computation.
        lead_shape = tuple(student_logits.shape[:2]) if not getattr(student_logits, "is_nested", False) else None
        logits = _flatten_positions(student_logits)
        # The payload is already [tokens, topk] — it does not carry a batch axis, so it
        # must NOT go through the logits flattener (which strips two leading dims and
        # would reshape it to tokens*topk).
        if teacher_ids.dim() != 2 or teacher_logprobs.dim() != 2:
            raise ValueError(
                f"teacher payload must be [tokens, topk], got {tuple(teacher_ids.shape)} and "
                f"{tuple(teacher_logprobs.shape)}"
            )
        # A flat 1-D nested row carries tokens*topk entries; restore the pair.
        topk = tu.get_non_tensor_data(data, key="teacher_topk", default=None)
        if topk is not None and teacher_ids.dim() == 1:
            topk = int(getattr(topk, "data", topk))
            teacher_ids = teacher_ids.reshape(-1, topk)
            teacher_logprobs = teacher_logprobs.reshape(-1, topk)
        if not (logits.shape[0] == teacher_ids.shape[0] == teacher_logprobs.shape[0]):
            raise ValueError(
                f"position counts disagree: logits={logits.shape[0]} "
                f"teacher_ids={teacher_ids.shape[0]} teacher_logprobs={teacher_logprobs.shape[0]}"
            )
        if logits.shape[0] == 0:
            return {DISTILLATION_LOSSES_KEY: logits.new_zeros((0,))}

        # The clamp lives on the nested DistillationLossConfig, which is where the
        # engine's own compute_forward_kl_topk reads it from. Reading only the
        # top-level field silently disables it and turns -inf pad rows into NaN.
        loss_cfg = getattr(self.config, "distillation_loss", None)
        clamp = getattr(loss_cfg, "log_prob_min_clamp", None)
        if clamp is None:
            clamp = getattr(self.config, "log_prob_min_clamp", None)
        # Per-token clip from the OPSD recipe (jsd_token_clip). A handful of style
        # tokens ("wait", "think") carry 6-15x the divergence of the maths tokens, so
        # an unclipped sum lets them dominate the gradient. Clipping each position's
        # divergence caps that tail without touching the rest of the distribution.
        token_clip = getattr(loss_cfg, "token_clip", None)
        if token_clip is None:
            token_clip = getattr(self.config, "token_clip", None)
        t_logp = teacher_logprobs.to(torch.float32)
        if clamp is not None:
            # exp(-inf) would be 0 and 0 * (-inf) is NaN; clamping is the engine's
            # answer to that, and it is what we inherit here.
            t_logp = t_logp.clamp_min(clamp)
        t_prob = t_logp.exp()
        # Full-vocabulary normaliser. TP=1 in this stack; a vocab-parallel logits shard
        # would need the all-reduce variant (vocab_parallel_topk.shard_logprobs).
        s_logp = torch.log_softmax(logits.to(torch.float32), dim=-1)
        s_at = s_logp.gather(-1, teacher_ids.to(torch.int64).clamp_min(0))
        per_position = (t_prob * (t_logp - s_at)).sum(dim=-1)
        if token_clip is not None and token_clip > 0:
            # Clip the magnitude, preserving the sign: a negative (truncation-induced)
            # divergence is bounded the same way as a positive one.
            per_position = per_position.clamp(min=-token_clip, max=token_clip)
        teacher_mass = t_prob.sum(dim=-1)
        if lead_shape is not None and per_position.numel() == lead_shape[0] * lead_shape[1]:
            per_position = per_position.reshape(lead_shape)
            teacher_mass = teacher_mass.reshape(lead_shape)
        return {DISTILLATION_LOSSES_KEY: per_position, "teacher_mass": teacher_mass}

    def _cross_entropy(self, student_logits, data) -> dict:
        """Fallback for batches without a teacher payload (i.e. validation data).

        Returns the same per-position container the distillation path returns, so the
        engine's masking and the reported number stay directly comparable: the value at
        position i is the NLL of the token after i, exactly like the divergence there.
        """
        lead_shape = tuple(student_logits.shape[:2]) if not getattr(student_logits, "is_nested", False) else None
        logits = _flatten_positions(student_logits)
        log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
        ids = _flatten_positions(data["input_ids"]).to(torch.int64)
        target = torch.roll(ids, shifts=-1)
        nll = -log_probs.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        if lead_shape is not None and nll.numel() == lead_shape[0] * lead_shape[1]:
            nll = nll.reshape(lead_shape)
        return {DISTILLATION_LOSSES_KEY: nll}

    def _as_loss(self, model_output, data):
        if DISTILLATION_LOSSES_KEY not in model_output:
            raise KeyError(
                f"the model output has no {DISTILLATION_LOSSES_KEY!r}: the batch must carry the "
                "non-tensor flag distillation_use_topk=True"
            )
        per_position = model_output[DISTILLATION_LOSSES_KEY]

        pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        dp_size = data["dp_size"]
        batch_num_tokens = data["batch_num_tokens"]

        if pad_mode == DatasetPadMode.NO_PADDING:
            per_position = (
                per_position.values() if getattr(per_position, "is_nested", False) else per_position.reshape(-1)
            )
            loss_mask = data["loss_mask"]
            loss_mask_flatten = (
                loss_mask.values() if getattr(loss_mask, "is_nested", False) else loss_mask.reshape(-1)
            )
            # Same convention as the CE loss: mask[i] marks target token i while the
            # per-position divergence at index i describes the token AFTER i, so the
            # mask is rolled once. Keeping the two losses on the same roll is what
            # makes a CE-vs-distillation comparison mean anything.
            loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
        else:
            # Padded batch: the divergence and the mask share the [batch, seq] grid, and
            # the same one-token convention applies — a divergence at (b, i) describes the
            # token after i — so the mask is rolled along the sequence axis, not globally.
            per_position = per_position.reshape(-1)
            raw_mask = data["loss_mask"] if "loss_mask" in data else data["response_mask"]
            loss_mask_flatten = torch.roll(raw_mask.to(bool), shifts=-1, dims=-1).reshape(-1)

        if per_position.numel() != loss_mask_flatten.numel():
            raise ValueError(
                f"distillation has {per_position.numel()} positions but the mask has "
                f"{loss_mask_flatten.numel()}; they must supervise the same tokens"
            )
        self.loss_calls += 1
        loss = masked_sum(per_position, loss_mask_flatten) / batch_num_tokens * dp_size
        with torch.no_grad():
            supervised = loss_mask_flatten.sum().clamp_min(1)
            metrics = {
                "distill/loss": (masked_sum(per_position, loss_mask_flatten) / supervised).detach().item(),
                "distill/teacher_mass": (
                    # the engine hands model_output values back in the batch's layout,
                    # which is nested here; flatten the same way as everything else
                    _flatten_positions(model_output["teacher_mass"])[loss_mask_flatten.to(bool)].mean().item()
                    if "teacher_mass" in model_output
                    else float("nan")
                ),
            }
        return loss, metrics
