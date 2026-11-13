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

import hashlib
import os

import torch

from verl.trainer.distillation.tail_kl import compute_tail_bucket_kl
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.utils.losses import masked_sum, sft_loss

DISTILLATION_LOSSES_KEY = "distillation_losses"

# Legacy process-wide default. Instances also carry their own path so worker
# serialization does not depend on inheriting this module's globals.
_TEACHER_PATH: str | None = os.environ.get("DISTILL_TEACHER_PATH")
_TEACHER_LOOKUP: dict | None = None


def set_teacher_data(path: str) -> None:
    """Point the distillation loss at the precomputed teacher payload."""
    global _TEACHER_PATH, _TEACHER_LOOKUP
    _TEACHER_PATH = str(path)
    _TEACHER_LOOKUP = None


# Training misses are counted and raised. Validation explicitly uses CE instead.
_MISSING_PAYLOAD = 0


def missing_payload_count() -> int:
    return _MISSING_PAYLOAD


def _teacher_lookup() -> dict | None:
    global _TEACHER_LOOKUP
    if _TEACHER_LOOKUP is None and _TEACHER_PATH is not None:
        raw = torch.load(_TEACHER_PATH, map_location="cpu")
        _TEACHER_LOOKUP = {k: (v["ids"], v["logps"]) if isinstance(v, dict) else (v[0], v[1]) for k, v in raw.items()}
    return _TEACHER_LOOKUP


def _payload_pair(value):
    return (value["ids"], value["logps"]) if isinstance(value, dict) else (value[0], value[1])


def _flatten_positions(t: torch.Tensor) -> torch.Tensor:
    """Bring a per-token tensor to ``[num_tokens, ...]`` regardless of its layout.

    Nested (jagged) batches keep their values in token order already; padded batches
    are row-major, so a plain reshape lands on the same order as long as the caller
    flattens both operands the same way.
    """
    if getattr(t, "is_nested", False):
        return t.values()
    return t.reshape(-1, *t.shape[2:])


def _flatten_payload(t: torch.Tensor) -> torch.Tensor:
    return t if not getattr(t, "is_nested", False) and t.dim() == 2 else _flatten_positions(t)


class SFTDistillationLoss:
    """Forward-KL against a precomputed teacher top-k, as an SFT loss."""

    def __init__(self, distillation_config, teacher_lookup=None, teacher_data_path=None):
        if distillation_config is None:
            raise ValueError("SFTDistillationLoss needs a DistillationConfig")
        self.config = distillation_config
        self.teacher_lookup = teacher_lookup
        self.teacher_data_path = str(teacher_data_path) if teacher_data_path is not None else None
        self.current = None
        self.loss_calls = 0

    def __call__(
        self,
        *,
        student_logits=None,
        data=None,
        data_format=None,
        model_output=None,
        dp_group=None,
        teacher_topk_ids=None,
        teacher_topk_logprobs=None,
        quant_align_size=None,
        cp_layout="zigzag",
        local_cp_size=None,
        min_local_rows=None,
        cp_chunk_index=0,
        cp_chunk_count=1,
        **kwargs,
    ):
        if student_logits is not None:
            return self._as_processor(
                student_logits,
                data,
                data_format,
                teacher_topk_ids=teacher_topk_ids,
                teacher_topk_logprobs=teacher_topk_logprobs,
                quant_align_size=quant_align_size,
                cp_layout=cp_layout,
                local_cp_size=local_cp_size,
                min_local_rows=min_local_rows,
                cp_chunk_index=cp_chunk_index,
                cp_chunk_count=cp_chunk_count,
            )
        if model_output is not None:
            return self._as_loss(model_output, data)
        raise TypeError(
            "SFTDistillationLoss is called either as a logits processor (student_logits=) "
            f"or as a loss (model_output=), got keys {sorted(kwargs)}"
        )

    def _lookup(self):
        if self.teacher_lookup is None and self.teacher_data_path is not None:
            self.teacher_lookup = torch.load(self.teacher_data_path, map_location="cpu")
        return self.teacher_lookup if self.teacher_lookup is not None else _teacher_lookup()

    def _lookup_rows(self, data):
        lookup = self._lookup()
        if lookup is None:
            if self.current is not None:
                return self.current
            raise KeyError("SFT distillation needs teacher_ids/teacher_logprobs or a teacher lookup")
        if "input_ids" not in data:
            raise KeyError("A teacher lookup requires the batch's input_ids")
        inputs = data["input_ids"]
        attention = data.get("attention_mask")
        rows = []
        for index, sequence in enumerate(inputs.unbind()):
            if attention is not None and not getattr(inputs, "is_nested", False):
                sequence = sequence[attention[index].to(device=sequence.device, dtype=torch.bool)]
            key = hashlib.sha1(str(sequence.tolist()).encode("utf8")).hexdigest()
            hit = lookup.get(key)
            if hit is None:
                global _MISSING_PAYLOAD
                _MISSING_PAYLOAD += 1
                raise KeyError(
                    f"Missing teacher payload for sample {index} ({sequence.numel()} tokens, sha1={key}); "
                    "training cannot fall back to cross-entropy"
                )
            ids, logps = _payload_pair(hit)
            if ids.shape != logps.shape or ids.dim() != 2 or ids.shape[0] != sequence.numel():
                raise ValueError(
                    f"teacher payload for sample {index} must have {sequence.numel()} position-aligned rows; "
                    f"got ids={tuple(ids.shape)} logprobs={tuple(logps.shape)}"
                )
            rows.append((ids, logps))
        return tuple(torch.nested.nested_tensor([row[i] for row in rows], layout=torch.jagged) for i in (0, 1))

    @staticmethod
    def _pack_payload(teacher_ids, teacher_logprobs, data, data_format, student_logits, **packing):
        """Apply the student's per-sequence padding and CP partition to a full payload."""
        if getattr(teacher_ids, "is_nested", False):
            if not getattr(teacher_logprobs, "is_nested", False) or [row.shape for row in teacher_ids.unbind()] != [
                row.shape for row in teacher_logprobs.unbind()
            ]:
                raise ValueError("teacher_ids and teacher_logprobs must have the same shape")
        elif teacher_ids.shape != teacher_logprobs.shape:
            raise ValueError("teacher_ids and teacher_logprobs must have the same shape")
        if not getattr(teacher_ids, "is_nested", False):
            # Padded/direct unit callers already have the logits' [batch, seq] grid.
            if data_format != "thd":
                return _flatten_payload(teacher_ids), _flatten_payload(teacher_logprobs)
            inputs = data.get("input_ids")
            if inputs is None:
                raise KeyError("Packing THD teacher_ids requires input_ids sequence boundaries")
            lengths = inputs.offsets().diff().tolist()
            if teacher_ids.dim() == 2:
                ids_rows = teacher_ids.split(lengths)
                lp_rows = teacher_logprobs.split(lengths)
            else:
                ids_rows = [row[:length] for row, length in zip(teacher_ids.unbind(), lengths, strict=True)]
                lp_rows = [row[:length] for row, length in zip(teacher_logprobs.unbind(), lengths, strict=True)]
            teacher_ids = torch.nested.nested_tensor(ids_rows, layout=torch.jagged)
            teacher_logprobs = torch.nested.nested_tensor(lp_rows, layout=torch.jagged)
        # Rebuild from rows: an older collator can put the ragged axis on top-k
        # when all sequence lengths happen to match. The engine packs along tokens.
        teacher_ids = torch.nested.nested_tensor(
            [row.to(student_logits.device) for row in teacher_ids.unbind()], layout=torch.jagged
        )
        teacher_logprobs = torch.nested.nested_tensor(
            [row.to(student_logits.device) for row in teacher_logprobs.unbind()], layout=torch.jagged
        )
        if data_format == "thd":
            from verl.models.mcore.util import preprocess_thd_engine

            ids = preprocess_thd_engine(teacher_ids, pre_process=True, **packing)[0]
            logps = preprocess_thd_engine(teacher_logprobs, pre_process=True, pad_value=float("-inf"), **packing)[0]
        else:
            if not torch.distributed.is_initialized():
                # Single-process callers have no CP partition or sequence-parallel
                # alignment; retain the logits' padded grid for numerical CPU tests.
                shape = (*student_logits.shape[:2], teacher_ids.values().shape[-1])
                ids = torch.zeros(shape, dtype=teacher_ids.dtype, device=student_logits.device)
                logps = torch.full(shape, float("-inf"), dtype=teacher_logprobs.dtype, device=student_logits.device)
                for index, (id_row, lp_row) in enumerate(
                    zip(teacher_ids.unbind(), teacher_logprobs.unbind(), strict=True)
                ):
                    ids[index, : id_row.shape[0]] = id_row
                    logps[index, : lp_row.shape[0]] = lp_row
                return _flatten_positions(ids), _flatten_positions(logps)
            # The logits already encode the minibatch padding target and CP degree.
            import megatron.core.parallel_state as mpu

            from verl.models.mcore.util import preprocess_bshd_engine

            cp_size = mpu.get_context_parallel_world_size()
            kwargs = dict(
                pre_process=True,
                use_fp8_padding=packing["quant_align_size"] is not None,
                forced_max_seqlen=student_logits.shape[1] * cp_size,
            )
            ids, _, _ = preprocess_bshd_engine(teacher_ids, **kwargs)
            logps, valid, _ = preprocess_bshd_engine(teacher_logprobs, **kwargs)
            logps = logps.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        return _flatten_positions(ids), _flatten_positions(logps)

    def _as_processor(
        self,
        student_logits,
        data,
        data_format,
        *,
        teacher_topk_ids=None,
        teacher_topk_logprobs=None,
        quant_align_size=None,
        cp_layout="zigzag",
        local_cp_size=None,
        min_local_rows=None,
        cp_chunk_index=0,
        cp_chunk_count=1,
    ) -> dict:
        if torch.distributed.is_initialized():
            from megatron.core import parallel_state as mpu

            if mpu.get_tensor_model_parallel_world_size() > 1:
                raise NotImplementedError(
                    "SFTDistillationLoss requires tensor_model_parallel_size=1; "
                    "its vocabulary normalizer does not support tensor-parallel logits"
                )
        if (teacher_topk_ids is None) != (teacher_topk_logprobs is None):
            raise ValueError("teacher_topk_ids and teacher_topk_logprobs must be supplied together")
        if teacher_topk_ids is not None:
            # This pair has already passed through the student's packing/CP path.
            teacher_ids = _flatten_payload(teacher_topk_ids)
            teacher_logprobs = _flatten_payload(teacher_topk_logprobs)
        else:
            teacher_ids, teacher_logprobs = data.get("teacher_ids"), data.get("teacher_logprobs")
            if (teacher_ids is None) != (teacher_logprobs is None):
                raise ValueError("teacher_ids and teacher_logprobs must be supplied together")
            if teacher_ids is None:
                teacher_ids, teacher_logprobs = self._lookup_rows(data)
            teacher_ids, teacher_logprobs = self._pack_payload(
                teacher_ids,
                teacher_logprobs,
                data,
                data_format,
                student_logits,
                quant_align_size=quant_align_size,
                cp_layout=cp_layout,
                local_cp_size=local_cp_size,
                min_local_rows=min_local_rows,
                cp_chunk_index=cp_chunk_index,
                cp_chunk_count=cp_chunk_count,
            )
        teacher_ids = teacher_ids.to(student_logits.device)
        teacher_logprobs = teacher_logprobs.to(student_logits.device)
        lead_shape = tuple(student_logits.shape[:2]) if not getattr(student_logits, "is_nested", False) else None
        logits = _flatten_positions(student_logits)
        if teacher_ids.dim() != 2 or teacher_logprobs.dim() != 2:
            raise ValueError("teacher payload must have shape [positions, topk]")
        if not (logits.shape[0] == teacher_ids.shape[0] == teacher_logprobs.shape[0]):
            raise ValueError(
                f"position counts disagree: logits={logits.shape[0]} "
                f"teacher_ids={teacher_ids.shape[0]} teacher_logprobs={teacher_logprobs.shape[0]}"
            )
        # The clamp lives on the nested DistillationLossConfig, which is where the
        # engine's own compute_forward_kl_topk reads it from. Reading only the
        # top-level field silently disables it and turns -inf pad rows into NaN.
        loss_cfg = getattr(self.config, "distillation_loss", None)
        clamp = getattr(loss_cfg, "log_prob_min_clamp", None)
        if clamp is None:
            clamp = getattr(self.config, "log_prob_min_clamp", None)
        include_tail = getattr(loss_cfg, "loss_mode", None) == "forward_kl_topk_tail"
        if include_tail and clamp is not None:
            raise ValueError("forward_kl_topk_tail requires log_prob_min_clamp=None")
        # Per-token clip from the OPSD recipe (jsd_token_clip). A handful of style
        # tokens ("wait", "think") carry 6-15x the divergence of the maths tokens, so
        # an unclipped sum lets them dominate the gradient. Clipping each position's
        # divergence caps that tail without touching the rest of the distribution.
        token_clip = getattr(loss_cfg, "token_clip", None)
        if token_clip is None:
            token_clip = getattr(self.config, "token_clip", None)
        t_logp = teacher_logprobs.to(torch.float32)
        live = ~torch.isneginf(t_logp)
        if clamp is not None:
            t_logp = t_logp.clamp_min(clamp)
        # -inf denotes absent top-k entries and physical alignment rows. Clamping
        # must not turn those sentinels into a live teacher distribution.
        t_logp = torch.where(live, t_logp, 0.0)
        t_prob = torch.where(live, t_logp.exp(), 0.0)
        # Full-vocabulary normaliser. TP=1 in this stack; a vocab-parallel logits shard
        # would need the all-reduce variant (vocab_parallel_topk.shard_logprobs).
        s_logp = torch.log_softmax(logits.to(torch.float32), dim=-1)
        safe_ids = torch.where(live, teacher_ids.to(torch.int64), 0)
        s_at = s_logp.gather(-1, safe_ids)
        if clamp is not None:
            s_at = s_at.clamp_min(clamp)
        per_position = (t_prob * (t_logp - s_at)).sum(dim=-1)
        teacher_mass = t_prob.sum(dim=-1)
        if include_tail:
            student_mass = torch.where(live, s_at.exp(), 0.0).sum(dim=-1)
            per_position = per_position + compute_tail_bucket_kl(
                student_topk_mass=student_mass,
                teacher_topk_mass=teacher_mass,
                tail_mass_eps=loss_cfg.tail_mass_eps,
            )
        if token_clip is not None and token_clip > 0:
            # Clip the magnitude, preserving the sign: a negative (truncation-induced)
            # divergence is bounded the same way as a positive one.
            per_position = per_position.clamp(min=-token_clip, max=token_clip)
        if lead_shape is not None and per_position.numel() == lead_shape[0] * lead_shape[1]:
            per_position = per_position.reshape(lead_shape)
            teacher_mass = teacher_mass.reshape(lead_shape)
        return {DISTILLATION_LOSSES_KEY: per_position, "teacher_mass": teacher_mass}

    def _as_loss(self, model_output, data):
        if tu.get_non_tensor_data(data, key="sft_distillation_validation", default=False):
            return sft_loss(config=None, model_output=model_output, data=data)
        if DISTILLATION_LOSSES_KEY not in model_output:
            raise KeyError(
                f"the model output has no {DISTILLATION_LOSSES_KEY!r}: the batch must carry the "
                "non-tensor flag needs_vocab_logits=True (or distillation_use_topk=True with a teacher payload)"
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
            loss_mask_flatten = loss_mask.values() if getattr(loss_mask, "is_nested", False) else loss_mask.reshape(-1)
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
