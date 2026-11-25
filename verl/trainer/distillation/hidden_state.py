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

"""Full-vocabulary forward KL from response-aligned final teacher hidden states."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open


def teacher_output_scale(config):
    """Resolve the multiplier used by the vLLM model's LogitsProcessor."""
    model = config.model_type
    multiplier = getattr(config, "logit_scale", 1.0)
    divisor = getattr(config, "logits_scaling", 1.0)
    if model in ("llama", "cohere", "cohere2"):
        return multiplier
    if model == "granite":
        return multiplier / divisor
    if model in ("granitemoe", "granitemoeshared", "granitemoehybrid"):
        return 1.0 / divisor
    for name in ("logit_scale", "logits_scaling", "output_multiplier_scale", "logits_scale"):
        if getattr(config, name, None) not in (None, 1.0):
            raise ValueError(f"Unsupported teacher output scaling {name!r} for model {model!r}")
    return 1.0


@lru_cache(maxsize=32)
def _head_info(model_path):
    from transformers import AutoConfig

    from verl.utils.fs import copy_to_local

    directory = Path(copy_to_local(model_path))
    if not directory.is_dir():
        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(model_path, allow_patterns=["config.json", "*.safetensors*"]))
    config = AutoConfig.from_pretrained(directory)
    text = getattr(config, "text_config", config)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        files = json.loads(index.read_text())["weight_map"]
    else:
        filename = directory / "model.safetensors"
        with safe_open(filename, framework="pt") as checkpoint:
            files = dict.fromkeys(checkpoint.keys(), filename.name)

    key = "lm_head.weight"
    if key not in files:
        embeds = [k for k in files if k.endswith(".embed_tokens.weight")]
        if not getattr(text, "tie_word_embeddings", config.tie_word_embeddings) or len(embeds) != 1:
            raise ValueError(f"Cannot resolve an unambiguous prediction head in {directory}")
        key = embeds[0]
    if "lm_head.bias" in files:
        raise ValueError(f"Hidden-state OPD requires a bias-free prediction head in {directory}")
    return (
        directory / files[key],
        key,
        text.vocab_size,
        getattr(text, "final_logit_softcapping", None),
        teacher_output_scale(text),
    )


@dataclass(frozen=True)
class _Head:
    path: str
    version: str
    tp_rank: int
    tp_size: int
    local_vocab: int
    fsdp_group: object = None


class _ResidentHead:
    """Keep only the active checkpoint's local weight shard on the actor device."""

    spec = None
    weight = None

    @classmethod
    def get(cls, spec, device):
        filename, key, vocab, softcap, scale = _head_info(spec.path)
        if spec.local_vocab * spec.tp_size < vocab:
            raise ValueError(f"Actor vocabulary shards cover fewer than the {vocab} teacher logits")
        fsdp_size = dist.get_world_size(spec.fsdp_group) if spec.fsdp_group is not None else 1
        fsdp_rank = dist.get_rank(spec.fsdp_group) if spec.fsdp_group is not None else 0
        shard_width = (vocab + fsdp_size - 1) // fsdp_size if fsdp_size > 1 else spec.local_vocab
        start = (fsdp_rank if fsdp_size > 1 else spec.tp_rank) * shard_width
        stop = min(start + shard_width, vocab)
        if cls.spec != (spec, device):
            cls.spec = None
            cls.weight = None
            with safe_open(filename, framework="pt") as checkpoint:
                weight = checkpoint.get_slice(key)
                if weight.get_shape()[0] != vocab:
                    raise ValueError(f"Head {key} does not match configured vocabulary {vocab}")
                local = weight[min(start, vocab) : stop].to(device)
            if fsdp_size > 1 and local.shape[0] < shard_width:
                local = F.pad(local, (0, 0, 0, shard_width - local.shape[0]))
            cls.weight = local
            cls.spec = (spec, device)
        if fsdp_size == 1:
            return cls.weight, softcap, scale

        # The actor's FSDP group can have different samples/teachers on each rank.
        # All ranks visit the same active teacher keys before the per-rank KL work.
        full = cls.weight.new_empty(shard_width * fsdp_size, cls.weight.shape[1])
        dist.all_gather_into_tensor(full, cls.weight.contiguous(), group=spec.fsdp_group)
        return full[:vocab], softcap, scale


def _log_normalizer(logits, tp_group):
    # Empty local vocab shards can occur when the actor pads a very small vocab.
    local_max = logits.amax(-1) if logits.shape[1] else logits.new_full((logits.shape[0],), -torch.inf)
    if tp_group is not None:
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
    total = (logits - local_max[:, None]).exp().sum(-1)
    if tp_group is not None:
        dist.all_reduce(total, group=tp_group)
    return local_max + total.log()


@dataclass
class _TeacherTokens:
    head: _Head
    # Local flat-logit row indices and corresponding CPU teacher signals.
    rows: list
    signal_kind: str = "hidden_states"


def _project(hidden, weight, softcap, scale):
    # Forward and backward must reconstruct identical logits from the checkpoint head.
    with torch.autocast(device_type=weight.device.type, enabled=False):
        logits = F.linear(hidden.to(device=weight.device, dtype=weight.dtype), weight)
        if softcap:
            logits = (logits / softcap).tanh() * softcap
        if scale != 1.0:
            logits *= scale
        return logits


@torch.no_grad()
def project_full_logits(model_path, version, hidden, device, chunk_size):
    """Produce the dense TransferQueue baseline on the teacher's output rank."""
    _, _, vocab, _, _ = _head_info(model_path)
    head = _Head(model_path, version, 0, 1, vocab)
    weight, softcap, scale = _ResidentHead.get(head, device)
    result = torch.empty(len(hidden), vocab, dtype=weight.dtype, device="cpu")
    for start in range(0, len(hidden), chunk_size):
        stop = start + chunk_size
        result[start:stop].copy_(_project(hidden[start:stop], weight, softcap, scale))
    return result


def _chunks(teachers, chunk_size, device):
    for teacher in teachers:
        if teacher.signal_kind == "full_logits":
            head = teacher.head
            lo = head.tp_rank * head.local_vocab
            for positions, logits in teacher.rows:
                if logits.size(1) > head.tp_size * head.local_vocab:
                    raise ValueError("Actor vocabulary shards cover fewer than the transferred teacher logits")
                for start in range(0, len(positions), chunk_size):
                    stop = start + chunk_size
                    shard = logits[start:stop, lo : lo + head.local_vocab]
                    # KL normalizes its input in place; preserve the CPU record for backward.
                    yield positions[start:stop].to(device), shard.to(device=device, dtype=torch.float32, copy=True)
            continue
        weight, softcap, scale = _ResidentHead.get(teacher.head, device)
        for positions, hidden in teacher.rows:
            for start in range(0, len(positions), chunk_size):
                stop = start + chunk_size
                teacher_logits = _project(hidden[start:stop], weight, softcap, scale)
                yield positions[start:stop].to(device), teacher_logits.float()
        del weight


class _HiddenStateKL(torch.autograd.Function):
    """Re-project the teacher in backward, saving only two log-normalizers per token."""

    @staticmethod
    def forward(ctx, student, teachers, chunk_size, tp_group):
        ctx.teachers = teachers
        ctx.chunk_size = chunk_size
        flat = student.reshape(-1, student.size(-1))
        loss = torch.zeros(flat.size(0), device=student.device, dtype=torch.float32)
        norms = loss.new_zeros((flat.size(0), 2))
        for indices, t in _chunks(teachers, chunk_size, student.device):
            s = flat[:, : t.size(1)].index_select(0, indices).float()
            log_t = _log_normalizer(t, tp_group)
            log_s = _log_normalizer(s, tp_group)
            norms[indices] = torch.stack((log_t, log_s), dim=1)
            t -= log_t[:, None]
            s -= log_s[:, None]
            chunk_loss = (t.exp() * (t - s)).sum(-1)
            if tp_group is not None:
                dist.all_reduce(chunk_loss, group=tp_group)
            loss[indices] = chunk_loss
        ctx.save_for_backward(student, norms)
        return loss.view(student.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):
        student, norms = ctx.saved_tensors
        flat = student.reshape(-1, student.size(-1))
        grad = torch.zeros_like(flat)
        scale = grad_output.reshape(-1)
        for indices, t in _chunks(ctx.teachers, ctx.chunk_size, student.device):
            width = t.size(1)
            s = flat[:, :width].index_select(0, indices).float()
            t -= norms[indices, 0, None]
            s -= norms[indices, 1, None]
            values = (s.exp() - t.exp()) * scale[indices, None]
            grad[indices, :width] = values.to(grad.dtype)
        return grad.view_as(student), None, None, None


def hidden_state_kl(actor, config, data, logits, token_indices=None, fsdp_group=None):
    from verl.experimental.teacher_loop.hidden_state_store import read_hidden_records

    offsets = data["input_ids"].offsets().tolist()
    tp_group = None
    rank, size = 0, 1
    if actor.strategy == "megatron":
        from megatron.core import parallel_state as mpu

        size = mpu.get_tensor_model_parallel_world_size()
        rank = mpu.get_tensor_model_parallel_rank()
        tp_group = mpu.get_tensor_model_parallel_group() if size > 1 else None
        if token_indices is None:
            raise ValueError("Missing Megatron predictor indices for hidden-state distillation")
    elif actor.strategy in ("fsdp", "fsdp2"):
        if token_indices is None:
            token_indices = torch.arange(1, offsets[-1] + 1).unsqueeze(0)
    else:
        raise ValueError(f"Hidden-state distillation requires a Megatron or FSDP actor, got {actor.strategy}")
    if token_indices.shape != logits.shape[:-1]:
        raise ValueError(f"Hidden-state predictor indices {token_indices.shape} do not match logits {logits.shape}")

    # Invert the local layout, keeping -1 for tokens owned by other CP/SP ranks.
    # Index zero is the padding sentinel in the layout, never a real token.
    ordinals = token_indices.detach().flatten().cpu()
    inverse = torch.full((offsets[-1] + 1,), -1, dtype=torch.long)
    valid = ordinals != 0
    inverse[ordinals[valid]] = torch.arange(ordinals.numel())[valid]

    rows = {}
    versions = {}
    signal_kind = getattr(getattr(config, "teacher_execution", None), "transfer_signal", "hidden_states")
    teacher_indices = {key: i for i, key in enumerate(config.teacher_models)}
    records = read_hidden_records(data)
    if len(records) != len(offsets) - 1:
        raise ValueError("Expected one teacher record per actor trajectory")
    for row, (meta, signal) in enumerate(records):
        if meta.get("signal_kind", "hidden_states") != signal_kind:
            raise ValueError(f"Unexpected teacher signal in record {row}, expected {signal_kind!r}")
        key = meta["teacher_key"]
        if key not in teacher_indices or meta["teacher_index"] != teacher_indices[key]:
            raise ValueError(f"Record {row} has an invalid teacher registry index for {key!r}")
        if len(teacher_indices) > 1 and data[config.teacher_key][row] != key:
            raise ValueError(f"Record {row} belongs to teacher {key!r}, conflicting with trajectory routing")
        version = meta["teacher_weight_version"]
        if key in versions and version != versions[key]:
            raise ValueError(f"Multiple weight versions in one step for teacher {key!r}")
        versions[key] = version
        positions = torch.as_tensor(meta["positions"], dtype=torch.long)
        local_rows = inverse[positions + offsets[row] + 1]
        keep = local_rows >= 0
        rows.setdefault(key, []).append((local_rows[keep], signal[keep]))
    if fsdp_group is not None and dist.get_world_size(fsdp_group) > 1:
        rank_versions = [None] * dist.get_world_size(fsdp_group)
        dist.all_gather_object(rank_versions, versions, group=fsdp_group)
        for other in rank_versions:
            for key, version in other.items():
                if key in versions and versions[key] != version:
                    raise ValueError(f"FSDP actors received conflicting versions for teacher {key!r}")
                versions[key] = version

    teachers = [
        _TeacherTokens(
            _Head(config.teacher_models[key].model_path, version, rank, size, logits.size(-1), fsdp_group),
            rows.get(key, []),
            signal_kind,
        )
        for key, version in sorted(versions.items())
    ]
    return _HiddenStateKL.apply(logits, teachers, config.distillation_loss.hidden_state_chunk_size, tp_group)
