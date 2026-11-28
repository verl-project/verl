# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import base64
import math
import pickle
from collections.abc import Mapping, Sequence
from typing import Any, Iterator, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist

from verl.utils.device import get_device_capability, get_device_name
from verl.workers.config.rollout import validate_sglang_score_centering_sampling
from verl.workers.rollout.utils import ensure_async_iterator

SGLANG_LORA_NAME = "verl_actor_lora_name"

_DEEPSEEK_V4_FUSION_MEMBERS = (
    ("wq_a.weight", "wkv.weight"),
    ("wq_a.scale", "wkv.scale"),
    ("wq_a.weight_scale_inv", "wkv.weight_scale_inv"),
    ("compressor.wkv.weight", "compressor.wgate.weight"),
    ("indexer.compressor.wkv.weight", "indexer.compressor.wgate.weight"),
)
DEEPSEEK_V4_FUSION_GROUPS = tuple(
    tuple(f".{attention}.{member}" for member in members)
    for attention in ("self_attn", "attn")
    for members in _DEEPSEEK_V4_FUSION_MEMBERS
)


def validate_sglang_topk_request(
    sampling_params: Mapping[str, Any],
    topk_log_probs: int,
    *,
    prompt_logprobs: int | None = None,
    return_original_logprob: bool = False,
) -> Literal["logprob", "sampling_mask"]:
    """Select the SGLang response that measures the effective sampling distribution."""
    if topk_log_probs <= 0:
        raise ValueError("rollout.topk_log_probs must be positive")
    if prompt_logprobs is not None:
        raise ValueError("prompt_logprobs cannot be combined with rollout.topk_log_probs in SGLang")
    temperature = sampling_params.get("temperature", 1.0)
    if not math.isfinite(temperature) or temperature < 1e-6:
        raise ValueError(
            "SGLang rollout.topk_log_probs requires stochastic sampling with temperature > 0 "
            "(SGLang treats temperatures below 1e-6 as greedy)"
        )
    top_p, min_p = sampling_params.get("top_p", 1.0), sampling_params.get("min_p", 0.0)
    if not (0.0 < top_p <= 1.0 and 0.0 <= min_p <= 1.0):
        raise ValueError("SGLang requires top_p in (0, 1] and min_p in [0, 1]")
    top_k = sampling_params.get("top_k", -1)
    validate_sglang_score_centering_sampling(top_k, top_p, min_p)
    if top_k > 0:
        return "sampling_mask"
    if return_original_logprob:
        raise ValueError("Unset SGLANG_RETURN_ORIGINAL_LOGPROB to record temperature-scaled SGLang log-probs")
    return "logprob"


def sglang_vocab_size(model_config: Any, tokenizer_manager: Any = None) -> int | None:
    """Get the sampler's vocabulary size. Tokenizer length can omit padded model IDs."""
    engine_config = getattr(tokenizer_manager, "model_config", None)
    hf_config = getattr(model_config, "hf_config", model_config)
    for config in (engine_config, getattr(hf_config, "text_config", None), hf_config):
        size = getattr(config, "vocab_size", None)
        if isinstance(size, int | np.integer) and not isinstance(size, bool) and size > 0:
            return int(size)
    return None


def _response_field(meta_info: Mapping[str, Any], name: str) -> Any:
    if name not in meta_info:
        raise ValueError(f"Missing SGLang response field {name}")
    return meta_info[name]


def _flat_response_array(meta_info: Mapping[str, Any], name: str, dtype: str) -> np.ndarray:
    b64_key = f"{name}_b64"
    if b64_key in meta_info:
        # SGLang sends float32 and int32 binary arrays in little-endian order.
        binary_dtype = np.dtype(dtype)
        if meta_info.get(f"{b64_key}_dtype", binary_dtype.name) != binary_dtype.name:
            raise ValueError(f"Invalid SGLang dtype for {b64_key}, expected {binary_dtype.name}")
        try:
            return np.frombuffer(base64.b64decode(meta_info[b64_key], validate=True), dtype=binary_dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid SGLang base64 array in {b64_key}") from error
    values = np.asarray(_response_field(meta_info, name))
    if values.ndim != 1:
        raise ValueError(f"SGLang {name} must be flat")
    return values


def _validate_token_ids(ids: np.ndarray, vocab_size: int | None) -> None:
    if ids.size and (
        ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids > np.iinfo(np.int32).max)
        or (vocab_size is not None and np.any(ids >= vocab_size))
    ):
        raise ValueError("SGLang token ids must be nonnegative int32 ids within the model vocabulary")


def _validate_logprobs(logprobs: np.ndarray) -> None:
    if (
        logprobs.dtype.kind not in "fiu"
        or np.any(np.isnan(logprobs))
        or np.any(np.isposinf(logprobs))
        or np.any(logprobs > 1e-6)
    ):
        raise ValueError("SGLang log-probs must be nonpositive and finite or -inf")


def _validate_ranked_rows(ids: np.ndarray, logprobs: np.ndarray, vocab_size: int | None) -> None:
    _validate_token_ids(ids, vocab_size)
    _validate_logprobs(logprobs)
    if np.any(logprobs[..., 1:] > logprobs[..., :-1] + 1e-6):
        raise ValueError("SGLang log-probs must be ordered by rank")
    sorted_ids = np.sort(ids, axis=-1)
    if np.any(sorted_ids[..., 1:] == sorted_ids[..., :-1]):
        raise ValueError("SGLang token ids must be unique within each row")


def _topk_output(token_ids: Sequence[int], k: int, vocab_size: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k <= 0 or (vocab_size is not None and k > vocab_size):
        raise ValueError("SGLang head size must be positive and no larger than the model vocabulary")
    return (
        np.empty((len(token_ids), k), dtype=np.int32),
        np.empty((len(token_ids), k), dtype=np.float32),
        np.empty(len(token_ids), dtype=np.bool_),
    )


def extract_sglang_topk_logprobs_with_coverage(
    meta_info: Mapping[str, Any],
    token_ids: Sequence[int],
    k: int,
    vocab_size: int | None = None,
    *,
    mass_atol: float = 1e-5,
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Extract the sampled log-probs and ranked head of an untruncated SGLang response."""
    ids, logprobs, complete = _topk_output(token_ids, k, vocab_size)
    if len(token_ids) == 0:
        return [], ids, logprobs, complete

    sampled_ids = np.asarray(token_ids)
    _validate_token_ids(sampled_ids, vocab_size)
    sampled_entries = _response_field(meta_info, "output_token_logprobs")
    if len(sampled_entries) != len(token_ids):
        raise ValueError("SGLang sampled log-probs must align with generated token ids")
    sampled_values, reported_ids, _ = zip(*sampled_entries, strict=True)
    sampled = np.asarray(sampled_values)
    reported_ids = np.asarray(reported_ids)
    _validate_logprobs(sampled)
    _validate_token_ids(reported_ids, vocab_size)
    if not np.array_equal(reported_ids, sampled_ids) or not np.all(np.isfinite(sampled)):
        raise ValueError("SGLang sampled log-probs must be finite and align with generated token ids")

    if "output_top_logprobs_shape" in meta_info:
        if (
            tuple(meta_info["output_top_logprobs_shape"]) != ids.shape
            or _response_field(meta_info, "output_top_logprobs_null_prefix") != 0
        ):
            raise ValueError("SGLang output top log-probs must have shape (generated tokens, k) and no null prefix")
        head_ids = _flat_response_array(meta_info, "output_top_logprobs_idx_flat", "<i4").reshape(ids.shape)
        head_logprobs = _flat_response_array(meta_info, "output_top_logprobs_val_flat", "<f4").reshape(ids.shape)
    else:
        rows = _response_field(meta_info, "output_top_logprobs")
        if len(rows) != len(token_ids) or any(row is None or len(row) != k for row in rows):
            raise ValueError("SGLang output top log-probs must have shape (generated tokens, k)")
        head_logprobs, head_ids, _ = zip(*(zip(*row, strict=True) for row in rows), strict=True)
        head_ids, head_logprobs = np.asarray(head_ids), np.asarray(head_logprobs)

    _validate_ranked_rows(head_ids, head_logprobs, vocab_size)
    mass = np.exp(head_logprobs, dtype=np.float64).sum(axis=-1)
    if not np.all(np.isfinite(head_logprobs).any(axis=-1)) or np.any(mass > 1 + mass_atol):
        raise ValueError("SGLang head must have positive probability mass no larger than one")
    rows, columns = np.nonzero(head_ids == sampled_ids[:, None])
    if not np.allclose(head_logprobs[rows, columns], sampled[rows], atol=1e-5, rtol=0):
        raise ValueError("SGLang sampled log-probs must match the corresponding top log-probs")
    # SGLang clamps -inf masked logits to finfo(dtype).min (<= -65504), which stays finite
    # and would falsely mark excluded tokens in-support if certified complete.
    has_clamped = np.isfinite(head_logprobs) & (head_logprobs <= -65504.0)
    complete[:] = (k == vocab_size) & ~has_clamped.any(axis=-1)
    if np.any(complete):
        sampled_in_head = (head_ids == sampled_ids[:, None]).any(axis=-1)
        if not np.all(sampled_in_head[complete]) or not np.allclose(mass[complete], 1.0, atol=mass_atol, rtol=0):
            raise ValueError(
                "SGLang full-vocabulary head must contain the sampled token and have unit probability mass"
            )
    ids[:] = head_ids
    logprobs[:] = head_logprobs
    return sampled.astype(float, copy=False).tolist(), ids, logprobs, complete


def extract_sglang_sampling_mask_with_coverage(
    meta_info: Mapping[str, Any], token_ids: Sequence[int], k: int, vocab_size: int | None = None
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Extract the head and sampled log-probs from the same full, normalized sampling support."""
    ids, logprobs, complete = _topk_output(token_ids, k, vocab_size)
    if len(token_ids) == 0:
        return [], ids, logprobs, complete
    _validate_token_ids(np.asarray(token_ids), vocab_size)

    mask_key = "output_token_sampling_mask"
    logprobs_key = "output_token_sampling_logprobs"
    length_key = f"{mask_key}_length"
    if f"{mask_key}_flat" in meta_info or f"{mask_key}_flat_b64" in meta_info:
        lengths = _flat_response_array(meta_info, f"{mask_key}_row_lengths", "<i4")
        flat_ids = _flat_response_array(meta_info, f"{mask_key}_flat", "<i4")
        flat_logprobs = _flat_response_array(meta_info, f"{logprobs_key}_flat", "<f4")
        if (
            _response_field(meta_info, length_key) != len(token_ids)
            or len(lengths) != len(token_ids)
            or lengths.dtype.kind not in "iu"
            or np.any(lengths <= 0)
            or sum(map(int, lengths)) != len(flat_ids)
            or len(flat_ids) != len(flat_logprobs)
        ):
            raise ValueError("SGLang flat sampling mask lengths must align with generated tokens and support entries")
        splits = np.cumsum(lengths, dtype=np.int64)[:-1]
        id_rows, logprob_rows = np.split(flat_ids, splits), np.split(flat_logprobs, splits)
    else:
        id_rows = _response_field(meta_info, mask_key)
        logprob_rows = _response_field(meta_info, logprobs_key)
        if (
            meta_info.get(length_key, len(token_ids)) != len(token_ids)
            or len(id_rows) != len(token_ids)
            or len(logprob_rows) != len(token_ids)
        ):
            raise ValueError("SGLang sampling mask row counts must match the number of generated tokens")

    sampled: list[float] = []
    logprobs.fill(-np.inf)
    for t, (token_id, row_ids, row_logprobs) in enumerate(zip(token_ids, id_rows, logprob_rows, strict=True)):
        row_ids, row_logprobs = np.asarray(row_ids), np.asarray(row_logprobs)
        if row_ids.ndim != 1 or row_ids.size == 0 or row_ids.shape != row_logprobs.shape:
            raise ValueError(f"SGLang sampling mask row {t} must have one log-prob for each support token")
        _validate_ranked_rows(row_ids, row_logprobs, vocab_size)
        weights = np.exp(row_logprobs, dtype=np.float64)
        if np.any(weights == 0):
            raise ValueError(f"SGLang sampling mask row {t} must contain only positive-probability support tokens")
        if not np.isclose(weights.sum(), 1.0, atol=1e-4, rtol=0) or weights[:k].sum() > 1 + 1e-5:
            raise ValueError(f"SGLang sampling mask row {t} must have unit support mass and head mass at most one")
        matching = np.flatnonzero(row_ids == token_id)
        if not matching.size:
            raise ValueError(f"SGLang sampled token {token_id} is missing from the sampling mask support at row {t}")
        sampled.append(float(row_logprobs[matching[0]]))
        width = min(row_ids.size, k)
        complete[t] = row_ids.size <= k
        ids[t, :width] = row_ids[:width]
        logprobs[t, :width] = row_logprobs[:width]
        if width < k:
            available = np.ones(k, dtype=np.bool_)
            available[row_ids[row_ids < k]] = False
            ids[t, width:] = np.flatnonzero(available)[: k - width]
    return sampled, ids, logprobs, complete


def _fusion_key(name: str, groups: tuple[tuple[str, ...], ...]):
    matches = [
        (index, suffix, len(group)) for index, group in enumerate(groups) for suffix in group if name.endswith(suffix)
    ]
    if len(matches) > 1:
        raise ValueError(f"{name!r} matches multiple fusion groups: {matches}")
    if not matches:
        return None
    index, suffix, size = matches[0]
    return name[: -len(suffix)], index, size


def normalize_peft_config_for_sglang(peft_config: dict) -> dict:
    """Normalize an engine's adapter config (enums to strings) for SGLang's adapter loader."""
    normalized = dict(peft_config)
    for key in ("task_type", "peft_type"):
        if key in normalized:
            normalized[key] = getattr(normalized[key], "value", normalized[key])
    if "peft_type" not in normalized:
        raise ValueError(
            "adapter config has no 'peft_type', which SGLang's adapter loader requires. See "
            "BaseEngine.get_per_tensor_param for the keys. Keys present: " + ", ".join(sorted(normalized))
        )
    # A bare string must stay one: list() would tear "all-linear" into characters.
    target_modules = normalized["target_modules"]
    normalized["target_modules"] = target_modules if isinstance(target_modules, str) else list(target_modules)
    return normalized


def lora_rank_of(model_config) -> int:
    """The LoRA rank, from whichever block carries it: megatron sets ``model.lora.rank``, fsdp
    the flat ``model.lora_rank``."""
    return max(int(getattr(model_config, "lora_rank", 0) or 0), int(model_config.lora.get("rank", 0) or 0))


def lora_served_as_adapter(model_config) -> bool:
    """Whether SGLang should serve LoRA as a hot-swappable adapter.

    ``HFModelConfig`` carries two LoRA config blocks that are never synced: megatron
    runs set ``model.lora.rank`` (dict) while fsdp runs set the flat ``model.lora_rank``,
    so both must be checked to detect that LoRA is enabled at all.

    With ``model.lora.merge=True`` the trainer merges the adapter into the base weights
    and pushes a full HF-keyed weight update (``peft_config=None``), so no adapter is ever
    loaded into SGLang: the engine must not be launched with ``enable_lora`` and requests
    must not carry a ``lora_path``.
    """
    lora_enabled = lora_rank_of(model_config) > 0
    return lora_enabled and not model_config.lora.get("merge", False)


def lora_base_kept_on_sleep(model_config) -> bool:
    """Whether sleep keeps the base weights on GPU and releases only the kv_cache.

    True in adapter mode (see :func:`lora_served_as_adapter`), where the trainer pushes only
    the adapter after the first sync. With ``model.lora.resync_base=True`` the trainer re-syncs
    the base before the adapter on every update, so sleep releases the weights too.
    """
    return lora_served_as_adapter(model_config) and not model_config.lora.get("resync_base", False)


def sglang_lora_target_modules(target_modules: Any) -> list[str]:
    """Render verl's ``model.target_modules`` as SGLang's ``lora_target_modules``."""
    if target_modules == "all-linear":
        return ["all"]
    if isinstance(target_modules, str):
        raise ValueError(
            f"SGLang cannot serve a regex `target_modules` ({target_modules!r}); PEFT matches it "
            f"against the whole parameter key. Use `all-linear`, or list the module names."
        )
    return list(target_modules)


def default_moe_runner_backend(quantization: Optional[str], engine_kwargs: Any, hf_config: Any = None) -> Optional[str]:
    """MoE runner verl asks SGLang for when the user leaves it to ``auto``; None keeps SGLang's choice.

    On Blackwell (compute capability 10.x) SGLang resolves ``moe_runner_backend="auto"`` to ``flashinfer_trtllm``
    for unquantized Qwen3-MoE- and DeepSeek-family checkpoints when no ``moe_a2a_backend`` is set. That runner keeps
    bf16 expert weights in a block layout, and ``update_weights_from_tensor`` restores the load-time shape only for
    ``flashinfer_trtllm_routed``, so the first weight sync fails in ``FusedMoE._load_w13`` and the scheduler exits
    (sgl-project/sglang#27787, still in sglang 0.5.21). Under the same conditions verl asks for the Triton runner,
    which SGLang's unquantized MoE method falls back to everywhere else and which takes weight updates. A quantized
    rollout or checkpoint uses SGLang's quantized MoE methods instead and is left alone.
    """
    if engine_kwargs.get("moe_runner_backend") not in (None, "auto"):
        return None  # the user picked a runner
    if engine_kwargs.get("moe_a2a_backend") not in (None, "none"):
        return None  # with an a2a backend SGLang does not pick flashinfer_trtllm
    if quantization is not None or getattr(hf_config, "quantization_config", None) is not None:
        return None  # quantized rollout or checkpoint: not the bf16 path
    if torch.version.hip is not None:
        return None
    major, _ = get_device_capability()
    return "triton" if major == 10 else None


def default_stop_token_ids(model_config: Any) -> tuple[int, ...]:
    """Collect default EOS token IDs from ``HFModelConfig`` for SGLang rollout.

    ``SGLangHttpServer`` defaults to ``skip_tokenizer_init=True``, so SGLang's scheduler runs
    with ``tokenizer=None`` and reads ``hf_eos_token_id`` only from top-level ``config.json``
    and ``generation_config.json`` on disk. Collecting from ``model_config`` preserves
    ``tokenizer.eos_token_id``, nested VLM ``text_config.eos_token_id``, and in-memory
    ``model.override_config`` / ``tokenizer_path`` / ``hf_config_path`` overrides.
    """
    hf_config = getattr(model_config, "hf_config", None)
    sources = (
        getattr(hf_config, "eos_token_id", None),
        getattr(getattr(hf_config, "text_config", None), "eos_token_id", None),
        getattr(getattr(model_config, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model_config, "tokenizer", None), "eos_token_id", None),
    )
    stop_ids: list[int] = []
    for src in sources:
        if src is None or isinstance(src, bool):
            continue
        items = (src,) if isinstance(src, int | np.integer) else src
        for item in items:
            if isinstance(item, int | np.integer) and not isinstance(item, bool):
                token_id = int(item)
                if token_id not in stop_ids:
                    stop_ids.append(token_id)
    return tuple(stop_ids)


def broadcast_pyobj(
    data: list[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    """from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(get_device_name() if not force_cpu_device else "cpu")

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, dtype=np.uint8)).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


def _compact_for_bucket(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor safe to retain in a weight-sync bucket without pinning extra memory.

    ``get_named_tensor_buckets`` keeps every tensor alive until its bucket is flushed. A tensor
    that is a *view* into a larger backing buffer would therefore keep that whole buffer resident
    (and ship the whole buffer downstream), so such views must be compacted with ``clone()``.

    However the weights synced here come from ``DTensor.full_tensor()`` (a fresh all-gather) and
    already own tight, contiguous storage. Cloning those allocates a second full-size buffer and
    transiently doubles the tensor's footprint -- which OOMs on multi-GiB fused MoE weights
    (e.g. ``[num_experts, ...]`` ``gate_up_proj``/``qkv``) while the actor params and rollout
    weights are both already resident. Skip the clone when the tensor already owns its storage.
    """
    return tensor.clone() if tensor._base is not None else tensor


async def get_named_tensor_buckets(
    iterable: Iterator[tuple[str, torch.Tensor]],
    bucket_bytes: int,
    fusion_groups: tuple[tuple[str, ...], ...] = (),
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """
    Group tensors into buckets based on a specified size in megabytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]

    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0
    # Members of a fused destination param (SGLang's DSv4 wqkv_a / wkv_gate) are
    # held here until the group is whole, then placed in one bucket: that loader
    # cats both halves inside a single load_weights call and asserts its cache
    # empty on return, so a bucket boundary between them makes it fire. A full
    # sync contains every member by construction, so this only ever delays a
    # tensor by its sibling -- nothing is dropped.
    staged: dict = {}

    def _place(entries):
        """Put an indivisible run of entries in a bucket, cutting before it if
        it would not fit rather than partway through."""
        nonlocal current_bucket, current_size
        total = sum(sz for _, _, sz in entries)
        if current_bucket and current_size + total > bucket_bytes:
            out, current_bucket, current_size = current_bucket, [], 0
            return out, entries
        return None, entries

    async for name, tensor in ensure_async_iterator(iterable):
        tensor_size = tensor.element_size() * tensor.numel()
        match = _fusion_key(name, fusion_groups)
        if match is not None:
            prefix, group_index, expected_size = match
            key = (prefix, group_index)
            slot = staged.setdefault(key, [])
            slot.append((name, _compact_for_bucket(tensor), tensor_size))
            if len(slot) < expected_size:
                continue
            entries = staged.pop(key)
        else:
            entries = [(name, _compact_for_bucket(tensor), tensor_size)]

        flushed, entries = _place(entries)
        if flushed:
            yield flushed
        for n, t, sz in entries:
            current_bucket.append((n, t))
            current_size += sz

    if staged:
        raise ValueError(f"fused parameter groups never completed in this sync: {sorted(staged)}")
    if current_bucket:
        yield current_bucket
