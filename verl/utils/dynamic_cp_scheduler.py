# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""
Dynamic Context Parallelism Scheduler for verl.

Uses Megatron-Core's Dynamic CP scheduler to assign CP sizes and micro-batches,
then adapts verl TensorDict batches through forward and reverse all-to-all routing.
"""

import hashlib
import logging
import os
import time
from collections import Counter

import torch
import torch.distributed
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MAX_SHAPE_DIMS = 8
DCP_RESPONSE_LENGTH_KEY = "_dcp_response_lengths"
_DCP_REPLICATED_FULL_SEQUENCE_OUTPUT_KEYS = frozenset({"routed_experts"})


def _get_megatron_dynamic_cp_scheduler_cls() -> type:
    """Load the Dynamic CP scheduler provided by Megatron-Core."""
    try:
        from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Dynamic CP requires a Megatron-Core build containing NVIDIA/Megatron-LM#5154 (merge commit d2e7ec5b)."
        ) from exc
    return DefaultDynamicCPScheduler


def _dcp_profile_enabled() -> bool:
    return os.getenv("VERL_DCP_PROFILE", "0").lower() in {"1", "true", "yes", "on"}


def _profile_now() -> float:
    device = get_torch_device()
    if hasattr(device, "synchronize"):
        device.synchronize()
    return time.perf_counter()


def _profile_elapsed_ms(start: float, group=None) -> float:
    device = get_torch_device()
    if hasattr(device, "synchronize"):
        device.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        dev = get_device_id() if hasattr(device, "synchronize") else "cpu"
        elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device=dev)
        torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX, group=group)
        elapsed_ms = float(elapsed.item())
    return elapsed_ms


def _profile_log(group, message: str) -> None:
    if group is None or group.rank() == 0:
        logger.info(message)


def _count_cp_groups(sample_id_groups: list[list[list[int]]]) -> dict[int, int]:
    """Count global CP group sizes from scheduler assignments."""
    counts = Counter()
    for sample_id_group in sample_id_groups:
        rank = 0
        total_ranks = len(sample_id_group)
        while rank < total_ranks:
            rank_ids = sample_id_group[rank]
            if not rank_ids:
                rank += 1
                continue
            first_sample_id = rank_ids[0]
            cp_size = 0
            while rank + cp_size < total_ranks and first_sample_id in sample_id_group[rank + cp_size]:
                cp_size += 1
            counts[cp_size] += 1
            rank += cp_size
    return dict(sorted(counts.items()))


# =============================================================================
# Data collection: gather global sequence lengths
# =============================================================================


def _get_global_seqlens_and_ids(
    local_seqlens: torch.Tensor, dp_group
) -> tuple[list[tuple[int, int]], torch.Tensor, torch.Tensor]:
    """Gather sequence lengths from all DP ranks and assign global IDs.

    Mirrors the private Megatron-Core helper of the same name in
    data_schedule_utils.py (int32 lengths, zero padding, dp-rank-major
    concatenation) so the imported scheduler sees the exact global ID
    protocol it was built for; it is not imported directly because it is
    private there and returns a different tuple.

    Args:
        local_seqlens: 1D int tensor of sequence lengths on this rank.
        dp_group: Data parallel process group.

    Returns:
        global_id_seqlens: List of (global_id, seq_len) for ALL samples across all ranks.
        global_ids_this_rank: 1D tensor of global IDs present on this rank.
        offsets: Cumulative offset tensor for each DP rank.
    """
    num_local = local_seqlens.shape[0]
    local_len = torch.tensor([num_local], dtype=torch.int32, device=local_seqlens.device)
    dp_counts = [torch.zeros_like(local_len) for _ in range(dp_group.size())]
    torch.distributed.all_gather(dp_counts, local_len, group=dp_group)

    dp_counts_cpu = torch.stack(dp_counts, dim=0).cpu().view(-1)
    max_count = int(dp_counts_cpu.max().item())

    # Pad local seqlens to max_count for all_gather
    if num_local < max_count:
        padded = torch.cat(
            [local_seqlens, torch.zeros(max_count - num_local, dtype=torch.int32, device=local_seqlens.device)],
            dim=0,
        )
    else:
        padded = local_seqlens

    gathered = [torch.empty_like(padded) for _ in range(dp_group.size())]
    torch.distributed.all_gather(gathered, padded, group=dp_group)

    # Trim each rank's data to actual length
    for rank_idx, seqlen_tensor in enumerate(gathered):
        gathered[rank_idx] = seqlen_tensor[: dp_counts_cpu[rank_idx]]

    all_seqlens = torch.cat(gathered, dim=0).cpu().tolist()

    # Compute offsets
    csum = torch.cumsum(dp_counts_cpu, dim=0, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum], dim=0)

    # Global IDs
    dp_rank = dp_group.rank()
    global_id_seqlens = [(i, all_seqlens[i]) for i in range(len(all_seqlens))]
    start_idx = offsets[dp_rank].item()
    end_idx = offsets[dp_rank + 1].item()
    global_ids_this_rank = torch.arange(start_idx, end_idx, dtype=torch.int32, device=local_seqlens.device)

    return global_id_seqlens, global_ids_this_rank, offsets


# =============================================================================
# All-to-all routing
# =============================================================================


def _unique_keys(keys: list[str]) -> list[str]:
    seen = set()
    unique = []
    for key in keys:
        if key not in seen:
            unique.append(key)
            seen.add(key)
    return unique


def _group_rank_indices(group, within_group) -> list[int]:
    """Return ranks from `group` expressed as rank indices in `within_group`."""
    within_ranks = torch.distributed.get_process_group_ranks(within_group)
    rank_to_idx = {rank: idx for idx, rank in enumerate(within_ranks)}
    return [rank_to_idx[rank] for rank in torch.distributed.get_process_group_ranks(group)]


def _infer_same_cp_dcp_rank(
    dp_rank: int,
    peer_dcp_rank: int,
    dp_size: int,
    cp_size: int,
    current_dp_rank_to_dcp_rank: list[int],
) -> int:
    """Map a DP rank to the DPxCP rank that shares `peer_dcp_rank`'s CP column."""
    if cp_size == 1:
        return current_dp_rank_to_dcp_rank[dp_rank]
    if dp_size == 1:
        return peer_dcp_rank

    dp_stride = current_dp_rank_to_dcp_rank[1] - current_dp_rank_to_dcp_rank[0]
    if abs(dp_stride) == cp_size:
        cp_rank = peer_dcp_rank % cp_size
        return dp_rank * cp_size + cp_rank
    if abs(dp_stride) == 1:
        cp_rank = peer_dcp_rank // dp_size
        return cp_rank * dp_size + dp_rank

    raise ValueError(
        "Cannot infer DPxCP rank layout from current DP group mapping "
        f"{current_dp_rank_to_dcp_rank} with dp_size={dp_size}, cp_size={cp_size}"
    )


def _dp_src_rank_for_gid(gid: int, offsets: torch.Tensor) -> int:
    return int(torch.bucketize(torch.tensor(gid, dtype=offsets.dtype), offsets[1:] - 1).item())


def _shape_row(tensor: torch.Tensor) -> list[int]:
    shape = list(tensor.shape)
    if len(shape) > MAX_SHAPE_DIMS:
        raise ValueError(f"DCP routing supports tensors with at most {MAX_SHAPE_DIMS} dims, got shape={shape}")
    return [len(shape), *shape, *([0] * (MAX_SHAPE_DIMS - len(shape)))]


def _missing_shape_row() -> list[int]:
    """Return the shape-metadata sentinel for a missing routed value."""
    return [-1, *([0] * MAX_SHAPE_DIMS)]


def _shape_row_is_missing(row: torch.Tensor, numel: int) -> bool:
    """Distinguish a missing value from a present tensor with zero elements."""
    ndim = int(row[0].item())
    if ndim < 0:
        if numel != 0:
            raise ValueError(f"DCP missing-value shape sentinel carried a non-empty payload ({numel} elements)")
        return True
    return False


def _shape_from_row(row: torch.Tensor) -> tuple[int, ...]:
    ndim = int(row[0].item())
    if ndim < 0:
        raise ValueError("Cannot decode the missing-value DCP shape sentinel as a tensor shape")
    if ndim == 0:
        return ()
    return tuple(int(x.item()) for x in row[1 : 1 + ndim])


def _elem_splits_per_rank(numels: list[int], counts_per_rank: list[int]) -> list[int]:
    """Sum flat per-sample element counts into one all-to-all split per peer rank."""
    splits = []
    cursor = 0
    for count in counts_per_rank:
        splits.append(sum(numels[cursor : cursor + count]))
        cursor += count
    return splits


def _stack_or_pad_samples(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-sample dense tensors, right-padding dimensions when DP ranks used different max lengths."""
    if not tensors:
        return torch.empty(0)
    shapes = [tuple(t.shape) for t in tensors]
    if len(set(shapes)) == 1:
        return torch.stack(tensors, dim=0)

    ndim = max(len(shape) for shape in shapes)
    if any(len(shape) != ndim for shape in shapes):
        raise ValueError(f"Cannot pad tensors with different ranks: {shapes}")

    max_shape = tuple(max(shape[dim] for shape in shapes) for dim in range(ndim))
    output = tensors[0].new_zeros((len(tensors), *max_shape))
    for i, tensor in enumerate(tensors):
        slices = (i, *[slice(0, size) for size in tensor.shape])
        output[slices] = tensor
    return output


def _collective_raise(dcp_group, local_error: str | None) -> None:
    """Raise on every rank when any rank rejected its local data.

    Data-dependent validation must fail collectively: a single rank raising
    before the scheduling collectives strands its peers in all_gather or
    all-to-all until the distributed timeout.
    """
    if dcp_group is None or dcp_group.size() <= 1:
        if local_error is not None:
            raise ValueError(local_error)
        return
    flag = torch.tensor([1.0 if local_error is not None else 0.0], dtype=torch.float32, device=get_device_id())
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=dcp_group)
    if flag.item() > 0:
        raise ValueError(local_error or "A peer DCP rank rejected its local batch during scheduling")


def _validate_routing_schema(schema: list[tuple[str, ...]], dcp_group) -> None:
    """Collectively validate the ordered routing schema before all-to-all.

    Every rank must enter the per-key all-to-all loop with the same keys in the
    same order and with matching routing categories and dtypes.  Compare a
    fixed-size digest so a schema mismatch itself cannot change the collective
    shape and strand peers in different all-to-all calls.
    """
    if dcp_group.size() <= 1:
        return

    digest = hashlib.sha256(repr(schema).encode()).digest()
    digest_words = [int.from_bytes(digest[i : i + 8], byteorder="little") & ((1 << 63) - 1) for i in range(0, 32, 8)]
    signature = torch.tensor([len(schema), *digest_words], dtype=torch.int64, device=get_device_id())
    gathered = [torch.empty_like(signature) for _ in range(dcp_group.size())]
    torch.distributed.all_gather(gathered, signature, group=dcp_group)

    if any(not torch.equal(peer, signature) for peer in gathered):
        raise ValueError(
            f"DCP routed tensor schema differs across ranks before all-to-all; local ordered schema={schema}"
        )


def _classify_routed_batch_fields(
    batch: TensorDict,
    ordered_keys: list[str],
    scalar_key_names: set[str],
    dcp_group,
) -> tuple[list[str], list[str], list[str]]:
    """Collectively validate and classify fields before routing collectives."""
    tensor_keys = []
    padded_keys = []
    scalar_keys = []
    errors = []
    schema = []

    input_ids = batch.get("input_ids", None)
    if not isinstance(input_ids, torch.Tensor):
        input_state = "missing_or_non_tensor"
        input_dtype = "-"
        input_ndim = "-"
        errors.append("Dynamic CP requires tensor input_ids")
    elif not input_ids.is_nested:
        input_state = "dense"
        input_dtype = str(input_ids.dtype)
        input_ndim = str(input_ids.dim())
        errors.append("Dynamic CP requires input_ids to be a NestedTensor")
    else:
        input_state = "nested"
        input_dtype = str(input_ids.dtype)
        input_ndim = str(input_ids.dim())
        if input_ids.offsets().numel() - 1 != len(batch):
            input_state = "nested_invalid_batch"
            errors.append("Dynamic CP input_ids nested batch size does not match the TensorDict batch size")
    schema.append(("__dcp_required_input_ids__", input_state, input_dtype, input_ndim))

    # The number of dimensions is part of the collective schema: ranks whose
    # same-named field disagrees in rank would otherwise only fail on the peers
    # that mix both layouts in one packed micro-batch, stranding the rest.
    for key in ordered_keys:
        val = batch[key]
        if not isinstance(val, torch.Tensor):
            if key in scalar_key_names:
                schema.append((key, "replicated_metadata", type(val).__qualname__, "-"))
                continue
            schema.append((key, "invalid_non_tensor", type(val).__qualname__, "-"))
            errors.append(f"DCP routed field {key!r} must be a tensor")
            continue

        dtype_name = str(val.dtype)
        ndim_name = str(val.dim())
        if key in scalar_key_names:
            if val.is_nested or val.dim() == 0 or val.shape[0] != len(batch) or val.numel() != len(batch):
                schema.append((key, "invalid_scalar_batch", dtype_name, ndim_name))
                errors.append(f"DCP scalar field {key!r} must contain exactly one dense value per sample")
            else:
                schema.append((key, "scalar", dtype_name, ndim_name))
                scalar_keys.append(key)
        elif val.is_nested:
            if val.offsets().numel() - 1 != len(batch):
                schema.append((key, "invalid_nested_batch", dtype_name, ndim_name))
                errors.append(f"DCP nested field {key!r} batch size does not match input_ids")
            else:
                schema.append((key, "nested", dtype_name, ndim_name))
                tensor_keys.append(key)
        elif val.dim() < 1 or val.shape[0] != len(batch):
            schema.append((key, "invalid_padded_batch", dtype_name, ndim_name))
            errors.append(f"DCP dense field {key!r} must have the TensorDict batch size in dimension 0")
        else:
            schema.append((key, "padded", dtype_name, ndim_name))
            padded_keys.append(key)

    _validate_routing_schema(schema, dcp_group)
    if errors:
        raise ValueError("Invalid DCP input routing schema: " + "; ".join(errors))
    return tensor_keys, padded_keys, scalar_keys


def _reroute_samples(
    local_samples: list[dict[str, torch.Tensor]],
    global_ids_this_rank: torch.Tensor,
    sample_id_groups: list[list[list[int]]],
    offsets: torch.Tensor,
    dp_group,
    dcp_group,
    tensor_keys: list[str],
    scalar_keys: list[str],
    padded_keys: list[str] | None = None,
    key_dtypes: dict[str, torch.dtype] | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Reroute samples to correct DCP ranks via all-to-all.

    Args:
        local_samples: List of per-sample dicts on this rank.
        global_ids_this_rank: Global IDs of samples on this rank.
        sample_id_groups: Per-microbatch, per-rank sample ID assignment.
            sample_id_groups[mb_idx][rank] = [global_id, ...]
        offsets: Cumulative sample count offsets per DP rank.
        dp_group: Data parallel process group that owns distinct data.
        dcp_group: DPxCP process group used for dynamic CP scheduling/routing.
        tensor_keys: Keys in sample dicts that are variable-length tensors.
        scalar_keys: Keys in sample dicts that are per-sample scalars.
        key_dtypes: Tensor dtypes from the source batch schema. This is needed
            when a DP rank has no local samples but still participates in the
            all-to-all payload collective.

    Returns:
        Dict mapping global_id -> sample dict for samples assigned to this rank.
    """
    total_dcp_gpus = dcp_group.size()
    dcp_rank = dcp_group.rank()
    dp_rank_to_dcp_rank = _group_rank_indices(dp_group, dcp_group)
    gid2local = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
    global_ids_set = set(gid2local.keys())

    def _gid_to_src_dcp_rank(gid: int) -> int:
        dp_src_rank = _dp_src_rank_for_gid(gid, offsets)
        return _infer_same_cp_dcp_rank(
            dp_src_rank,
            dcp_rank,
            dp_group.size(),
            dcp_group.size() // dp_group.size(),
            dp_rank_to_dcp_rank,
        )

    # Build combined sample ID groups per rank
    combined = [[] for _ in range(total_dcp_gpus)]
    for sample_id_group in sample_id_groups:
        for d in range(total_dcp_gpus):
            combined[d].extend(sample_id_group[d])
    for d in range(total_dcp_gpus):
        combined[d].sort()

    # Each CP replica owns the same local samples for its DP rank. To avoid duplicate
    # sends, a rank only serves destinations in its DP group expressed inside the
    # DPxCP group, matching Megatron-LM's hybrid DPxCP routing.
    send_groups = [[] for _ in range(total_dcp_gpus)]
    for dest_rank in dp_rank_to_dcp_rank:
        for gid in combined[dest_rank]:
            if gid in global_ids_set:
                send_groups[dest_rank].append(gid)
    send_ids_sorted = [gid for group in send_groups for gid in group]
    send_num_per_rank = [len(group) for group in send_groups]

    # Build recv plan
    recv_groups = [[] for _ in range(total_dcp_gpus)]
    for gid in combined[dcp_rank]:
        src_rank = _gid_to_src_dcp_rank(gid)
        recv_groups[src_rank].append(gid)

    recv_ids_sorted = [gid for d in range(total_dcp_gpus) for gid in recv_groups[d]]
    recv_counts = [len(recv_groups[d]) for d in range(total_dcp_gpus)]

    if padded_keys is None:
        padded_keys = []
    all_keys = _unique_keys(tensor_keys + scalar_keys + padded_keys)

    key_categories: dict[str, list[str]] = {}
    for category, keys in (("nested", tensor_keys), ("scalar", scalar_keys), ("padded", padded_keys)):
        for key in keys:
            key_categories.setdefault(key, []).append(category)

    resolved_dtypes: dict[str, torch.dtype | None] = {}
    schema_errors = []
    routing_schema = []
    for key in all_keys:
        categories = _unique_keys(key_categories.get(key, []))
        category = categories[0] if len(categories) == 1 else f"ambiguous:{','.join(categories)}"
        if len(categories) != 1:
            schema_errors.append(f"key {key!r} has routing categories {categories}")

        dtype = key_dtypes.get(key) if key_dtypes is not None else None
        local_dtypes = {sample[key].dtype for sample in local_samples if key in sample}
        if dtype is None and len(local_dtypes) == 1:
            dtype = next(iter(local_dtypes))
        if len(local_dtypes) > 1 or (dtype is not None and any(local_dtype != dtype for local_dtype in local_dtypes)):
            local_dtype_names = sorted(str(local_dtype) for local_dtype in local_dtypes)
            schema_errors.append(f"key {key!r} has declared dtype {dtype} but local sample dtypes {local_dtype_names}")
            dtype_name = f"mismatch:{dtype}:{','.join(local_dtype_names)}"
        elif dtype is None:
            schema_errors.append(f"cannot determine dtype for key {key!r}")
            dtype_name = "missing"
        else:
            dtype_name = str(dtype)
        resolved_dtypes[key] = dtype
        routing_schema.append((key, category, dtype_name))

    _validate_routing_schema(routing_schema, dcp_group)
    if schema_errors:
        raise ValueError("Invalid DCP routed tensor schema: " + "; ".join(schema_errors))

    recv_samples = [{k: None for k in all_keys} for _ in range(sum(recv_counts))]
    dev = get_device_id()
    profile = _dcp_profile_enabled()
    profile_key_ms: dict[str, float] = {}

    def _pack_key_payload(key: str):
        parts = []
        numels = []
        shapes = []
        for gid in send_ids_sorted:
            sample = local_samples[gid2local[gid]]
            if key not in sample:
                numels.append(0)
                shapes.append(_missing_shape_row())
                continue
            t = sample[key].to(dev, non_blocking=True)
            numels.append(t.numel())
            shapes.append(_shape_row(t))
            parts.append(t.reshape(-1))

        dtype = resolved_dtypes[key]
        if dtype is None:
            raise ValueError(
                f"Cannot determine dtype for DCP-routed key {key!r}; "
                "pass the source batch schema when this rank has no local samples"
            )

        send_tensor = torch.cat(parts, dim=0) if parts else torch.empty(0, device=dev, dtype=dtype)
        send_numels = torch.tensor(numels, dtype=torch.int64, device=dev)
        send_shapes = torch.tensor(shapes, dtype=torch.int64, device=dev).reshape(-1)
        return send_tensor, send_numels, send_shapes, numels

    shape_width = MAX_SHAPE_DIMS + 1
    for key in all_keys:
        key_start = _profile_now() if profile else None
        send_tensor, send_numels, send_shapes, send_numels_list = _pack_key_payload(key)

        recv_numels = torch.empty(sum(recv_counts), dtype=torch.int64, device=dev)
        torch.distributed.all_to_all_single(
            output=recv_numels,
            input=send_numels,
            output_split_sizes=recv_counts,
            input_split_sizes=send_num_per_rank,
            group=dcp_group,
        )

        recv_shapes = torch.empty(sum(recv_counts) * shape_width, dtype=torch.int64, device=dev)
        torch.distributed.all_to_all_single(
            output=recv_shapes,
            input=send_shapes,
            output_split_sizes=[count * shape_width for count in recv_counts],
            input_split_sizes=[count * shape_width for count in send_num_per_rank],
            group=dcp_group,
        )
        # One device-to-host copy per key: the split/unpack loops below read
        # these values element-wise, which would otherwise synchronize once per
        # sample. The send-side numels never need a GPU round-trip at all.
        recv_numels_list = recv_numels.tolist()
        recv_shapes = recv_shapes.reshape(-1, shape_width).cpu()

        input_elem_splits = _elem_splits_per_rank(send_numels_list, send_num_per_rank)
        output_elem_splits = _elem_splits_per_rank(recv_numels_list, recv_counts)

        recv_size = sum(output_elem_splits)
        recv_tensor = torch.empty(recv_size, device=dev, dtype=send_tensor.dtype)

        torch.distributed.all_to_all_single(
            output=recv_tensor,
            input=send_tensor,
            output_split_sizes=output_elem_splits,
            input_split_sizes=input_elem_splits,
            group=dcp_group,
        )

        cursor = 0
        for i, _gid in enumerate(recv_ids_sorted):
            numel = recv_numels_list[i]
            shape_row = recv_shapes[i]
            if _shape_row_is_missing(shape_row, numel):
                recv_samples[i].pop(key, None)
                continue
            shape = _shape_from_row(shape_row)
            recv_samples[i][key] = recv_tensor[cursor : cursor + numel].reshape(shape)
            cursor += numel
        if profile:
            profile_key_ms[key] = _profile_elapsed_ms(key_start, dcp_group)

    if profile:
        _profile_log(
            dcp_group,
            "DCP profile reroute_samples: "
            f"rank={dcp_rank}, send_counts={send_num_per_rank}, recv_counts={recv_counts}, "
            f"keys_ms={profile_key_ms}",
        )

    return {recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)}


# =============================================================================
# Micro-batch building
# =============================================================================


def _build_micro_batches_from_samples(
    samples_with_id: dict[int, dict[str, torch.Tensor]],
    sample_id_groups: list[list[list[int]]],
    dcp_rank: int,
) -> tuple[list[dict[str, object]], list[int]]:
    """Build packed micro-batches from scheduled samples.

    Args:
        samples_with_id: Mapping from global_id -> sample dict.
        sample_id_groups: Per-microbatch, per-rank sample ID assignment.
        dcp_rank: This rank's index in the DCP group.

    Returns:
        Tuple of:
            - List of packed sample dicts (one per micro-batch).
            - List of local_cp_size per micro-batch.
    """
    packed_batches = []
    local_cp_sizes = []
    for group in sample_id_groups:
        my_ids = group[dcp_rank]
        # local_cp_size = number of ranks that share the same first sample
        cp_size = sum(1 for rank_ids in group if my_ids[0] in rank_ids)
        local_cp_sizes.append(cp_size)
        packed_batches.append({"_samples": [samples_with_id[gid] for gid in my_ids]})

    return packed_batches, local_cp_sizes


# =============================================================================
# NestedTensor <-> per-sample conversion utilities
# =============================================================================


def _nested_tensor_to_samples(
    batch: TensorDict,
    tensor_keys: list[str],
    scalar_keys: list[str],
    padded_keys: list[str] | None = None,
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    """Convert a TensorDict with NestedTensors into per-sample dicts.

    Args:
        batch: TensorDict containing NestedTensor fields.
        tensor_keys: Keys that are variable-length (NestedTensor).
        scalar_keys: Keys that are per-sample scalars.
        padded_keys: Keys that are padded tensors (dim 0 = batch size).

    Returns:
        Tuple of:
            - List of per-sample dicts.
            - 1D tensor of sequence lengths.
    """
    if padded_keys is None:
        padded_keys = []

    input_ids = batch["input_ids"]
    if not input_ids.is_nested:
        raise ValueError("Dynamic CP requires input_ids to be a NestedTensor")
    seq_lens = input_ids.offsets().diff()
    num_samples = len(seq_lens)

    # Unbind NestedTensors
    unbound = {}
    for key in tensor_keys:
        if key in batch.keys():
            val = batch[key]
            if isinstance(val, torch.Tensor) and val.is_nested:
                unbound[key] = val.unbind()
            else:
                unbound[key] = None
        else:
            unbound[key] = None

    samples = []
    for i in range(num_samples):
        sample = {}
        for key in tensor_keys:
            if unbound[key] is not None:
                sample[key] = unbound[key][i]
            else:
                # If key doesn't exist or is not nested, skip
                pass
        for key in scalar_keys:
            if key in batch.keys():
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    sample[key] = val[i : i + 1] if val.dim() > 0 else val.unsqueeze(0)
        for key in padded_keys:
            if key in batch.keys():
                val = batch[key]
                if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == num_samples:
                    sample[key] = val[i]
        samples.append(sample)

    return samples, seq_lens


def _get_response_lengths(batch: TensorDict, seq_lens: torch.Tensor) -> torch.Tensor:
    """Return the true response span length for every sample before DCP routing."""
    num_samples = len(seq_lens)
    prompts = batch.get("prompts", None)
    responses = batch.get("responses", None)
    response_mask = batch.get("response_mask", None)
    attention_mask = batch.get("attention_mask", None)

    if isinstance(responses, torch.Tensor) and responses.is_nested:
        response_lens = responses.offsets().diff()
    elif isinstance(prompts, torch.Tensor) and prompts.is_nested:
        response_lens = seq_lens - prompts.offsets().diff().to(seq_lens.device)
    elif isinstance(response_mask, torch.Tensor) and response_mask.is_nested:
        # A jagged response mask encodes the true response span in its offsets;
        # its values may still contain internal zeros from tool calls or
        # rejection sampling and therefore must not be summed for the span.
        response_lens = response_mask.offsets().diff()
    elif (
        isinstance(prompts, torch.Tensor)
        and not prompts.is_nested
        and isinstance(attention_mask, torch.Tensor)
        and not attention_mask.is_nested
    ):
        response_lens = attention_mask[:, prompts.shape[1] :].sum(dim=-1)
    elif (
        isinstance(responses, torch.Tensor)
        and not responses.is_nested
        and isinstance(attention_mask, torch.Tensor)
        and not attention_mask.is_nested
    ):
        response_lens = attention_mask[:, -responses.shape[1] :].sum(dim=-1)
    else:
        max_response_len = tu.get_non_tensor_data(batch, key="max_response_len", default=None)
        if max_response_len is not None and isinstance(attention_mask, torch.Tensor) and not attention_mask.is_nested:
            response_lens = attention_mask[:, -int(max_response_len) :].sum(dim=-1)
        else:
            raise ValueError(
                "Dynamic CP loss routing requires prompts/responses or attention_mask with max_response_len "
                "to preserve the true response span"
            )

    response_lens = response_lens.to(device=seq_lens.device, dtype=torch.int64).reshape(-1)
    if response_lens.numel() != num_samples:
        raise ValueError(f"DCP response length count must match batch size: {response_lens.numel()} != {num_samples}")
    if torch.any(response_lens < 0) or torch.any(response_lens > seq_lens):
        raise ValueError(
            f"Invalid DCP response lengths {response_lens.tolist()} for sequence lengths {seq_lens.tolist()}"
        )
    return response_lens


def _samples_to_nested_tensor_batch(
    packed_batch: dict[str, object],
    tensor_keys: list[str],
    scalar_keys: list[str],
    padded_keys: list[str],
    local_cp_size: int,
    non_tensor_data: dict,
) -> TensorDict:
    """Convert a packed sample dict back to a TensorDict with NestedTensors.

    Args:
        packed_batch: Packed sample dict with concatenated tensors.
        tensor_keys: Variable-length tensor keys.
        scalar_keys: Per-sample scalar keys.
        local_cp_size: CP size for this micro-batch.
        non_tensor_data: Non-tensor metadata to preserve.

    Returns:
        TensorDict suitable for verl's forward pass.
    """
    samples = packed_batch["_samples"]
    num_seqs = len(samples)
    data_dict = {}

    for key in tensor_keys:
        per_seq = [sample[key] for sample in samples if key in sample and sample[key] is not None]
        if not per_seq:
            continue
        if len(per_seq) != num_seqs:
            raise ValueError(f"Key {key!r} is missing from some DCP-routed samples")
        data_dict[key] = torch.nested.as_nested_tensor(per_seq, layout=torch.jagged)

    for key in scalar_keys:
        tensors = [sample[key].reshape(-1) for sample in samples if key in sample and sample[key] is not None]
        if tensors:
            if len(tensors) != num_seqs:
                raise ValueError(f"Key {key!r} is missing from some DCP-routed samples")
            data_dict[key] = torch.cat(tensors, dim=0)

    for key in padded_keys:
        tensors = [sample[key] for sample in samples if key in sample and sample[key] is not None]
        if tensors:
            if len(tensors) != num_seqs:
                raise ValueError(f"Key {key!r} is missing from some DCP-routed samples")
            data_dict[key] = _stack_or_pad_samples(tensors)

    td = TensorDict(data_dict, batch_size=[num_seqs])

    # Assign non-tensor metadata
    for k, v in non_tensor_data.items():
        tu.assign_non_tensor(td, **{k: v})
    tu.assign_non_tensor_data(td, "local_cp_size", local_cp_size)

    return td


# =============================================================================
# Main scheduler class
# =============================================================================


def _build_reverse_routing_plans(routing_info: dict, dp_group, dcp_group):
    """Pre-compute send/recv plans for reverse routing (reused across keys).

    Returns:
        send_by_dest: send_by_dest[dest_rank] = [gid, ...] — ordered by dest rank
        recv_by_src:  recv_by_src[src_rank]  = [gid, ...] — ordered by src rank
        my_output_gids: global IDs whose outputs sit on this rank (scheduler order)
        send_ids_flat: flattened gid list in send order
        recv_ids_flat: flattened gid list in recv order
    """
    sample_id_groups = routing_info["sample_id_groups"]
    offsets = routing_info["offsets"]
    global_ids_this_rank = routing_info["global_ids_this_rank"]
    total = dcp_group.size()
    dcp_rank = dcp_group.rank()
    dp_rank_to_dcp_rank = _group_rank_indices(dp_group, dcp_group)
    dp_size = dp_group.size()
    cp_size = dcp_group.size() // dp_size

    def _gid_to_output_rank(gid: int) -> int:
        dp_src_rank = _dp_src_rank_for_gid(gid, offsets)
        return _infer_same_cp_dcp_rank(dp_src_rank, 0, dp_size, cp_size, dp_rank_to_dcp_rank)

    # Combined sample assignment per rank (same as forward routing)
    combined = [[] for _ in range(total)]
    for grp in sample_id_groups:
        for d in range(total):
            combined[d].extend(grp[d])

    # my_output_gids preserves micro-batch concatenation order (matches postprocess output)
    my_output_gids = list(combined[dcp_rank])

    # Sort combined for consistent all-to-all communication plans
    for d in range(total):
        combined[d].sort()

    original_gids = set(int(g) for g in global_ids_this_rank)

    # Reverse: outputs go from scheduled rank → original rank
    send_by_dest = [[] for _ in range(total)]
    for gid in my_output_gids:
        send_by_dest[_gid_to_output_rank(gid)].append(gid)
    for dest_rank in range(total):
        send_by_dest[dest_rank].sort()

    recv_by_src = [[] for _ in range(total)]
    for src_rank in range(total):
        for gid in combined[src_rank]:
            if gid in original_gids and _gid_to_output_rank(gid) == dcp_rank:
                recv_by_src[src_rank].append(gid)

    send_ids_flat = [gid for dest in range(total) for gid in send_by_dest[dest]]
    recv_ids_flat = [gid for src in range(total) for gid in recv_by_src[src]]

    # The canonical rank that collects this rank's original samples.
    output_rank = _infer_same_cp_dcp_rank(dp_group.rank(), 0, dp_size, cp_size, dp_rank_to_dcp_rank)

    return send_by_dest, recv_by_src, my_output_gids, send_ids_flat, recv_ids_flat, output_rank


def _build_reverse_routing_schema(
    model_output: dict[str, object], merge_duplicate_gids: bool, expected_local_batch_size: int
) -> tuple[list[tuple[str, str, str]], list[str], bool]:
    """Describe reverse-routing collectives without using variable-size metadata.

    Compact-routing metadata is listed first in its actual collective order.
    Remaining outputs retain their insertion order, and non-tensor values are
    included so a tensor/non-tensor disagreement cannot make only some ranks
    enter a collective.
    """
    compact_keys = ("_dcp_local_token_indices", "_dcp_full_seq_lens")
    compact_presence = [key in model_output for key in compact_keys]
    schema_errors = []

    if any(compact_presence) and not all(compact_presence):
        schema_errors.append(
            "compact DCP reverse routing requires both _dcp_local_token_indices and _dcp_full_seq_lens"
        )

    def _value_schema(key: str, value: object) -> tuple[str, str]:
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                schema_errors.append(
                    f"DCP reverse-routed tensor {key!r} must have a batch dimension; got a scalar tensor"
                )
                return "tensor:invalid_batch", str(value.dtype)
            actual_batch_size = int(value.shape[0])
            if actual_batch_size != expected_local_batch_size:
                schema_errors.append(
                    f"DCP reverse-routed tensor {key!r} batch size must match the scheduled local sample count: "
                    f"{actual_batch_size} != {expected_local_batch_size}"
                )
                return "tensor:invalid_batch", str(value.dtype)
            return "tensor:valid_batch", str(value.dtype)
        value_type = type(value)
        return "non_tensor", f"{value_type.__module__}.{value_type.__qualname__}"

    compact_route = bool(merge_duplicate_gids and all(compact_presence))
    schema = [
        (
            "reverse_route_protocol",
            f"merge_duplicate_gids={merge_duplicate_gids};compact_route={compact_route}",
            "v2",
        )
    ]

    for position, key in enumerate(compact_keys):
        if key not in model_output:
            schema.append((key, f"compact_metadata[{position}]:absent", "-"))
            continue
        value_kind, value_detail = _value_schema(key, model_output[key])
        schema.append((key, f"compact_metadata[{position}]:{value_kind}", value_detail))
        if not value_kind.startswith("tensor:"):
            schema_errors.append(f"compact DCP reverse-routing metadata {key!r} must be a tensor")

    output_position = 0
    for key, value in model_output.items():
        if key in compact_keys:
            continue
        value_kind, value_detail = _value_schema(key, value)
        schema.append((key, f"output[{output_position}]:{value_kind}", value_detail))
        output_position += 1

    return schema, schema_errors, compact_route


def _reconstruct_compact_sample(
    gid: int,
    value_parts: list[torch.Tensor],
    index_parts: list[torch.Tensor],
    length_parts: list[torch.Tensor],
) -> torch.Tensor:
    """Reconstruct one full-sequence output and validate its CP shards."""
    if len(index_parts) != len(value_parts):
        raise ValueError(
            f"Cannot reconstruct compact DCP output for gid={gid}: "
            f"{len(value_parts)} value parts but {len(index_parts)} index parts"
        )
    if not length_parts:
        raise ValueError(f"Missing full sequence length for compact DCP output gid={gid}")
    if len(length_parts) != len(value_parts):
        raise ValueError(
            f"Cannot reconstruct compact DCP output for gid={gid}: "
            f"{len(value_parts)} value parts but {len(length_parts)} full-length parts"
        )

    full_lengths = []
    for length in length_parts:
        if length.numel() != 1:
            raise ValueError(
                f"Cannot reconstruct compact DCP output for gid={gid}: "
                f"full sequence length metadata must be scalar, got shape {tuple(length.shape)}"
            )
        full_lengths.append(int(length.reshape(-1)[0].item()))
    if len(set(full_lengths)) != 1:
        raise ValueError(
            f"Cannot reconstruct compact DCP output for gid={gid}: inconsistent full sequence lengths {full_lengths}"
        )
    full_len = full_lengths[0]
    if full_len < 0:
        raise ValueError(f"Cannot reconstruct compact DCP output for gid={gid}: negative full length {full_len}")

    trailing_shape = tuple(value_parts[0].shape[1:])
    normalized_indices = []
    for values, indices in zip(value_parts, index_parts, strict=True):
        if tuple(values.shape[1:]) != trailing_shape:
            raise ValueError(
                f"Cannot reconstruct compact DCP output for gid={gid}: "
                f"trailing shape mismatch {tuple(values.shape[1:])} != {trailing_shape}"
            )
        indices = indices.to(device=values.device, dtype=torch.long).reshape(-1)
        if values.shape[0] != indices.numel():
            raise ValueError(
                f"Cannot reconstruct compact DCP output for gid={gid}: "
                f"value length {values.shape[0]} != index length {indices.numel()}"
            )
        normalized_indices.append(indices)

    all_indices = (
        torch.cat(normalized_indices, dim=0) if normalized_indices else value_parts[0].new_empty(0, dtype=torch.long)
    )
    expected_indices = torch.arange(full_len, dtype=torch.long, device=all_indices.device)
    if all_indices.numel() != full_len or not torch.equal(all_indices.sort().values, expected_indices):
        raise ValueError(
            f"Cannot reconstruct compact DCP output for gid={gid}: token indices must cover "
            f"[0, {full_len}) exactly once, got {all_indices.tolist()}"
        )

    result = value_parts[0].new_empty((full_len, *trailing_shape))
    for values, indices in zip(value_parts, normalized_indices, strict=True):
        result[indices.to(result.device)] = values
    return result


def reverse_route_outputs(
    model_output: dict[str, torch.Tensor],
    routing_info: dict,
    dp_group,
    dcp_group=None,
    merge_duplicate_gids: bool = False,
) -> dict[str, torch.Tensor]:
    """Reverse-route outputs back to original DP ranks via all-to-all.

    The forward scheduling sent samples from their original ranks to scheduled
    ranks. After the forward pass, outputs must be returned to the original
    ranks so that the caller sees them in the original sample order.

    Args:
        model_output: Dict of NestedTensors (keys like 'log_probs', 'entropy').
        routing_info: Dict returned by DynamicCPScheduler.schedule().
        dp_group: Data parallel process group.
        merge_duplicate_gids: If True, outputs from duplicated CP ranks for the
            same sample are merged instead of keeping the last one. Compact DCP
            outputs are reconstructed from token indices when present; otherwise
            zero-filled full-sequence local outputs are added.

    Returns:
        Dict of NestedTensors in the original sample order for this rank.
    """
    if dp_group is None:
        raise ValueError("Dynamic CP reverse routing requires a data-parallel process group")
    if dcp_group is None:
        dcp_group = dp_group
    total_dcp_gpus = dcp_group.size()
    dev = get_device_id()

    send_by_dest, recv_by_src, my_output_gids, send_ids_flat, recv_ids_flat, output_rank = (
        _build_reverse_routing_plans(routing_info, dp_group, dcp_group)
    )

    routing_schema, schema_errors, compact_route = _build_reverse_routing_schema(
        model_output,
        merge_duplicate_gids,
        expected_local_batch_size=len(my_output_gids),
    )
    _validate_routing_schema(routing_schema, dcp_group)
    if schema_errors:
        raise ValueError("Invalid DCP reverse-routing tensor schema: " + "; ".join(schema_errors))

    send_counts = [len(send_by_dest[d]) for d in range(total_dcp_gpus)]
    recv_counts = [len(recv_by_src[d]) for d in range(total_dcp_gpus)]

    original_gids_sorted = (
        sorted(int(g) for g in routing_info["global_ids_this_rank"]) if dcp_group.rank() == output_rank else []
    )
    is_output_rank = dcp_group.rank() == output_rank
    shape_width = MAX_SHAPE_DIMS + 1
    profile = _dcp_profile_enabled()
    profile_total_start = _profile_now() if profile else None
    profile_key_ms: dict[str, float] = {}

    def _route_tensor_values(val: torch.Tensor) -> dict[int, list[torch.Tensor]]:
        per_sample = list(val.unbind()) if val.is_nested else [val[i] for i in range(val.shape[0])]
        gid_to_out = {}
        for i, gid in enumerate(my_output_gids):
            if i < len(per_sample):
                gid_to_out[gid] = per_sample[i]

        parts = []
        send_numel_values = []
        send_shape_values = []
        for gid in send_ids_flat:
            if gid in gid_to_out:
                out = gid_to_out[gid].to(dev)
                parts.append(out.reshape(-1))
                send_numel_values.append(out.numel())
                send_shape_values.append(_shape_row(out))
            else:
                send_numel_values.append(0)
                send_shape_values.append(_missing_shape_row())
        send_tensor = torch.cat(parts, dim=0) if parts else torch.empty(0, device=dev, dtype=val.dtype)
        send_numels = torch.tensor(send_numel_values, dtype=torch.int64, device=dev)
        recv_numels = torch.empty(len(recv_ids_flat), dtype=torch.int64, device=dev)
        torch.distributed.all_to_all_single(
            output=recv_numels,
            input=send_numels,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=dcp_group,
        )

        send_shapes = torch.tensor(send_shape_values, dtype=torch.int64, device=dev).reshape(-1)
        recv_shapes = torch.empty(len(recv_ids_flat) * shape_width, dtype=torch.int64, device=dev)
        torch.distributed.all_to_all_single(
            output=recv_shapes,
            input=send_shapes,
            output_split_sizes=[count * shape_width for count in recv_counts],
            input_split_sizes=[count * shape_width for count in send_counts],
            group=dcp_group,
        )
        # One device-to-host copy per key instead of one synchronization per
        # sample; the send-side numels are already Python ints.
        recv_numels_list = recv_numels.tolist()
        recv_shapes = recv_shapes.reshape(-1, shape_width).cpu()

        send_elem_splits = _elem_splits_per_rank(send_numel_values, send_counts)
        recv_elem_splits = _elem_splits_per_rank(recv_numels_list, recv_counts)

        recv_tensor = torch.empty(sum(recv_elem_splits), device=dev, dtype=send_tensor.dtype)

        torch.distributed.all_to_all_single(
            output=recv_tensor,
            input=send_tensor,
            output_split_sizes=recv_elem_splits,
            input_split_sizes=send_elem_splits,
            group=dcp_group,
        )

        gid_to_result: dict[int, list[torch.Tensor]] = {}
        cursor = 0
        for i, gid in enumerate(recv_ids_flat):
            n = recv_numels_list[i]
            shape_row = recv_shapes[i]
            if _shape_row_is_missing(shape_row, n):
                continue
            shape = _shape_from_row(shape_row)
            result = recv_tensor[cursor : cursor + n].reshape(shape)
            gid_to_result.setdefault(gid, []).append(result)
            cursor += n

        return gid_to_result

    def _merge_duplicate_results(gid: int, results: list[torch.Tensor]) -> torch.Tensor:
        result = results[0]
        for other in results[1:]:
            if result.shape != other.shape:
                raise ValueError(
                    f"Cannot merge DCP outputs for gid={gid}: "
                    f"shape mismatch {tuple(result.shape)} vs {tuple(other.shape)}"
                )
            if result.dtype == torch.bool:
                result = result | other
            else:
                result = result + other
        return result

    def _rebuild_nested(gid_to_result: dict[int, torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
        if not original_gids_sorted:
            if is_output_rank:
                if fallback.is_nested:
                    values = fallback.values()
                    empty = torch.nested.nested_tensor_from_jagged(
                        values.new_empty((0, *values.shape[1:])),
                        offsets=fallback.offsets().new_zeros(1),
                        min_seqlen=0,
                        max_seqlen=0,
                    )
                    ragged_idx = getattr(fallback, "_ragged_idx", None)
                    if ragged_idx is not None:
                        empty._ragged_idx = ragged_idx
                    return empty
                return fallback.new_empty((0, *fallback.shape[1:]))
            return fallback
        missing = [gid for gid in original_gids_sorted if gid not in gid_to_result]
        if missing:
            raise ValueError(f"Missing DCP reverse-routed outputs for global ids: {missing}")
        result_tensors = [gid_to_result[gid] for gid in original_gids_sorted]
        return torch.nested.as_nested_tensor(result_tensors, layout=torch.jagged)

    def _reconstruct_compact_results(
        routed_values: dict[int, list[torch.Tensor]],
        routed_indices: dict[int, list[torch.Tensor]],
        routed_lens: dict[int, list[torch.Tensor]],
    ) -> dict[int, torch.Tensor]:
        gid_to_result = {}
        for gid, value_parts in routed_values.items():
            gid_to_result[gid] = _reconstruct_compact_sample(
                gid,
                value_parts,
                routed_indices.get(gid, []),
                routed_lens.get(gid, []),
            )
        return gid_to_result

    compact_indices = model_output.get("_dcp_local_token_indices", None)
    compact_lens = model_output.get("_dcp_full_seq_lens", None)

    # All-to-all for each output key
    reversed_output = {}
    if compact_route:
        profile_start = _profile_now() if profile else None
        routed_indices = _route_tensor_values(compact_indices)
        if profile:
            profile_key_ms["_dcp_local_token_indices"] = _profile_elapsed_ms(profile_start, dcp_group)
        profile_start = _profile_now() if profile else None
        routed_lens = _route_tensor_values(compact_lens)
        if profile:
            profile_key_ms["_dcp_full_seq_lens"] = _profile_elapsed_ms(profile_start, dcp_group)
    else:
        routed_indices = {}
        routed_lens = {}

    for key, val in model_output.items():
        if key in {"_dcp_local_token_indices", "_dcp_full_seq_lens"}:
            continue
        if not isinstance(val, torch.Tensor):
            reversed_output[key] = val
            continue

        profile_start = _profile_now() if profile else None
        routed_values = _route_tensor_values(val)
        replicated_full_sequence = key in _DCP_REPLICATED_FULL_SEQUENCE_OUTPUT_KEYS
        if compact_route and not replicated_full_sequence:
            gid_to_result = _reconstruct_compact_results(routed_values, routed_indices, routed_lens)
            reversed_output[key] = _rebuild_nested(gid_to_result, val)
            if profile:
                profile_key_ms[key] = _profile_elapsed_ms(profile_start, dcp_group)
            continue

        gid_to_result = {}
        for gid, results in routed_values.items():
            should_merge = merge_duplicate_gids and not replicated_full_sequence
            gid_to_result[gid] = _merge_duplicate_results(gid, results) if should_merge else results[-1]
        reversed_output[key] = _rebuild_nested(gid_to_result, val)
        if profile:
            profile_key_ms[key] = _profile_elapsed_ms(profile_start, dcp_group)

    if profile:
        total_ms = _profile_elapsed_ms(profile_total_start, dcp_group)
        _profile_log(
            dcp_group,
            "DCP profile reverse_route_outputs: "
            f"merge_duplicate_gids={merge_duplicate_gids}, compact_route={compact_route}, "
            f"send_counts={send_counts}, recv_counts={recv_counts}, keys_ms={profile_key_ms}, total_ms={total_ms:.3f}",
        )

    return reversed_output


# =============================================================================
# PP broadcast for dynamic CP metadata
# =============================================================================


def broadcast_dcp_metadata_to_pp(
    micro_batches: list[TensorDict],
    pp_group,
) -> list[TensorDict]:
    """Broadcast dynamic CP metadata from first/last PP stage to middle stages.

    Middle PP stages need local_cp_size and cu_seqlens for attention computation
    but do not receive full batch data. This broadcasts the metadata from the
    first PP stage (rank 0 in pp_group).

    Reference: the equivalent Megatron-Core PP metadata broadcast lives inline in
    data_schedule.py get_batch_on_this_rank_for_sequence_packing(), built on
    data_schedule_utils.broadcast_tensor().

    Args:
        micro_batches: List of TensorDict micro-batches (populated on first/last PP).
        pp_group: Pipeline parallel process group.

    Returns:
        micro_batches with metadata populated on all PP stages.
    """
    if pp_group.size() <= 2:
        # Only first and last stage — no middle stages to broadcast to
        return micro_batches

    pp_src_rank = torch.distributed.get_process_group_ranks(pp_group)[0]
    dev = get_device_id()

    if pp_group.rank() == 0:
        # First PP stage: pack metadata and broadcast
        num_mb = len(micro_batches)
        cp_sizes = []
        cu_seqlens_list = []
        for mb in micro_batches:
            cp_size = tu.get_non_tensor_data(mb, key="local_cp_size", default=1)
            cp_sizes.append(int(cp_size))
            # Extract cu_seqlens from NestedTensor offsets
            input_ids = mb["input_ids"]
            if not input_ids.is_nested:
                raise ValueError("Dynamic CP PP metadata requires nested input_ids on the first pipeline stage")
            offsets = input_ids.offsets().to(dtype=torch.int32, device=dev)
            cu_seqlens_list.append(offsets)

        header = torch.tensor(
            [num_mb, *cp_sizes, *[int(cu.numel()) for cu in cu_seqlens_list]],
            dtype=torch.int32,
            device=dev,
        )
        info = torch.cat([header, *cu_seqlens_list], dim=0)

        info_len = torch.tensor(info.shape[0], dtype=torch.int32, device=dev)
        torch.distributed.broadcast(info_len, pp_src_rank, group=pp_group)
        torch.distributed.broadcast(info, pp_src_rank, group=pp_group)
    else:
        info_len = torch.tensor(0, dtype=torch.int32, device=dev)
        torch.distributed.broadcast(info_len, pp_src_rank, group=pp_group)
        info = torch.empty(int(info_len.item()), dtype=torch.int32, device=dev)
        torch.distributed.broadcast(info, pp_src_rank, group=pp_group)

        if pp_group.rank() != pp_group.size() - 1:
            # Middle PP stage: unpack metadata
            num_mb = int(info[0].item())
            cp_sizes = info[1 : 1 + num_mb].tolist()
            cu_lens = info[1 + num_mb : 1 + 2 * num_mb].tolist()
            cu_start = 1 + 2 * num_mb

            if micro_batches and len(micro_batches) != num_mb:
                raise ValueError(
                    "DCP PP metadata count does not match existing middle-stage micro-batches: "
                    f"received {num_mb}, existing {len(micro_batches)}"
                )
            existing_micro_batches = micro_batches
            micro_batches = []
            for i in range(num_mb):
                cu_len = int(cu_lens[i])
                cu_seqlens = info[cu_start : cu_start + cu_len].to(device=dev, dtype=torch.int64)
                cu_start += cu_len
                seq_lens = cu_seqlens.diff().tolist()
                dummy = [torch.zeros(int(seq_len), dtype=torch.long, device=dev) for seq_len in seq_lens]
                input_ids = torch.nested.as_nested_tensor(dummy, layout=torch.jagged)
                if existing_micro_batches:
                    td = existing_micro_batches[i]
                    if td.batch_size != torch.Size([len(dummy)]):
                        raise ValueError(
                            "DCP PP metadata batch size does not match the existing middle-stage micro-batch: "
                            f"received {len(dummy)}, existing {list(td.batch_size)}"
                        )
                    # Keep already-routed fields such as routed_experts. Only
                    # input_ids and the DCP metadata originate from PP rank 0.
                    td["input_ids"] = input_ids
                else:
                    td = TensorDict({"input_ids": input_ids}, batch_size=[len(dummy)])
                tu.assign_non_tensor_data(td, "local_cp_size", int(cp_sizes[i]))
                tu.assign_non_tensor_data(td, "_dcp_scheduled", True)
                micro_batches.append(td)

    return micro_batches


# =============================================================================
# Main scheduler class
# =============================================================================


class DynamicCPScheduler:
    """Dynamic Context Parallelism Scheduler.

    Collects sequence lengths from all DP ranks, schedules samples into
    micro-batches with optimal CP sizes, routes samples via all-to-all,
    and builds packed micro-batches.

    Args:
        max_seqlen_per_dp_cp_rank: Max sequence length per DPxCP rank.
        dp_size: Data parallel world size.
        min_cp_size: Minimum CP group size (default 1).
        microbatch_group_size_per_vp_stage: For VPP alignment (default None).
    """

    # verl data keys. Keep routing narrow: DCP all-to-all is paid once for each
    # tensor key, so only send tensors that model forward or the local loss can
    # actually consume on the scheduled rank.
    FORWARD_KEYS = ["input_ids"]
    TRAIN_LOSS_KEYS = [
        "loss_mask",
        "response_mask",
        "old_log_probs",
        "advantages",
        "rollout_is_weights",
        "ref_log_prob",
        "values",
        "returns",
    ]
    ROUTER_REPLAY_KEYS = ["routed_experts"]
    DISTILLATION_KEYS = ["teacher_logprobs", "teacher_ids"]
    SCALAR_KEYS = ["temperature"]

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank: int,
        dp_size: int,
        cp_size: int = 1,
        min_cp_size: int = 1,
        microbatch_group_size_per_vp_stage: int | None = None,
    ):
        for name, value in (
            ("max_seqlen_per_dp_cp_rank", max_seqlen_per_dp_cp_rank),
            ("dp_size", dp_size),
            ("cp_size", cp_size),
            ("min_cp_size", min_cp_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        self.max_seqlen_per_dp_cp_rank = max_seqlen_per_dp_cp_rank
        self.dp_size = dp_size
        self.cp_size = cp_size
        self.total_dcp_gpus = dp_size * cp_size
        self.min_cp_size = min_cp_size
        if min_cp_size > self.total_dcp_gpus:
            raise ValueError(f"min_cp_size cannot exceed the DPxCP group size: {min_cp_size} > {self.total_dcp_gpus}")
        self.microbatch_group_size_per_vp_stage = microbatch_group_size_per_vp_stage

    def _routing_key_order(self, batch: TensorDict, non_tensor_data: dict) -> list[str]:
        """Return tensor keys that should be routed for this batch."""
        compute_loss = bool(non_tensor_data.get("compute_loss", False))
        route_router_replay = bool(non_tensor_data.get("enable_routing_replay", False))

        keys = list(self.FORWARD_KEYS)
        if compute_loss:
            keys.extend(self.TRAIN_LOSS_KEYS)
        if route_router_replay:
            keys.extend(self.ROUTER_REPLAY_KEYS)
        # Both top-k distillation and the full-logprob estimators consume these
        # fields. Presence in the batch is the authoritative routing signal.
        keys.extend(self.DISTILLATION_KEYS)
        keys.extend(self.SCALAR_KEYS)

        return _unique_keys([key for key in keys if key in batch.keys()])

    def _get_groups_and_subsamples(self, sample_id_seqlens: list[tuple[int, int]]) -> list[list[list[int]]]:
        """Schedule samples with Megatron-Core's Dynamic CP implementation.

        Args:
            sample_id_seqlens: List of (global_id, seq_len) tuples.

        Returns:
            sample_id_groups: Per-microbatch, per-rank assignment of sample IDs.
                sample_id_groups[mb_idx][rank] = [global_id, ...]
        """
        if not sample_id_seqlens:
            raise ValueError("Dynamic CP received no samples to schedule across the DP group")
        zero_length_ids = [sample_id for sample_id, seq_len in sample_id_seqlens if seq_len <= 0]
        if zero_length_ids:
            raise ValueError(
                "Dynamic CP cannot schedule zero-length sequences (Megatron-Core requires positive "
                f"sequence lengths): global sample ids {zero_length_ids[:8]}. Filter empty samples upstream."
            )
        max_sample_id, max_seq_len = max(sample_id_seqlens, key=lambda id_len: id_len[1])
        required_chunks = -(-max_seq_len // self.max_seqlen_per_dp_cp_rank)
        required_cp = 1
        while required_cp < required_chunks:
            required_cp *= 2
        if required_cp > self.total_dcp_gpus:
            raise ValueError(
                f"Sequence with global id {max_sample_id} has {max_seq_len} tokens and needs a CP group of "
                f"{required_cp} ranks (power-of-two ceil of {max_seq_len} / max_seqlen_per_dp_cp_rank="
                f"{self.max_seqlen_per_dp_cp_rank}), but the DPxCP group only has {self.total_dcp_gpus} ranks. "
                "Increase max_seqlen_per_dp_cp_rank or cap the rollout sequence length."
            )

        scheduler_cls = _get_megatron_dynamic_cp_scheduler_cls()
        scheduler = scheduler_cls(
            max_seqlen_per_dp_cp_rank=self.max_seqlen_per_dp_cp_rank,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            min_cp_size=self.min_cp_size,
            microbatch_group_size_per_vp_stage=self.microbatch_group_size_per_vp_stage,
        )
        sample_id_groups = scheduler.get_groups_and_subsamples(sample_id_seqlens)

        bad_group_lengths = {
            mb_idx: len(group) for mb_idx, group in enumerate(sample_id_groups) if len(group) != self.total_dcp_gpus
        }
        if bad_group_lengths:
            raise RuntimeError(
                "Megatron-Core Dynamic CP returned micro-batches whose rank-assignment length does not match "
                f"the DPxCP group size {self.total_dcp_gpus}: {{micro_batch: length}}={bad_group_lengths}."
            )
        if any(not rank_ids for group in sample_id_groups for rank_ids in group):
            raise RuntimeError(
                "Megatron-Core Dynamic CP returned an empty rank. Use a build containing "
                "NVIDIA/Megatron-LM#5154 (merge commit d2e7ec5b)."
            )
        for mb_idx, group in enumerate(sample_id_groups):
            for rank_ids in group:
                # Same membership rule as _build_micro_batches_from_samples: the CP group of a rank
                # is the set of ranks whose assignment shares its first sample.
                cp_group_size = sum(1 for other_ids in group if rank_ids[0] in other_ids)
                if cp_group_size != self.total_dcp_gpus and (cp_group_size & (cp_group_size - 1)) != 0:
                    raise RuntimeError(
                        f"Megatron-Core Dynamic CP produced a CP group of {cp_group_size} ranks in micro-batch "
                        f"{mb_idx}; only power-of-two sizes or the full DPxCP group ({self.total_dcp_gpus}) have "
                        f"process groups: {group}"
                    )

        expected_ids = {sample_id for sample_id, _ in sample_id_seqlens}
        assigned_ids = {sample_id for group in sample_id_groups for rank_ids in group for sample_id in rank_ids}
        if assigned_ids != expected_ids:
            raise RuntimeError(
                "Megatron-Core Dynamic CP did not preserve the input sample set: "
                f"expected={sorted(expected_ids)}, assigned={sorted(assigned_ids)}"
            )
        return sample_id_groups

    def schedule(
        self,
        batch: TensorDict,
        dp_group,
        dcp_group=None,
        non_tensor_data: dict | None = None,
    ) -> tuple[list[TensorDict], dict]:
        """Run the full dynamic CP scheduling pipeline.

        Args:
            batch: Local TensorDict with NestedTensor data for this DP rank.
            dp_group: Data parallel process group with distinct data.
            dcp_group: DPxCP process group used by dynamic context parallelism.
            non_tensor_data: Non-tensor metadata to propagate to micro-batches.

        Returns:
            Tuple of:
                - micro_batches: List of TensorDict, each with local_cp_size
                  and _dcp_scheduled=True set as non-tensor data.
                - routing_info: Dict with routing metadata for reverse_route_outputs().
        """
        if dp_group is None:
            raise ValueError("Dynamic CP scheduling requires a data-parallel process group")
        non_tensor_data = dict(non_tensor_data or {})
        if dcp_group is None:
            dcp_group = dp_group
        if dcp_group.size() != self.total_dcp_gpus:
            raise ValueError(
                f"DCP group size {dcp_group.size()} does not match dp_size * cp_size "
                f"({self.dp_size} * {self.cp_size} = {self.total_dcp_gpus})"
            )
        profile = _dcp_profile_enabled()
        profile_total_start = _profile_now() if profile else None
        profile_ms: dict[str, float] = {}

        # Determine which keys are required on the scheduled rank. Their actual
        # tensor layout still decides routing: jagged NestedTensors are sent as
        # variable-length tensors; dense tensors with batch dimension are routed
        # as per-sample padded tensors.
        profile_start = _profile_now() if profile else None
        ordered_keys = self._routing_key_order(batch, non_tensor_data)
        tensor_keys, padded_keys, scalar_keys = _classify_routed_batch_fields(
            batch,
            ordered_keys,
            set(self.SCALAR_KEYS),
            dcp_group,
        )
        if profile:
            profile_ms["classify_keys"] = _profile_elapsed_ms(profile_start, dcp_group)

        # Step 1: Convert NestedTensor batch to per-sample format
        profile_start = _profile_now() if profile else None
        local_samples, local_seqlens = _nested_tensor_to_samples(batch, tensor_keys, scalar_keys, padded_keys)
        key_dtypes = {
            key: batch[key].dtype
            for key in _unique_keys(tensor_keys + scalar_keys + padded_keys)
            if isinstance(batch.get(key, None), torch.Tensor)
        }
        if non_tensor_data.get("compute_loss", False) and "response_mask" in batch.keys():
            # Response spans depend on local data (and on non-routed keys such as
            # prompts/responses), so their validation must reject collectively.
            # Catch broadly: rank-local shape errors raise RuntimeError and bad
            # metadata raises TypeError, and any escape strands the peers.
            response_lens = None
            local_error = None
            try:
                response_lens = _get_response_lengths(batch, local_seqlens)
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            _collective_raise(dcp_group, local_error)
            for sample, response_len in zip(local_samples, response_lens, strict=True):
                sample[DCP_RESPONSE_LENGTH_KEY] = response_len.reshape(1)
            scalar_keys.append(DCP_RESPONSE_LENGTH_KEY)
            key_dtypes[DCP_RESPONSE_LENGTH_KEY] = response_lens.dtype
        if profile:
            profile_ms["to_samples"] = _profile_elapsed_ms(profile_start, dcp_group)

        # Step 2: Gather global sequence lengths from all DP ranks
        profile_start = _profile_now() if profile else None
        local_seqlens_gpu = local_seqlens.to(dtype=torch.int32, device=get_device_id())
        global_id_seqlens, global_ids_this_rank, offsets = _get_global_seqlens_and_ids(local_seqlens_gpu, dp_group)
        if profile:
            profile_ms["gather_seqlens"] = _profile_elapsed_ms(profile_start, dcp_group)

        # Step 3: Schedule samples into groups
        profile_start = _profile_now() if profile else None
        sample_id_groups = self._get_groups_and_subsamples(global_id_seqlens)
        if profile:
            profile_ms["schedule_groups"] = _profile_elapsed_ms(profile_start, dcp_group)

        # Step 4: Reroute samples via all-to-all
        profile_start = _profile_now() if profile else None
        samples_this_rank = _reroute_samples(
            local_samples,
            global_ids_this_rank,
            sample_id_groups,
            offsets,
            dp_group,
            dcp_group,
            tensor_keys,
            scalar_keys,
            padded_keys,
            key_dtypes,
        )
        if profile:
            profile_ms["reroute_samples"] = _profile_elapsed_ms(profile_start, dcp_group)

        # Step 5: Build packed micro-batches
        profile_start = _profile_now() if profile else None
        dcp_rank = dcp_group.rank()
        packed_batches, local_cp_sizes = _build_micro_batches_from_samples(
            samples_this_rank,
            sample_id_groups,
            dcp_rank,
        )
        if profile:
            profile_ms["build_micro_batches"] = _profile_elapsed_ms(profile_start, dcp_group)

        # Mark micro-batches as scheduler-managed for backend-specific loss and output handling.
        non_tensor_data["_dcp_scheduled"] = True

        # Step 6: Convert packed batches to TensorDict with NestedTensor
        profile_start = _profile_now() if profile else None
        micro_batches = []
        for packed, cp_size in zip(packed_batches, local_cp_sizes, strict=True):
            td = _samples_to_nested_tensor_batch(
                packed,
                tensor_keys,
                scalar_keys,
                padded_keys,
                local_cp_size=cp_size,
                non_tensor_data=non_tensor_data,
            )
            micro_batches.append(td)
        if profile:
            profile_ms["to_tensordict"] = _profile_elapsed_ms(profile_start, dcp_group)

        # Build routing info for reverse_route_outputs()
        routing_info = {
            "global_ids_this_rank": global_ids_this_rank,
            "sample_id_groups": sample_id_groups,
            "offsets": offsets,
        }

        logger.info(
            f"DynamicCPScheduler: {len(global_id_seqlens)} samples -> "
            f"{len(micro_batches)} micro-batches, "
            f"local_cp_sizes={local_cp_sizes}, "
            f"global_cp_size_counts={_count_cp_groups(sample_id_groups)}, "
            f"max_seqlen_per_rank={self.max_seqlen_per_dp_cp_rank}"
        )
        if profile:
            profile_ms["total"] = _profile_elapsed_ms(profile_total_start, dcp_group)
            _profile_log(
                dcp_group,
                "DCP profile schedule: "
                f"samples={len(global_id_seqlens)}, micro_batches={len(micro_batches)}, "
                f"local_cp_sizes={local_cp_sizes}, tensor_keys={tensor_keys}, padded_keys={padded_keys}, "
                f"scalar_keys={scalar_keys}, ms={profile_ms}",
            )

        return micro_batches, routing_info
