"""Streaming safetensors loading for Kimi-K3 FSDP-Turbo models.

The legacy Kimi loader first assembles every expert into a CPU full state dict
and lets FSDP-Turbo broadcast/shard that state.  This module builds the final
DTensor layout first and reads only the slice owned by the current rank.

The loader intentionally does not call ``state_dict`` or ``full_tensor``.
Those operations are both correctness hazards for a large EP model and the
source of the multi-hundred-GB initialization peak seen with the old path.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass

import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


@dataclass
class _TensorSpec:
    name: str
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_tensor: torch.Tensor


@dataclass
class _LoadTask:
    target_name: str
    source_key: str
    destination: torch.Tensor
    source_slices: tuple[slice, ...]
    expected_source_shape: tuple[int, ...]
    transpose: bool = False


def _as_shape(value) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _local_spec(name: str, tensor: torch.Tensor) -> _TensorSpec:
    if isinstance(tensor, DTensor):
        local_tensor = tensor.to_local()
        local_shape, global_offset = compute_local_shape_and_global_offset(
            tensor.shape,
            tensor.device_mesh,
            tensor.placements,
        )
        local_shape = _as_shape(local_shape)
        global_offset = _as_shape(global_offset)
    else:
        local_tensor = tensor
        local_shape = _as_shape(tensor.shape)
        global_offset = (0,) * len(local_shape)

    global_shape = _as_shape(tensor.shape)
    if _as_shape(local_tensor.shape) != local_shape:
        raise RuntimeError(
            f"Local DTensor shape mismatch for {name}: "
            f"computed={local_shape}, actual={tuple(local_tensor.shape)}, "
            f"global={global_shape}"
        )
    if any(
        offset < 0 or offset + size > global_size
        for offset, size, global_size in zip(global_offset, local_shape, global_shape, strict=True)
    ):
        raise RuntimeError(
            f"Invalid local DTensor range for {name}: "
            f"offset={global_offset}, local={local_shape}, global={global_shape}"
        )
    return _TensorSpec(
        name=name,
        global_shape=global_shape,
        local_shape=local_shape,
        global_offset=global_offset,
        local_tensor=local_tensor,
    )


def _slice_tuple(offset: tuple[int, ...], shape: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(start, start + size) for start, size in zip(offset, shape, strict=True))


def _add_task(
    tasks_by_file: dict[str, list[_LoadTask]],
    weight_map: dict[str, str],
    task: _LoadTask,
) -> None:
    tasks_by_file[weight_map[task.source_key]].append(task)


def _add_direct_task(
    spec: _TensorSpec,
    source_key: str,
    weight_map: dict[str, str],
    tasks_by_file,
) -> None:
    source_slices = _slice_tuple(spec.global_offset, spec.local_shape)
    _add_task(
        tasks_by_file,
        weight_map,
        _LoadTask(
            target_name=spec.name,
            source_key=source_key,
            destination=spec.local_tensor,
            source_slices=source_slices,
            expected_source_shape=spec.global_shape,
        ),
    )


def _add_kimi_packed_tasks(spec: _TensorSpec, weight_map, tasks_by_file) -> None:
    """Map local [expert, input, output] slices to transposed HF matrices."""
    if len(spec.global_shape) != 3:
        raise ValueError(f"Packed Kimi parameter must be 3-D: {spec.name}")
    _, rows, columns = spec.global_shape
    prefix, projection = spec.name.rsplit(".", 1)
    if projection == "gate_up_proj":
        if columns % 2:
            raise ValueError(f"Packed gate/up width must be even: {spec.name}")
        projections = (("w1", 0, columns // 2), ("w3", columns // 2, columns))
    else:
        projections = (("w2", 0, columns),)

    expert_start, row_start, column_start = spec.global_offset
    expert_count, row_count, column_count = spec.local_shape
    for local_expert in range(expert_count):
        for source_projection, begin, end in projections:
            left = max(column_start, begin)
            right = min(column_start + column_count, end)
            if left >= right:
                continue
            _add_task(
                tasks_by_file,
                weight_map,
                _LoadTask(
                    target_name=spec.name,
                    source_key=f"{prefix}.{expert_start + local_expert}.{source_projection}.weight",
                    destination=spec.local_tensor[local_expert, :, left - column_start : right - column_start],
                    source_slices=(slice(left - begin, right - begin), slice(row_start, row_start + row_count)),
                    expected_source_shape=(end - begin, rows),
                    transpose=True,
                ),
            )


def _read_index(checkpoint_path: str) -> dict[str, str]:
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        single_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.isfile(single_path):
            # Read only metadata; no tensor payload is materialized.
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                return {key: "model.safetensors" for key in handle.keys()}
        raise FileNotFoundError(f"No safetensors index or single checkpoint found in {checkpoint_path}")
    with open(index_path, encoding="utf-8") as index_file:
        index = json.load(index_file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid safetensors weight_map in {index_path}")
    return {str(key): str(value) for key, value in weight_map.items()}


def _check_shape(task: _LoadTask, source_shape: tuple[int, ...]) -> None:
    if source_shape != task.expected_source_shape:
        raise RuntimeError(
            f"Checkpoint shape mismatch for {task.source_key} -> {task.target_name}: "
            f"checkpoint={source_shape}, expected={task.expected_source_shape}"
        )
    for axis, item in enumerate(task.source_slices):
        start = 0 if item.start is None else item.start
        stop = source_shape[axis] if item.stop is None else item.stop
        if start < 0 or stop < start or stop > source_shape[axis]:
            raise RuntimeError(
                f"Source slice out of range for {task.source_key}: slices={task.source_slices}, shape={source_shape}"
            )


def _copy_task(task: _LoadTask, accessor) -> None:
    source_shape = _as_shape(accessor.get_shape())
    if (
        task.source_key.endswith(".self_attn.A_log")
        and len(source_shape) == len(task.expected_source_shape) == 1
        and source_shape[0] > task.expected_source_shape[0]
    ):
        # The source pads 96 active KDA heads to 128; only zero padding may be discarded.
        if torch.count_nonzero(accessor[task.expected_source_shape[0] :]).item():
            raise ValueError(f"Nonzero inactive A_log entries in {task.source_key}")
        source_shape = task.expected_source_shape
    _check_shape(task, source_shape)
    source = accessor[task.source_slices] if task.source_slices else accessor[()]
    if task.transpose:
        if source.ndim != 2:
            raise RuntimeError(f"Transpose task is not 2-D: {task.source_key} shape={tuple(source.shape)}")
        source = source.transpose(0, 1)
    destination = task.destination
    if tuple(source.shape) != tuple(destination.shape):
        raise RuntimeError(
            f"Local slice shape mismatch for {task.source_key} -> {task.target_name}: "
            f"source={tuple(source.shape)}, destination={tuple(destination.shape)}"
        )
    destination.copy_(source)


@torch.no_grad()
def load_kimi_k3_checkpoint_to_local_shards(
    model: torch.nn.Module,
    checkpoint_path: str | None,
    *,
    materialize_device,
) -> None:
    """Load raw Kimi checkpoint weights directly into local DTensor shards.

    ``model`` must already have the final FSDP-Turbo/TP/EP wrappers applied.
    When checkpoint_path is None, zero the local parameter shards for a smoke test.
    Missing parameters and mismatched checkpoint shapes are errors when loading weights.
    """

    # Materialize meta tensors without discarding initialized rotary/cache buffers.
    model._apply(
        lambda tensor: torch.empty_like(tensor, device=materialize_device)
        if tensor.is_meta
        else tensor.to(device=materialize_device)
    )

    if checkpoint_path is None:
        for param in model.parameters():
            local_param = param.to_local() if isinstance(param, DTensor) else param
            local_param.zero_()
        return

    weight_map = _read_index(checkpoint_path)
    tasks_by_file: dict[str, list[_LoadTask]] = defaultdict(list)
    missing_parameters: list[str] = []

    for target_name, parameter in model.named_parameters():
        spec = _local_spec(target_name, parameter)
        source_key = target_name
        if source_key in weight_map:
            _add_direct_task(spec, source_key, weight_map, tasks_by_file)
            continue

        if target_name.endswith((".experts.gate_up_proj", ".experts.down_proj")) and len(spec.global_shape) == 3:
            _add_kimi_packed_tasks(spec, weight_map, tasks_by_file)
            continue

        missing_parameters.append(target_name)

    for target_name, buffer in model.named_buffers():
        if target_name in weight_map:
            _add_direct_task(_local_spec(target_name, buffer), target_name, weight_map, tasks_by_file)

    if missing_parameters:
        raise RuntimeError(
            "Streaming Kimi checkpoint is missing target parameters: " + ", ".join(missing_parameters[:32])
        )

    for shard_name in sorted(tasks_by_file):
        shard_path = shard_name if os.path.isabs(shard_name) else os.path.join(checkpoint_path, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for task in tasks_by_file[shard_name]:
                _copy_task(task, handle.get_slice(task.source_key))
