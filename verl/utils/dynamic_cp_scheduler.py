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

"""Thin TensorDict adapter for Megatron-Core's Dynamic CP scheduler."""

import math
from collections import Counter

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.py_functional import append_to_dict

DCP_GROUP_LEADER = "_dcp_group_leader"
DCP_LOCAL_NUM_TOKENS = "_dcp_local_num_tokens"
DCP_PADDING_MASK = "_dcp_padding_mask"
DCP_SAMPLE_IDS = "_dcp_sample_ids"


def get_megatron_dynamic_cp_scheduler_cls() -> type:
    """Return the scheduler implementation owned by Megatron-Core."""
    try:
        from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Dynamic CP requires a Megatron-Core build containing NVIDIA/Megatron-LM#5154 (merge commit d2e7ec5b)."
        ) from exc
    return DefaultDynamicCPScheduler


def _local_padding_mask(
    seq_lens: list[int],
    *,
    cp_size: int,
    cp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Build the MCore router padding mask for one local THD shard."""
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    padded_lens = [math.ceil(seq_len / align_size) * align_size for seq_len in seq_lens]

    masks = []
    for seq_len, padded_len in zip(seq_lens, padded_lens, strict=True):
        local_padded_len = padded_len // cp_size
        mask = torch.ones(local_padded_len, dtype=torch.bool)
        if cp_size == 1:
            mask[:seq_len] = False
            masks.append(mask)
            continue

        half_chunk = padded_len // (2 * cp_size)
        front_start = cp_rank * half_chunk
        back_start = padded_len - (cp_rank + 1) * half_chunk
        front_tokens = max(0, min(seq_len - front_start, half_chunk))
        back_tokens = max(0, min(seq_len - back_start, half_chunk))
        mask[:front_tokens] = False
        mask[half_chunk : half_chunk + back_tokens] = False
        masks.append(mask)
    return torch.cat(masks)


def _cp_members(rank_assignments: list[list[int]], rank: int) -> list[int]:
    rank_sample_ids = rank_assignments[rank]
    if not rank_sample_ids:
        raise RuntimeError("Megatron-Core Dynamic CP returned an empty rank assignment")
    first_sample_id = rank_sample_ids[0]
    return [peer for peer, sample_ids in enumerate(rank_assignments) if first_sample_id in sample_ids]


class DynamicCPScheduler:
    """Apply Megatron-Core scheduling to the replicated batch used by verl #5057."""

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank: int,
        dp_size: int,
        cp_size: int,
    ):
        values = {
            "max_seqlen_per_dp_cp_rank": max_seqlen_per_dp_cp_rank,
            "dp_size": dp_size,
            "cp_size": cp_size,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        self.max_seqlen_per_dp_cp_rank = max_seqlen_per_dp_cp_rank
        self.dp_size = dp_size
        self.cp_size = cp_size
        self.total_ranks = dp_size * cp_size

    def schedule(
        self,
        batch: TensorDict,
        dcp_group,
        *,
        tp_size: int,
    ) -> list[TensorDict]:
        """Return this rank's scheduled micro-batches without rerouting batch data."""
        input_ids = batch.get("input_ids", None)
        if not isinstance(input_ids, torch.Tensor) or not input_ids.is_nested:
            raise ValueError("Dynamic CP requires input_ids to be a jagged NestedTensor")
        if len(batch) == 0:
            raise ValueError("Dynamic CP cannot schedule an empty batch")
        if dcp_group.size() != self.total_ranks:
            raise ValueError(
                f"DPxCP group size {dcp_group.size()} does not match "
                f"dp_size * cp_size ({self.dp_size} * {self.cp_size})"
            )

        seq_lens = [int(seq_len) for seq_len in input_ids.offsets().diff().tolist()]
        if any(seq_len <= 0 for seq_len in seq_lens):
            raise ValueError("Dynamic CP requires every input sequence to contain at least one token")

        scheduler_cls = get_megatron_dynamic_cp_scheduler_cls()
        scheduler = scheduler_cls(
            max_seqlen_per_dp_cp_rank=self.max_seqlen_per_dp_cp_rank,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            microbatch_group_size_per_vp_stage=None,
        )
        sample_id_groups = scheduler.get_groups_and_subsamples(list(enumerate(seq_lens)))

        expected_ids = set(range(len(seq_lens)))
        assigned_ids = {sample_id for group in sample_id_groups for ids in group for sample_id in ids}
        if assigned_ids != expected_ids:
            raise RuntimeError(
                "Megatron-Core Dynamic CP did not preserve the input sample set: "
                f"expected={sorted(expected_ids)}, assigned={sorted(assigned_ids)}"
            )

        dcp_rank = dcp_group.rank()
        micro_batches = []
        for rank_assignments in sample_id_groups:
            if len(rank_assignments) != self.total_ranks:
                raise RuntimeError(
                    "Megatron-Core Dynamic CP returned an assignment with "
                    f"{len(rank_assignments)} ranks, expected {self.total_ranks}"
                )
            if any(not sample_ids for sample_ids in rank_assignments):
                raise RuntimeError("Megatron-Core Dynamic CP returned an empty rank assignment")

            sample_ids = rank_assignments[dcp_rank]
            members = _cp_members(rank_assignments, dcp_rank)
            local_cp_size = len(members)
            cp_rank = members.index(dcp_rank)
            if members != list(range(members[0], members[0] + local_cp_size)):
                raise RuntimeError(f"Dynamic CP group members must be contiguous, got {members}")
            if members[0] % local_cp_size != 0:
                raise RuntimeError(f"Dynamic CP group {members} is not aligned to its size")
            if any(rank_assignments[member] != sample_ids for member in members):
                raise RuntimeError(f"Dynamic CP group {members} does not share one packed sample list")

            micro_batch = tu.index_select_tensor_dict(batch, sample_ids)
            selected_lens = [seq_lens[sample_id] for sample_id in sample_ids]
            padding_mask = _local_padding_mask(
                selected_lens,
                cp_size=local_cp_size,
                cp_rank=cp_rank,
                tp_size=tp_size,
            )
            tu.assign_non_tensor_data(micro_batch, "local_cp_size", local_cp_size)
            tu.assign_non_tensor_data(micro_batch, DCP_SAMPLE_IDS, list(sample_ids))
            tu.assign_non_tensor_data(micro_batch, DCP_GROUP_LEADER, dcp_rank == members[0])
            tu.assign_non_tensor_data(micro_batch, DCP_PADDING_MASK, padding_mask)
            tu.assign_non_tensor_data(
                micro_batch,
                DCP_LOCAL_NUM_TOKENS,
                padding_mask.numel() - int(padding_mask.sum().item()),
            )
            micro_batches.append(micro_batch)
        return micro_batches


def _detach_model_output(model_output: dict, sample_ids: list[int]) -> dict[str, list[torch.Tensor]]:
    detached = {}
    for key, value in model_output.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Dynamic CP model output {key!r} must be a tensor, got {type(value).__name__}")
        parts = list(value.unbind()) if value.is_nested else list(value.unbind(0))
        if len(parts) != len(sample_ids):
            raise ValueError(f"Dynamic CP output {key!r} has {len(parts)} samples for ids {sample_ids}")
        detached[key] = [part.detach().cpu() for part in parts]
    return detached


def postprocess_dynamic_cp_batch(output_lst: list[dict], batch_size: int, dcp_group) -> dict:
    """Collect one full-output copy per CP group and restore the original sample order."""
    output_device = next(
        (
            value.device
            for output in output_lst
            for value in output.get("model_output", {}).values()
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    local_records = []
    for output in output_lst:
        if not output.get(DCP_GROUP_LEADER, False):
            continue
        sample_ids = [int(sample_id) for sample_id in output[DCP_SAMPLE_IDS]]
        local_records.append(
            {
                "sample_ids": sample_ids,
                "loss": output.get("loss", None),
                "metrics": output.get("metrics", {}),
                "model_output": _detach_model_output(output.get("model_output", {}), sample_ids),
            }
        )

    records_by_rank = [None for _ in range(dcp_group.size())]
    torch.distributed.all_gather_object(records_by_rank, local_records, group=dcp_group)
    records = [record for rank_records in records_by_rank for record in rank_records]

    id_counts = Counter(sample_id for record in records for sample_id in record["sample_ids"])
    expected_ids = set(range(batch_size))
    if set(id_counts) != expected_ids or any(count != 1 for count in id_counts.values()):
        raise RuntimeError(
            "Dynamic CP output collection must contain every sample exactly once: "
            f"expected={sorted(expected_ids)}, counts={dict(sorted(id_counts.items()))}"
        )

    records.sort(key=lambda record: min(record["sample_ids"]))
    by_key: dict[str, dict[int, torch.Tensor]] = {}
    losses = []
    metrics = {}
    for record in records:
        if record["loss"] is not None:
            losses.append(record["loss"])
        append_to_dict(metrics, record["metrics"])
        for key, parts in record["model_output"].items():
            values = by_key.setdefault(key, {})
            for sample_id, part in zip(record["sample_ids"], parts, strict=True):
                values[sample_id] = part

    model_output = {}
    for key, values in by_key.items():
        missing = expected_ids.difference(values)
        if missing:
            raise RuntimeError(f"Dynamic CP output {key!r} is missing sample ids {sorted(missing)}")
        model_output[key] = torch.nested.as_nested_tensor(
            [values[sample_id] for sample_id in range(batch_size)], layout=torch.jagged
        )
        if output_device is not None:
            model_output[key] = model_output[key].to(output_device)

    return {"model_output": model_output, "loss": losses, "metrics": metrics}
