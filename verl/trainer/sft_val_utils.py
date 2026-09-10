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

"""Helpers shared by the SPMD and Ray SFT trainers for building and reducing validation.

The invariant this module leans on: a row whose loss mask is all zeros is invisible to
``sft_loss``. It adds nothing to ``masked_sum(log_prob, loss_mask)`` and nothing to
``batch_num_tokens`` (the all-reduced ``loss_mask.sum()`` that loss is divided by), so such
rows can be appended freely to even out shard lengths and batch sizes without moving
``val/loss`` by a single token.
"""

import math
from typing import Optional

import torch
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.utils import tensordict_utils as tu

# Keys whose zeroing takes a row out of the loss. ``loss_mask`` is the one every backend
# derives ``batch_num_tokens`` from; ``response_mask`` is what the padded pad modes reduce.
_LOSS_MASK_KEYS = ("loss_mask", "response_mask")


def resolve_sft_val_batch_size(data_config, val_dataset_len: int) -> int:
    """Pick the SFT validation dataloader batch size.

    ``data.val_batch_size`` when it is set to a positive value, otherwise the full val set.
    ``null``/``0``/``-1`` all mean "the whole dataset", matching how ``train_max_samples``
    and ``val_max_samples`` read ``-1`` in the same config. ``micro_batch_size_per_gpu`` is
    an engine split size, not a dataloader knob.
    """
    val_batch_size = data_config.get("val_batch_size", None)
    if val_batch_size is None or int(val_batch_size) <= 0:
        return max(1, int(val_dataset_len))
    return int(val_batch_size)


def sft_val_batch_divisor(data_config, dp_size: int = 1) -> int:
    """Row count a validation batch has to be a multiple of before it reaches the engine.

    Two asserts stand between a val batch and its forward pass:

    * ``chunk_tensordict`` requires ``len(batch) % dp_size == 0`` wherever a worker group
      splits a dispatched batch across the DP mesh. Pass ``dp_size=1`` where the dataloader
      is already sharded per rank (the SPMD trainer) and the real DP size where the driver
      hands a whole batch to the worker group (the Ray trainer).
    * with ``use_dynamic_bsz=False``, ``prepare_micro_batches`` requires the per-rank rows
      to be a multiple of ``micro_batch_size_per_gpu``. The dynamic path packs micro-batches
      by token budget and has no such constraint.

    ``force_group_size`` is left out on purpose: it is a reward-model knob that the SFT
    trainers never set, so it stays at its default of 1.
    """
    dp_size = max(1, int(dp_size))
    if data_config.get("use_dynamic_bsz", True):
        return dp_size
    micro_batch_size_per_gpu = data_config.get("micro_batch_size_per_gpu", 1) or 1
    return dp_size * max(1, int(micro_batch_size_per_gpu))


def pad_sft_val_batch(batch, divisor: int):
    """Pad a collated validation batch up to a multiple of ``divisor``.

    The padding rows are copies of the first row with their loss mask zeroed, so the loss
    and the token count of the padded batch are exactly those of ``batch``. That keeps every
    real val row evaluated exactly once, which rounding the batch size down (or bringing
    ``drop_last=True`` back) would not.

    Returns ``(batch, pad_size)``; ``batch`` is handed back untouched when it already fits.
    """
    num_rows = len(batch)
    if divisor <= 1 or num_rows % divisor == 0:
        return batch, 0

    pad_size = divisor - num_rows % divisor
    padded = tu.index_select_tensor_dict(batch, list(range(num_rows)) + [0] * pad_size)
    for key in _LOSS_MASK_KEYS:
        if key not in padded.keys():
            continue
        mask = padded[key]
        if mask.is_nested:
            rows = list(mask.unbind(0))
            for i in range(num_rows, num_rows + pad_size):
                rows[i] = torch.zeros_like(rows[i])
            padded[key] = tu.nested_tensor_from_tensor_list(
                rows, ragged_idx=getattr(mask, "_ragged_idx", mask.dim() - 1)
            )
        else:
            mask = mask.clone()
            mask[num_rows:] = 0
            padded[key] = mask
    return padded, pad_size


def sft_val_num_tokens(batch) -> int:
    """Loss-mask token count of a collated SFT validation batch.

    This is this rank's share of the ``batch_num_tokens`` the engine divides the loss by, so
    it is the weight to aggregate per-batch validation losses with.
    """
    loss_mask = batch["loss_mask"]
    return int(loss_mask.values().sum() if loss_mask.is_nested else loss_mask.sum())


def reduce_sft_val_loss(losses_and_token_counts: list[tuple[float, int]]) -> Optional[float]:
    """Token-weighted mean of per-batch validation losses. ``None`` if there were no tokens.

    ``sft_loss`` normalizes by ``batch_num_tokens``, an all-reduced ``loss_mask.sum()`` over
    the DP group, so each per-batch loss reported back is already the global per-token NLL
    of that batch. Combining batches is therefore ``sum(loss_b * T_b) / sum(T_b)``; a
    per-sample or per-batch mean over-weights the batches made of short sequences.
    """
    total_tokens = sum(num_tokens for _, num_tokens in losses_and_token_counts)
    if total_tokens <= 0:
        return None
    return sum(float(loss) * num_tokens for loss, num_tokens in losses_and_token_counts) / total_tokens


class LossMaskedPaddedDataset(Dataset):
    """A val dataset grown to ``target_len`` rows with loss-masked-out copies of real rows.

    ``DistributedSampler(drop_last=False)`` pads its index list by repeating from the head so
    every rank receives ``ceil(N / D)`` indices. Those repeats are real samples: they get
    evaluated twice and counted twice, and because ``shuffle=False`` and the val sampler
    never sees ``set_epoch``, it is the same rows at every validation step -- a stable bias
    towards whatever sits at the front of the val set, worst for exactly the small val sets
    #7464 is about. Rounding the dataset itself up to a multiple of the DP size keeps the
    per-rank index counts equal -- which the engine's collectives need -- while the extra
    rows carry no tokens and no loss.
    """

    def __init__(self, dataset, target_len: int):
        assert len(dataset) > 0, "cannot pad an empty val dataset"
        assert target_len >= len(dataset), f"target_len {target_len} is below len(dataset) {len(dataset)}"
        self.dataset = dataset
        self.target_len = int(target_len)

    def __len__(self):
        return self.target_len

    def __getitem__(self, index):
        real_len = len(self.dataset)
        if index < real_len:
            return self.dataset[index]
        item = dict(self.dataset[index % real_len])
        for key in _LOSS_MASK_KEYS:
            if key in item:
                item[key] = torch.zeros_like(item[key])
        return item


def pad_val_dataset_for_dp(dataset, dp_size: int):
    """Round a val dataset up to a multiple of ``dp_size`` with loss-masked-out rows."""
    dp_size = max(1, int(dp_size))
    target_len = math.ceil(len(dataset) / dp_size) * dp_size
    if target_len == len(dataset):
        return dataset
    return LossMaskedPaddedDataset(dataset, target_len)


def build_sft_val_dataloader(
    *,
    dataset,
    data_config,
    dp_rank: int,
    dp_size: int,
    collate_fn,
    num_workers: int,
    device_name: str,
) -> StatefulDataLoader:
    """Build the SFT validation dataloader.

    ``drop_last=False`` on both the sampler and the loader, so a val set smaller than one
    batch still yields a batch instead of an empty loader and ``val/loss=NaN`` (#7464). The
    dataset is padded to a multiple of ``dp_size`` first, so that ``drop_last=False`` does
    not make the sampler repeat -- and the metric double-count -- real samples.
    """
    dataset = pad_val_dataset_for_dp(dataset, dp_size)
    val_batch_size = resolve_sft_val_batch_size(data_config, len(dataset))
    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=False)
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=val_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        pin_memory_device=device_name,
    )
    assert len(dataloader) >= 1, "Validation dataloader is empty!"
    return dataloader
