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

import pytest
import torch
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.sft_val_utils import (
    build_sft_val_dataloader,
    pad_sft_val_batch,
    reduce_sft_val_loss,
    resolve_sft_val_batch_size,
    sft_val_batch_divisor,
    sft_val_num_tokens,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator

# gsm8k test.parquet: the val set the reported validation crash was hit with.
GSM8K_VAL_ROWS = 1319


class _Toy(Dataset):
    """Val dataset whose row ``i`` carries exactly ``i + 1`` loss-mask tokens."""

    def __init__(self, num_rows: int = 200):
        self.num_rows = num_rows

    def __len__(self):
        return self.num_rows

    def __getitem__(self, i):
        n = i + 1
        return {
            "input_ids": torch.arange(n),
            "attention_mask": torch.ones(n, dtype=torch.long),
            "position_ids": torch.arange(n),
            "loss_mask": torch.ones(n, dtype=torch.long),
        }

    def total_tokens(self) -> int:
        return sum(i + 1 for i in range(self.num_rows))


def _batch(token_counts: list[int]):
    """A collated no-padding SFT batch whose rows carry the given loss-mask token counts."""
    return tu.get_tensordict(
        tensor_dict={
            "input_ids": torch.nested.as_nested_tensor([torch.arange(t) for t in token_counts], layout=torch.jagged),
            "loss_mask": torch.nested.as_nested_tensor(
                [torch.ones(t, dtype=torch.long) for t in token_counts], layout=torch.jagged
            ),
        },
        non_tensor_dict={"use_dynamic_bsz": True},
    )


def _num_rows(collated_batch) -> int:
    """Row count of a batch as the collator hands it out (a plain dict of tensors)."""
    return int(collated_batch["input_ids"].shape[0])


def _make_val_loader(*, num_replicas: int, rank: int = 0, dataset=None, data_config=None):
    """Mirror of the validation dataloader both SFT trainers build."""
    return build_sft_val_dataloader(
        dataset=dataset if dataset is not None else _Toy(),
        data_config=data_config if data_config is not None else {},
        dp_rank=rank,
        dp_size=num_replicas,
        collate_fn=SFTTensorCollator(DatasetPadMode.NO_PADDING),
        num_workers=0,
        device_name="cpu",
    )


# ----------------------------------------------------------------------------------
# validation batch size resolution
# ----------------------------------------------------------------------------------


def test_resolve_prefers_explicit_val_batch_size():
    assert resolve_sft_val_batch_size({"val_batch_size": 16}, 200) == 16


def test_resolve_defaults_to_full_val_set():
    assert resolve_sft_val_batch_size({}, 200) == 200
    assert resolve_sft_val_batch_size({"micro_batch_size_per_gpu": 4}, 200) == 200


def test_resolve_treats_non_positive_as_full_val_set():
    """``-1`` means "use the full dataset" everywhere else in the SFT data config."""
    assert resolve_sft_val_batch_size({"val_batch_size": -1}, GSM8K_VAL_ROWS) == GSM8K_VAL_ROWS
    assert resolve_sft_val_batch_size({"val_batch_size": 0}, GSM8K_VAL_ROWS) == GSM8K_VAL_ROWS
    assert resolve_sft_val_batch_size({"val_batch_size": None}, GSM8K_VAL_ROWS) == GSM8K_VAL_ROWS


# ----------------------------------------------------------------------------------
# batch divisibility: dispatch chunking (DP) and static micro-batch splitting
# ----------------------------------------------------------------------------------


def test_divisor_tracks_dp_size_and_micro_batch_size():
    dynamic = {"use_dynamic_bsz": True, "micro_batch_size_per_gpu": 4}
    assert sft_val_batch_divisor(dynamic, dp_size=1) == 1
    assert sft_val_batch_divisor(dynamic, dp_size=2) == 2

    static = {"use_dynamic_bsz": False, "micro_batch_size_per_gpu": 4}
    assert sft_val_batch_divisor(static, dp_size=1) == 4
    assert sft_val_batch_divisor(static, dp_size=2) == 8


def test_full_val_batch_is_chunkable_after_padding():
    """A single-batch val set must survive the DP chunk the worker group performs.

    ``chunk_tensordict`` asserts ``len(td) % chunks == 0``, and 1319 gsm8k rows over DP=2
    does not divide.
    """
    with pytest.raises(AssertionError, match="divisible by chunks"):
        tu.chunk_tensordict(_batch([5] * GSM8K_VAL_ROWS), 2)

    padded, pad_size = pad_sft_val_batch(_batch([5] * GSM8K_VAL_ROWS), sft_val_batch_divisor({}, dp_size=2))
    assert pad_size == 1
    assert len(padded) == GSM8K_VAL_ROWS + 1
    assert [len(chunk) for chunk in tu.chunk_tensordict(padded, 2)] == [660, 660]


def test_full_val_batch_is_micro_batchable_after_padding():
    """The ``use_dynamic_bsz=False`` path additionally asserts divisibility by
    ``micro_batch_size_per_gpu``."""
    from verl.workers.engine.utils import prepare_micro_batches

    static = {"use_dynamic_bsz": False, "micro_batch_size_per_gpu": 4}
    batch = _batch([5] * GSM8K_VAL_ROWS)
    tu.assign_non_tensor(batch, use_dynamic_bsz=False, micro_batch_size_per_gpu=4)
    with pytest.raises(AssertionError, match="micro_batch_size_per_gpu"):
        prepare_micro_batches(batch, dp_group=None)

    padded, pad_size = pad_sft_val_batch(batch, sft_val_batch_divisor(static, dp_size=1))
    assert pad_size == 1
    micro_batches, _ = prepare_micro_batches(padded, dp_group=None)
    assert sum(len(mb) for mb in micro_batches) == GSM8K_VAL_ROWS + 1
    assert all(len(mb) % 4 == 0 for mb in micro_batches)


def test_padding_rows_carry_no_tokens_and_no_loss():
    """Padding must not move the loss: pad rows are loss-masked out, so they contribute
    neither to ``masked_sum(log_prob, loss_mask)`` nor to the ``batch_num_tokens`` divisor.
    """
    batch = _batch([3, 4, 5])
    padded, pad_size = pad_sft_val_batch(batch, 4)
    assert pad_size == 1
    assert sft_val_num_tokens(padded) == sft_val_num_tokens(batch) == 12
    rows = list(padded["loss_mask"].unbind(0))
    assert int(rows[-1].sum()) == 0
    assert all(int(row.sum()) > 0 for row in rows[:-1])


def test_padding_is_a_noop_when_already_divisible():
    batch = _batch([3, 4, 5, 6])
    padded, pad_size = pad_sft_val_batch(batch, 4)
    assert pad_size == 0
    assert padded is batch
    assert pad_sft_val_batch(batch, 1)[1] == 0


# ----------------------------------------------------------------------------------
# val loss aggregation
# ----------------------------------------------------------------------------------


def test_reduce_sft_val_loss_is_token_weighted():
    """``sft_loss`` divides by the all-reduced ``loss_mask.sum()``, so a batch's reported
    loss is already that batch's global per-token NLL. Aggregating across batches is
    therefore ``sum(loss_b * T_b) / sum(T_b)`` and not a per-sample mean.

    Worked example: token counts ``[10, 10, 10, 10, 1000]`` with a per-token NLL of 2.0 for
    the four short rows and 1.0 for the long one, at ``val_batch_size=2``. Weighting by
    sample count instead reports 1.8 (+73%).
    """
    batches = [_batch([10, 10]), _batch([10, 10]), _batch([1000])]
    per_token_nll = [2.0, 2.0, 1.0]

    token_weighted = reduce_sft_val_loss(
        [(nll, sft_val_num_tokens(batch)) for nll, batch in zip(per_token_nll, batches, strict=True)]
    )
    assert token_weighted == pytest.approx(1080 / 1040)

    sample_weighted = reduce_sft_val_loss(
        [(nll, len(batch)) for nll, batch in zip(per_token_nll, batches, strict=True)]
    )
    assert sample_weighted == pytest.approx(1.8)
    assert token_weighted != pytest.approx(sample_weighted)


def test_reduce_sft_val_loss_without_tokens_is_none():
    assert reduce_sft_val_loss([]) is None
    assert reduce_sft_val_loss([(1.0, 0)]) is None


def test_num_tokens_counts_loss_mask_not_sequences():
    batch = _batch([3, 4, 5])
    assert len(batch) == 3
    assert sft_val_num_tokens(batch) == 12


# ----------------------------------------------------------------------------------
# dataloader construction
# ----------------------------------------------------------------------------------


def test_drop_last_true_with_train_batch_is_empty():
    """The #7464 bug itself: a train-sized batch with ``drop_last=True`` yields no batch."""
    dataset = _Toy()
    loader = StatefulDataLoader(
        dataset,
        batch_size=256,
        sampler=DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False, drop_last=True),
        drop_last=True,
    )
    assert len(loader) == 0


def test_drop_last_false_keeps_short_val_set():
    """Regression test for #7464: a val set smaller than one batch must still yield a batch."""
    loader = _make_val_loader(num_replicas=1, data_config={"val_batch_size": 256})
    assert len(loader) == 1
    assert sum(_num_rows(batch) for batch in loader) == 200


def test_val_loader_evaluates_every_sample_exactly_once():
    """``DistributedSampler(drop_last=False)`` repeats real samples so every rank gets
    ``ceil(N / D)`` indices, which evaluates and counts those rows twice. The rows that
    even out the shards must be loss-masked out instead, so ``val/loss`` stays a mean over
    distinct samples.
    """
    dataset = _Toy(num_rows=10)
    dp_size = 4

    batch_counts, tokens = [], 0
    for rank in range(dp_size):
        loader = _make_val_loader(num_replicas=dp_size, rank=rank, dataset=dataset)
        batch_counts.append(len(loader))
        for batch in loader:
            tokens += sft_val_num_tokens(batch)

    # every rank must run the same number of batches, or the engine's collectives deadlock
    assert len(set(batch_counts)) == 1
    assert tokens == dataset.total_tokens() == 55


def test_val_loader_shards_a_prime_val_set_without_dropping_rows():
    dataset = _Toy(num_rows=GSM8K_VAL_ROWS)
    dp_size = 8
    tokens = 0
    for rank in range(dp_size):
        loader = _make_val_loader(num_replicas=dp_size, rank=rank, dataset=dataset)
        for batch in loader:
            tokens += sft_val_num_tokens(batch)
    assert tokens == dataset.total_tokens()


def _per_token_nll(collated_batch) -> float:
    """Stand-in for what the engine reports back per batch.

    ``sft_loss`` returns ``-masked_sum(log_prob, loss_mask) / batch_num_tokens * dp_size``,
    which after the DP average in ``_postprocess_output`` is the batch's global per-token
    NLL. Here every token of a row of length ``n`` is given a log prob of ``-n``, so the
    exact answer over a set of rows is ``sum(n^2) / sum(n)``.
    """
    rows = collated_batch["input_ids"].unbind(0)
    masks = collated_batch["loss_mask"].unbind(0)
    weighted = sum(float(row.shape[0]) * int(mask.sum()) for row, mask in zip(rows, masks, strict=True))
    tokens = sum(int(mask.sum()) for mask in masks)
    return weighted / tokens if tokens else 0.0


def test_reported_val_loss_is_the_exact_per_token_nll_of_the_val_set():
    """End-to-end check of the validation math the SPMD trainer runs.

    Shards a val set over DP, pads every batch to what the engine needs, reports each batch's
    global per-token NLL, and weights by DP-local token counts the way the trainer's
    ``all_reduce(SUM)`` pair does. The result has to be the per-token NLL over the distinct
    val rows -- no dropped rows, no double-counted rows, no per-sample skew.
    """
    dataset = _Toy(num_rows=10)
    dp_size, divisor = 4, sft_val_batch_divisor({"use_dynamic_bsz": False, "micro_batch_size_per_gpu": 2})
    data_config = {"val_batch_size": 2, "use_dynamic_bsz": False, "micro_batch_size_per_gpu": 2}

    per_rank_batches = []
    for rank in range(dp_size):
        loader = _make_val_loader(num_replicas=dp_size, rank=rank, dataset=dataset, data_config=data_config)
        batches = []
        for batch in loader:
            padded, _ = pad_sft_val_batch(tu.get_tensordict(tensor_dict=batch), divisor)
            assert len(padded) % divisor == 0
            batches.append(padded)
        per_rank_batches.append(batches)

    numerator, denominator = 0.0, 0
    for batches in zip(*per_rank_batches, strict=True):
        # one validation step: the loss the engine reports is global over the DP group
        global_tokens = sum(sft_val_num_tokens(b) for b in batches)
        global_nll = sum(_per_token_nll(b) * sft_val_num_tokens(b) for b in batches) / global_tokens
        for batch in batches:
            numerator += global_nll * sft_val_num_tokens(batch)
            denominator += sft_val_num_tokens(batch)

    expected = sum((i + 1) ** 2 for i in range(len(dataset))) / dataset.total_tokens()
    assert numerator / denominator == pytest.approx(expected)
    assert denominator == dataset.total_tokens()
