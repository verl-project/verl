# Copyright 2026 Amazon.com Inc and/or its affiliates
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
CPU tests for the DPO trainer's data path: dpo_trainer.yaml loading, the packed-batch helpers,
the packed-pair expansion in prepare_micro_batches, and DPODataset -> collator -> micro-batching
-> dpo_loss end to end.
"""

import os
import random

import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from verl.trainer.dpo.dpo_losses import DPOLossConfig
from verl.trainer.dpo.utils import DPO_EXPAND_FLAG, expand_packed_sequences
from verl.trainer.dpo_trainer import expanded_seqlens, prepare_packed_batch, reduce_dpo_metrics
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.dataset_utils import DPOTensorCollator
from verl.utils.dataset.dpo_dataset import DPODataConfig, DPODataset
from verl.workers.engine.utils import prepare_micro_batches
from verl.workers.utils.losses import dpo_loss

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "verl", "trainer", "config"))


# ─── Helpers ─────────────────────────────────────────────────────────────


def _packed_sample(pair_idx, chosen_len, rejected_len, prompt_len=2):
    """A collator input. Token value 2*pair+1 marks chosen, 2*pair+2 marks rejected."""
    ids = torch.cat([torch.full((chosen_len,), 2 * pair_idx + 1), torch.full((rejected_len,), 2 * pair_idx + 2)]).long()
    mask = torch.ones_like(ids)
    mask[:prompt_len] = 0
    mask[chosen_len : chosen_len + prompt_len] = 0
    return {"input_ids": ids, "loss_mask": mask, "boundary_offset": torch.tensor(chosen_len, dtype=torch.long)}


def _packed_batch(lengths, with_ref=True, **meta):
    collated = DPOTensorCollator()([_packed_sample(i, c, r) for i, (c, r) in enumerate(lengths)])
    packed = prepare_packed_batch(collated)
    if with_ref:
        # ref_logps[i] = (-(2i+1), -(2i+2)) so each value names its own row.
        packed["ref_logps"] = -torch.arange(1, 2 * len(lengths) + 1, dtype=torch.float32).reshape(-1, 2)
    meta = {"pad_mode": "no_padding", "temperature": 1.0, DPO_EXPAND_FLAG: True, **meta}
    return tu.get_tensordict(tensor_dict=packed, non_tensor_dict=meta)


def _assert_interleaved(micro_batches, num_pairs, with_ref=True):
    seen = []
    for mb in micro_batches:
        firsts = [row[0].item() for row in mb["input_ids"].unbind()]
        assert len(firsts) % 2 == 0, firsts
        for chosen, rejected in zip(firsts[0::2], firsts[1::2], strict=True):
            assert chosen % 2 == 1 and rejected == chosen + 1, f"pair broken or reordered: {firsts}"
            seen.append((chosen - 1) // 2)
        assert all(p.tolist() == list(range(p.numel())) for p in mb["position_ids"].unbind())
        if with_ref:
            assert mb["ref_logps"].tolist() == [-float(v) for v in firsts]
        assert "boundary_offset" not in mb.keys()
        assert tu.get_non_tensor_data(mb, DPO_EXPAND_FLAG, default=None) is None
        assert tu.get_non_tensor_data(mb, "temperature", default=None) == 1.0
    assert sorted(seen) == list(range(num_pairs))


# ─── Config ──────────────────────────────────────────────────────────────


def test_dpo_trainer_yaml_builds_every_config():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="dpo_trainer")
    for section in ("engine", "optim", "checkpoint", "profiler"):
        omega_conf_to_dataclass(cfg[section])
    data_config = omega_conf_to_dataclass(cfg.data, DPODataConfig)
    assert data_config.pad_mode == "no_padding"
    DPOLossConfig(
        loss_type=cfg.trainer.loss_type,
        beta=cfg.trainer.beta,
        label_smoothing=cfg.trainer.label_smoothing,
        reference_free=cfg.trainer.reference_free,
    )


# ─── Batch helpers ───────────────────────────────────────────────────────


def test_prepare_packed_batch_and_seqlens():
    collated = DPOTensorCollator()([_packed_sample(0, 5, 3), _packed_sample(1, 4, 6)])
    packed = prepare_packed_batch(collated)
    assert [p.tolist() for p in packed["position_ids"].unbind()] == [list(range(8)), list(range(10))]
    assert packed["boundary_offset"].tolist() == [5, 4]
    assert expanded_seqlens(packed) == [5, 3, 4, 6]


def test_expand_packed_sequences_rejects_bad_offset():
    with pytest.raises(ValueError, match="outside"):
        expand_packed_sequences([torch.arange(4)], [torch.ones(4)], [4])


def test_reduce_dpo_metrics():
    metrics = {"dpo/loss": [[0.2, 0.3], [0.4, 0.1]], "dpo/accuracy": [0.25, 0.5], "loss": 0.5, "lr": 1e-5}
    reduced = reduce_dpo_metrics(metrics)
    assert reduced["dpo/loss"] == pytest.approx(0.5)  # (0.5 + 0.5) / 2 ranks
    assert reduced["dpo/accuracy"] == pytest.approx(0.75)  # one rank: sum of micro-batches
    assert reduced["loss"] == 0.5 and reduced["lr"] == 1e-5


# ─── prepare_micro_batches with the DPO flag ─────────────────────────────


@pytest.mark.parametrize("seed", range(10))
def test_dynamic_batching_expands_whole_pairs(seed):
    rng = random.Random(seed)
    num_pairs = rng.choice([1, 2, 3, 8, 16])
    lengths = [(rng.choice([4, 40, 300]), rng.choice([4, 40, 300])) for _ in range(num_pairs)]
    data = _packed_batch(lengths, use_dynamic_bsz=True, max_token_len_per_gpu=1024)
    micro_batches, indices = prepare_micro_batches(data)
    _assert_interleaved(micro_batches, num_pairs)
    assert sorted(i for idx in indices for i in idx) == list(range(num_pairs))


@pytest.mark.parametrize("pairs_per_micro_batch", [1, 2, 4])
def test_fixed_batching_expands_whole_pairs(pairs_per_micro_batch):
    data = _packed_batch(
        [(5 + i, 9 - i) for i in range(8)], use_dynamic_bsz=False, micro_batch_size_per_gpu=pairs_per_micro_batch
    )
    micro_batches, _ = prepare_micro_batches(data)
    assert len(micro_batches) == 8 // pairs_per_micro_batch
    _assert_interleaved(micro_batches, 8)


def test_equal_offsets_expand_correctly():
    """Every chosen side has the same length: the case that broke on tensordict 0.8 with int offsets."""
    data = _packed_batch([(6, 3), (6, 9), (6, 4), (6, 6)], use_dynamic_bsz=False, micro_batch_size_per_gpu=2)
    micro_batches, _ = prepare_micro_batches(data)
    _assert_interleaved(micro_batches, 4)
    assert micro_batches[0]["input_ids"].offsets().diff().tolist() == [6, 3, 6, 9]


def test_reference_free_batch_without_ref_logps():
    data = _packed_batch([(5, 3), (4, 6)], with_ref=False, use_dynamic_bsz=False, micro_batch_size_per_gpu=1)
    micro_batches, _ = prepare_micro_batches(data)
    _assert_interleaved(micro_batches, 2, with_ref=False)
    assert all("ref_logps" not in mb.keys() for mb in micro_batches)


def test_without_flag_nothing_changes():
    data = _packed_batch([(5, 3), (4, 6)], use_dynamic_bsz=False, micro_batch_size_per_gpu=1)
    tu.assign_non_tensor(data, **{DPO_EXPAND_FLAG: False})
    micro_batches, _ = prepare_micro_batches(data)
    assert [mb["input_ids"].offsets().diff().tolist() for mb in micro_batches] == [[8], [10]]
    assert all("boundary_offset" in mb.keys() for mb in micro_batches)


# ─── Dataset -> collator -> micro-batching -> loss ───────────────────────


class _CharTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        return {"input_ids": torch.tensor([[ord(c) % 200 + 1 for c in text] or [1]], dtype=torch.long)}


@pytest.mark.parametrize("dynamic", [True, False])
def test_dataset_to_loss_matches_single_micro_batch(tmp_path, dynamic):
    """Micro-batched losses over real DPODataset rows must sum to the loss of the whole batch."""
    path = os.path.join(tmp_path, "pairs.parquet")
    pd.DataFrame(
        {
            "prompt": ["Q0", "Q1 longer", "Q2", "Q3 much longer prompt", "Q4", "Q5"],
            "chosen": ["yes", "the answer is 4", "ok", "fine", "a", "same len"],
            "rejected": ["no", "5", "not ok at all", "bad", "bbbbbbbbbb", "same len!"],
        }
    ).to_parquet(path)
    config = DPODataConfig(max_length=64, truncation="right", shuffle=False)
    dataset = DPODataset(data_files=path, tokenizer=_CharTokenizer(), config=config)
    collated = next(iter(DataLoader(dataset, batch_size=6, collate_fn=DPOTensorCollator())))

    packed = prepare_packed_batch(collated)
    num_pairs = 6
    gen = torch.Generator().manual_seed(0)
    # Fake per-token log-probs and reference log-probs, fixed per expanded row.
    expanded = expand_packed_sequences(
        list(packed["input_ids"].unbind()), list(packed["loss_mask"].unbind()), packed["boundary_offset"].tolist()
    )
    rows = expanded["input_ids"].unbind()
    token_logps = {i: -torch.rand(r.numel(), generator=gen) for i, r in enumerate(rows)}
    packed["ref_logps"] = -torch.rand(num_pairs, 2, generator=gen) * 5
    meta = {"pad_mode": "no_padding", "dp_size": 1, "global_batch_size": num_pairs, DPO_EXPAND_FLAG: True}
    meta.update(
        {"use_dynamic_bsz": True, "max_token_len_per_gpu": 80}
        if dynamic
        else {"use_dynamic_bsz": False, "micro_batch_size_per_gpu": 2}
    )
    data = tu.get_tensordict(tensor_dict=packed, non_tensor_dict=meta)

    def log_probs_for(micro_batch):
        # Identify each expanded row by its tokens, then look up its fixed log-probs.
        out = []
        for row in micro_batch["input_ids"].unbind():
            match = [i for i, r in enumerate(rows) if r.numel() == row.numel() and torch.equal(r, row)]
            out.append(token_logps[match[0]])
        return torch.nested.as_nested_tensor(out, layout=torch.jagged)

    config = DPOLossConfig(beta=0.1)
    micro_batches, _ = prepare_micro_batches(data)
    assert len(micro_batches) > 1
    total = sum(dpo_loss(config, {"log_probs": log_probs_for(mb)}, mb)[0].item() for mb in micro_batches)

    whole = tu.get_tensordict(
        tensor_dict=packed, non_tensor_dict={**meta, "use_dynamic_bsz": False, "micro_batch_size_per_gpu": num_pairs}
    )
    (single,), _ = prepare_micro_batches(whole)
    expected = dpo_loss(config, {"log_probs": log_probs_for(single)}, single)[0].item()
    assert total == pytest.approx(expected, rel=1e-6)
