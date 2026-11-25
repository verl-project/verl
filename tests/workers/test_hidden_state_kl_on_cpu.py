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

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file

from verl.trainer.distillation.hidden_state import (
    _Head,
    _HiddenStateKL,
    _ResidentHead,
    _TeacherTokens,
    project_full_logits,
)


def _make_checkpoints(directory):
    rng = torch.Generator().manual_seed(99)
    for i in range(2):
        path = directory / str(i)
        path.mkdir()
        weight = torch.randn(7, 3 + i, generator=rng)
        key = "lm_head.weight" if i == 0 else "model.embed_tokens.weight"
        save_file({key: weight}, path / "model.safetensors")
        (path / "config.json").write_text(
            json.dumps({"model_type": "llama", "vocab_size": 7, "tie_word_embeddings": i == 1})
        )


def _compare(rank, size, layout, directory, autocast=False):
    if size > 1:
        dist.init_process_group("gloo", init_method=f"file://{directory}/init", rank=rank, world_size=size)
    try:
        tp = layout == "tp"
        group = dist.group.WORLD if size > 1 else None
        rng = torch.Generator().manual_seed(73 if tp else 73 + rank)
        dense_student = torch.randn(7, 8, generator=rng, requires_grad=True)
        shard = slice(rank * 4, (rank + 1) * 4) if tp and size == 2 else slice(None)
        local = dense_student.detach()[:, shard].clone().requires_grad_(True)
        groups = []
        full_logit_groups = []
        expected = dense_student.new_zeros(7)
        for i in range(2):
            path = Path(directory) / str(i)
            head = _Head(str(path), str(i), rank if tp else 0, size if tp else 1, local.size(1), None if tp else group)
            # An FSDP peer can have no samples assigned to the currently gathered teacher.
            rows = []
            full_logit_rows = []
            if tp or i == rank or size == 1:
                indices = torch.tensor([0, 2, 5] if i == 0 else [1, 4])
                h = torch.randn(len(indices), 3 + i, generator=rng)
                weight = next(iter(load_file(path / "model.safetensors").values()))
                teacher = (h @ weight.T).log_softmax(-1)
                student = dense_student[indices, :7].log_softmax(-1)
                expected[indices] = ((teacher - student) * teacher.exp()).sum(-1)
                rows.append((indices, h))
                full_logits = project_full_logits(str(path), str(i), h, torch.device("cpu"), 2)
                full_logit_rows.append((indices, full_logits))
            groups.append(_TeacherTokens(head, rows))
            full_logit_groups.append(_TeacherTokens(head, full_logit_rows, signal_kind="full_logits"))
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            actual = _HiddenStateKL.apply(local.unsqueeze(0), groups, 2, group if tp else None).squeeze(0)
        factors = torch.randn(7, generator=rng)
        expected.backward(factors)
        actual.backward(factors)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(local.grad, dense_student.grad[:, shard])

        # Dense records can be consumed repeatedly on CPU, even when the head is unavailable.
        transferred = local.detach().clone().requires_grad_(True)
        with patch.object(_ResidentHead, "get", side_effect=AssertionError("actor loaded a teacher head")):
            dense_kl = _HiddenStateKL.apply(transferred.unsqueeze(0), full_logit_groups, 2, group if tp else None)
            dense_kl.squeeze(0).backward(factors)
            reread = _HiddenStateKL.apply(transferred.unsqueeze(0), full_logit_groups, 2, group if tp else None)
        torch.testing.assert_close(dense_kl.squeeze(0), actual)
        torch.testing.assert_close(transferred.grad, local.grad)
        torch.testing.assert_close(reread, dense_kl)
    finally:
        if size > 1:
            dist.destroy_process_group()


@pytest.mark.parametrize("layout,world", [("tp", 1), ("tp", 2), ("fsdp", 2)])
def test_hidden_state_kl_matches_dense_with_masked_rows_and_padded_vocab(tmp_path, layout, world):
    _make_checkpoints(tmp_path)
    if world == 1:
        _compare(0, world, layout, tmp_path)
    else:
        mp.spawn(_compare, args=(world, layout, tmp_path), nprocs=world)


def test_hidden_state_kl_bf16_autocast_matches_dense_gradient(tmp_path):
    _make_checkpoints(tmp_path)
    _compare(0, 1, "tp", tmp_path, autocast=True)


def test_hidden_state_kl_selects_unpadded_causal_rows(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from verl.experimental.teacher_loop import hidden_state_store
    from verl.trainer.distillation.hidden_state import hidden_state_kl
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    _make_checkpoints(tmp_path)
    # Non-adjacent predictions cover an interior masked token and two trajectories.
    hidden_b = torch.tensor([[1.0, 2.0, 0.5, 2.5], [4.0, -1.0, 2.0, 1.0]])
    hidden_a = torch.tensor([[5.0, 2.0, -4.0]])
    records = [
        (dict(teacher_key="b", teacher_index=1, teacher_weight_version="1", positions=[1, 3]), hidden_b),
        (dict(teacher_key="a", teacher_index=0, teacher_weight_version="1", positions=[0]), hidden_a),
    ]
    monkeypatch.setattr(hidden_state_store, "read_hidden_records", lambda data: records)
    model_config = SimpleNamespace(
        teacher_key="route",
        teacher_models={
            "a": SimpleNamespace(model_path=str(tmp_path / "0")),
            "b": SimpleNamespace(model_path=str(tmp_path / "1")),
        },
        distillation_loss=SimpleNamespace(hidden_state_chunk_size=1),
    )
    logits = torch.randn(1, 9, 8, requires_grad=True)
    actor = SimpleNamespace(strategy="fsdp")
    data = list_of_dict_to_tensordict(
        [
            {"input_ids": torch.arange(6), "route": "b"},
            {"input_ids": torch.arange(3), "route": "a"},
        ]
    )
    actual = hidden_state_kl(actor, model_config, data, logits).flatten()
    expected = torch.zeros(9)
    for i, selected, h in [(1, [1, 3], hidden_b), (0, [6], hidden_a)]:
        w = next(iter(load_file(tmp_path / str(i) / "model.safetensors").values()))
        teacher = (h @ w.T).log_softmax(-1)
        expected[selected] = (teacher.exp() * (teacher - logits[0, selected, :7].log_softmax(-1))).sum(-1)
    torch.testing.assert_close(actual, expected)

    # With static padding to twelve tokens and two Ulysses ranks, the
    # second local shard contains the last trajectory plus three pad tokens.
    shard = torch.nn.functional.pad(logits[:, 6:], (0, 0, 0, 3))
    local_ordinals = torch.tensor([[7, 8, 9, 0, 0, 0]])
    local_loss = hidden_state_kl(actor, model_config, data, shard, token_indices=local_ordinals)
    torch.testing.assert_close(local_loss.flatten(), torch.nn.functional.pad(expected[6:], (0, 3)))

    records[0][0]["teacher_index"] = 0
    with pytest.raises(ValueError, match="registry index"):
        hidden_state_kl(actor, model_config, data, logits)
    records[0][0]["teacher_key"] = "a"
    with pytest.raises(ValueError, match="trajectory routing"):
        hidden_state_kl(actor, model_config, data, logits)


def test_reconstructed_logit_scale_matches_teacher_backend(tmp_path):
    from types import SimpleNamespace

    from transformers import CohereConfig

    from verl.trainer.distillation.hidden_state import _chunks, teacher_output_scale

    # HF supplies a non-unit Cohere default even if config.json omits logit_scale.
    _make_checkpoints(tmp_path)
    config = {"model_type": "cohere", "vocab_size": 7, "tie_word_embeddings": False}
    path = tmp_path / "0"
    (path / "config.json").write_text(json.dumps(config))
    h = torch.tensor([[1.0, -2.5, 0.125]])
    head = _Head(str(path), "1", 0, 1, 7)
    groups = [_TeacherTokens(head, [(torch.tensor([0]), h)])]
    _, actual = next(_chunks(groups, 1, torch.device("cpu")))
    weight = load_file(path / "model.safetensors")["lm_head.weight"]
    torch.testing.assert_close(actual, (h @ weight.T) * CohereConfig().logit_scale)
    assert teacher_output_scale(SimpleNamespace(model_type="granite", logit_scale=2.0, logits_scaling=4.0)) == 0.5
    with pytest.raises(ValueError, match="Unsupported teacher output scaling"):
        teacher_output_scale(SimpleNamespace(model_type="unrecognized", logits_scaling=4.0))


@pytest.mark.parametrize("global_tokens", [0, 4], ids=["all-ranks-empty", "local-rank-empty"])
def test_direct_kl_without_local_targets(global_tokens):
    from types import SimpleNamespace

    from tensordict import TensorDict

    from verl.trainer.distillation.losses import distillation_ppo_loss
    from verl.utils.tensordict_utils import assign_non_tensor
    from verl.workers.config import DistillationLossConfig

    data = TensorDict(
        {
            "prompts": torch.tensor([[10]]),
            "responses": torch.tensor([[20, 21]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "response_mask": torch.zeros(1, 2, dtype=torch.bool),
        },
        batch_size=[1],
    )
    assign_non_tensor(data, dp_size=2, batch_num_tokens=global_tokens, global_batch_size=2)
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    loss_config = DistillationLossConfig(
        loss_mode="forward_kl_hidden_state", use_task_rewards=False, use_policy_gradient=False
    )
    packed_losses = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss, metrics = distillation_ppo_loss(
        config, SimpleNamespace(distillation_loss=loss_config), {"distillation_losses": packed_losses}, data
    )
    assert loss.item() == 0
    assert set(metrics) == {"distillation/loss"}
    assert metrics["distillation/loss"].aggregate() == 0
    loss.backward()
    assert torch.equal(packed_losses.grad, torch.zeros_like(packed_losses))


def test_direct_kl_excludes_masked_assistant_and_environment_targets():
    from types import SimpleNamespace

    from tensordict import TensorDict

    from verl.trainer.distillation.losses import distillation_ppo_loss
    from verl.utils.tensordict_utils import assign_non_tensor
    from verl.workers.config import DistillationLossConfig

    data = TensorDict(
        {
            "prompts": torch.tensor([[10, 11]]),
            "responses": torch.tensor([[20, 21, 40, 22]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0, 1]]),
            "loss_mask": torch.tensor([[1, 0, 0, 1]]),
        },
        batch_size=[1],
    )
    assign_non_tensor(data, dp_size=1, batch_num_tokens=2, global_batch_size=1)
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    loss_config = DistillationLossConfig(
        loss_mode="forward_kl_hidden_state", use_task_rewards=False, use_policy_gradient=False, loss_max_clamp=None
    )
    # Predict tokens 20 and 22 from positions 1 and 4 in the full sequence.
    packed_losses = torch.tensor([0.0, 2.0, 5.0, 7.0, 4.0, 0.0], requires_grad=True)
    loss, metrics = distillation_ppo_loss(
        config, SimpleNamespace(distillation_loss=loss_config), {"distillation_losses": packed_losses}, data
    )
    assert loss.item() == 3.0
    assert metrics["distillation/loss_min"].aggregate() == 2.0
    assert metrics["distillation/loss_max"].aggregate() == 4.0
    loss.backward()
    torch.testing.assert_close(packed_losses.grad, torch.tensor([0.0, 0.5, 0.0, 0.0, 0.5, 0.0]))


def test_producer_full_logits_bf16_with_soft_cap_and_scale(tmp_path):
    # Deliberately use different values for every parameter to detect operation-order changes.
    config = dict(
        model_type="llama", vocab_size=5, tie_word_embeddings=True, final_logit_softcapping=1.5, logit_scale=0.75
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight = torch.arange(-5, 5, dtype=torch.bfloat16).reshape(5, 2) / 3
    save_file({"model.embed_tokens.weight": weight}, tmp_path / "model.safetensors")
    hidden = torch.tensor([[1, -2], [3, 1], [4, -2]], dtype=torch.bfloat16)
    for selected in (hidden, hidden[:0]):
        projected = project_full_logits(str(tmp_path), "v1", selected, torch.device("cpu"), 2)
        expected = ((selected @ weight.T) / 1.5).tanh() * 1.5 * 0.75
        assert projected.device == torch.device("cpu")
        torch.testing.assert_close(projected, expected, rtol=0, atol=0)
