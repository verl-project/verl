import types

import torch

from verl.model_merger.fsdp_model_merger import FSDPModelMerger

A_LOG = "model.layers.0.linear_attn.A_log"
UP_PROJ = "model.layers.0.mlp.up_proj.weight"


def _merge(tmp_path, fp32_modules):
    # two ranks holding the same replicated weights, as a world-size-2 checkpoint; values bf16 cannot represent
    weights = {A_LOG: torch.tensor([0.1234567, 1.7654321]), UP_PROJ: torch.tensor([[0.1234567, 1.7654321]])}
    for rank in range(2):
        torch.save(dict(weights), tmp_path / f"model_world_size_2_rank_{rank}.pt")
    merger = FSDPModelMerger.__new__(FSDPModelMerger)
    merger.config = types.SimpleNamespace(local_dir=str(tmp_path), fp32_modules=fp32_modules)
    return weights, merger._load_and_merge_state_dicts(2, 2, (2,), ("fsdp",))


def test_the_fp32_modules_keep_their_fp32_values(tmp_path):
    weights, merged = _merge(tmp_path, ["A_log"])

    assert merged[A_LOG].dtype == torch.float32 and torch.equal(merged[A_LOG], weights[A_LOG])
    assert merged[UP_PROJ].dtype == torch.bfloat16


def test_without_fp32_modules_every_weight_is_bf16(tmp_path):
    _, merged = _merge(tmp_path, [])

    assert {tensor.dtype for tensor in merged.values()} == {torch.bfloat16}
