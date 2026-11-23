"""Offline CUDA preflight for final-state capture and backward, no downloads."""

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from examples.search_r1.turn_hidden_diversity import PAIR_MASK, PREVIOUS, capture_decisions, distance_sums
from verl.utils import tensordict_utils as tu


def main(output):
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    nt = lambda xs: torch.nested.as_nested_tensor(xs, layout=torch.jagged, device="cuda")
    data = tu.get_tensordict(
        {
            "prompts": nt([ids[0, :2]]),
            PAIR_MASK: nt([torch.tensor([False, False, True], device="cuda")]),
            PREVIOUS: nt([torch.tensor([-1, -1, 0], device="cuda")]),
        },
        {"turn_hidden_margin": 0.1},
    )
    with capture_decisions(model, data) as captured:
        result = model(ids, use_cache=False, output_hidden_states=True)
    expected = distance_sums(result.hidden_states[-1][:, 4], result.hidden_states[-1][:, 2], 0.1)
    torch.testing.assert_close(captured["loss_sum"], expected["loss_sum"])
    assert captured["pairs"] == 1
    assert not model.lm_head._forward_pre_hooks
    # Activate the hinge to verify nonzero gradient through checkpointed blocks.
    tu.assign_non_tensor(data, turn_hidden_margin=2.0)
    with capture_decisions(model, data) as captured:
        model(ids, use_cache=False)
    captured["loss_sum"].backward()
    grad = model.model.layers[0].self_attn.q_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    report = {
        "passed": True,
        "cuda": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": str(model.dtype),
        "gradient_checkpointing": True,
        "pairs": captured["pairs"],
    }
    Path(output).write_text(json.dumps(report, indent=2))
    print("TURN_HIDDEN_PREFLIGHT_OK", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    main(parser.parse_args().output)
