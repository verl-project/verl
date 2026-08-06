"""Tests for DetachActorWorker FSDP1 handler routing (#7249).

Verifies that FSDP1 strategy uses fsdp1_sharded_save/load_to_cpu
instead of the FSDP2-only DTensor variants, and that the FSDP1
save/load round-trip preserves parameter values.
"""

import torch


def test_fsdp1_save_load_round_trip():
    """fsdp1_sharded_save_to_cpu / fsdp1_sharded_load_from_cpu preserves values."""
    from verl.utils.fsdp_utils import (
        fsdp1_sharded_save_to_cpu,
        fsdp1_sharded_load_from_cpu,
    )

    model = torch.nn.Linear(8, 4, bias=True)
    orig_weight = model.weight.data.clone()
    orig_bias = model.bias.data.clone()

    cpu_state = fsdp1_sharded_save_to_cpu(model)
    assert isinstance(cpu_state, dict)
    assert all(t.device == torch.device("cpu") for t in cpu_state.values())

    # Corrupt the model
    model.weight.data.fill_(999.0)
    model.bias.data.fill_(-999.0)
    assert not torch.equal(model.weight.data, orig_weight)

    # Restore from CPU
    fsdp1_sharded_load_from_cpu(model, cpu_state)
    assert torch.equal(model.weight.data, orig_weight)
    assert torch.equal(model.bias.data, orig_bias)


def test_fsdp1_save_is_detached_copy():
    """Saved state must be a detached copy, not a reference to live parameters."""
    from verl.utils.fsdp_utils import fsdp1_sharded_save_to_cpu

    model = torch.nn.Linear(4, 2)
    cpu_state = fsdp1_sharded_save_to_cpu(model)

    saved_weight = cpu_state["weight"].clone()
    model.weight.data.fill_(0.0)
    assert torch.equal(cpu_state["weight"], saved_weight), (
        "Saved state was mutated when model parameters changed"
    )


def test_fsdp1_handler_routing_in_source():
    """engine_workers.py must route 'fsdp' to fsdp1 functions, not fsdp2."""
    import pathlib

    src = pathlib.Path("verl/experimental/separation/engine_workers.py").read_text()

    # FSDP1 should get its own branch
    assert "fsdp1_sharded_save_to_cpu" in src
    assert "fsdp1_sharded_load_from_cpu" in src

    # 'fsdp' must NOT be in the same list as 'fsdp2'
    # The old bug: if strategy in ["fsdp", "fsdp2", "veomni"]:
    assert '["fsdp", "fsdp2", "veomni"]' not in src, (
        "'fsdp' must not be grouped with 'fsdp2' in strategy routing"
    )


def test_fsdp2_routing_preserved_in_source():
    """'fsdp2' and 'veomni' must still route to fsdp2 functions."""
    import pathlib

    src = pathlib.Path("verl/experimental/separation/engine_workers.py").read_text()
    assert '["fsdp2", "veomni"]' in src
    assert "fsdp2_sharded_save_to_cpu" in src
    assert "fsdp2_sharded_load_from_cpu" in src


def test_multi_layer_save_restore():
    """Round-trip works for models with multiple named parameter groups."""
    from verl.utils.fsdp_utils import (
        fsdp1_sharded_save_to_cpu,
        fsdp1_sharded_load_from_cpu,
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4),
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 2),
    )
    originals = {name: p.data.clone() for name, p in model.named_parameters()}

    cpu_state = fsdp1_sharded_save_to_cpu(model)
    assert len(cpu_state) == len(originals)

    for p in model.parameters():
        p.data.fill_(0.0)

    fsdp1_sharded_load_from_cpu(model, cpu_state)
    for name, p in model.named_parameters():
        assert torch.equal(p.data, originals[name]), f"Mismatch for {name}"


def test_restore_from_cpu_branches_fsdp1():
    """restore_model_from_cpu must NOT unpack (state, spec) tuple for fsdp strategy."""
    import pathlib

    src = pathlib.Path("verl/experimental/separation/engine_workers.py").read_text()
    # For FSDP1, the saved state is a plain dict (not a tuple of (state, spec))
    # so restore_model_from_cpu must not destructure it as a tuple
    assert 'strategy == "fsdp"' in src or "strategy == 'fsdp'" in src
