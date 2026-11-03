"""CPU checks for the MaxText engine's batch, loss, and weight layout.

These modules are loaded by file path so collection does not import the engine
package, which pulls in transformer engine.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[4]


def _load(name: str, relative: str):
    path = _ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_batch = _load("maxtext_batch_mod", "verl/workers/engine/maxtext/batch.py")
_export = _load("maxtext_export_mod", "verl/workers/engine/maxtext/export.py")
_loss = _load("maxtext_loss_mod", "verl/workers/engine/maxtext/loss.py")
_ckpt = _load("maxtext_ckpt_mod", "verl/workers/engine/maxtext/checkpoint.py")
_params = _load("maxtext_params_mod", "verl/workers/engine/maxtext/params.py")
_hf_import = _load("maxtext_hf_import_mod", "verl/workers/engine/maxtext/hf_import.py")
_count = _load("accelerator_count_mod", "verl/single_controller/accelerator_count.py")


def test_one_process_owns_the_node():
    spec = {"global_pool": [1], "rollout": [8]}
    assert _count.accelerator_total(spec, {"global_pool": 8}) == 16


def test_response_logprob_sits_one_position_before_the_token():
    mask = np.array([[1, 1, 1, 1, 0]])
    response = np.array([[0, 0, 1, 1, 0]])
    positions = _batch.response_positions(mask, response)
    assert positions.tolist() == [[0, 1, 1, 0, 0]]


def test_next_token_logprob_is_zero_when_the_token_is_certain():
    tokens = np.array([[1, 2, 0]])
    logits = np.full((1, 3, 4), -80.0, dtype=np.float32)
    logits[0, 0, 2] = 20.0
    logits[0, 1, 0] = 20.0
    log_probs = _batch.next_token_logprobs(logits, tokens)
    assert log_probs.shape == (1, 3)
    assert log_probs[0, 0] > -1e-4
    assert log_probs[0, 1] > -1e-4
    assert log_probs[0, 2] == 0.0


def test_next_token_entropy_matches_the_definition():
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((2, 5, 7)).astype(np.float32)
    got = _batch.next_token_entropy(logits)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    want = -(probs * np.log(probs)).sum(-1)
    np.testing.assert_allclose(got[:, :-1], want[:, :-1], atol=1e-5)
    assert (got[:, -1] == 0).all()


def test_clipped_loss_matches_the_vanilla_policy_loss():
    import torch

    log_prob = np.array([[0.0, 2.302585], [-5.0, 0.0]], dtype=np.float32)
    old = np.zeros_like(log_prob)
    advantages = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    mask = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    loss, metrics = _loss.clipped_policy_loss(np, log_prob, old, advantages, mask, 0.2, 0.2, 3.0)

    reference = _vanilla_policy_loss(
        torch.tensor(old),
        torch.tensor(log_prob),
        torch.tensor(advantages),
        torch.tensor(mask),
    )
    assert abs(float(loss) - float(reference[0])) < 1e-5
    assert abs(float(metrics["pg_clipfrac"]) - float(reference[1])) < 1e-5
    assert abs(float(metrics["ppo_kl"]) - float(reference[2])) < 1e-5


def _vanilla_policy_loss(old_log_prob, log_prob, advantages, response_mask):
    """token-mean dual-clip policy loss from compute_policy_loss_vanilla."""
    import torch

    negative_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_kl)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
    upper = torch.maximum(unclipped, clipped)
    lower = torch.minimum(-advantages * 3.0, upper)
    losses = torch.where(advantages < 0, lower, upper)
    token_count = response_mask.sum().clamp_min(1)
    loss = (losses * response_mask).sum() / token_count
    clipfrac = ((clipped > unclipped).float() * response_mask).sum() / token_count
    ppo_kl = ((-negative_kl) * response_mask).sum() / token_count
    return loss, clipfrac, ppo_kl


def test_qwen3_export_inverts_the_scanned_layout():
    rng = np.random.default_rng(0)
    layers, heads, kv, dim, experts = 2, 2, 2, 4, 2
    hidden, intermediate, vocab = 8, 4, 6
    hf = {
        "model.embed_tokens.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
        "model.norm.weight": rng.standard_normal((hidden,)).astype(np.float32),
        "lm_head.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        hf[f"{prefix}.input_layernorm.weight"] = rng.standard_normal((hidden,)).astype(np.float32)
        hf[f"{prefix}.post_attention_layernorm.weight"] = rng.standard_normal((hidden,)).astype(np.float32)
        hf[f"{prefix}.self_attn.q_proj.weight"] = rng.standard_normal((heads * dim, hidden)).astype(np.float32)
        hf[f"{prefix}.self_attn.k_proj.weight"] = rng.standard_normal((kv * dim, hidden)).astype(np.float32)
        hf[f"{prefix}.self_attn.v_proj.weight"] = rng.standard_normal((kv * dim, hidden)).astype(np.float32)
        hf[f"{prefix}.self_attn.o_proj.weight"] = rng.standard_normal((hidden, heads * dim)).astype(np.float32)
        hf[f"{prefix}.self_attn.q_norm.weight"] = rng.standard_normal((dim,)).astype(np.float32)
        hf[f"{prefix}.self_attn.k_norm.weight"] = rng.standard_normal((dim,)).astype(np.float32)
        hf[f"{prefix}.mlp.gate.weight"] = rng.standard_normal((experts, hidden)).astype(np.float32)
        for expert in range(experts):
            name = f"{prefix}.mlp.experts.{expert}"
            hf[f"{name}.gate_proj.weight"] = rng.standard_normal((intermediate, hidden)).astype(np.float32)
            hf[f"{name}.up_proj.weight"] = rng.standard_normal((intermediate, hidden)).astype(np.float32)
            hf[f"{name}.down_proj.weight"] = rng.standard_normal((hidden, intermediate)).astype(np.float32)
    restored = _export.hf_from_stacked(
        _export.stack_qwen3_for_test(hf, layers, heads, kv, dim, experts), heads, kv, dim
    )
    for name, value in hf.items():
        np.testing.assert_allclose(restored[name], value, atol=1e-5)


def _tiny_hf(layers=3, heads=2, kv=2, dim=4, experts=2, hidden=8, inter=4, vocab=6):
    rng = np.random.default_rng(1)
    hf = {
        "model.embed_tokens.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
        "model.norm.weight": rng.standard_normal((hidden,)).astype(np.float32),
        "lm_head.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
    }
    for layer in range(layers):
        p = f"model.layers.{layer}"
        for name, shape in {
            "input_layernorm": (hidden,),
            "post_attention_layernorm": (hidden,),
            "self_attn.q_proj": (heads * dim, hidden),
            "self_attn.k_proj": (kv * dim, hidden),
            "self_attn.v_proj": (kv * dim, hidden),
            "self_attn.o_proj": (hidden, heads * dim),
            "self_attn.q_norm": (dim,),
            "self_attn.k_norm": (dim,),
            "mlp.gate": (experts, hidden),
        }.items():
            hf[f"{p}.{name}.weight"] = rng.standard_normal(shape).astype(np.float32)
        for e in range(experts):
            hf[f"{p}.mlp.experts.{e}.gate_proj.weight"] = rng.standard_normal((inter, hidden)).astype(np.float32)
            hf[f"{p}.mlp.experts.{e}.up_proj.weight"] = rng.standard_normal((inter, hidden)).astype(np.float32)
            hf[f"{p}.mlp.experts.{e}.down_proj.weight"] = rng.standard_normal((hidden, inter)).astype(np.float32)
    return hf


def test_streaming_export_yields_one_layer_at_a_time():
    layers, heads, kv, dim, experts = 3, 2, 2, 4, 2
    hf = _tiny_hf(layers, heads, kv, dim, experts)
    stacked = _export.stack_qwen3_for_test(hf, layers, heads, kv, dim, experts)
    seen = {}
    for name, value in _export.iter_hf_from_stacked(stacked, heads, kv, dim):
        seen[name] = value
    assert set(seen) == set(hf)
    for name, value in hf.items():
        np.testing.assert_allclose(seen[name], value, atol=1e-5)


def test_export_names_the_layout_problem():
    import pytest

    with pytest.raises(KeyError):
        _export.check_layout({"decoder": {"layers": {}}}, 32, 128)


def test_maxtext_batch_pads_and_places_advantages_on_response_positions():
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int32)
    token_mask = np.ones_like(tokens)
    response = np.array([[0, 0, 1, 1]], dtype=np.float32)
    advantages = np.array([[0.5, -0.5]], dtype=np.float32)
    batch = _batch.maxtext_batch(
        tokens, token_mask, response, old_log_probs=advantages, advantages=advantages, batch_size=2, seq_len=6
    )
    assert batch["inputs"].shape == (2, 6)
    assert batch["inputs"][1].sum() == 0
    # Response tokens are positions 2 and 3, so their log-probs sit at 1 and 2.
    assert batch["response_mask"][0].tolist() == [0, 1, 1, 0, 0, 0]
    assert batch["advantages"][0, 1] == np.float32(0.5)
    assert batch["advantages"][0, 2] == np.float32(-0.5)


def test_hf_checkpoint_loads_into_the_scanned_layout(tmp_path):
    import json

    import torch
    from safetensors.torch import save_file

    layers, heads, kv, dim, experts = 3, 2, 2, 4, 2
    hf = _tiny_hf(layers, heads, kv, dim, experts)
    names = sorted(hf)
    half = len(names) // 2
    weight_map = {}
    for shard, chosen in (("a.safetensors", names[:half]), ("b.safetensors", names[half:])):
        save_file({n: torch.from_numpy(hf[n]) for n in chosen}, str(tmp_path / shard))
        weight_map.update({n: shard for n in chosen})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    expected = _export.stack_qwen3_for_test(hf, layers, heads, kv, dim, experts)
    like = _zeros_like(expected)
    loaded = _hf_import.load_stacked_from_hf(_hf_import.HFReader(str(tmp_path)), like, lambda old, host: host)

    def check(a, b):
        if isinstance(a, dict):
            assert set(a) == set(b)
            for key in a:
                check(a[key], b[key])
        else:
            np.testing.assert_allclose(a, b, atol=1e-6)

    check(expected, loaded)


def _zeros_like(tree):
    if isinstance(tree, dict):
        return {key: _zeros_like(value) for key, value in tree.items()}
    return np.zeros_like(tree, dtype=np.float32)


def test_checkpoint_roundtrip_and_prune(tmp_path):
    import json

    import torch

    layers, heads, kv, dim, experts = 3, 2, 2, 4, 2
    hf = _tiny_hf(layers, heads, kv, dim, experts)
    tensors = [(name, torch.from_numpy(value)) for name, value in hf.items()]
    out = tmp_path / "global_step_10" / "actor"
    stats = _ckpt.write_hf_checkpoint(tensors, str(out), shard_bytes=2048)
    assert stats["tensors"] == len(hf)
    assert not (tmp_path / "global_step_10" / "actor.tmp").exists()
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(hf)
    assert len(set(index["weight_map"].values())) > 1

    expected = _export.stack_qwen3_for_test(hf, layers, heads, kv, dim, experts)
    like = _zeros_like(expected)
    loaded = _hf_import.load_stacked_from_hf(_hf_import.HFReader(str(out)), like, lambda old, host: host)
    np.testing.assert_allclose(
        loaded["decoder"]["layers"]["moe_block"]["wi_0"], expected["decoder"]["layers"]["moe_block"]["wi_0"], atol=1e-6
    )

    for step in (20, 30):
        (tmp_path / f"global_step_{step}").mkdir()
    for base, dirs, files in os.walk(out):
        for name in [*dirs, *files]:
            assert not os.stat(os.path.join(base, name)).st_mode & 0o002, "checkpoint must not be world-writable"
    removed = _ckpt.prune_checkpoints(str(tmp_path), current_step=30, keep=2)
    assert [Path(p).name for p in removed] == ["global_step_10"]
    assert (tmp_path / "global_step_20").exists() and (tmp_path / "global_step_30").exists()


def _engine_config(path, **overrides):
    from types import SimpleNamespace

    return SimpleNamespace(config_path=str(path), overrides=overrides, maxtext_path=None, env={}, mem_fraction=0.85)


def test_maxtext_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(_params, "bind_maxtext", lambda config: None)
    cfg = tmp_path / "run.yml"
    cfg.write_text("base_config: base.yml\nper_device_batch_size: 2\n")
    argv = _params.maxtext_argv(_engine_config(cfg, model_name="qwen3-30b-a3b", sparse_matmul=False, ici=[1, 2]))
    assert argv[:2] == ["maxtext", str(cfg)]
    assert "model_name=qwen3-30b-a3b" in argv
    assert "sparse_matmul=false" in argv
    assert "ici=[1, 2]" in argv
    # The eval batch follows the train batch when the config leaves it unset.
    assert "eval_per_device_batch_size=2" in argv

    cfg.write_text("per_device_batch_size: 2\neval_per_device_batch_size: 4\n")
    assert not any(a.startswith("eval_per_device_batch_size") for a in _params.maxtext_argv(_engine_config(cfg)))

    # Keys are inherited through base_config, as MaxText resolves it.
    child = tmp_path / "child.yml"
    child.write_text("base_config: run.yml\nmax_target_length: 64\n")
    values = _params._user_values(str(child), {"steps": 3})
    assert values["per_device_batch_size"] == 2 and values["max_target_length"] == 64 and values["steps"] == 3

    bad = tmp_path / "run.yaml"
    bad.write_text("per_device_batch_size: 1\n")
    try:
        _params.maxtext_argv(_engine_config(bad))
    except ValueError as exc:
        assert ".yml" in str(exc)
    else:
        raise AssertionError("a .yaml config must be rejected")


def test_merge_xla_flags():
    merged = _params._merge_xla_flags(
        "--xla_gpu_autotune_level=0 --keep_me=1", ["--xla_gpu_autotune_level=4", "--new=true"]
    )
    assert merged.split() == ["--keep_me=1", "--xla_gpu_autotune_level=4", "--new=true"]


def test_maxtext_actor_rejects_losses_the_engine_ignores():
    from verl.workers.config.actor import MaxTextActorConfig

    base = {"ppo_mini_batch_size": 8, "ppo_micro_batch_size_per_gpu": 4, "rollout_n": 4}
    MaxTextActorConfig(**base)
    for bad in ({"use_kl_loss": True}, {"entropy_coeff": 0.01}, {"loss_agg_mode": "seq-mean-token-sum"}):
        with pytest.raises(ValueError, match="vanilla token-mean"):
            MaxTextActorConfig(**base, **bad)


def test_checkpoint_device_follows_worker_unless_overridden(monkeypatch):
    import torch

    from verl.checkpoint_engine import nccl_checkpoint_engine as nccl

    monkeypatch.delenv("VERL_CHECKPOINT_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert nccl._checkpoint_device_index() == 3
    monkeypatch.setenv("VERL_CHECKPOINT_DEVICE", "0")
    assert nccl._checkpoint_device_index() == 0
    monkeypatch.delenv("VERL_CHECKPOINT_DEVICE")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert nccl._checkpoint_device_index() == 0
