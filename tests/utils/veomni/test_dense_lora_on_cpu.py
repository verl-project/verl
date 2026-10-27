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
"""Real native LoRA / Qwen3 math; no claim of GPU or rollout execution."""

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

pytest.importorskip("veomni")
from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
from veomni.lora.layers import LoraLinear


def _native_model(dtype=torch.float32):
    torch.manual_seed(17)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    base = Qwen3ForCausalLM(cfg).to(dtype)
    oracle = copy.deepcopy(base)
    model = VeOmniLoraModel(base, VeOmniLoraConfig(r=4, lora_alpha=12, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.lora_B["default"].weight.normal_(std=0.03)
    return model, oracle


def _export(model):
    from verl.utils.veomni.lora import merged_dense_lora_weights

    return dict(merged_dense_lora_weights(model, device="cpu"))


def test_disable_adapter_restores_nested_and_exception_state():
    model, base = _native_model()
    x = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    base.eval()
    adapted = model(x).logits.detach()
    assert not torch.equal(adapted, base(x).logits)
    with pytest.raises(RuntimeError, match="inside"):
        with model.disable_adapter():
            torch.testing.assert_close(model(x).logits, base(x).logits, rtol=0, atol=0)
            with model.disable_adapter():
                torch.testing.assert_close(model(x).logits, base(x).logits, rtol=0, atol=0)
            torch.testing.assert_close(model(x).logits, base(x).logits, rtol=0, atol=0)
            raise RuntimeError("inside")
    torch.testing.assert_close(model(x).logits, adapted, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merged_export_matches_forward_and_preserves_live_parameters(dtype):
    model, receiver = _native_model(dtype)
    model.eval()
    receiver.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    exported = _export(model)
    assert set(exported) == set(receiver.state_dict())
    assert not any("lora_" in k or "base_layer" in k for k in exported)
    receiver.load_state_dict(exported, strict=True)
    x = torch.tensor([[1, 3, 5, 7]])
    torch.testing.assert_close(receiver(x).logits, model(x).logits, rtol=0.025, atol=0.002)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    for k, v in _export(model).items():
        torch.testing.assert_close(v, exported[k], rtol=0, atol=0)


def _step(model, optimizer, x):
    optimizer.zero_grad()
    loss = model(x, labels=x).loss
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_qwen3_training_export_reference_and_resume(tmp_path):
    model, receiver = _native_model()
    x = torch.tensor([[1, 3, 5, 7, 9]])
    base_state = {k: v.clone() for k, v in model.named_parameters() if not v.requires_grad}
    adapter_before = {k: v.clone() for k, v in model.named_parameters() if v.requires_grad}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    _step(model, optimizer, x)
    assert any(not torch.equal(v, dict(model.named_parameters())[k]) for k, v in adapter_before.items())
    for k, v in base_state.items():
        torch.testing.assert_close(dict(model.named_parameters())[k], v, rtol=0, atol=0)
    receiver.load_state_dict(_export(model), strict=True)
    torch.testing.assert_close(receiver(x).logits, model(x).logits, rtol=1e-5, atol=1e-6)
    with model.disable_adapter():
        reference = model(x).logits.detach().clone()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, tmp_path / "resume.pt")
    expected_loss = _step(model, optimizer, x)
    resumed, _ = _native_model()
    resumed_opt = torch.optim.AdamW((p for p in resumed.parameters() if p.requires_grad), lr=0.01)
    saved = torch.load(tmp_path / "resume.pt", weights_only=True)
    resumed.load_state_dict(saved["model"], strict=True)
    resumed_opt.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(_step(resumed, resumed_opt, x), expected_loss, rtol=0, atol=0)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[k], v, rtol=0, atol=0)
    with resumed.disable_adapter():
        torch.testing.assert_close(resumed(x).logits, reference, rtol=0, atol=0)


def test_checkpoint_manager_recognizes_native_adapter():
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    model, _ = _native_model()
    manager = object.__new__(FSDPCheckpointManager)
    manager.model = model
    assert manager._has_lora()
    assert manager._get_lora_train_meta(model) == {"r": 4, "lora_alpha": 12, "task_type": "CAUSAL_LM"}


@pytest.mark.parametrize(("world_size", "init_device", "mode"), [(1, "cuda", "ddp"), (2, "meta", "fsdp2")])
def test_actual_accelerator_config_accepts_single_device_and_fsdp_initialization(
    monkeypatch, world_size, init_device, mode
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from verl.workers.config import VeOmniEngineConfig
    from verl.workers.engine.veomni import transformer_impl as impl

    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setattr(impl.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(impl.dist, "get_world_size", lambda: world_size)
    initialize = MagicMock()
    monkeypatch.setattr(
        impl,
        "parallel_state",
        SimpleNamespace(
            init_parallel_state_from_config=initialize,
            get_parallel_state=lambda: SimpleNamespace(sp_enabled=False),
        ),
    )
    engine = impl.VeOmniEngine(
        model_config=SimpleNamespace(lora_rank=8, lora={"merge": True}, use_remove_padding=True),
        engine_config=VeOmniEngineConfig(init_device=init_device, use_torch_compile=False),
        optimizer_config=None,
        checkpoint_config=None,
    )
    accelerator = initialize.call_args.args[0]
    assert accelerator.fsdp_config.fsdp_mode == mode
    assert accelerator.init_device == init_device
    assert engine.data_parallel_mode == mode


@pytest.mark.parametrize("offload_field", ["param_offload", "enable_fsdp_offload"])
def test_single_device_rejects_fsdp_model_offload_before_mesh_creation(monkeypatch, offload_field):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from verl.workers.config import VeOmniEngineConfig
    from verl.workers.engine.veomni import transformer_impl as impl

    monkeypatch.setattr(impl.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(impl.dist, "get_world_size", lambda: 1)
    initialize = MagicMock()
    monkeypatch.setattr(
        impl,
        "parallel_state",
        SimpleNamespace(
            init_parallel_state_from_config=initialize,
            get_parallel_state=lambda: SimpleNamespace(sp_enabled=False),
        ),
    )
    with pytest.raises(ValueError, match="single-device.*offload"):
        impl.VeOmniEngine(
            model_config=SimpleNamespace(lora_rank=8, lora={"merge": True}, use_remove_padding=True),
            engine_config=VeOmniEngineConfig(init_device="cuda", use_torch_compile=False, **{offload_field: True}),
            optimizer_config=None,
            checkpoint_config=None,
        )
    initialize.assert_not_called()


def _engine_for(model):
    from types import SimpleNamespace

    from verl.workers.config import VeOmniEngineConfig
    from verl.workers.engine.veomni.transformer_impl import VeOmniEngine

    engine = object.__new__(VeOmniEngine)
    engine._is_lora = True
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = True
    engine.engine_config = VeOmniEngineConfig(forward_only=True, attn_implementation="eager")
    engine.model_config = SimpleNamespace(
        local_hf_config_path="/model",
        local_path="/model",
        enable_gradient_checkpointing=False,
        lora_rank=4,
        lora_alpha=12,
        target_modules=["q_proj", "v_proj"],
        target_parameters=None,
        exclude_modules=None,
        lora={"merge": True},
        lora_adapter_path=None,
        use_shm=False,
    )
    engine.module = model
    return engine


def test_engine_injects_native_lora_before_parallelization(monkeypatch):
    from veomni.arguments import OpsImplementationConfig

    from verl.workers.engine.veomni import transformer_impl as impl

    _, base = _native_model()
    engine = _engine_for(base)
    monkeypatch.setattr(impl, "_build_ops_implementation_config", lambda _: OpsImplementationConfig())
    monkeypatch.setattr(impl, "build_foundation_model", lambda **_: base)
    monkeypatch.setattr(impl, "load_safetensors_index", lambda _: {})
    monkeypatch.setattr(impl, "log_gpu_memory_usage", lambda *a, **k: None)
    seen = {}

    def parallelize(module, **kwargs):
        assert isinstance(module, VeOmniLoraModel)
        seen.update(kwargs)
        return module

    monkeypatch.setattr(impl, "build_parallelize_model", parallelize)
    engine._build_model_optimizer()
    assert seen["is_peft_model"] is True
    assert seen["adapter_path"] is None
    assert any(p.requires_grad for p in engine.module.parameters())
    assert all("lora_" in name for name, p in engine.module.named_parameters() if p.requires_grad)


def test_engine_exports_merged_weights_and_disables_reference(monkeypatch):
    from verl.workers.engine.veomni import transformer_impl as impl

    model, receiver = _native_model()
    engine = _engine_for(model)
    monkeypatch.setattr(impl, "get_device_id", lambda: "cpu")
    weights, peft_config = engine.get_per_tensor_param(base_sync_done=True)
    assert peft_config is None
    receiver.load_state_dict(dict(weights), strict=True)
    x = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(receiver(x).logits, model(x).logits, rtol=1e-5, atol=1e-6)
    with engine.disable_adapter():
        assert all(m.disable_adapters for m in model.modules() if isinstance(m, LoraLinear))


def test_engine_rejects_adapter_only_sync_before_build(monkeypatch):
    from verl.workers.engine.veomni import transformer_impl as impl

    _, base = _native_model()
    engine = _engine_for(base)
    engine.model_config.lora = {"merge": False}
    monkeypatch.setattr(impl, "build_foundation_model", lambda **_: pytest.fail("model must not be built"))
    with pytest.raises(ValueError, match="merge"):
        engine._build_model_optimizer()


@pytest.mark.parametrize("adapter_only", [False, True])
def test_native_checkpoint_manager_roundtrip_next_step(monkeypatch, tmp_path, adapter_only):
    from verl.trainer.config import CheckpointConfig
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    model, _ = _native_model()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    x = torch.tensor([[1, 2, 3, 4]])
    _step(model, optimizer, x)
    config = CheckpointConfig(
        save_contents=["model", "optimizer", "extra", "hf_model"],
        load_contents=["model", "optimizer", "extra"],
        save_lora_only=adapter_only,
    )
    manager = FSDPCheckpointManager(
        model=model,
        optimizer=optimizer,
        checkpoint_config=config,
        per_tensor_param_fn=lambda: (iter(_export(model).items()), None),
    )
    path = str(tmp_path / "checkpoint")
    manager.save_checkpoint(path, global_step=1)
    import json

    from safetensors.torch import load_file

    hf_dir = tmp_path / "checkpoint/huggingface"
    index = json.loads((hf_dir / "model.safetensors.index.json").read_text())
    saved = {}
    for shard in set(index["weight_map"].values()):
        saved.update(load_file(str(hf_dir / shard)))
    # A fresh bare model is the receiver; strict keys prove HF normalization.
    _, receiver = _native_model()
    receiver.load_state_dict(saved, strict=True)
    torch.testing.assert_close(receiver(x).logits, model(x).logits, rtol=1e-5, atol=1e-6)
    assert (tmp_path / "checkpoint/lora_train_meta.json").exists()
    expected = _step(model, optimizer, x)
    resumed, _ = _native_model()
    resumed_optimizer = torch.optim.AdamW((p for p in resumed.parameters() if p.requires_grad), lr=0.01)
    restore_manager = FSDPCheckpointManager(
        model=resumed,
        optimizer=resumed_optimizer,
        checkpoint_config=config,
        per_tensor_param_fn=lambda: (iter(_export(resumed).items()), None),
    )
    restore_manager.load_checkpoint(path, del_local_after_load=False)
    torch.testing.assert_close(_step(resumed, resumed_optimizer, x), expected, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], tensor, rtol=0, atol=0)


def _dtensor_export_worker(rank, init_file):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model, _ = _native_model()
        expected = {k: v.clone() for k, v in _export(model).items()}
        mesh = init_device_mesh("cpu", (2,))
        for name, parameter in list(model.named_parameters()):
            parent_name, _, attr = name.rpartition(".")
            sharded = distribute_tensor(parameter.detach(), mesh, [Shard(0)])
            setattr(
                model.get_submodule(parent_name),
                attr,
                torch.nn.Parameter(sharded, requires_grad=parameter.requires_grad),
            )
        for name, tensor in _export(model).items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_real_two_rank_dtensor_export(tmp_path):
    """Exercise actual gather collectives, without claiming FSDP2 GPU coverage."""
    torch.multiprocessing.spawn(_dtensor_export_worker, args=(str(tmp_path / "gloo-init"),), nprocs=2, join=True)


def test_default_all_linear_targets_exclude_lm_head():
    _, base = _native_model()
    engine = _engine_for(base)
    engine.model_config.target_modules = "all-linear"
    wrapped, _ = engine._build_native_lora_module(base)
    targets = [name for name, m in wrapped.named_modules() if isinstance(m, LoraLinear)]
    assert len(targets) == 14  # seven projections in each of two decoder layers
    assert not any("lm_head" in name for name in targets)


def test_materialized_model_loads_pretrained_nonzero_adapter(tmp_path):
    model, base = _native_model()
    adapter_path = tmp_path / "adapter"
    model.save_pretrained(str(adapter_path))
    engine = _engine_for(base)
    engine.model_config.lora_adapter_path = str(adapter_path)
    engine.model_config.use_shm = False
    loaded, _ = engine._build_native_lora_module(base)
    x = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    loaded.eval()
    torch.testing.assert_close(loaded(x).logits, model(x).logits, rtol=0, atol=0)


def test_exported_tensors_do_not_alias_trainable_model_storage():
    model, _ = _native_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    exported = _export(model)
    exported["model.embed_tokens.weight"].zero_()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_actual_veomni_qwen3_meta_loader_and_ppo_logprob_gradients(tmp_path, monkeypatch):
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model
    from veomni.models.checkpoint.weights import load_model_weights
    from veomni.models.loss_utils import chunk_logprobs

    # Exercise the real torch chunk-logprob fallback on CPU even when the
    # host also has FlashAttention installed (its CE kernel is CUDA-only).
    monkeypatch.setattr(chunk_logprobs, "_FA_CE_AVAILABLE", False)

    _, oracle = _native_model()
    oracle.config.architectures = ["Qwen3ForCausalLM"]
    oracle.save_pretrained(str(tmp_path / "base"))
    ops = OpsImplementationConfig(attn_implementation="eager", moe_implementation="eager")
    base = build_foundation_model(
        config_path=str(tmp_path / "base"),
        init_device="meta",
        torch_dtype="float32",
        ops_implementation=ops,
    )
    assert "veomni.models" in type(base).__module__
    engine = _engine_for(base)
    engine.model_config.target_modules = "all-linear"
    model, _ = engine._build_native_lora_module(base)
    load_model_weights(model, str(tmp_path / "base"), init_device="cpu", is_peft_model=True)
    assert not any(p.is_meta for p in model.parameters())
    x = torch.tensor([[1, 2, 3, 4]])
    # B=0 after native loader initialization: patched model equals bare HF model.
    torch.testing.assert_close(model(x).logits, oracle(x).logits, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, LoraLinear):
                layer.lora_B["default"].weight.normal_(std=0.03)
    # Actual patched VeOmni PPO chunk-logprob path, including its backward.
    output = model(x, labels=x, return_log_probs=True, temperature=0.7)
    aux = output.fused_linear_aux
    assert aux is not None
    loss = -(aux.log_probs.mean() + 0.01 * aux.entropy.mean())
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    engine.module = model
    oracle.load_state_dict(_export(model), strict=True)
    torch.testing.assert_close(model(x).logits, oracle(x).logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tied", [False, True])
def test_engine_rejects_lm_head_adapter_targets(tied):
    _, base = _native_model()
    base.config.tie_word_embeddings = tied
    if tied:
        base.tie_weights()
    engine = _engine_for(base)
    engine.model_config.target_modules = ["lm_head"]
    with pytest.raises(ValueError, match="lm_head|output embedding"):
        engine._build_native_lora_module(base)


def test_rejects_non_lora_type_instead_of_silently_training_lora():
    _, base = _native_model()
    engine = _engine_for(base)
    engine.model_config.lora["type"] = "dora"
    with pytest.raises(ValueError, match="type|DoRA"):
        engine._build_native_lora_module(base)


def test_rejects_nested_rank_instead_of_silently_full_finetuning(monkeypatch):
    from types import SimpleNamespace

    from verl.workers.config import VeOmniEngineConfig
    from verl.workers.engine.veomni import transformer_impl as impl

    model_config = SimpleNamespace(lora_rank=0, lora={"rank": 8, "merge": True})
    # Invalid configuration should fail before distributed model setup.
    monkeypatch.setattr(impl.dist, "get_rank", lambda: pytest.fail("configuration must be checked first"))
    with pytest.raises(ValueError, match="lora_rank"):
        impl.VeOmniEngine(model_config, VeOmniEngineConfig(), None, None)


@pytest.mark.parametrize(
    ("feature", "value"),
    [("alora_invocation_tokens", [1, 2]), ("use_qalora", True), ("lora_bias", True), ("megatron_config", {"x": 1})],
)
def test_rejects_pretrained_adapter_features_that_cannot_be_merged(tmp_path, feature, value):
    import json

    model, base = _native_model()
    model.get_lora_config().save_pretrained(str(tmp_path))
    config_path = tmp_path / "adapter_config.json"
    settings = json.loads(config_path.read_text())
    settings[feature] = value
    config_path.write_text(json.dumps(settings))
    engine = _engine_for(base)
    engine.model_config.lora_adapter_path = str(tmp_path)
    with pytest.raises(ValueError, match="unsupported LoRA features"):
        engine._build_native_lora_module(base)
