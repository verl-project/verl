"""Scheduled GPU regression of real grouped save/load and full-Adam resume.

This small-module test isolates the checkpoint compatibility contract. It is
not a substitute for the unchanged eight-node full-model regression.
"""

import pytest
import torch
import torch.distributed as dist


@pytest.fixture
def model_parallel(tmp_path):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    assert torch.cuda.is_available()
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("precision", ["bf16", "nvfp4", "fp8_delayed"])
def test_grouped_checkpoint_full_adam_resume(model_parallel, tmp_path, precision, monkeypatch):
    import transformer_engine.pytorch as te
    from megatron.core import dist_checkpointing
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformer_engine.common.recipe import DelayedScaling, NVFP4BlockScaling

    recipe = (
        NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
            disable_2d_quantization=True,
            row_scaled_activation=True,
            backward_override="dequantized",
            nvfp4_4over6="none",
        )
        if precision == "nvfp4"
        else DelayedScaling()
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_grouped_gemm=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
    )

    def build():
        return TEColumnParallelGroupedLinear(
            2,
            256,
            128,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=True,
        )

    torch.manual_seed(1234)
    layer = build()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001, foreach=False)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)

    def step(module, opt):
        opt.zero_grad(set_to_none=True)
        module.is_first_microbatch = True
        with te.autocast(enabled=precision != "bf16", recipe=recipe):
            y, _ = module(x, [64, 64])
            loss = y.float().square().mean()
            loss.backward()
        assert torch.isfinite(loss)
        opt.step()
        return y.detach()

    step(layer, optimizer)
    extra = layer.get_extra_state()
    assert extra.dtype == torch.uint8
    assert (extra.numel() == 0) == (precision != "fp8_delayed")
    if precision != "bf16":
        assert layer.fp8 or layer.fp8_meta["fp8_checkpoint"]
    print("CHECKPOINT_EXTRA_STATE", precision, extra.numel(), flush=True)
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    metadata = {"dp_cp_group": dist.group.WORLD}
    state = layer.sharded_state_dict(metadata=metadata)
    dist_checkpointing.save(state, str(checkpoint))
    torch.save(optimizer.state_dict(), tmp_path / "adam.pt")

    restored = build()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001, foreach=False)
    loaded = dist_checkpointing.load(restored.sharded_state_dict(metadata=metadata), str(checkpoint))
    if precision == "fp8_delayed":
        # TE protects legacy stateful FP8 pickle payloads by default. Only this
        # freshly generated fixture is trusted; never opt in in the runtime.
        with monkeypatch.context() as scoped:
            scoped.delenv("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", raising=False)
            with pytest.raises(RuntimeError, match="Refusing to load pickled"):
                restored.load_state_dict(loaded, strict=True)
            scoped.setenv("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
            restored.load_state_dict(loaded, strict=True)
    else:
        restored.load_state_dict(loaded, strict=True)
    restored_optimizer.load_state_dict(torch.load(tmp_path / "adam.pt", weights_only=True))
    for original, resumed in zip(layer.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original, resumed, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[original][key], restored_optimizer.state[resumed][key], rtol=0, atol=0
            )
    # Continuing from restored model + optimizer must match uninterrupted training.
    expected = step(layer, optimizer)
    actual = step(restored, restored_optimizer)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for original, resumed in zip(layer.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original, resumed, rtol=0, atol=0)
    print("GROUPED_FULL_ADAM_RESUME_PASS", precision, flush=True)
