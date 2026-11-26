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

"""GPU regression tests for MegatronCheckpointManager model-only resume (#8604).

Verifies with a real Megatron-Core BF16 GPTModel + MixedPrecision/Distributed
optimizer that resuming with ``load_contents`` excluding ``"optimizer"``
refreshes the optimizer's FP32 master weights, so the first ``optimizer.step()``
advances from the loaded checkpoint weights instead of silently reverting to
the initial weights.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as McoreDDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from verl.trainer.config import CheckpointConfig
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")

SEQUENCE_LENGTH = 32
VOCAB_SIZE = 128


class _MegatronSub:
    dist_ckpt_optim_fully_reshardable = False
    distrib_optim_fully_reshardable_mem_efficient = False


class _RoleCfg:
    megatron = _MegatronSub()


class _MinimalModel:
    path = "/tmp/fake_model"


class _MinimalConfig:
    model = _MinimalModel()
    actor = _RoleCfg()


@contextmanager
def _megatron_single_gpu_env(tmp_path):
    if dist.is_initialized() and "nccl" not in dist.get_backend():
        pytest.skip("Requires an NCCL-capable process group; run GPU tests separately from *_on_cpu.py tests.")
    torch.cuda.set_device(0)
    created_pg = False
    created_mpu = False
    if not dist.is_initialized():
        rendezvous_file = tmp_path / "rdzv_ckpt_reload"
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            init_method=f"file://{rendezvous_file}",
            rank=0,
            world_size=1,
        )
        created_pg = True
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel()
        created_mpu = True
    model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        if created_mpu and mpu.model_parallel_is_initialized():
            mpu.destroy_model_parallel()
        if created_pg and dist.is_initialized():
            dist.destroy_process_group()


def _build_model_and_optimizer(
    *,
    use_distributed_optimizer: bool,
    lr: float = 1e-2,
    seed: int = 42,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    gpt_model = (
        GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=VOCAB_SIZE,
            max_sequence_length=SEQUENCE_LENGTH,
        )
        .cuda()
        .bfloat16()
    )
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=use_distributed_optimizer)
    model_chunk = McoreDDP(transformer_config, ddp_config, gpt_model)
    model_chunks = [model_chunk]

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=lr,
        min_lr=lr,
        clip_grad=1.0,
        weight_decay=0.0,
        use_distributed_optimizer=use_distributed_optimizer,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    optimizer = get_megatron_optimizer(optimizer_config, model_chunks)
    return transformer_config, model_chunks, optimizer


def _make_manager(
    *,
    transformer_config,
    model_chunks,
    optimizer,
    save_contents,
    load_contents,
    use_distributed_optimizer: bool,
    use_dist_checkpointing: bool,
    bridge=None,
):
    ckpt_config = CheckpointConfig(
        save_contents=list(save_contents),
        load_contents=list(load_contents),
        async_save=False,
    )
    return MegatronCheckpointManager(
        config=_MinimalConfig(),
        checkpoint_config=ckpt_config,
        model_config=MagicMock(),
        transformer_config=transformer_config,
        role="actor",
        model=model_chunks,
        arch="GPTForCausalLM",
        hf_config=MagicMock(),
        param_dtype=torch.bfloat16,
        share_embeddings_and_output_weights=False,
        processing_class=None,
        optimizer=optimizer,
        optimizer_scheduler=None,
        use_distributed_optimizer=use_distributed_optimizer,
        use_dist_checkpointing=use_dist_checkpointing,
        bridge=bridge,
    )


def _snapshot_params(model_chunks) -> dict[str, torch.Tensor]:
    return {name: p.detach().float().cpu().clone() for name, p in model_chunks[0].named_parameters() if p.requires_grad}


def _run_one_step(model_chunks, optimizer) -> None:
    optimizer.zero_grad()
    for chunk in model_chunks:
        chunk.zero_grad_buffer()

    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH), device="cuda")
    position_ids = torch.arange(SEQUENCE_LENGTH, device="cuda").unsqueeze(0).expand(2, -1)
    # Megatron boolean attention mask uses True to mask out positions; strict upper triangle = causal mask.
    attention_mask = torch.triu(
        torch.ones((1, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    output = model_chunks[0](input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
    loss = output.float().pow(2).mean()
    loss.backward()

    update_successful, _, _ = optimizer.step()
    assert update_successful


def _assert_step_continued_from_checkpoint(
    init_weights: dict[str, torch.Tensor],
    ckpt_weights: dict[str, torch.Tensor],
    after_weights: dict[str, torch.Tensor],
    *,
    shift: float,
    lr: float,
) -> None:
    moved_params = 0
    for name in init_weights:
        dist_to_ckpt = (after_weights[name] - ckpt_weights[name]).abs().max().item()
        dist_to_init = (after_weights[name] - init_weights[name]).abs().max().item()
        # With shift=2.0 and lr=1e-2, one Adam step changes each param by ~lr (~0.01)
        # while a stale-master-weight overwrite snaps weights back by ~shift (~2.0).
        assert dist_to_ckpt < 10 * lr, f"{name} drifted too far from checkpoint weights: {dist_to_ckpt}"
        assert dist_to_init > shift * 0.5, f"{name} snapped back toward initial weights: {dist_to_init}"
        if dist_to_ckpt > 0.0:
            moved_params += 1
    assert moved_params > 0, "optimizer.step() did not update any model parameters"


@pytest.mark.parametrize(
    ("use_distributed_optimizer", "with_offload"),
    [
        (True, False),
        (True, True),
        (False, False),
    ],
)
def test_dist_ckpt_model_only_resume_preserves_loaded_weights_after_step(
    tmp_path, use_distributed_optimizer, with_offload
):
    """Real Megatron dist_ckpt save -> model-only resume -> 1 step must continue from checkpoint weights."""
    shift = 2.0
    lr = 1e-2
    ckpt_dir = str(tmp_path / "global_step_1")

    with _megatron_single_gpu_env(tmp_path):
        # 1. Build model A, shift weights away from initial values, save model-only dist_ckpt.
        tf_cfg_a, model_a, opt_a = _build_model_and_optimizer(
            use_distributed_optimizer=use_distributed_optimizer, lr=lr, seed=42
        )
        init_weights = _snapshot_params(model_a)
        with torch.no_grad():
            for p in model_a[0].parameters():
                if p.requires_grad:
                    p.add_(shift)
        ckpt_weights = _snapshot_params(model_a)

        save_mgr = _make_manager(
            transformer_config=tf_cfg_a,
            model_chunks=model_a,
            optimizer=opt_a,
            save_contents=["model"],
            load_contents=["model"],
            use_distributed_optimizer=use_distributed_optimizer,
            use_dist_checkpointing=True,
        )
        save_mgr.save_checkpoint(ckpt_dir, global_step=1)
        del save_mgr, opt_a, model_a

        # 2. Fresh model B with identical initial weights (so optimizer FP32 master weights hold W_init).
        tf_cfg_b, model_b, opt_b = _build_model_and_optimizer(
            use_distributed_optimizer=use_distributed_optimizer, lr=lr, seed=42
        )
        for name, p in model_b[0].named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.detach().float().cpu(), init_weights[name], rtol=0, atol=0)

        load_mgr = _make_manager(
            transformer_config=tf_cfg_b,
            model_chunks=model_b,
            optimizer=opt_b,
            save_contents=["model"],
            load_contents=["model"],
            use_distributed_optimizer=use_distributed_optimizer,
            use_dist_checkpointing=True,
        )

        # Mirror MegatronEngine.load_checkpoint() offload/onload lifecycle when enabled.
        if with_offload:
            offload_megatron_model_to_cpu(model_b)
            offload_megatron_optimizer(opt_b)
            load_megatron_model_to_gpu(model_b)
            load_megatron_optimizer(opt_b)

        load_mgr.load_checkpoint(ckpt_dir)

        if with_offload:
            offload_megatron_model_to_cpu(model_b)
            offload_megatron_optimizer(opt_b)
            load_megatron_model_to_gpu(model_b)
            load_megatron_optimizer(opt_b)

        # 3. Run one real forward/backward/optimizer.step() and verify weights continue from W_ckpt.
        _run_one_step(model_b, opt_b)
        after_weights = _snapshot_params(model_b)
        _assert_step_continued_from_checkpoint(init_weights, ckpt_weights, after_weights, shift=shift, lr=lr)


def test_hf_bridge_model_only_resume_preserves_loaded_weights_after_step(tmp_path):
    """Bridge HF model-only resume -> 1 step with real DistributedOptimizer must continue from loaded weights."""
    shift = 2.0
    lr = 1e-2
    ckpt_dir = tmp_path / "global_step_1"
    ckpt_dir.mkdir()

    with _megatron_single_gpu_env(tmp_path):
        tf_cfg, model_chunks, optimizer = _build_model_and_optimizer(use_distributed_optimizer=True, lr=lr, seed=42)
        init_weights = _snapshot_params(model_chunks)
        ckpt_weights = {name: (w + shift).to(torch.bfloat16).float() for name, w in init_weights.items()}

        bridge = MagicMock()

        def _fake_load_hf_weights(models, _hf_model_path):
            with torch.no_grad():
                for name, p in models[0].named_parameters():
                    if p.requires_grad:
                        p.copy_(ckpt_weights[name].to(device=p.device, dtype=p.dtype))

        bridge.load_hf_weights.side_effect = _fake_load_hf_weights

        load_mgr = _make_manager(
            transformer_config=tf_cfg,
            model_chunks=model_chunks,
            optimizer=optimizer,
            save_contents=["model"],
            load_contents=["model"],
            use_distributed_optimizer=True,
            use_dist_checkpointing=False,
            bridge=bridge,
        )
        load_mgr.load_checkpoint(str(ckpt_dir))
        bridge.load_hf_weights.assert_called_once()

        _run_one_step(model_chunks, optimizer)
        after_weights = _snapshot_params(model_chunks)
        _assert_step_continued_from_checkpoint(init_weights, ckpt_weights, after_weights, shift=shift, lr=lr)
