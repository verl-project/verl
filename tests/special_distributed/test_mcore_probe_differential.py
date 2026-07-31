# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""TP>1 differential oracle for the comm-stubbed mcore probe.

For every conversion task of a tiny model the REAL ``megatron_to_hf`` (true
collectives, run in lockstep by all ranks) must equal the probe assembly:
each rank runs its comm-stubbed probe on its own local shard, the non-NaN
contributions are merged across ranks, and the merged tensor is compared to
the real output BITWISE. This is the regression oracle for the probe's two
assumptions (communication confined to the stubbed helpers; transforms
rearrange rather than blend) -- rerun it whenever Megatron-Bridge is upgraded.

Run under torchrun on >=2 GPUs, e.g.::

    MODEL_KIND=qwen2 torchrun --nproc_per_node=2 tests/special_distributed/test_mcore_probe_differential.py
    MODEL_KIND=nemotron_h torchrun --nproc_per_node=2 tests/special_distributed/test_mcore_probe_differential.py

``MODEL_KIND``: ``qwen2`` (Column/Row/QKV/GatedMLP/Replicated), ``qwen3_moe``
(adds fused-MoE expert mappings; uses etp=TP), ``nemotron_h`` (Mamba mixers --
the tp_size-dependent de-interleave the probe must reproduce).
"""

import os
import pathlib
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ.get("VERL_PATH", "/home/changyi/verl_ab_pre"))

MODEL_KIND = os.environ.get("MODEL_KIND", "qwen2")
TP = int(os.environ.get("TP_SIZE", "2"))


def _build_tiny_hf_dir(rank: int) -> str:
    from transformers import AutoConfig

    cfg_dir = f"/tmp/tiny_{MODEL_KIND}_probe_diff_v2"
    if MODEL_KIND == "qwen2":
        hf_config = AutoConfig.for_model(
            "qwen2",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["Qwen2ForCausalLM"]
    elif MODEL_KIND == "qwen3_moe":
        hf_config = AutoConfig.for_model(
            "qwen3_moe",
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["Qwen3MoeForCausalLM"]
    elif MODEL_KIND == "nemotron_h":
        # hybrid mamba2+attention+mlp stack; sizes divisible by TP=2
        # (n_groups and mamba_num_heads both TP-divisible).
        hf_config = AutoConfig.for_model(
            "nemotron_h",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            hybrid_override_pattern="M*M-",
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_head_dim=16,
            ssm_state_size=16,
            conv_kernel=4,
            expand=2,
            mamba_num_heads=8,
            mamba_head_dim=16,
            n_groups=2,
            chunk_size=32,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["NemotronHForCausalLM"]
    elif MODEL_KIND == "falcon_h1":
        # mamba2+attention parallel-hybrid; mamba_n_heads/n_groups TP=2 divisible
        hf_config = AutoConfig.for_model(
            "falcon_h1",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            mamba_d_ssm=64,
            mamba_n_heads=8,
            mamba_d_head=8,
            mamba_n_groups=2,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_chunk_size=32,
            mamba_norm_before_gate=False,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["FalconH1ForCausalLM"]
    else:
        raise ValueError(f"unknown MODEL_KIND {MODEL_KIND!r}")

    if MODEL_KIND in ("nemotron_h", "falcon_h1"):
        # NemotronH's mixer pulls causal-conv1d / mamba-ssm kernels from the hub
        # at layer init; offline runs must fall back to the reference path.
        try:
            import transformers.integrations.hub_kernels as _hk

            _hk.lazy_load_kernel = lambda *a, **k: None
            import importlib

            _mnh = importlib.import_module(f"transformers.models.{MODEL_KIND}.modeling_{MODEL_KIND}")

            for _sym in ("lazy_load_kernel", "get_kernel"):
                if hasattr(_mnh, _sym):
                    setattr(_mnh, _sym, lambda *a, **k: None)
        except Exception as e:  # pragma: no cover
            print(f"kernel stub patch failed: {e}")

    if rank == 0 and not pathlib.Path(cfg_dir, "model.safetensors").exists():
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_config(hf_config).to(torch.bfloat16)
        m.save_pretrained(cfg_dir, safe_serialization=True)
        del m
    dist.barrier()
    return cfg_dir


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    assert world >= 2, "differential needs TP>1 to be meaningful"

    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(tensor_model_parallel_size=TP, pipeline_model_parallel_size=1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)

    cfg_dir = _build_tiny_hf_dir(rank)

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(cfg_dir, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = TP
    provider.pipeline_model_parallel_size = 1
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    model = [m.cuda().to(torch.bfloat16) for m in (model if isinstance(model, list) else [model])]
    # real HF weights, identically loaded on every rank: replicated params
    # (layernorms, routers) must be bit-identical across TP ranks or the
    # differential would flag test artifacts instead of probe bugs.
    bridge.load_hf_weights(model)

    from verl.workers.engine.megatron.delta_export import make_probe

    tasks = bridge.get_conversion_tasks(model)
    n_checked = n_pass = 0
    failures = []
    for t in tasks:
        p = t.param_weight
        if p is None or not p.is_floating_point():
            continue
        local = p.data.to(torch.bfloat16)

        # REAL path: true collectives, all ranks in lockstep; full tensors everywhere.
        real = {k: v.detach().clone() for k, v in t.mapping.megatron_to_hf(local, t.megatron_module).items()}

        # PROBE path: comm-stubbed, purely local. Feed the real local shard so
        # the surviving (non-NaN) elements carry real values.
        probe = make_probe(t.mapping, t.megatron_module)
        mine = {k: v.detach().to("cpu") for k, v in probe.megatron_to_hf(local.clone(), t.megatron_module).items()}

        # Assemble: union of every rank's non-NaN contributions.
        all_outs: list = [None] * world
        dist.all_gather_object(all_outs, mine)
        if rank == 0:
            names = {k for d in all_outs for k in d}
            for name in sorted(names):
                ref = real.get(name)
                if ref is None:
                    failures.append(f"{t.global_param_name}: probe slot {name} absent from real output")
                    continue
                merged = torch.full_like(ref.cpu(), float("nan"))
                covered = torch.zeros(ref.shape, dtype=torch.int32)
                for d in all_outs:
                    if name not in d:
                        continue
                    part = d[name]
                    mask = ~torch.isnan(part)
                    # replicated params legitimately arrive from every rank --
                    # overlap only counts as failure when the values CONFLICT.
                    both = mask & (covered > 0)
                    if both.any() and not torch.equal(merged[both].view(torch.int16), part[both].view(torch.int16)):
                        failures.append(f"{t.global_param_name}/{name}: conflicting rank contributions")
                    merged[mask] = part[mask]
                    covered[mask] += 1
                n_checked += 1
                if torch.isnan(merged).any():
                    failures.append(f"{t.global_param_name}/{name}: uncovered positions remain NaN")
                elif not torch.equal(merged.view(torch.int16), ref.cpu().view(torch.int16)):
                    failures.append(f"{t.global_param_name}/{name}: bitwise mismatch")
                else:
                    n_pass += 1
        dist.barrier()

    if rank == 0:
        print("=" * 60)
        for f in failures[:20]:
            print("FAIL:", f)
        print(f"PROBE DIFFERENTIAL [{MODEL_KIND} tp={TP}]: {n_pass}/{n_checked} bitwise-equal")
        print("PROBE DIFFERENTIAL PASSED" if (n_checked > 0 and not failures) else "PROBE DIFFERENTIAL FAILED")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and (failures or n_checked == 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
