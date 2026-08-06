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
PP = int(os.environ.get("PP_SIZE", "1"))
EP = int(os.environ.get("EP_SIZE", "1"))


def _build_tiny_hf_dir(rank: int) -> str:
    from transformers import AutoConfig

    cfg_dir = f"/tmp/tiny_{MODEL_KIND}_probe_diff_v4"
    if MODEL_KIND in ("qwen2", "qwen2_tied"):
        hf_config = AutoConfig.for_model(
            "qwen2",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=(MODEL_KIND == "qwen2_tied"),
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
        if MODEL_KIND in ("nemotron_h", "falcon_h1"):
            # newer transformers' serialization rename table emits
            # "backbone.embedding.weight" (singular) while the bridge mappings
            # (and the RELEASED checkpoints) use "backbone.embeddings.weight".
            # Normalize the stray key so the tiny checkpoint matches the
            # released format the bridge expects.
            from safetensors.torch import load_file, save_file

            f = str(pathlib.Path(cfg_dir, "model.safetensors"))
            sd = load_file(f)
            fixed = {k.replace("backbone.embedding.", "backbone.embeddings."): v for k, v in sd.items()}
            if fixed.keys() != sd.keys():
                save_file(fixed, f, metadata={"format": "pt"})
    dist.barrier()
    return cfg_dir


def _pp_differential(bridge, model, rank: int, world: int) -> None:
    """PP>1 differential: run the PRODUCTION export path (global directory +
    slot pre-seed + probe entries, a full-delta sync where every local element
    is 'changed') and compare the merged wire-view against
    ``bridge.export_hf_weights`` -- the bridge's own PP-broadcast full export,
    an independent chain. Bitwise, per HF tensor, with directory-lockstep and
    placeholder-row checks on the way."""
    from verl.workers.engine.megatron.delta_export import build_export_index, mcore_hf_delta_entry

    slot_cache: dict = {}
    index = build_export_index(bridge, model, slot_cache)

    # 1. directory lockstep: same ROW COUNT everywhere (mcore expert param
    # NAMES legitimately differ per ep rank -- the row is the unit).
    names = [r.megatron_name for r in index]
    all_names: list = [None] * world
    dist.all_gather_object(all_names, names)
    failures = []
    if rank == 0 and any(len(n) != len(names) for n in all_names):
        failures.append(f"directory row counts diverge: {[len(n) for n in all_names]}")

    # 1b. per-row slot-table identity: the batched gather merges BY SLOT
    # POSITION within a row and rank 0 names merged pieces from its own list,
    # so the row's table must be identical on every rank (the ep_size>1
    # collision this oracle now guards).
    row_tables = [r.slots for r in index]
    all_tables: list = [None] * world
    dist.all_gather_object(all_tables, row_tables)
    if rank == 0:
        for r_i, tbls in enumerate(all_tables):
            bad = [k for k in range(len(row_tables)) if tbls[k] != row_tables[k]][:3]
            if bad:
                failures.append(f"row slot tables diverge on rank {r_i} (rows {bad})")
                break

    # 2. entries: owned params ship a full delta (every local element), rows
    # owned by other stages must come back as zero-count lockstep entries.
    mine: dict = {}  # slot name -> (shape, positions int64 cpu, values bf16 cpu)
    for rec in index:
        if rec.param is None:
            e_idx = torch.empty(0, dtype=torch.int64, device="cuda")
            e_val = torch.empty(0, dtype=torch.bfloat16, device="cuda")
            slots, _dt, counts, hf_idx, _hf_val = mcore_hf_delta_entry(rec, 0, e_idx, e_val, slot_cache)
            if int(counts.sum()) != 0 or hf_idx.numel() != 0:
                failures.append(f"{rec.megatron_name}: placeholder row shipped a non-empty delta")
            continue
        if not rec.param.is_floating_point():
            continue
        local = rec.param.data.to(torch.bfloat16).reshape(-1)
        lidx = torch.arange(local.numel(), dtype=torch.int64, device=local.device)
        slots, _dt, counts, hf_idx, hf_val = mcore_hf_delta_entry(rec, 0, lidx, local, slot_cache)
        off = 0
        for (sname, sshape), c in zip(slots, counts.tolist(), strict=True):
            if c:
                pos = hf_idx[off : off + c].to(torch.int64).cpu()
                val = hf_val[off : off + c].cpu()
                off += c
                prev = mine.get(sname)
                if prev is None:
                    mine[sname] = (tuple(sshape), pos, val)
                else:
                    mine[sname] = (tuple(sshape), torch.cat([prev[1], pos]), torch.cat([prev[2], val]))

    # 3. independent reference: the bridge's own full export (PP broadcasts
    # inside; full tensors on every rank).
    exported = bridge.export_hf_weights(model, cpu=True, show_progress=False)
    ref = {name: t.detach().to(torch.bfloat16).cpu() for name, t in exported}

    all_pieces: list = [None] * world
    dist.all_gather_object(all_pieces, mine)
    n_checked = n_pass = 0
    if rank == 0:
        slot_names = {k for d in all_pieces for k in d}
        missing_ref = sorted(set(ref) - slot_names)
        if missing_ref:
            more = max(0, len(missing_ref) - 5)
            failures.append(f"HF tensors never covered by any entry: {missing_ref[:5]} (+{more})")
        for sname in sorted(slot_names):
            if sname not in ref:
                failures.append(f"entry slot {sname} absent from bridge export")
                continue
            full = ref[sname]
            merged = torch.full_like(full, float("nan"))
            covered = torch.zeros(full.numel(), dtype=torch.int32)
            mflat = merged.view(-1)
            for d in all_pieces:
                got = d.get(sname)
                if got is None:
                    continue
                sshape, pos, val = got
                if tuple(sshape) != tuple(full.shape):
                    failures.append(f"{sname}: slot shape {sshape} != export shape {tuple(full.shape)}")
                    continue
                both = covered[pos] > 0
                if both.any() and not torch.equal(mflat[pos[both]].view(torch.int16), val[both].view(torch.int16)):
                    failures.append(f"{sname}: conflicting rank contributions")
                mflat[pos] = val
                covered[pos] += 1
            n_checked += 1
            if torch.isnan(mflat).any():
                failures.append(f"{sname}: uncovered positions remain NaN")
            elif not torch.equal(mflat.view(torch.int16), full.view(-1).view(torch.int16)):
                failures.append(f"{sname}: bitwise mismatch")
            else:
                n_pass += 1
        print("=" * 60)
        for f in failures[:20]:
            print("FAIL:", f)
        print(f"PROBE DIFFERENTIAL [{MODEL_KIND} tp={TP} pp={PP}]: {n_pass}/{n_checked} bitwise-equal")
        print("PROBE DIFFERENTIAL PASSED" if (n_checked > 0 and not failures) else "PROBE DIFFERENTIAL FAILED")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and (failures or n_checked == 0):
        sys.exit(1)


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    assert world >= 2, "differential needs TP>1 to be meaningful"

    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1 if EP > 1 else None,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)

    cfg_dir = _build_tiny_hf_dir(rank)

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(cfg_dir, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = TP
    provider.pipeline_model_parallel_size = PP
    if EP > 1:
        provider.expert_model_parallel_size = EP
        provider.expert_tensor_parallel_size = 1
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    model = [m.cuda().to(torch.bfloat16) for m in (model if isinstance(model, list) else [model])]
    # real HF weights, identically loaded on every rank: replicated params
    # (layernorms, routers) must be bit-identical across TP ranks or the
    # differential would flag test artifacts instead of probe bugs.
    bridge.load_hf_weights(model)

    if PP > 1 or EP > 1 or os.environ.get("ENGINE_DIFF"):
        _pp_differential(bridge, model, rank, world)
        return

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
