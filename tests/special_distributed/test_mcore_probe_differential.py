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

Two oracles run here. The TASK differential compares one mapping's probe output
against its real ``megatron_to_hf``; the ENGINE differential compares the whole
production export path (directory + slot tables + delta entries) against
``bridge.export_hf_weights``. Only the second one sees how entries from
different tasks AGGREGATE, so every MoE kind runs it unconditionally: a mapping
can be per-task correct while the rows collide downstream (Megatron-Bridge's
``is_grouped_export`` protocol collapses many per-expert tasks onto ONE shared
HF name, which no per-task comparison can catch).

Run under torchrun on >=2 GPUs, e.g.::

    MODEL_KIND=qwen2 torchrun --nproc_per_node=2 tests/special_distributed/test_mcore_probe_differential.py
    MODEL_KIND=nemotron_h torchrun --nproc_per_node=2 tests/special_distributed/test_mcore_probe_differential.py
    MODEL_KIND=glm4_moe_fused EP_SIZE=4 ETP_SIZE=2 TP_SIZE=1 torchrun --nproc_per_node=8 ...
    MODEL_KIND=glm4_moe TP_SIZE=1 ETP_SIZE=2 torchrun --nproc_per_node=2 ...

``MODEL_KIND``: ``qwen2`` (Column/Row/QKV/GatedMLP/Replicated), ``qwen3_moe``
(per-expert HF names), ``glm4_moe`` / ``glm4_moe_fused`` (same weights saved
per-expert vs stacked -- the stacked form is what flips the bridge onto its
``is_grouped_export`` mappings), ``nemotron_h`` (Mamba mixers -- the
tp_size-dependent de-interleave the probe must reproduce).

``ETP_SIZE`` sets expert-tensor parallelism independently of ``TP_SIZE``;
``TP_SIZE=1 ETP_SIZE=2`` is its own routing path (a param that is neither
tp-sharded nor ep-sharded yet still split across ranks).
"""

import os
import pathlib
import re
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ.get("VERL_PATH", "/home/changyi/verl_ab_pre"))

MODEL_KIND = os.environ.get("MODEL_KIND", "qwen2")
TP = int(os.environ.get("TP_SIZE", "2"))
PP = int(os.environ.get("PP_SIZE", "1"))
EP = int(os.environ.get("EP_SIZE", "1"))
# Previously pinned: 1 under EP, otherwise implicitly TP. Both defaults are kept
# so existing invocations are unchanged, but ETP is now addressable on its own.
ETP = int(os.environ.get("ETP_SIZE", "1" if EP > 1 else str(TP)))
MOE_KINDS = ("qwen3_moe", "glm4_moe", "glm4_moe_fused")


_EXPERT_KEY = re.compile(r"^(?P<prefix>.+\.mlp\.experts)\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")


def _stack_expert_keys(path: pathlib.Path) -> None:
    """Rewrite a checkpoint's per-expert keys into GLM4.5's released stacked form.

    ``experts.{e}.gate_proj|up_proj.weight`` -> ``experts.gate_up_proj [E, 2I, H]``
    ``experts.{e}.down_proj.weight``         -> ``experts.down_proj   [E, H, I]``

    transformers always serializes the per-expert form, so this is what makes
    the bridge select its grouped-export mappings. ``shared_experts`` carry no
    index and pass through untouched.
    """
    from safetensors.torch import load_file, save_file

    sd = load_file(str(path))
    out: dict = {}
    experts: dict = {}
    for name, t in sd.items():
        m = _EXPERT_KEY.match(name)
        if m is None:
            out[name] = t
        else:
            experts.setdefault(m["prefix"], {}).setdefault(int(m["e"]), {})[m["proj"]] = t
    for prefix, per in experts.items():
        order = sorted(per)
        out[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat([per[e]["gate_proj"], per[e]["up_proj"]]) for e in order]
        )
        out[f"{prefix}.down_proj"] = torch.stack([per[e]["down_proj"] for e in order])
    assert experts, f"{path}: no per-expert keys to stack -- the fused fixture would silently be a per-expert one"
    save_file({k: v.contiguous() for k, v in out.items()}, str(path), metadata={"format": "pt"})


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
    elif MODEL_KIND in ("glm4_moe", "glm4_moe_fused"):
        # Same weights, two on-disk layouts. GLM45Bridge picks its
        # FusedGatedExpertMapping / FusedExpertMapping (is_grouped_export=True)
        # only when the checkpoint carries stacked expert keys, so the "_fused"
        # kind rewrites them after save and the plain kind keeps transformers'
        # per-expert form -- an A/B over the two mapping families on identical
        # values. moe_intermediate_size stays divisible by ETP=2.
        hf_config = AutoConfig.for_model(
            "glm4_moe",
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=512,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=1,
            n_group=1,
            topk_group=1,
            max_position_embeddings=256,
            tie_word_embeddings=False,
            use_qk_norm=False,
        )
        hf_config.architectures = ["Glm4MoeForCausalLM"]
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
        if MODEL_KIND == "glm4_moe_fused":
            _stack_expert_keys(pathlib.Path(cfg_dir, "model.safetensors"))
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


def _engine_differential(bridge, model, rank: int, world: int):
    """Engine-level differential: run the PRODUCTION export path (global
    directory + slot pre-seed + probe entries, a full-delta sync where every
    local element is 'changed') and compare the merged wire-view against
    ``bridge.export_hf_weights`` -- the bridge's own full export, an
    independent chain. Bitwise, per HF tensor, with directory-lockstep and
    placeholder-row checks on the way.

    This is the only oracle that sees how entries from DIFFERENT tasks land on
    the same HF tensor, so it is the one that catches aggregation bugs: slot
    tables that disagree across ranks, rows that overwrite each other, and
    grouped-export families where many per-expert tasks share one HF name.

    Returns ``(failures, n_pass, n_checked)``; only rank 0's counts are
    meaningful."""
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
        # Routing counts as much as coordinates: the engine consumes ONLY what
        # reaches global rank 0 -- its own entry for ``gather_group=None`` rows,
        # otherwise the gather over rank 0's OWN group (sibling replica groups
        # gather to their own roots and are dropped). A row whose spec sends it
        # down the wrong path still emits perfectly correct coordinates on every
        # rank, so merging everyone's pieces would hide the bug; only this filter
        # turns "a replicated row that was actually sharded" into lost elements.
        group = rec.spec.gather_group
        if group is None:
            ships = rank == 0
        else:
            ships = rec.spec.contributes and 0 in dist.get_process_group_ranks(group)
        if not ships:
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
    dist.barrier()
    return failures, n_pass, n_checked


def _task_differential(bridge, model, rank: int, world: int):
    """Per-task differential: for one mapping, the REAL ``megatron_to_hf``
    (true collectives, all ranks in lockstep) must equal the merged non-NaN
    contributions of every rank's comm-stubbed probe, bitwise.

    Guards the probe's own two assumptions -- communication confined to the
    stubbed helpers, transforms rearrange rather than blend -- and nothing
    beyond one task, which is why MoE kinds also run the engine oracle.

    Returns ``(failures, n_pass, n_checked)``."""
    from verl.workers.engine.megatron.delta_export import make_probe

    tasks = bridge.get_conversion_tasks(model)
    n_checked = n_pass = 0
    failures = []
    for t in tasks:
        p = t.param_weight
        if p is None or not p.is_floating_point():
            continue
        if EP > 1 and getattr(t.mapping, "is_grouped_export", False):
            # Not comparable task-by-task under EP: the real mapping all-gathers
            # over the expert group and returns ``[ep_size, *slice]`` (which
            # ``_accumulate_grouped_export`` then scatters to global expert ids),
            # while the probe stubs that gather away and returns this rank's bare
            # 2D slice. The engine oracle checks these rows end to end instead.
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
    return failures, n_pass, n_checked


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
        expert_tensor_parallel_size=ETP,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)

    cfg_dir = _build_tiny_hf_dir(rank)

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(cfg_dir, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = TP
    provider.pipeline_model_parallel_size = PP
    if EP > 1 or MODEL_KIND in MOE_KINDS:
        provider.expert_model_parallel_size = EP
        provider.expert_tensor_parallel_size = ETP
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    model = [m.cuda().to(torch.bfloat16) for m in (model if isinstance(model, list) else [model])]
    # real HF weights, identically loaded on every rank: replicated params
    # (layernorms, routers) must be bit-identical across TP ranks or the
    # differential would flag test artifacts instead of probe bugs.
    bridge.load_hf_weights(model)

    # The per-task oracle needs every rank to hold the param (PP=1 only). The
    # engine oracle always runs for MoE: a grouped-export family is per-task
    # correct and only collides once the rows are merged, so gating it on EP>1
    # (as this file used to) leaves the collision unobserved.
    results = []
    if PP == 1:
        results.append(("task", _task_differential(bridge, model, rank, world)))
    if PP > 1 or EP > 1 or MODEL_KIND in MOE_KINDS or os.environ.get("ENGINE_DIFF"):
        results.append(("engine", _engine_differential(bridge, model, rank, world)))

    failed = False
    if rank == 0:
        print("=" * 60)
        cfg = f"{MODEL_KIND} tp={TP} pp={PP} ep={EP} etp={ETP}"
        for label, (failures, n_pass, n_checked) in results:
            for f in failures[:20]:
                print(f"FAIL [{label}]:", f)
            print(f"PROBE DIFFERENTIAL [{label}] [{cfg}]: {n_pass}/{n_checked} bitwise-equal")
            failed = failed or bool(failures) or n_checked == 0
        if not results:
            print(f"PROBE DIFFERENTIAL [{cfg}]: no oracle ran")
            failed = True
        print("PROBE DIFFERENTIAL PASSED" if not failed else "PROBE DIFFERENTIAL FAILED")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
