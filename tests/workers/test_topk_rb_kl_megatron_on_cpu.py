# Copyright 2026 Individual Contributor
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
"""Megatron ``_lm_head_logits_processor`` with ``kl_loss_type: topk_rb`` under real tensor-parallel groups (gloo).

Runs the engine's actual logits processor on vocab-sharded logits, together with Megatron's log-prob cross
entropy, which shifts the logits in place, and compares outputs and the gradient of
``coarse + log_q_tail + log_probs`` with the single-device computation.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.trainer.ppo.topk_rb_kl import compute_coarse_reverse_kl, compute_topk_log_probs

pytest.importorskip("megatron.core")

N, V, K = 9, 24, 4


def _case():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(N, V, generator=g) * 2
    ref = logits + torch.randn(N, V, generator=g)
    label = torch.randint(0, V, (N,), generator=g)
    temperature = torch.full((N,), 0.7)
    return logits, ref, label, temperature


def _worker(rank, world_size, rendezvous_file, out_dir, mode):
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size)
    from megatron.core import parallel_state as mpu

    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    try:
        mpu.initialize_model_parallel(tensor_model_parallel_size=world_size)
        logits, ref, label, temperature = _case()
        ref_lp, ref_ids = compute_topk_log_probs(ref / temperature.unsqueeze(-1), K)
        shard = V // world_size
        vp = logits[:, rank * shard : (rank + 1) * shard].clone().unsqueeze(0).requires_grad_(True)
        engine = object.__new__(MegatronEngineWithLMHead)
        engine.engine_config = SimpleNamespace(entropy_from_logits_with_chunking=False)
        if mode == "actor":
            flags = dict(kl_topk_rb=True, ref_topk_ids=ref_ids.to(torch.int32)[None], ref_topk_log_probs=ref_lp[None])
        else:
            flags = dict(ref_topk=K)
        # `vp * 1.0`: the processor scales the logits in place, which a leaf tensor does not allow
        ret = engine._lm_head_logits_processor(
            vp * 1.0,
            label[None],
            temperature.clone()[None],
            calculate_sum_pi_squared=False,
            calculate_entropy=False,
            distillation_use_topk=False,
            distillation_only=False,
            score_centering=False,
            logits_processor_func=None,
            batch=None,
            data_format="thd",
            **flags,
        )
        out = {k: v.detach().squeeze(0) for k, v in ret.items()}
        if mode == "actor":
            (ret["kl_topk_coarse"] * 1.3 + ret["kl_topk_log_q_tail"] + ret["log_probs"]).sum().backward()
            out["grad"] = vp.grad.squeeze(0)
        torch.save(out, f"{out_dir}/rank{rank}.pt")
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def _run(tmp_path, world_size, mode):
    mp.spawn(_worker, args=(world_size, str(tmp_path / "rdzv"), str(tmp_path), mode), nprocs=world_size)
    return [torch.load(tmp_path / f"rank{r}.pt") for r in range(world_size)]


@pytest.mark.parametrize("world_size", [1, 2])
def test_megatron_ref_pass_emits_global_topk(tmp_path, world_size):
    logits, _, _, temperature = _case()
    log_probs, ids = compute_topk_log_probs(logits / temperature.unsqueeze(-1), K)  # single-device reference
    for out in _run(tmp_path, world_size, "ref"):
        assert out["ref_topk_ids"].dtype == torch.int32
        torch.testing.assert_close(out["ref_topk_ids"].long(), ids)
        torch.testing.assert_close(out["ref_topk_log_probs"], log_probs, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("world_size", [1, 2])
def test_megatron_actor_coarse_kl_and_gradient_match_single_device(tmp_path, world_size):
    logits, ref, label, temperature = _case()
    ref_lp, ref_ids = compute_topk_log_probs(ref / temperature.unsqueeze(-1), K)
    z = logits.clone().requires_grad_(True)
    scaled = z / temperature.unsqueeze(-1)
    coarse, log_q_tail = compute_coarse_reverse_kl(scaled, ref_ids, ref_lp)
    log_probs = scaled.log_softmax(-1).gather(-1, label.unsqueeze(-1)).squeeze(-1)
    (coarse * 1.3 + log_q_tail + log_probs).sum().backward()

    shards = _run(tmp_path, world_size, "actor")
    tol = dict(atol=2e-5, rtol=2e-5)
    for out in shards:
        torch.testing.assert_close(out["kl_topk_coarse"], coarse.detach(), **tol)
        torch.testing.assert_close(out["kl_topk_log_q_tail"], log_q_tail.detach(), **tol)
        torch.testing.assert_close(out["log_probs"], log_probs.detach(), **tol)
    torch.testing.assert_close(torch.cat([s["grad"] for s in shards], dim=-1), z.grad, **tol)
