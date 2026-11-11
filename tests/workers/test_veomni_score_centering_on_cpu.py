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

"""CPU unit tests for Score Centering on the VeOmni actor (chunked LM head, SC/SC+TIS/SC+MIS, Ulysses SP)."""

import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.score_centering import pad_rollout_topk, score_centering_ppo_loss
from verl.utils import tensordict_utils as tu
from verl.utils.config import _validate_score_centering_config
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.ulysses import set_ulysses_sequence_parallel_group
from verl.utils.veomni.score_centering import (
    chunked_selected_linear_log_probs,
    make_veomni_score_centering_adapter,
    veomni_score_centering_context,
)
from verl.workers.config import ActorConfig, PolicyLossConfig, VeOmniEngineConfig


def _import_veomni_impl():
    for mod in (
        "veomni",
        "veomni.arguments",
        "veomni.distributed",
        "veomni.distributed.torch_parallelize",
        "veomni.models",
        "veomni.models.auto",
        "veomni.models.checkpoint_tensor_loading",
        "veomni.optim",
        "veomni.utils",
        "veomni.utils.seqlen_pos_transform_utils",
    ):
        sys.modules.setdefault(mod, MagicMock())
    from verl.workers.engine.veomni import transformer_impl

    return transformer_impl


def _actor_config(rollout_is, rollout_is_threshold=2.0):
    rc = {
        "bypass_mode": True,
        "loss_type": "reinforce",
        "rollout_is": rollout_is,
        "rollout_is_threshold": rollout_is_threshold,
        "score_centering": True,
    }
    return ActorConfig(
        strategy="veomni",
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=2,
        use_dynamic_bsz=False,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=rc),
    )


@pytest.mark.parametrize("use_fused_kernels", [True, False])
def test_validate_score_centering_config_accepts_veomni(use_fused_kernels):
    rc = {"bypass_mode": True, "loss_type": "reinforce", "rollout_is": "token", "rollout_is_threshold": 2.0}
    cfg = OmegaConf.create(
        {
            "algorithm": {"rollout_correction": {**rc, "score_centering": True}},
            "actor_rollout_ref": {
                "model": {"use_remove_padding": True, "use_fused_kernels": use_fused_kernels},
                "actor": {
                    "strategy": "veomni",
                    "veomni": {"cross_entropy_loss_implementation": "chunk_loss"},
                    "policy_loss": {"loss_mode": "bypass_mode", "rollout_correction": {**rc, "score_centering": True}},
                },
                "rollout": {"name": "vllm", "topk_log_probs": 128, "temperature": 1.0},
            },
            "distillation": {"enabled": False},
            "trainer": {"use_v1": True},
        }
    )
    _validate_score_centering_config(cfg)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [1.0, 0.8])
def test_chunked_selected_linear_log_probs_matches_dense_forward_and_backward(dtype, temperature):
    torch.manual_seed(7)
    num_tokens, hidden_dim, vocab_size, k = 9, 16, 29, 5
    hidden = torch.randn(1, num_tokens, hidden_dim, dtype=dtype, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, vocab_size, (1, num_tokens))
    labels[0, 3] = -100
    labels[0, -1] = -100
    topk_ids = torch.stack([torch.randperm(vocab_size)[:k] for _ in range(num_tokens)])

    lp, ent, head_lp = chunked_selected_linear_log_probs(
        hidden, weight, labels, topk_ids, temperature=temperature, chunk_size=3, ignore_index=-100
    )

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    logits_ref = (hidden_ref.reshape(-1, hidden_dim) @ weight_ref.t()).float() / temperature
    full_logp = logits_ref.log_softmax(dim=-1)
    valid = labels.reshape(-1) != -100
    lp_ref = torch.where(valid, full_logp.gather(-1, labels.reshape(-1).clamp(min=0).unsqueeze(-1)).squeeze(-1), 0.0)
    ent_ref = torch.where(valid, -(full_logp.exp() * full_logp).sum(dim=-1), 0.0)
    head_lp_ref = full_logp.gather(-1, topk_ids)

    torch.testing.assert_close(lp, lp_ref.view_as(labels), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ent, ent_ref.view_as(labels), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(head_lp, head_lp_ref, atol=1e-5, rtol=1e-5)

    w_lp = torch.randn_like(lp)
    w_ent = torch.randn_like(ent)
    w_head = torch.randn_like(head_lp)
    ((lp * w_lp).sum() + (ent * w_ent).sum() + (head_lp * w_head).sum()).backward()
    (
        (lp_ref.view_as(labels) * w_lp).sum() + (ent_ref.view_as(labels) * w_ent).sum() + (head_lp_ref * w_head).sum()
    ).backward()

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(hidden.grad.float(), hidden_ref.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(weight.grad.float(), weight_ref.grad.float(), atol=atol, rtol=rtol)


def _build_ragged_micro_batch(vocab_size=19, k=4, support_complete=False):
    torch.manual_seed(11)
    prompts = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    responses = [torch.tensor([6, 7, 8]), torch.tensor([9, 10])]
    input_seqs = [torch.cat([p, r]) for p, r in zip(prompts, responses, strict=True)]
    pos_seqs = [torch.arange(len(s)) for s in input_seqs]

    sampler_logits = [torch.randn(len(r), vocab_size) for r in responses]
    topk_ids_list, topk_lp_list, complete_list, rollout_lp_list = [], [], [], []
    for r, slogits in zip(responses, sampler_logits, strict=True):
        if support_complete:
            # Keep only top-(k-1) tokens in sampler support; mark remaining slot -inf.
            vals, order = torch.topk(slogits, k=k, dim=-1)
            vals[:, -1] = float("-inf")
            # Ensure the sampled token is inside the kept support.
            order[:, 0] = r
            vals[:, 0] = slogits.gather(-1, r.unsqueeze(-1)).squeeze(-1)
            slp = torch.log_softmax(vals, dim=-1)
            sampled_lp = slp[:, 0]
            full_ids, full_lp = pad_rollout_topk(order, slp, len(input_seqs[len(topk_ids_list)]) - len(r))
            comp = torch.zeros(len(full_ids), dtype=torch.bool)
            start = len(full_ids) - len(r) - 1
            comp[start : start + len(r)] = True
        else:
            full_slp = torch.log_softmax(slogits, dim=-1)
            sampled_lp = full_slp.gather(-1, r.unsqueeze(-1)).squeeze(-1)
            ids = torch.topk(full_slp, k=k, dim=-1).indices
            full_ids, full_lp = pad_rollout_topk(
                ids, full_slp.gather(-1, ids), len(input_seqs[len(topk_ids_list)]) - len(r)
            )
            comp = torch.zeros(len(full_ids), dtype=torch.bool)
        topk_ids_list.append(full_ids)
        topk_lp_list.append(full_lp)
        complete_list.append(comp)
        rollout_lp_list.append(sampled_lp)

    batch = TensorDict(
        {
            "prompts": torch.nested.as_nested_tensor(prompts, layout=torch.jagged),
            "input_ids": torch.nested.as_nested_tensor(input_seqs, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(pos_seqs, layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor(responses, layout=torch.jagged),
            "response_mask": torch.nested.as_nested_tensor(
                [torch.ones_like(r) for r in responses], layout=torch.jagged
            ),
            "advantages": torch.nested.as_nested_tensor(
                [torch.tensor([1.2, -0.5, 0.8]), torch.tensor([-0.9, 0.4])], layout=torch.jagged
            ),
            "old_log_probs": torch.nested.as_nested_tensor(rollout_lp_list, layout=torch.jagged),
            "rollout_log_probs": torch.nested.as_nested_tensor(rollout_lp_list, layout=torch.jagged),
            "rollout_topk_ids": torch.nested.as_nested_tensor(topk_ids_list, layout=torch.jagged),
            "rollout_topk_log_probs": torch.nested.as_nested_tensor(topk_lp_list, layout=torch.jagged),
            "rollout_support_complete": torch.nested.as_nested_tensor(complete_list, layout=torch.jagged),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        batch,
        temperature=1.0,
        use_remove_padding=True,
        pad_mode=DatasetPadMode.NO_PADDING,
        score_centering=True,
        calculate_entropy=True,
        dp_size=1,
        batch_num_tokens=5,
        global_batch_size=2,
    )
    return batch


class _TinyVeOmniCausalLM(torch.nn.Module):
    """Minimal causal LM matching VeOmni's eager and return_log_probs forward contracts."""

    def __init__(self, vocab_size: int, hidden_dim: int, adapted_chunk_fn):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_moe")
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        self._adapted_chunk_fn = adapted_chunk_fn

    def forward(self, input_ids, labels=None, shift_labels=None, return_log_probs=False, temperature=1.0, **kwargs):
        hidden = self.embed(input_ids)
        if return_log_probs:
            log_probs, entropy = self._adapted_chunk_fn(
                hidden,
                self.lm_head.weight,
                labels,
                chunk_size=3,
                ignore_index=-100,
                shift_labels=shift_labels,
                temperature=temperature,
            )
            return SimpleNamespace(log_probs=log_probs, entropy=entropy, logits=None)
        return SimpleNamespace(logits=self.lm_head(hidden))


@pytest.mark.parametrize(
    "rollout_is, threshold",
    [
        (None, 2.0),  # SC
        ("token", 2.0),  # SC + TIS
        ("token", "0.5_5.0"),  # SC + MIS
    ],
)
@pytest.mark.parametrize("support_complete", [False, True])
def test_veomni_fused_score_centering_matches_eager_across_sc_tis_mis(rollout_is, threshold, support_complete):
    veomni_impl = _import_veomni_impl()
    VeOmniEngineWithLMHead = veomni_impl.VeOmniEngineWithLMHead

    vocab_size, hidden_dim = 19, 12
    adapted_fn = make_veomni_score_centering_adapter(lambda *a, **kw: None)

    torch.manual_seed(23)
    model_fused = _TinyVeOmniCausalLM(vocab_size, hidden_dim, adapted_fn)
    model_eager = _TinyVeOmniCausalLM(vocab_size, hidden_dim, adapted_fn)
    model_eager.load_state_dict(model_fused.state_dict())

    actor_cfg = _actor_config(rollout_is, threshold)
    loss_fn = partial(score_centering_ppo_loss, config=actor_cfg)

    def _run(model, use_fused):
        engine = object.__new__(VeOmniEngineWithLMHead)
        engine.module = model
        engine.engine_config = VeOmniEngineConfig(cross_entropy_loss_implementation="chunk_loss", batch_invariant=False)
        engine.use_ulysses_sp = False
        engine.ulysses_sequence_parallel_size = 1
        engine.pad_to_length = False
        engine.pad_to_length_bucket = 0
        engine._router_replay = None
        engine._autocast_dtype = torch.float32
        engine.compute_entropy_from_logits = verl_F.entropy_from_logits
        engine._logprobs_from_logits = lambda logits, labels, **kw: verl_F.logprobs_from_logits_v2(logits, labels)
        engine.get_data_parallel_group = lambda: None

        batch = _build_ragged_micro_batch(vocab_size=vocab_size, k=4, support_complete=support_complete)
        tu.assign_non_tensor(batch, use_fused_kernels=use_fused)
        with (
            patch.object(
                veomni_impl.parallel_state, "get_parallel_state", return_value=SimpleNamespace(sp_enabled=False)
            ),
            patch("verl.utils.veomni.score_centering.install_veomni_score_centering_adapter"),
            patch.object(veomni_impl, "get_device_name", return_value="cpu"),
            patch.object(veomni_impl, "get_device_id", return_value="cpu"),
            patch("verl.workers.engine.fsdp.transformer_impl.get_device_name", return_value="cpu"),
            patch("verl.workers.engine.fsdp.transformer_impl.get_device_id", return_value="cpu"),
            patch.object(engine, "_apply_veomni_input_transforms", lambda model_inputs, mb: None),
        ):
            loss, metrics = engine.forward_step(batch, loss_function=loss_fn, forward_only=False)
        loss.backward()
        return loss.detach(), metrics

    with patch.dict("os.environ", {"VERL_DISABLE_FLASH_ATTN_CE": "1"}):
        loss_fused, metrics_fused = _run(model_fused, use_fused=True)
        loss_eager, metrics_eager = _run(model_eager, use_fused=False)

    torch.testing.assert_close(loss_fused, loss_eager, atol=1e-5, rtol=1e-5)
    for key in (
        "actor/sc_correction",
        "actor/sc_sampler_head_mass",
        "actor/sc_train_head_mass",
        "actor/sc_support_aligned",
        "actor/sc_sampler_tail_mass",
    ):
        if key not in metrics_fused and key not in metrics_eager:
            continue
        v_fused = metrics_fused[key].value if hasattr(metrics_fused[key], "value") else metrics_fused[key]
        v_eager = metrics_eager[key].value if hasattr(metrics_eager[key], "value") else metrics_eager[key]
        torch.testing.assert_close(torch.as_tensor(v_fused), torch.as_tensor(v_eager), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model_fused.embed.weight.grad, model_eager.embed.weight.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model_fused.lm_head.weight.grad, model_eager.lm_head.weight.grad, atol=1e-5, rtol=1e-5)


def _sp2_worker(rank: int, world_size: int, port: int, state_dict: dict, batch_dict: dict, result_queue):
    veomni_impl = _import_veomni_impl()

    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        adapted_fn = make_veomni_score_centering_adapter(lambda *a, **kw: None)
        model = _TinyVeOmniCausalLM(batch_dict["vocab_size"], batch_dict["hidden_dim"], adapted_fn)
        model.load_state_dict(state_dict)

        engine = object.__new__(veomni_impl.VeOmniEngineWithLMHead)
        engine.module = model
        engine.engine_config = VeOmniEngineConfig(cross_entropy_loss_implementation="chunk_loss", batch_invariant=False)
        engine.use_ulysses_sp = True
        engine.ulysses_sequence_parallel_size = world_size
        engine.pad_to_length = False
        engine.pad_to_length_bucket = 0
        engine._router_replay = None
        engine._autocast_dtype = torch.float32
        engine.compute_entropy_from_logits = verl_F.entropy_from_logits
        engine._logprobs_from_logits = lambda logits, labels, **kw: verl_F.logprobs_from_logits_v2(logits, labels)
        engine.get_data_parallel_group = lambda: None

        batch = _build_ragged_micro_batch(vocab_size=batch_dict["vocab_size"], k=4, support_complete=False)
        tu.assign_non_tensor(batch, use_fused_kernels=True)
        loss_fn = partial(score_centering_ppo_loss, config=_actor_config("token", 2.0))

        with (
            patch.object(
                veomni_impl.parallel_state,
                "get_parallel_state",
                return_value=SimpleNamespace(sp_enabled=True, sp_size=world_size, sp_rank=rank),
            ),
            patch("verl.utils.veomni.score_centering.install_veomni_score_centering_adapter"),
            patch("verl.workers.engine.fsdp.transformer_impl.get_device_name", return_value="cpu"),
            patch("verl.workers.engine.fsdp.transformer_impl.get_device_id", return_value="cpu"),
            # Skip flash-attn position_ids helper on CPU; input_ids is already SP-sliced by super().
            patch.object(engine, "_apply_veomni_input_transforms", lambda model_inputs, mb: None),
        ):
            loss, _ = engine.forward_step(batch, loss_function=loss_fn, forward_only=False)
        loss.backward()
        # Sum local unsharded parameter gradients across the 2 SP shards to compare against SP=1.
        dist.all_reduce(model.lm_head.weight.grad, op=dist.ReduceOp.SUM)
        # Gather scales upstream grad by sp_world_size for FSDP avg-reduce; divide by world_size here.
        if rank == 0:
            result_queue.put((float(loss.detach()), (model.lm_head.weight.grad.detach() / world_size).tolist()))
    finally:
        set_ulysses_sequence_parallel_group(None)
        dist.destroy_process_group()


def test_veomni_fused_score_centering_ulysses_sp2_matches_sp1():
    veomni_impl = _import_veomni_impl()
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    vocab_size, hidden_dim = 19, 12
    adapted_fn = make_veomni_score_centering_adapter(lambda *a, **kw: None)
    torch.manual_seed(31)
    ref_model = _TinyVeOmniCausalLM(vocab_size, hidden_dim, adapted_fn)
    state_dict = {k: v.clone() for k, v in ref_model.state_dict().items()}

    # Single-rank SP=1 reference (total_nnz = 9 tokens, so SP=2 exercises non-zero SP padding!).
    engine = object.__new__(veomni_impl.VeOmniEngineWithLMHead)
    engine.module = ref_model
    engine.engine_config = VeOmniEngineConfig(cross_entropy_loss_implementation="chunk_loss", batch_invariant=False)
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.pad_to_length = False
    engine.pad_to_length_bucket = 0
    engine._router_replay = None
    engine._autocast_dtype = torch.float32
    engine.get_data_parallel_group = lambda: None
    batch = _build_ragged_micro_batch(vocab_size=vocab_size, k=4, support_complete=False)
    tu.assign_non_tensor(batch, use_fused_kernels=True)
    loss_fn = partial(score_centering_ppo_loss, config=_actor_config("token", 2.0))
    with (
        patch.object(veomni_impl.parallel_state, "get_parallel_state", return_value=SimpleNamespace(sp_enabled=False)),
        patch("verl.utils.veomni.score_centering.install_veomni_score_centering_adapter"),
        patch("verl.workers.engine.fsdp.transformer_impl.get_device_name", return_value="cpu"),
        patch("verl.workers.engine.fsdp.transformer_impl.get_device_id", return_value="cpu"),
        patch.object(engine, "_apply_veomni_input_transforms", lambda model_inputs, mb: None),
    ):
        loss_sp1, _ = engine.forward_step(batch, loss_function=loss_fn, forward_only=False)
    loss_sp1.backward()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(
        _sp2_worker,
        args=(2, port, state_dict, {"vocab_size": vocab_size, "hidden_dim": hidden_dim}, queue),
        nprocs=2,
        join=True,
    )
    loss_sp2, grad_sp2 = queue.get(timeout=5)
    torch.testing.assert_close(torch.tensor(loss_sp2), loss_sp1.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.tensor(grad_sp2), ref_model.lm_head.weight.grad, atol=1e-5, rtol=1e-5)


def test_veomni_score_centering_context_cleans_up_on_error():
    with pytest.raises(RuntimeError, match="boom"):
        with veomni_score_centering_context() as req:
            req.topk_ids = torch.zeros((2, 4), dtype=torch.long)
            raise RuntimeError("boom")
    adapted = make_veomni_score_centering_adapter(lambda *a, **kw: ("orig_lp", "orig_ent"))
    assert adapted(torch.zeros(1, 2, 4), torch.zeros(5, 4), torch.zeros(1, 2, dtype=torch.long)) == (
        "orig_lp",
        "orig_ent",
    )
