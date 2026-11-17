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

"""Chunk-driver scaling and forward-only behavior against the real MCore schedule."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from megatron.core.pipeline_parallel.outer_chunk import validate_outer_chunk_config
from megatron.core.pipeline_parallel.schedules import forward_step_calc_loss
from megatron.core.transformer.transformer_config import TransformerConfig

from verl.workers.engine.megatron.transformer_impl import (
    MegatronEngineWithLMHead,
    _check_cp_seq_chunk_config,
    _check_cp_seq_chunk_models,
)


def _case(monkeypatch, *, per_token, microbatches=1, loss_scale=1):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=2)
    config.calculate_per_token_loss = per_token
    config.grad_scale_func = lambda loss: loss * loss_scale
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    model.config = config
    with torch.no_grad():
        model.weight.fill_(1)
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.get_context_parallel_group = lambda: None
    phases = []

    def forward_step(*args):
        index = engine._dsv41_chunk_ctx[0]
        phases.append((index, torch.is_grad_enabled()))
        value = model.weight.sum()
        values = torch.cat([value.expand(4), value.new_zeros(4)])
        if index:
            values = values.roll(4)
        output = {"log_probs": torch.nested.nested_tensor_from_jagged(values, offsets=torch.tensor([0, 8]))}

        def loss_func(output):
            total = output["log_probs"].values().sum()
            logged = {
                "loss": float(total.detach()) / 8,
                "metrics": {"mean": float(total.detach()) / 8},
                "model_output": output,
            }
            if per_token:
                return total, torch.tensor(8, dtype=torch.int), logged
            return total / 8 * microbatches, logged

        return output, loss_func

    engine.forward_step = forward_step
    batch = {"num_micro_batch": microbatches}
    return engine, model, config, batch, phases


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("microbatches", [1, 3])
@pytest.mark.parametrize("loss_scale", [1, 8])
def test_chunk_gradient_and_token_divisor_match_single_pass(monkeypatch, per_token, microbatches, loss_scale):
    engine, model, config, batch, phases = _case(
        monkeypatch, per_token=per_token, microbatches=microbatches, loss_scale=loss_scale
    )
    seed, closure = engine._dsv41_chunked_forward_step(batch, model, None, None, 2)
    records = []
    outer_loss, tokens = forward_step_calc_loss(
        model, seed, closure, config, None, False, microbatches, records, cp_group_size=1, is_last_stage=True
    )
    (outer_loss * loss_scale).backward()
    gradient = model.weight.grad / tokens.clamp(min=1)
    torch.testing.assert_close(gradient, torch.full_like(gradient, loss_scale))
    assert tokens.item() == (8 if per_token else 0)
    assert records[0]["loss"] == 1.0
    assert records[0]["metrics"] == {"mean": 1.0}
    torch.testing.assert_close(records[0]["model_output"]["log_probs"].values(), torch.ones(8, dtype=torch.float64))
    assert phases == [(0, False), (1, False), (1, True), (0, True)]


@pytest.mark.parametrize("per_token", [False, True])
def test_chunk_inference_preserves_outputs_and_never_runs_backward(monkeypatch, per_token):
    engine, model, _, batch, phases = _case(monkeypatch, per_token=per_token)
    model.weight.grad = torch.full_like(model.weight, 17)
    with torch.no_grad():
        seed, closure = engine._dsv41_chunked_forward_step(batch, model, None, None, 2, forward_only=True)
        result = closure(seed)
    assert not seed.requires_grad
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 17))
    torch.testing.assert_close(result[-1]["model_output"]["log_probs"].values(), torch.ones(8, dtype=torch.float64))
    assert phases == [(0, False), (1, False)]


@pytest.mark.parametrize("engine_replay,batch_replay", [(True, False), (False, True)])
def test_chunk_rejects_unpartitioned_router_replay(monkeypatch, engine_replay, batch_replay):
    engine, model, post = _guard_case(monkeypatch, cp_size=2)
    engine.engine_config.router_replay.mode = "R2" if engine_replay else "disabled"
    with pytest.raises(NotImplementedError, match="router replay"):
        _check_cp_seq_chunk_config(engine.engine_config, engine.tf_config)
        engine.forward_step(iter([{"enable_routing_replay": batch_replay}]), model, None, post)


@pytest.mark.parametrize(
    "overrides,observed",
    [
        ({"dynamic_context_parallel": True}, "dynamic_context_parallel=True"),
        ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size=2"),
        ({"use_remove_padding": False}, "use_remove_padding=False"),
        ({"use_fused_kernels": True}, "use_fused_kernels=True"),
        ({"overlap_grad_reduce": True}, "overlap_grad_reduce=True"),
        ({"transformer_config": SimpleNamespace(experimental_attention_variant="mla")}, "variant='mla'"),
    ],
)
def test_core_rejects_unsupported_outer_chunk_config(overrides, observed):
    config = {"context_parallel_size": 2, "pipeline_model_parallel_size": 1, **overrides}
    with pytest.raises(NotImplementedError, match=observed):
        validate_outer_chunk_config(**config)


@pytest.mark.parametrize(
    "cp_layout,chunk_positions",
    [
        ("contiguous", (((0, 1), (2, 3)), ((4, 5), (6, 7)), ((8, 9), (10, 11)))),
        ("zigzag", (((0, 11), (1, 10)), ((2, 9), (3, 8)), ((4, 7), (5, 6)))),
    ],
)
def test_chunk_thd_packing_restores_logical_rows_and_local_gradients(monkeypatch, cp_layout, chunk_positions):
    from verl.models.mcore import util

    group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", group.size)
    monkeypatch.setattr(util.mpu, "get_context_parallel_rank", group.rank)
    monkeypatch.setattr(util.mpu, "get_context_parallel_group", lambda: group)

    logical_ids = torch.arange(1, 8)
    input_ids = torch.nested.as_nested_tensor([logical_ids], layout=torch.jagged)
    padded_ids = torch.cat([logical_ids, torch.zeros(5, dtype=logical_ids.dtype)])
    # Give padding nonzero model output so an incorrect restored position stays visible.
    padded_values = torch.arange(1.0, 25.0).reshape(12, 2)

    for chunk_index, (peer_positions, local_positions) in enumerate(chunk_positions):
        chunk = {"cp_chunk_index": chunk_index, "cp_chunk_count": 3, "cp_layout": cp_layout}
        local_ids, packed, _ = util.preprocess_thd_engine(input_ids, **chunk)
        torch.testing.assert_close(local_ids[0], padded_ids[list(local_positions)])
        assert packed.cu_seqlens_q_padded.tolist() == [0, 12]

        peer = padded_values[list(peer_positions)].requires_grad_()
        local = padded_values[list(local_positions)].requires_grad_()

        def gather(destinations, shard, group, *, expected_local=local, expected_peer=peer):
            assert group.size() == len(destinations) == 2
            assert not shard.requires_grad
            torch.testing.assert_close(shard, expected_local)
            destinations[0].copy_(expected_peer.detach())
            destinations[1].copy_(shard)

        monkeypatch.setattr(torch.distributed, "all_gather", gather)
        restored = util.postprocess_thd_engine(local.unsqueeze(0), packed, input_ids, batch_size=1, **chunk)

        visible = [position for position in (*peer_positions, *local_positions) if position < len(logical_ids)]
        expected = torch.zeros_like(padded_values[: len(logical_ids)])
        expected[visible] = padded_values[visible]
        torch.testing.assert_close(restored.values(), expected)
        restored.values().sum().backward()
        expected_grad = (torch.tensor(local_positions) < len(logical_ids)).to(local.dtype)
        torch.testing.assert_close(local.grad, expected_grad[:, None].expand_as(local))
        assert peer.grad is None


@pytest.mark.parametrize("chunks", [2, 3])
@pytest.mark.parametrize("objective", ["gspo", "geo_mean", "sequence_mean"])
def test_chunk_differentiates_the_complete_sequence_objective(monkeypatch, chunks, objective):
    from verl.trainer.ppo.core_algos import compute_policy_loss_geo_mean, compute_policy_loss_gspo
    from verl.workers.config import ActorConfig

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=2)
    model = torch.nn.Linear(6, 1, bias=False, dtype=torch.float64)
    model.config = config
    with torch.no_grad():
        model.weight.fill_(-1)
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(dynamic_context_parallel=False)
    engine.tf_config = config
    engine.get_context_parallel_group = lambda: None
    engine.get_data_parallel_group = lambda: None
    engine.prepare_model_outputs = lambda output, data: output
    actor = ActorConfig(strategy="megatron", rollout_n=1, ppo_micro_batch_size_per_gpu=1, clip_ratio=0.2)
    mask = torch.ones((1, 6), dtype=torch.float64)
    old_log_probs = -mask
    calls = []

    def loss_function(model_output, data, dp_group=None):
        calls.append(getattr(engine, "_dsv41_chunk_ctx", None))
        values = model_output["log_probs"].values().reshape(1, 6)
        if objective == "sequence_mean":
            # A custom coupled objective needs the same generic treatment as PPO.
            return values.mean().square(), {}
        function = compute_policy_loss_gspo if objective == "gspo" else compute_policy_loss_geo_mean
        return function(old_log_probs, values, mask, mask, loss_agg_mode="token-mean", config=actor)

    def output_for(index=None):
        values = model.weight.flatten()
        if index is not None:
            owned = torch.zeros_like(values)
            owned[index * (6 // chunks) : (index + 1) * (6 // chunks)] = 1
            values = values * owned
        return {"log_probs": torch.nested.nested_tensor_from_jagged(values, offsets=torch.tensor([0, 6]))}

    reference_loss, _ = loss_function(output_for(), None)
    reference_loss.backward()
    reference_gradient = model.weight.grad.clone()
    assert reference_gradient.count_nonzero() == 6
    model.zero_grad()
    calls.clear()
    batch = {"input_ids": torch.arange(6).unsqueeze(0), "num_micro_batch": 1}
    post = partial(engine.postprocess_micro_batch_func, forward_only=False, loss_function=loss_function)
    engine.forward_step = lambda *args: (output_for(engine._dsv41_chunk_ctx[0]), partial(post, data=batch))
    seed, closure = engine._dsv41_chunked_forward_step(batch, model, None, post, chunks)

    torch.testing.assert_close(model.weight.grad, reference_gradient, rtol=1e-12, atol=1e-12)
    assert closure(seed)[-1]["loss"] == pytest.approx(reference_loss.item(), abs=1e-12)
    assert calls == [None]


def test_chunk_replay_restores_model_rng_and_preserves_loss_rng_consumption(monkeypatch):
    engine, model, _, batch, _ = _case(monkeypatch, per_token=False)
    forward = engine.forward_step

    def stochastic_forward(*args):
        output, _ = forward(*args)
        values = output["log_probs"].values() * torch.rand(8, dtype=torch.float64)
        output = {"log_probs": torch.nested.nested_tensor_from_jagged(values, offsets=torch.tensor([0, 8]))}

        def stochastic_loss(output):
            loss = torch.rand((), dtype=torch.float64) * output["log_probs"].values().sum() / 8
            return loss, {"loss": loss.detach().item()}

        return output, stochastic_loss

    engine.forward_step = stochastic_forward
    torch.manual_seed(946)
    first, second = torch.rand(8, dtype=torch.float64), torch.rand(8, dtype=torch.float64)
    expected_gradient = (first[:4].sum() + second[4:].sum()) / 8 * torch.rand((), dtype=torch.float64)
    expected_next = torch.rand((), dtype=torch.float64)
    torch.manual_seed(946)
    engine._dsv41_chunked_forward_step(batch, model, None, None, 2)

    torch.testing.assert_close(model.weight.grad.squeeze(), expected_gradient, rtol=1e-12, atol=1e-12)
    assert torch.equal(torch.rand((), dtype=torch.float64), expected_next)


def _guard_case(monkeypatch, cp_size):
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        cp_seq_chunks=2,
        dynamic_context_parallel=False,
        pipeline_model_parallel_size=1,
        use_remove_padding=True,
        use_fused_kernels=False,
        context_parallel_size=cp_size,
        forward_only=False,
        use_megatron_fsdp=False,
        override_ddp_config={},
        router_replay=SimpleNamespace(mode="disabled"),
    )
    engine.tf_config = SimpleNamespace(experimental_attention_variant="dsv4_hybrid", calculate_per_token_loss=True)
    engine.enable_routing_replay = False
    engine._dsv41_chunked_forward_step = lambda *args, **kwargs: "chunk-driver"
    return engine, torch.nn.Linear(1, 1), SimpleNamespace(keywords={"forward_only": False})


def _constructed_chunk_step(engine, model, post):
    # These fixtures bypass engine.__init__ and Bridge; exercise their construction
    # hook explicitly before handing the already-constructed model to a step.
    engine.engine_config.forward_only = post.keywords["forward_only"]
    model.config = engine.tf_config
    _check_cp_seq_chunk_models([model], engine_config=engine.engine_config)
    return engine.forward_step(iter([{}]), model, None, post)


def test_chunk_rejects_cp1_before_native_compression(monkeypatch):
    engine, model, post = _guard_case(monkeypatch, cp_size=1)
    with pytest.raises(NotImplementedError, match="physical context_parallel_size > 1"):
        _constructed_chunk_step(engine, model, post)


@pytest.mark.parametrize("key", ["pixel_values", "image_grid_hws", "image_grid_thw"])
def test_chunk_rejects_images_with_an_older_core(monkeypatch, key):
    engine, model, post = _guard_case(monkeypatch, cp_size=2)
    batch = {"multi_modal_inputs": [{key: torch.ones(1, 2)}]}
    with pytest.raises(NotImplementedError, match="upgrade Core for chunk-aware image placement"):
        engine.forward_step(iter([batch]), model, None, post)


def test_chunk_allows_images_with_a_chunk_aware_core(monkeypatch):
    engine, model, post = _guard_case(monkeypatch, cp_size=2)
    model.supports_cp_chunk_images = True
    batch = {"multi_modal_inputs": [{"pixel_values": torch.ones(9, 3, 2, 2), "image_grid_hws": torch.tensor([[3, 3]])}]}
    assert engine.forward_step(iter([batch]), model, None, post) == "chunk-driver"


@pytest.mark.parametrize("key", ["pixel_values_videos", "video_grid_thw"])
@pytest.mark.parametrize("image_support", [False, True])
def test_chunk_keeps_video_inputs_unsupported(monkeypatch, key, image_support):
    engine, model, post = _guard_case(monkeypatch, cp_size=2)
    model.supports_cp_chunk_images = image_support
    with pytest.raises(NotImplementedError, match="does not support video inputs"):
        engine.forward_step(iter([{"multi_modal_inputs": [{key: torch.ones(1, 2)}]}]), model, None, post)


@pytest.mark.parametrize("metadata", [[], [{"vision_token_types": torch.full((1, 6), -1)}]])
def test_chunk_keeps_text_only_vlm_batches_supported(monkeypatch, metadata):
    engine, model, post = _guard_case(monkeypatch, cp_size=2)
    assert engine.forward_step(iter([{"multi_modal_inputs": metadata}]), model, None, post) == "chunk-driver"


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize(
    "routing_types,coefficients,enabled",
    [
        ("aux_loss", 0.01, True),
        ("seq_aux_loss", 0.01, True),
        ("global_aux_loss", 0.01, True),
        (["aux_loss", "seq_aux_loss"], [0.0, 0.01], True),
        (["global_aux_loss", "aux_loss"], [0.01, 0.0], True),
        ("aux_loss", 0.0, False),
        ("seq_aux_loss", 0.0, False),
        ("global_aux_loss", 0.0, False),
        (["aux_loss", "seq_aux_loss", "global_aux_loss"], [0.0, 0.0, 0.0], False),
        ("none", 0.01, False),
        (["none", "aux_loss"], [0.01, 0.0], False),
        ("aux_loss", None, False),
    ],
)
def test_chunk_rejects_only_enabled_nonadditive_balancing_losses(
    monkeypatch, per_token, routing_types, coefficients, enabled
):
    engine, model, post = _guard_case(monkeypatch, cp_size=2)
    engine.tf_config.calculate_per_token_loss = per_token
    engine.tf_config.moe_router_load_balancing_type = routing_types
    engine.tf_config.moe_aux_loss_coeff = coefficients
    if enabled:
        with pytest.raises(
            NotImplementedError, match="expert-frequency products require the complete logical sequence"
        ):
            _constructed_chunk_step(engine, model, post)
    else:
        assert _constructed_chunk_step(engine, model, post) == "chunk-driver"


@pytest.mark.parametrize("per_token", [False, True])
def test_chunk_z_loss_keeps_its_per_token_requirement(monkeypatch, per_token):
    engine, model, post = _guard_case(monkeypatch, cp_size=2)
    engine.tf_config.calculate_per_token_loss = per_token
    engine.tf_config.moe_z_loss_coeff = 0.01
    if per_token:
        assert _constructed_chunk_step(engine, model, post) == "chunk-driver"
    else:
        with pytest.raises(NotImplementedError, match="MoE z-loss requires calculate_per_token_loss=True"):
            _constructed_chunk_step(engine, model, post)


@pytest.mark.parametrize("routing_type", ["aux_loss", "seq_aux_loss", "global_aux_loss"])
def test_chunk_inference_can_use_a_model_configured_with_balancing_aux(monkeypatch, routing_type):
    engine, model, post = _guard_case(monkeypatch, cp_size=2)
    post.keywords["forward_only"] = True
    engine.tf_config.moe_router_load_balancing_type = routing_type
    engine.tf_config.moe_aux_loss_coeff = 0.01
    assert _constructed_chunk_step(engine, model, post) == "chunk-driver"


def _routing_guard_case(monkeypatch, *, mode="none", topk=1, capacity=None, training=True, forward_only=False):
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.router import TopKRouter

    engine, model, post = _guard_case(monkeypatch, cp_size=2)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_moe_experts=2,
        moe_router_topk=topk,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0,
        perform_initialization=False,
    )
    # All native router fields are initialized on CPU. QB's GPU-only buffers
    # are unnecessary for testing its entry guard (the real Core GPU repro
    # separately checks quantile accumulation and the finalized bias).
    router = TopKRouter(config, pg_collection=ProcessGroupCollection(tp=None, cp=None, tp_cp=None, tp_dp_cp=None))
    router.routing_type = mode
    router.config.moe_expert_capacity_factor = capacity
    router.train(training)
    model.add_module("router", router)
    post.keywords["forward_only"] = forward_only
    return engine, model, post, router


@pytest.mark.parametrize("forward_only", [False, True])
def test_chunk_rejects_sinkhorn_using_the_actual_router_training_mode(monkeypatch, forward_only):
    engine, model, post, _ = _routing_guard_case(monkeypatch, mode="sinkhorn", forward_only=forward_only)
    with pytest.raises(NotImplementedError, match="training-mode Sinkhorn"):
        _constructed_chunk_step(engine, model, post)


@pytest.mark.parametrize("training,topk,hash_router", [(False, 1, False), (True, 2, False), (True, 1, True)])
def test_chunk_keeps_tokenwise_sinkhorn_paths(monkeypatch, training, topk, hash_router):
    engine, model, post, router = _routing_guard_case(monkeypatch, mode="sinkhorn", training=training, topk=topk)
    router.is_hash_layer = hash_router
    assert _constructed_chunk_step(engine, model, post) == "chunk-driver"


@pytest.mark.parametrize("per_token", [False, True])
def test_chunk_rejects_quantile_updates_independently_of_the_token_loss_regime(monkeypatch, per_token):
    engine, model, post, _ = _routing_guard_case(monkeypatch, mode="quantile_balancing")
    engine.tf_config.calculate_per_token_loss = per_token
    with pytest.raises(NotImplementedError, match="averaging chunk quantiles"):
        _constructed_chunk_step(engine, model, post)


@pytest.mark.parametrize(
    "training,forward_only,ema,hash_router",
    [(False, False, 0.0, False), (True, True, 0.0, False), (True, False, 1.0, False), (True, False, 0.0, True)],
)
def test_chunk_keeps_quantile_routing_without_state_updates(monkeypatch, training, forward_only, ema, hash_router):
    engine, model, post, router = _routing_guard_case(
        monkeypatch, mode="quantile_balancing", training=training, forward_only=forward_only
    )
    router.config.moe_router_quantile_balancing_ema = ema
    router.is_hash_layer = hash_router
    assert _constructed_chunk_step(engine, model, post) == "chunk-driver"


@pytest.mark.parametrize("topk,factor", [(1, 0.01), (1, 1.0), (1, 1.99), (2, 0.99)])
@pytest.mark.parametrize("forward_only", [False, True])
def test_chunk_rejects_competitive_capacity_even_in_inference(monkeypatch, topk, factor, forward_only):
    engine, model, post, _ = _routing_guard_case(monkeypatch, topk=topk, capacity=factor, forward_only=forward_only)
    with pytest.raises(NotImplementedError, match="competitive MoE token dropping"):
        _constructed_chunk_step(engine, model, post)


@pytest.mark.parametrize("topk,factor", [(1, None), (1, 0.0), (1, 2.0), (1, 3.0), (2, 1.0)])
def test_chunk_keeps_disabled_zero_and_nonbinding_capacity(monkeypatch, topk, factor):
    engine, model, post, _ = _routing_guard_case(monkeypatch, topk=topk, capacity=factor)
    assert _constructed_chunk_step(engine, model, post) == "chunk-driver"


def test_chunk_does_not_reject_additive_expert_bias_counts(monkeypatch):
    engine, model, post, router = _routing_guard_case(monkeypatch)
    router.enable_expert_bias = True
    assert _constructed_chunk_step(engine, model, post) == "chunk-driver"
