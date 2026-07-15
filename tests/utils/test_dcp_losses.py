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

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import verl.models.mcore.model_forward_fused as fused_forward_module
import verl.trainer.distillation.losses as distillation_losses_module
import verl.trainer.distillation.megatron.losses as megatron_distillation_losses
import verl.trainer.ppo.core_algos as core_algos_module
import verl.workers.engine.megatron.losses as megatron_loss_module
import verl.workers.engine.megatron.transformer_impl as transformer_module
from verl.trainer.distillation.losses import (
    DISTILLATION_LOSS_REGISTRY,
    DISTILLATION_SETTINGS_REGISTRY,
    DistillationLossSettings,
    distillation_ppo_loss,
    register_distillation_loss,
)
from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.engine.megatron.losses import (
    _prepare_dcp_loss_function,
    call_megatron_loss,
    validate_dcp_loss_normalization,
    validate_dcp_policy_loss,
)
from verl.workers.engine.megatron.transformer_impl import (
    MegatronEngineWithLMHead,
    _aggregate_dcp_loss_for_logging,
    _aggregate_dcp_metrics_for_logging,
    _apply_dcp_local_token_mask_for_loss,
    _normalize_temperature_for_thd,
    _prepare_dcp_temperature,
    _synchronize_dcp_metric_error,
    _validate_dcp_multi_modal_inputs,
)
from verl.workers.utils.losses import ppo_loss, sft_loss, value_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def _nested(parts):
    return torch.nested.as_nested_tensor(parts, layout=torch.jagged)


def _mark_dcp(data: TensorDict) -> TensorDict:
    tu.assign_non_tensor(data, _dcp_scheduled=True)
    return data


def test_engine_applies_dcp_local_mask_before_value_loss():
    data = TensorDict(
        {
            "values": torch.zeros(1, 3),
            "returns": torch.zeros(1, 3),
            "response_mask": torch.ones(1, 3, dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=3,
        global_batch_size=1,
        _dcp_scheduled=True,
    )
    model_output = {
        "values": _nested([torch.tensor([0.0, 10.0, 100.0, 1000.0, 10000.0])]),
        "_dcp_local_token_mask": _nested([torch.tensor([False, True, False, True, False])]),
    }

    _apply_dcp_local_token_mask_for_loss(model_output, data)
    config = CriticConfig(strategy="megatron", ppo_micro_batch_size_per_gpu=1)
    _loss, metrics = call_megatron_loss(partial(value_loss, config=config), model_output, data)

    assert metrics["critic/vpred_mean"].aggregate() == pytest.approx(505.0)
    assert "_dcp_local_token_mask" not in model_output


def test_engine_applies_dcp_local_mask_before_sft_loss_shift():
    data = TensorDict(
        {
            "loss_mask": _nested([torch.tensor([False, False, True, True, True])]),
        },
        batch_size=[1],
    )
    model_output = {
        "_dcp_local_token_mask": _nested([torch.tensor([False, True, False, True, False])]),
    }

    _apply_dcp_local_token_mask_for_loss(model_output, data)

    shifted = torch.roll(data["loss_mask"].values(), shifts=-1, dims=0)
    expected = torch.tensor([False, True, False, True, False])
    torch.testing.assert_close(shifted, expected)


def test_engine_compacts_dcp_local_mask_for_forward_only_sft_loss():
    data = TensorDict(
        {
            "loss_mask": _nested([torch.tensor([True, False, True, False, False, True])]),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": _nested([torch.tensor([1.0, 2.0, 3.0])]),
        "_dcp_local_token_mask": _nested([torch.tensor([True, True, False, False, True, False])]),
    }

    _apply_dcp_local_token_mask_for_loss(model_output, data)

    assert data["loss_mask"].values().numel() == model_output["log_probs"].values().numel()
    shifted = torch.roll(data["loss_mask"].values(), shifts=-1, dims=0)
    expected = torch.tensor([False, True, True])
    torch.testing.assert_close(shifted, expected)


def test_dcp_sft_handles_dense_loss_mask_from_left_right_no_padding():
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2],
            "responses": input_ids[:, 2:],
            "attention_mask": torch.ones_like(input_ids),
            "response_mask": torch.ones(1, 3, dtype=torch.bool),
            "position_ids": torch.arange(5).unsqueeze(0),
        },
        batch_size=[1],
    )
    data = left_right_2_no_padding(data)
    # Dynamic routing intentionally keeps only model/loss inputs, so exercise
    # the response-length metadata path rather than prompts/responses fallback.
    data.pop("prompts")
    data.pop("responses")
    data["_dcp_response_lengths"] = torch.tensor([3])
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=3, _dcp_scheduled=True)
    model_output = {
        "log_probs": _nested([torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])]),
        "_dcp_local_token_mask": _nested([torch.tensor([False, True, False, True, False])]),
    }

    _apply_dcp_local_token_mask_for_loss(model_output, data)
    loss, _ = call_megatron_loss(partial(sft_loss, config=None), model_output, data)

    assert not data["loss_mask"].is_nested
    torch.testing.assert_close(data["loss_mask"], torch.tensor([[True, False, True]]))
    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_dcp_no_padding_uses_response_mask_for_response_span():
    data = TensorDict(
        {
            "response_mask": torch.ones(1, 3, dtype=torch.bool),
            "loss_mask": torch.tensor([[False, False, False, True, False]]),
        },
        batch_size=[1],
    )
    model_output = _nested([torch.tensor([0.0, 10.0, 100.0, 1000.0, 10000.0])])

    padded = no_padding_2_padding(model_output, data)

    torch.testing.assert_close(padded, torch.tensor([[10.0, 100.0, 1000.0]]))


def test_dcp_no_padding_preserves_internal_zero_response_span():
    data = TensorDict(
        {
            "_dcp_response_lengths": torch.tensor([5]),
            "response_mask": torch.tensor([[True, False, True, False, True]]),
        },
        batch_size=[1],
    )
    model_output = _nested([torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])])

    padded = no_padding_2_padding(model_output, data)

    torch.testing.assert_close(padded, torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]]))


def test_no_padding_rejects_token_count_mismatch():
    data = TensorDict(
        {
            "prompts": _nested([torch.tensor([1, 2])]),
            "responses": _nested([torch.tensor([3, 4])]),
        },
        batch_size=[1],
    )
    model_output = _nested([torch.tensor([0.0, 10.0, 20.0])])

    with pytest.raises(ValueError, match="Token count mismatch"):
        no_padding_2_padding(model_output, data)


def test_engine_slices_rollout_is_weights_to_dcp_response_span():
    data = TensorDict(
        {
            "_dcp_response_lengths": torch.tensor([3]),
            "response_mask": torch.tensor([[True, True, True, False, False]]),
            "rollout_is_weights": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        },
        batch_size=[1],
    )
    model_output = {
        "_dcp_local_token_mask": _nested([torch.tensor([False, True, True, True])]),
    }

    _apply_dcp_local_token_mask_for_loss(model_output, data)

    torch.testing.assert_close(data["rollout_is_weights"], torch.tensor([[1.0, 2.0, 3.0]]))


def test_engine_applies_local_ownership_to_nested_response_mask():
    response_mask = _nested(
        [
            torch.tensor([True, False, True]),
            torch.tensor([False, True]),
        ]
    )
    data = TensorDict(
        {
            "response_mask": response_mask,
            "old_log_probs": torch.arange(16, dtype=torch.float32).reshape(2, 8),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, max_response_len=8)
    model_output = {
        "_dcp_local_token_mask": _nested(
            [
                torch.tensor([False, True, False, True, False]),
                torch.tensor([True, False, True]),
            ]
        )
    }

    _apply_dcp_local_token_mask_for_loss(model_output, data)

    torch.testing.assert_close(data["response_token_counts"], torch.tensor([2, 1]))
    torch.testing.assert_close(
        tu.get_non_tensor_data(data, key="_dcp_local_num_tokens", default=None),
        torch.tensor(4, dtype=torch.int),
    )
    assert data["response_mask"].is_nested
    torch.testing.assert_close(
        data["response_mask"].values(),
        torch.tensor([True, False, True, False, False]),
    )
    assert data["old_log_probs"].shape == (2, 3)

    padded_log_probs = no_padding_2_padding(
        _nested(
            [
                torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0]),
                torch.tensor([50.0, 60.0, 70.0]),
            ]
        ),
        data,
    )
    padded_response_mask = data["response_mask"].to_padded_tensor(False)
    assert padded_log_probs.shape == padded_response_mask.shape == (2, 3)
    torch.testing.assert_close(
        padded_log_probs,
        torch.tensor([[10.0, 20.0, 30.0], [50.0, 60.0, 0.0]]),
    )


def test_megatron_dcp_loss_config_is_copied_per_call():
    config = SimpleNamespace(global_batch_info={"existing": 1})
    response_token_counts = torch.tensor([4])

    def fake_loss(config, model_output, data, dp_group=None):
        assert config.global_batch_info["sequence_token_counts"] is response_token_counts
        assert model_output == {"log_probs": "sentinel"}
        assert dp_group == "dp"
        return torch.tensor(2.0), {}

    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, dp_size=2, batch_num_tokens=4, global_batch_size=1)
    loss_function, dcp_config = _prepare_dcp_loss_function(
        partial(fake_loss, config=config), data, response_token_counts
    )
    loss, _metrics = loss_function(
        model_output={"log_probs": "sentinel"},
        data=data,
        dp_group="dp",
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))
    assert config.global_batch_info == {"existing": 1}
    assert dcp_config is not config
    assert dcp_config.global_batch_info == {
        "existing": 1,
        "dp_size": 2,
        "batch_num_tokens": 4,
        "global_batch_size": 1,
        "sequence_token_counts": response_token_counts,
    }


def test_megatron_dcp_distillation_config_is_copied_per_call():
    config = SimpleNamespace(global_batch_info={})
    distillation_loss_config = SimpleNamespace(global_batch_info={"persistent": 1})
    distillation_config = SimpleNamespace(distillation_loss=distillation_loss_config)
    response_token_counts = torch.tensor([2])

    def fake_loss(config, distillation_config, **_kwargs):
        distillation_config.distillation_loss.global_batch_info["micro_batch"] = 2
        return torch.tensor(0.0), {}

    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=2, global_batch_size=1)
    loss_function, _ = _prepare_dcp_loss_function(
        partial(fake_loss, config=config, distillation_config=distillation_config),
        data,
        response_token_counts,
    )

    loss_function(model_output={}, data=data)

    assert distillation_loss_config.global_batch_info == {"persistent": 1}


def test_megatron_generic_loss_path_is_unchanged_without_scheduler_marker():
    calls = []

    def custom_loss(model_output, data, dp_group=None):
        calls.append((model_output, data, dp_group))
        return torch.tensor(3.0), {"custom": True}

    data = TensorDict({"response_token_counts": torch.tensor([2])}, batch_size=[1])
    model_output = {"sentinel": True}

    loss, metrics = call_megatron_loss(custom_loss, model_output, data, dp_group="dp")

    torch.testing.assert_close(loss, torch.tensor(3.0))
    assert metrics == {"custom": True}
    assert calls == [(model_output, data, "dp")]


def test_megatron_dcp_does_not_apply_policy_gates_to_sft():
    config = SimpleNamespace(
        global_batch_info={},
        policy_loss={"loss_mode": "gspo"},
        loss_agg_mode="seq-mean-token-sum-norm",
        loss_scale_factor=None,
    )
    loss_function = partial(sft_loss, config=config)

    validate_dcp_policy_loss(loss_function)
    validate_dcp_loss_normalization(loss_function, TensorDict({}, batch_size=[]))


def test_megatron_dcp_normalized_sequence_sum_requires_global_horizon():
    config = SimpleNamespace(
        global_batch_info={},
        policy_loss={"loss_mode": "vanilla"},
        loss_agg_mode="seq-mean-token-sum-norm",
        loss_scale_factor=None,
    )
    data = _mark_dcp(TensorDict({"response_token_counts": torch.tensor([2])}, batch_size=[1]))

    with pytest.raises(ValueError, match="requires a global loss_scale_factor or max_response_len"):
        call_megatron_loss(partial(ppo_loss, config=config), model_output={}, data=data)


@pytest.mark.parametrize("loss_scale_factor", [0, -1, True, 1.5, torch.tensor([2, 3])])
def test_megatron_dcp_normalized_sequence_sum_requires_positive_integer_horizon(loss_scale_factor):
    config = SimpleNamespace(
        global_batch_info={},
        policy_loss={"loss_mode": "vanilla"},
        loss_agg_mode="seq-mean-token-sum-norm",
        loss_scale_factor=loss_scale_factor,
    )
    data = _mark_dcp(TensorDict({"response_token_counts": torch.tensor([2])}, batch_size=[1]))

    with pytest.raises(ValueError, match="positive integer scalar"):
        call_megatron_loss(partial(ppo_loss, config=config), model_output={}, data=data)


@pytest.mark.parametrize("loss_mode", ["gspo", "geo_mean", "clip_cov", "clip-cov", "kl_cov", "kl-cov"])
def test_megatron_dcp_rejects_full_sequence_policy_losses(loss_mode):
    config = SimpleNamespace(global_batch_info={}, policy_loss={"loss_mode": loss_mode})
    data = _mark_dcp(TensorDict({"response_token_counts": torch.tensor([2])}, batch_size=[1]))

    with pytest.raises(NotImplementedError, match="sequence- or batch-level statistics"):
        call_megatron_loss(partial(ppo_loss, config=config), model_output={}, data=data)


@pytest.mark.parametrize(
    "rollout_correction",
    [
        {"rollout_is": "sequence"},
        {"rollout_is": "batch"},
        {"rollout_is": "token", "rollout_is_batch_normalize": True},
        {"rollout_is": "token", "rollout_is_batch_normalize": False},
        {"rollout_is": None, "rollout_rs": "seq_mean_k1"},
    ],
)
def test_megatron_dcp_rejects_bypass_mode(rollout_correction):
    config = SimpleNamespace(
        global_batch_info={},
        policy_loss={"loss_mode": "bypass_mode", "rollout_correction": rollout_correction},
    )
    data = _mark_dcp(TensorDict({"response_token_counts": torch.tensor([2])}, batch_size=[1]))

    with pytest.raises(NotImplementedError, match="policy_loss.loss_mode='bypass_mode'"):
        call_megatron_loss(partial(ppo_loss, config=config), model_output={}, data=data)


def test_megatron_dcp_rejects_unverified_custom_policy_loss():
    config = SimpleNamespace(global_batch_info={}, policy_loss={"loss_mode": "external_custom_loss"})
    data = _mark_dcp(TensorDict({"response_token_counts": torch.tensor([2])}, batch_size=[1]))

    with pytest.raises(NotImplementedError, match="has not been verified to decompose over token shards"):
        call_megatron_loss(partial(ppo_loss, config=config), model_output={}, data=data)


def test_megatron_dcp_rejects_unknown_top_level_loss_even_in_vanilla_mode():
    config = SimpleNamespace(global_batch_info={}, policy_loss={"loss_mode": "vanilla"})

    def custom_loss(**_kwargs):
        raise AssertionError("unknown top-level loss must be rejected before execution")

    with pytest.raises(NotImplementedError, match="top-level loss callable"):
        validate_dcp_policy_loss(partial(custom_loss, config=config))


def test_megatron_dcp_skips_unused_actor_policy_for_distillation_only():
    config = SimpleNamespace(global_batch_info={}, policy_loss={"loss_mode": "gspo"})
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode="kl",
            use_task_rewards=False,
            use_policy_gradient=False,
        )
    )

    validate_dcp_policy_loss(partial(distillation_ppo_loss, config=config, distillation_config=distillation_config))


def test_megatron_dcp_rejects_custom_distillation_loss_without_opt_in():
    loss_name = "test_unverified_dcp_distillation_loss"

    @register_distillation_loss(DistillationLossSettings(names=loss_name, use_estimator=True))
    def custom_distillation_loss(*_args, **_kwargs):
        raise AssertionError("unverified distillation loss must be rejected before execution")

    config = SimpleNamespace(global_batch_info={}, policy_loss={"loss_mode": "vanilla"})
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode=loss_name,
            use_task_rewards=False,
            use_policy_gradient=False,
        )
    )
    try:
        with pytest.raises(NotImplementedError, match="has not been explicitly verified"):
            validate_dcp_policy_loss(
                partial(distillation_ppo_loss, config=config, distillation_config=distillation_config)
            )
    finally:
        DISTILLATION_LOSS_REGISTRY.pop(loss_name)
        DISTILLATION_SETTINGS_REGISTRY.pop(loss_name)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"loss_scale_factor": 7, "max_response_len": 4}, 7),
        ({"max_response_len": 4}, 4),
    ],
)
def test_dcp_critic_forwards_global_loss_scale_factor(monkeypatch, metadata, expected):
    captured = {}

    def fake_compute_value_loss(**kwargs):
        captured.update(kwargs)
        return torch.tensor(0.0), torch.tensor(0.0)

    monkeypatch.setattr(megatron_loss_module, "compute_value_loss", fake_compute_value_loss)
    config = CriticConfig(
        strategy="megatron",
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode="seq-mean-token-sum-norm",
    )
    data = TensorDict(
        {
            "values": torch.zeros(1, 4),
            "returns": torch.zeros(1, 4),
            "response_mask": torch.ones(1, 4, dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=4, global_batch_size=1, _dcp_scheduled=True, **metadata)
    model_output = {
        "values": _nested([torch.arange(5, dtype=torch.float32)]),
        "_dcp_local_token_mask": _nested([torch.tensor([False, True, False, True, False])]),
    }
    _apply_dcp_local_token_mask_for_loss(model_output, data)

    call_megatron_loss(partial(value_loss, config=config), model_output=model_output, data=data)

    assert captured["loss_scale_factor"] == expected


def test_megatron_topk_distillation_uses_routed_local_cp_size(monkeypatch):
    preprocess_cp_sizes = []

    def fake_preprocess(value, *, pre_process, local_cp_size=None, **_kwargs):
        assert pre_process
        preprocess_cp_sizes.append(local_cp_size)
        dense = value.values().reshape(1, -1, value.values().shape[-1])[:, :2]
        return dense, SimpleNamespace(), None

    def fake_vocab_parallel_kl(student_logits, teacher_logprobs, teacher_ids, _clamp):
        assert teacher_logprobs.shape[:2] == teacher_ids.shape[:2] == student_logits.shape[:2]
        output = torch.zeros(student_logits.shape[:2], dtype=student_logits.dtype)
        return output, output, output, output, output

    monkeypatch.setattr(megatron_distillation_losses, "preprocess_thd_engine", fake_preprocess)
    monkeypatch.setattr(megatron_distillation_losses._VocabParallelKLDivergence, "apply", fake_vocab_parallel_kl)

    data = TensorDict(
        {
            "teacher_logprobs": _nested([torch.zeros(4, 2)]),
            "teacher_ids": _nested([torch.zeros(4, 2, dtype=torch.long)]),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor_data(data, "local_cp_size", 2)
    config = SimpleNamespace(strategy="megatron")
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=None))

    output = distillation_losses_module.compute_topk_loss(
        config=config,
        distillation_config=distillation_config,
        data=data,
        student_logits=torch.zeros(1, 2, 4),
        data_format="thd",
    )

    assert preprocess_cp_sizes == [2, 2]
    assert set(output) == {
        "distillation_losses",
        "student_mass",
        "teacher_mass",
        "overlap_count",
        "overlap_token_advantage",
    }


def test_megatron_topk_distillation_forwards_fp8_padding_to_teacher(monkeypatch):
    captured_fp8_flags = []

    def fake_preprocess(value, *, pre_process, use_fp8_padding=False, **_kwargs):
        assert pre_process
        captured_fp8_flags.append(use_fp8_padding)
        dense = value.values().reshape(1, -1, value.values().shape[-1])[:, :2]
        return dense, SimpleNamespace(), None

    def fake_vocab_parallel_kl(student_logits, teacher_logprobs, teacher_ids, _clamp):
        output = torch.zeros(student_logits.shape[:2], dtype=student_logits.dtype)
        return output, output, output, output, output

    monkeypatch.setattr(megatron_distillation_losses, "preprocess_thd_engine", fake_preprocess)
    monkeypatch.setattr(megatron_distillation_losses._VocabParallelKLDivergence, "apply", fake_vocab_parallel_kl)

    data = TensorDict(
        {
            "teacher_logprobs": _nested([torch.zeros(4, 2)]),
            "teacher_ids": _nested([torch.zeros(4, 2, dtype=torch.long)]),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor_data(data, "_distillation_use_fp8_padding", True)

    distillation_losses_module.compute_topk_loss(
        config=SimpleNamespace(strategy="megatron"),
        distillation_config=SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=None)),
        data=data,
        student_logits=torch.zeros(1, 2, 4),
        data_format="thd",
    )

    # The teacher THD stream must be padded exactly like the FP8-padded student.
    assert captured_fp8_flags == [True, True]


def test_megatron_topk_distillation_rejects_dynamic_cp_bshd():
    teacher_logprobs = _nested([torch.zeros(4, 2)])
    teacher_ids = _nested([torch.zeros(4, 2, dtype=torch.long)])

    with pytest.raises(NotImplementedError, match="only supports THD"):
        megatron_distillation_losses.compute_forward_kl_topk(
            student_logits=torch.zeros(1, 2, 4),
            teacher_topk_log_probs=teacher_logprobs,
            teacher_topk_ids=teacher_ids,
            config=SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=None)),
            data_format="bshd",
            local_cp_size=2,
        )


def test_empty_local_distillation_range_is_finite_before_dcp_reduce():
    metrics = distillation_losses_module.compute_distillation_loss_range(
        distillation_losses=torch.zeros(1, 3),
        response_mask=torch.zeros(1, 3, dtype=torch.bool),
        dcp_scheduled=True,
    )

    assert metrics["distillation/loss_min"].aggregation == AggregationType.MIN
    assert metrics["distillation/loss_min"].values == [0.0]
    assert metrics["distillation/loss_max"].aggregation == AggregationType.MAX
    assert metrics["distillation/loss_max"].values == [0.0]


def test_empty_local_topk_distillation_metrics_are_dcp_safe():
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "response_mask": torch.zeros(1, 2, dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, _dcp_scheduled=True)
    model_output = {
        key: _nested([torch.zeros(3)])
        for key in [
            "distillation_losses",
            "student_mass",
            "teacher_mass",
            "overlap_count",
            "overlap_token_advantage",
        ]
    }

    _, metrics = distillation_losses_module.compute_forward_kl_topk(
        config=SimpleNamespace(),
        distillation_config=SimpleNamespace(distillation_loss=SimpleNamespace(topk=2)),
        model_output=model_output,
        data=data,
    )

    for key in [
        "distillation/student_mass",
        "distillation/teacher_mass",
        "distillation/overlap_ratio",
        "distillation/overlap_token_advantage",
    ]:
        assert metrics[key].aggregation == AggregationType.MEAN
        assert metrics[key].values == [0.0]
        assert metrics[key].dcp_weight == 0.0
    for key in ["distillation/student_mass_min", "distillation/teacher_mass_min"]:
        assert metrics[key].aggregation == AggregationType.MIN
        assert metrics[key].values == [0.0]
    for key in ["distillation/student_mass_max", "distillation/teacher_mass_max"]:
        assert metrics[key].aggregation == AggregationType.MAX
        assert metrics[key].values == [0.0]


def test_overlap_advantage_preserves_conditional_mean_and_weight():
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "response_mask": torch.ones(1, 2, dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, _dcp_scheduled=True)
    model_output = {
        "distillation_losses": _nested([torch.zeros(3)]),
        "student_mass": _nested([torch.ones(3)]),
        "teacher_mass": _nested([torch.ones(3)]),
        "overlap_count": _nested([torch.tensor([1.0, 0.0, 0.0])]),
        "overlap_token_advantage": _nested([torch.tensor([-0.4, 0.0, 0.0])]),
    }

    _, metrics = distillation_losses_module.compute_forward_kl_topk(
        config=SimpleNamespace(),
        distillation_config=SimpleNamespace(distillation_loss=SimpleNamespace(topk=2)),
        model_output=model_output,
        data=data,
    )

    overlap_ratio = metrics["distillation/overlap_ratio"]
    overlap_advantage = metrics["distillation/overlap_token_advantage"]
    assert overlap_ratio.aggregate() == pytest.approx(0.25)
    assert overlap_ratio.dcp_weight == 2.0
    assert overlap_advantage.aggregate() == pytest.approx(-0.4)
    assert overlap_advantage.dcp_weight == 1.0


def test_topk_distillation_metrics_keep_legacy_types_without_dcp():
    """Without the DCP marker the generic path must keep the pre-DCP metric forms."""
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "response_mask": torch.ones(1, 2, dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "distillation_losses": _nested([torch.zeros(3)]),
        "student_mass": _nested([torch.ones(3)]),
        "teacher_mass": _nested([torch.ones(3)]),
        "overlap_count": _nested([torch.tensor([1.0, 0.0, 0.0])]),
        "overlap_token_advantage": _nested([torch.tensor([-0.4, 0.0, 0.0])]),
    }

    _, metrics = distillation_losses_module.compute_forward_kl_topk(
        config=SimpleNamespace(),
        distillation_config=SimpleNamespace(distillation_loss=SimpleNamespace(topk=2)),
        model_output=model_output,
        data=data,
    )

    assert isinstance(metrics["distillation/overlap_ratio"], float)
    assert isinstance(metrics["distillation/overlap_token_advantage"], float)
    assert isinstance(metrics["distillation/student_mass"], float)
    assert isinstance(metrics["distillation/teacher_mass"], float)
    for key in ["distillation/student_mass_min", "distillation/teacher_mass_max"]:
        assert not hasattr(metrics[key], "dcp_weight")


@pytest.mark.parametrize(
    "loss_agg_mode",
    [
        "token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-sum-norm",
        "seq-mean-token-mean",
    ],
)
def test_dcp_distillation_only_initializes_global_denominators_before_loss(monkeypatch, loss_agg_mode):
    loss_mat = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    local_mask = torch.tensor([[True, False, False, True]])
    sequence_token_counts = torch.tensor([4])
    config = SimpleNamespace(
        global_batch_info={"sequence_token_counts": sequence_token_counts},
        loss_agg_mode=loss_agg_mode,
        loss_scale_factor=None,
    )
    loss_config = SimpleNamespace(
        loss_mode="kl",
        loss_max_clamp=None,
        use_policy_gradient=False,
        use_task_rewards=False,
        distillation_loss_coef=1.0,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)
    data = TensorDict(
        {
            "response_mask": local_mask,
            "response_token_counts": sequence_token_counts,
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        data,
        dp_size=2,
        batch_num_tokens=4,
        global_batch_size=1,
        max_response_len=4,
        _dcp_scheduled=True,
    )

    seen_configs = []

    def fake_distillation_loss(**kwargs):
        dcp_config = kwargs["config"]
        seen_configs.append(dcp_config)
        expected_scale = 4 if loss_agg_mode == "seq-mean-token-sum-norm" else None
        expected_global_batch_info = {
            "sequence_token_counts": sequence_token_counts,
            "dp_size": 2,
            "batch_num_tokens": 4,
            "global_batch_size": 1,
        }
        if expected_scale is not None:
            expected_global_batch_info["loss_scale_factor"] = expected_scale
        assert dcp_config is not config
        assert dcp_config.global_batch_info == expected_global_batch_info
        return loss_mat, {}

    monkeypatch.setattr(
        distillation_losses_module,
        "get_distillation_loss_fn",
        lambda _loss_mode: fake_distillation_loss,
    )

    loss, _metrics = call_megatron_loss(
        partial(
            distillation_losses_module.distillation_ppo_loss,
            config=config,
            distillation_config=distillation_config,
        ),
        model_output={},
        data=data,
    )

    expected_scale = 4 if loss_agg_mode == "seq-mean-token-sum-norm" else None
    expected = agg_loss(
        loss_mat,
        local_mask,
        loss_agg_mode,
        dp_size=2,
        batch_num_tokens=4,
        global_batch_size=1,
        loss_scale_factor=expected_scale,
        sequence_token_counts=sequence_token_counts,
    )
    torch.testing.assert_close(loss, expected)
    assert len(seen_configs) == 1
    assert config.global_batch_info == {"sequence_token_counts": sequence_token_counts}


def test_dcp_logging_loss_reconstructs_static_global_loss(monkeypatch):
    monkeypatch.setattr(transformer_module, "get_device_id", lambda: "cpu")

    def fake_all_reduce(value, op, group):
        assert op == torch.distributed.ReduceOp.SUM
        assert group == "dp-cp"
        value.mul_(4)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    result = _aggregate_dcp_loss_for_logging([0.5, 1.0], loss_normalization_world_size=2, dcp_group="dp-cp")

    assert result == [3.0]


def test_dcp_logging_metrics_aggregate_shards_by_metric_semantics(monkeypatch):
    monkeypatch.setattr(transformer_module, "get_device_id", lambda: "cpu")
    dcp_group = SimpleNamespace(size=lambda: 2)
    outputs = [
        {
            "_dcp_metric_weight": 2,
            "metrics": {
                "sum": Metric(value=2.0, aggregation=AggregationType.SUM),
                "mean": Metric(value=4.0, aggregation=AggregationType.MEAN),
            },
        },
        {
            "_dcp_metric_weight": 1,
            "metrics": {
                "sum": Metric(value=3.0, aggregation=AggregationType.SUM),
                "mean": Metric(value=10.0, aggregation=AggregationType.MEAN),
            },
        },
    ]

    def fake_all_gather(signatures, signature, group):
        assert group is dcp_group
        assert signature.shape == (2,)
        assert signature.dtype == torch.int64
        for peer in signatures:
            peer.copy_(signature)

    def fake_all_reduce(value, op, group):
        assert group is dcp_group
        assert op == torch.distributed.ReduceOp.SUM
        if value.dim() == 1:
            value.mul_(4)
        else:
            # The local buffer must contain the weight-packed rows
            # (values * weights, weights); a reduce that ignored the weights
            # would ship (values, weights) and slip through a hardcoded result.
            torch.testing.assert_close(value, value.new_tensor([[8.0, 10.0], [2.0, 1.0]]))
            value.add_(value.new_tensor([[27.0, 10.0], [3.0, 3.0]]))

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    _aggregate_dcp_metrics_for_logging(outputs, loss_normalization_world_size=2, dcp_group=dcp_group)

    assert "_dcp_metric_weight" not in outputs[0]
    assert "_dcp_metric_weight" not in outputs[1]
    assert outputs[0]["metrics"]["sum"].values == [4.0]
    assert outputs[1]["metrics"]["sum"].values == [6.0]
    assert outputs[0]["metrics"]["mean"].values == [7.0]
    assert outputs[1]["metrics"]["mean"].values == [5.0]


def test_dcp_logging_metric_schema_mismatch_fails_before_reduce(monkeypatch):
    monkeypatch.setattr(transformer_module, "get_device_id", lambda: "cpu")
    dcp_group = SimpleNamespace(size=lambda: 2)
    outputs = [
        {
            "_dcp_metric_weight": 1,
            "metrics": {"mean": Metric(value=4.0, aggregation=AggregationType.MEAN)},
        }
    ]
    reduce_called = False

    def fake_all_gather(signatures, signature, group):
        assert group is dcp_group
        assert all(peer.shape == (2,) for peer in signatures)
        signatures[0].copy_(signature)
        signatures[1].copy_(signature)
        signatures[1][1] ^= 1

    def fake_all_reduce(*_args, **_kwargs):
        nonlocal reduce_called
        reduce_called = True

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    error = _aggregate_dcp_metrics_for_logging(outputs, loss_normalization_world_size=1, dcp_group=dcp_group)

    assert "metric schema differs across ranks" in error
    assert not reduce_called


def test_dcp_metric_error_is_propagated_to_other_pipeline_stages(monkeypatch):
    monkeypatch.setattr(transformer_module, "get_device_id", lambda: "cpu")

    def fake_all_reduce(error_flag, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        assert group == "tp-pp"
        error_flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    with pytest.raises(RuntimeError, match="another pipeline stage"):
        _synchronize_dcp_metric_error(None, "tp-pp")


def test_sequence_token_counts_reconstruct_global_sequence_mean():
    losses = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    full_mask = torch.ones_like(losses, dtype=torch.bool)
    rank0_mask = torch.tensor([[True, False, False, True]])
    rank1_mask = torch.tensor([[False, True, True, False]])
    token_counts = torch.tensor([4])

    full_loss = agg_loss(losses, full_mask, "seq-mean-token-mean", global_batch_size=1)
    rank0_loss = agg_loss(
        losses,
        rank0_mask,
        "seq-mean-token-mean",
        global_batch_size=1,
        sequence_token_counts=token_counts,
    )
    rank1_loss = agg_loss(
        losses,
        rank1_mask,
        "seq-mean-token-mean",
        global_batch_size=1,
        sequence_token_counts=token_counts,
    )

    torch.testing.assert_close(rank0_loss + rank1_loss, full_loss)


def test_agg_loss_preserves_existing_optional_positional_argument_order():
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.ones_like(losses, dtype=torch.bool)
    sequence_token_counts = torch.tensor([4])

    positional = agg_loss(
        losses,
        mask,
        "seq-mean-token-sum-norm",
        1,
        4,
        1,
        4,
        sequence_token_counts,
    )
    keyword = agg_loss(
        losses,
        mask,
        "seq-mean-token-sum-norm",
        dp_size=1,
        batch_num_tokens=4,
        global_batch_size=1,
        loss_scale_factor=4,
        sequence_token_counts=sequence_token_counts,
    )

    torch.testing.assert_close(positional, keyword)


def test_compute_value_loss_preserves_sequence_counts_positional_argument(monkeypatch):
    captured = {}

    def fake_agg_loss(*_args, sequence_token_counts=None, loss_scale_factor=None, **_kwargs):
        captured["sequence_token_counts"] = sequence_token_counts
        captured["loss_scale_factor"] = loss_scale_factor
        return torch.tensor(0.0)

    monkeypatch.setattr(core_algos_module, "agg_loss", fake_agg_loss)
    sequence_token_counts = torch.tensor([2])
    values = torch.zeros(1, 2)
    compute_value_loss(
        values,
        values,
        values,
        torch.ones_like(values, dtype=torch.bool),
        0.2,
        "token-mean",
        1,
        None,
        None,
        None,
        sequence_token_counts,
    )

    assert captured["sequence_token_counts"] is sequence_token_counts
    assert captured["loss_scale_factor"] is None


@pytest.mark.parametrize(
    "loss_agg_mode",
    [
        "token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-sum-norm",
        "seq-mean-token-mean",
    ],
)
def test_dcp_per_token_gradient_normalization_matches_static_loss(loss_agg_mode):
    parameter = torch.tensor(2.0, requires_grad=True)
    coefficients = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    full_mask = torch.ones_like(coefficients, dtype=torch.bool)
    local_masks = [
        torch.tensor([[True, False, False, True]]),
        torch.tensor([[False, True, True, False]]),
    ]
    common = {
        "batch_num_tokens": 4,
        "global_batch_size": 1,
        "loss_scale_factor": 4,
        "sequence_token_counts": torch.tensor([4]),
    }

    full_loss = agg_loss(parameter * coefficients, full_mask, loss_agg_mode, dp_size=1, **common)
    full_grad = torch.autograd.grad(full_loss, parameter, retain_graph=True)[0]
    local_grads = []
    local_token_counts = []
    for local_mask in local_masks:
        local_loss = agg_loss(parameter * coefficients, local_mask, loss_agg_mode, dp_size=1, **common)
        local_sum = local_loss * coefficients.numel()
        local_grads.append(torch.autograd.grad(local_sum, parameter, retain_graph=True)[0])
        local_token_counts.append(local_mask.sum())

    # Megatron SUM-reduces DDP gradients, then finalize_model_grads divides by
    # the all-reduced exact local ownership counts returned by each callback.
    dcp_grad = torch.stack(local_grads).sum() / torch.stack(local_token_counts).sum()
    torch.testing.assert_close(dcp_grad, full_grad)


def test_dcp_per_token_callback_uses_exact_local_token_count(monkeypatch):
    parameter = torch.tensor(2.0, requires_grad=True)
    data = TensorDict({"input_ids": _nested([torch.arange(7)])}, batch_size=[1])
    tu.assign_non_tensor(
        data,
        num_micro_batch=1,
        _dcp_scheduled=True,
        _dcp_local_num_tokens=torch.tensor(1, dtype=torch.int),
        routed_num_tokens=7,
        dp_size=1,
    )
    engine = SimpleNamespace(
        tf_config=SimpleNamespace(calculate_per_token_loss=True),
        engine_config=SimpleNamespace(use_remove_padding=True, context_parallel_size=4),
        prepare_model_outputs=lambda output, _data: output,
        get_data_parallel_group=lambda: "dp",
    )

    monkeypatch.setattr(
        transformer_module,
        "call_megatron_loss",
        lambda _loss_function, model_output, **_kwargs: (model_output["task_loss"], {}),
    )

    loss_sum, local_num_tokens, _ = MegatronEngineWithLMHead.postprocess_micro_batch_func(
        engine,
        {"task_loss": parameter * 3},
        data,
        forward_only=False,
        loss_function=lambda **_kwargs: None,
        local_cp_size=4,
        dcp_local_output_only=True,
    )

    torch.testing.assert_close(loss_sum, parameter * 21)
    torch.testing.assert_close(local_num_tokens, torch.tensor(1, dtype=torch.int))
    torch.testing.assert_close(torch.autograd.grad(loss_sum, parameter)[0], torch.tensor(21.0))


def test_dcp_critic_shards_reconstruct_static_value_loss():
    config = CriticConfig(
        strategy="megatron",
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode="seq-mean-token-mean",
    )
    values = _nested([torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])])
    static_data = TensorDict(
        {
            "values": torch.zeros(1, 4),
            # Non-zero returns give every response position a non-zero loss
            # contribution, so a dropped position cannot pass unnoticed.
            "returns": torch.full((1, 4), 0.5),
            "response_mask": torch.ones(1, 4, dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(static_data, dp_size=1, batch_num_tokens=4, global_batch_size=1)
    static_loss, _ = value_loss(config, {"values": values}, static_data)

    # Ownership masks index model-output (predicting) positions, which the
    # padding helper slices as [0, 4) for this 5-token sequence. The two shards
    # tile that window as {0, 3} and {1, 2}.
    local_masks = [
        torch.tensor([True, False, False, True, False]),
        torch.tensor([False, True, True, False, False]),
    ]
    local_losses = []
    for local_mask in local_masks:
        data = TensorDict(
            {
                "values": torch.zeros(1, 4),
                "returns": torch.full((1, 4), 0.5),
                "response_mask": torch.ones(1, 4, dtype=torch.bool),
            },
            batch_size=[1],
        )
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=4, global_batch_size=1, _dcp_scheduled=True)
        model_output = {"values": values, "_dcp_local_token_mask": _nested([local_mask])}
        _apply_dcp_local_token_mask_for_loss(model_output, data)
        local_loss, _ = call_megatron_loss(partial(value_loss, config=config), model_output, data)
        local_losses.append(local_loss)

    torch.testing.assert_close(sum(local_losses), static_loss)


@pytest.mark.parametrize("loss_agg_mode", ["token-mean", "seq-mean-token-mean"])
def test_dcp_actor_shards_reconstruct_static_ppo_loss(loss_agg_mode):
    config = ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode=loss_agg_mode,
    )
    log_probs = _nested([torch.tensor([0.05, -0.10, 0.20, -0.30, 0.15])])

    def _make_data():
        return TensorDict(
            {
                "response_mask": torch.ones(1, 4, dtype=torch.bool),
                "old_log_probs": torch.tensor([[0.10, -0.20, 0.05, -0.15]]),
                "advantages": torch.tensor([[1.0, -0.5, 0.25, 2.0]]),
            },
            batch_size=[1],
        )

    static_data = _make_data()
    tu.assign_non_tensor(static_data, dp_size=1, batch_num_tokens=4, global_batch_size=1)
    static_loss, _ = ppo_loss(config, {"log_probs": log_probs}, static_data)

    # Ownership masks index model-output (predicting) positions: the response
    # span of a length-5 sequence with a 4-token response window is sliced as
    # positions [0, 4). The two shards tile that window as {0, 3} and {1, 2}.
    local_masks = [
        torch.tensor([True, False, False, True, False]),
        torch.tensor([False, True, True, False, False]),
    ]
    local_losses = []
    for local_mask in local_masks:
        data = _make_data()
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=4, global_batch_size=1, _dcp_scheduled=True)
        model_output = {"log_probs": log_probs, "_dcp_local_token_mask": _nested([local_mask])}
        _apply_dcp_local_token_mask_for_loss(model_output, data)
        local_loss, _ = call_megatron_loss(partial(ppo_loss, config=config), model_output, data)
        local_losses.append(local_loss)

    torch.testing.assert_close(sum(local_losses), static_loss)


def test_temperature_is_normalized_once_for_fused_and_non_fused_paths():
    input_ids = _nested([torch.arange(3), torch.arange(2)])

    nested_temperature, fused_temperature = _normalize_temperature_for_thd(0.7, input_ids)
    torch.testing.assert_close(nested_temperature.values(), torch.full((5,), 0.7))
    assert fused_temperature == pytest.approx(0.7)

    nested_temperature, fused_temperature = _normalize_temperature_for_thd(torch.tensor([[0.5], [1.0]]), input_ids)
    torch.testing.assert_close(nested_temperature.values(), torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0]))
    assert fused_temperature is None

    wrapped = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(wrapped, temperature=0.25)
    nested_temperature, fused_temperature = _normalize_temperature_for_thd(wrapped["temperature"], input_ids)
    torch.testing.assert_close(nested_temperature.values(), torch.full((5,), 0.25))
    assert fused_temperature == pytest.approx(0.25)


def test_temperature_rejects_non_per_sample_shape():
    input_ids = _nested([torch.arange(3), torch.arange(2)])

    with pytest.raises(ValueError, match="one temperature per sample"):
        _normalize_temperature_for_thd(torch.ones(3), input_ids)


@pytest.mark.parametrize(
    "temperature",
    [0.0, -0.1, float("nan"), float("inf"), torch.tensor([1.0, 0.0])],
)
def test_temperature_rejects_non_positive_or_non_finite_values(temperature):
    input_ids = _nested([torch.arange(3), torch.arange(2)])

    with pytest.raises(ValueError, match="strictly positive and finite"):
        _normalize_temperature_for_thd(temperature, input_ids)


def test_temperature_legacy_path_clamps_non_positive_and_requires_presence():
    input_ids = _nested([torch.arange(3), torch.arange(2)])

    nested_temperature, fused_temperature = _normalize_temperature_for_thd(
        torch.tensor([0.5, 0.0]), input_ids, strict=False
    )
    torch.testing.assert_close(nested_temperature.values(), torch.tensor([0.5, 0.5, 0.5, 1e-8, 1e-8]))
    assert fused_temperature is None

    with pytest.raises(ValueError, match="requires a 'temperature' entry"):
        _normalize_temperature_for_thd(None, input_ids, strict=False)

    with pytest.raises(ValueError, match="strictly positive and finite"):
        _normalize_temperature_for_thd(float("nan"), input_ids, strict=False)


@pytest.mark.parametrize(
    "temperature",
    [0.0, float("nan"), float("inf"), torch.tensor([0.5, 0.0]), torch.tensor([0.5, float("nan")])],
)
def test_dcp_rejects_invalid_temperature_before_scheduling(temperature):
    batch_size = temperature.numel() if isinstance(temperature, torch.Tensor) and temperature.ndim > 0 else 2
    data = TensorDict({}, batch_size=[batch_size])
    tu.assign_non_tensor(data, temperature=temperature)

    with pytest.raises(ValueError, match="strictly positive and finite"):
        _prepare_dcp_temperature(data)


@pytest.mark.parametrize("temperatures", [torch.tensor([0.5]), torch.empty(0)])
def test_dcp_routes_singleton_and_empty_tensor_temperatures(temperatures):
    data = TensorDict({}, batch_size=[temperatures.numel()])
    tu.assign_non_tensor(data, temperature=temperatures)

    scalar_temperature = _prepare_dcp_temperature(data)

    assert scalar_temperature is None
    assert isinstance(data["temperature"], torch.Tensor)
    assert data["temperature"].shape == temperatures.shape
    torch.testing.assert_close(data["temperature"], temperatures)


@pytest.mark.parametrize("temperature", [0.5, torch.tensor(0.5)])
def test_dcp_keeps_python_and_zero_dimensional_temperature_as_metadata(temperature):
    data = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor(data, temperature=temperature)

    scalar_temperature = _prepare_dcp_temperature(data)

    # Zero-dimensional tensors are stored back as Python floats so the metadata
    # never enters the routed schema, whose scalar fields are one value per sample.
    assert isinstance(scalar_temperature, float)
    assert scalar_temperature == 0.5
    stored = tu.get_non_tensor_data(data, key="temperature", default=None)
    assert isinstance(stored, float)
    assert stored == 0.5


def test_dcp_rejects_mixed_temperature_routing_classification(monkeypatch):
    class FakeGroup:
        def size(self):
            return 2

    data = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor(data, temperature=torch.tensor([0.5]))
    monkeypatch.setattr(transformer_module, "get_device_id", lambda: torch.device("cpu"))

    def fake_all_gather(states, state, *, group):
        assert isinstance(group, FakeGroup)
        states[0].copy_(state)
        states[1].copy_(state)
        states[1][1] = 0

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    with pytest.raises(ValueError, match="mixed routing classifications"):
        _prepare_dcp_temperature(data, dcp_group=FakeGroup())


def test_dcp_rejects_unequal_replicated_scalar_temperatures(monkeypatch):
    class FakeGroup:
        def size(self):
            return 2

    data = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor(data, temperature=0.5)
    monkeypatch.setattr(transformer_module, "get_device_id", lambda: torch.device("cpu"))

    def fake_all_gather(states, state, *, group):
        assert isinstance(group, FakeGroup)
        states[0].copy_(state)
        states[1].copy_(state)
        states[1][3] = 0.75

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    with pytest.raises(ValueError, match="same value on every rank"):
        _prepare_dcp_temperature(data, dcp_group=FakeGroup())


def test_dcp_rejects_multimodal_batches(monkeypatch):
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, multi_modal_inputs=[{"pixel_values": torch.ones(1)}])
    monkeypatch.setattr(transformer_module, "extract_multi_modal_inputs", lambda _inputs: {"pixel_values": object()})

    with pytest.raises(NotImplementedError, match="does not yet support multi_modal_inputs"):
        _validate_dcp_multi_modal_inputs(data)


def test_dcp_multimodal_rejection_is_propagated_from_peer_rank(monkeypatch):
    data = TensorDict({}, batch_size=[0])
    dcp_group = object()
    monkeypatch.setattr(transformer_module, "get_device_id", lambda: torch.device("cpu"))

    def fake_all_reduce(flag, *, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is dcp_group
        flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    with pytest.raises(NotImplementedError, match="does not yet support multi_modal_inputs"):
        _validate_dcp_multi_modal_inputs(data, dcp_group=dcp_group)


def test_fused_forward_propagates_dynamic_cp_metadata(monkeypatch):
    packed_seq_params = SimpleNamespace()
    preprocess_local_cp_sizes = []
    postprocess_calls = []

    def fake_preprocess(value, *, pre_process, need_roll=False, use_fp8_padding=False, local_cp_size=None, **_kwargs):
        preprocess_local_cp_sizes.append(local_cp_size)
        return torch.tensor([[1, 2]], dtype=value.dtype), packed_seq_params, None

    def fake_local_postprocess(
        output,
        _packed_seq_params,
        _input_ids,
        _batch_size,
        *,
        post_process,
        local_cp_size,
        compact,
    ):
        postprocess_calls.append((local_cp_size, compact))
        return _nested([output.reshape(-1)])

    monkeypatch.setattr(fused_forward_module, "preprocess_thd_engine", fake_preprocess)
    monkeypatch.setattr(fused_forward_module, "postprocess_thd_engine_local", fake_local_postprocess)
    monkeypatch.setattr(
        fused_forward_module,
        "build_thd_local_token_indices",
        lambda *_args, **_kwargs: _nested([torch.tensor([0, 1])]),
    )
    monkeypatch.setattr(
        fused_forward_module,
        "build_thd_full_seq_lens",
        lambda *_args, **_kwargs: _nested([torch.tensor([2])]),
    )
    monkeypatch.setattr(
        fused_forward_module,
        "build_thd_local_token_mask",
        lambda *_args, **_kwargs: _nested([torch.tensor([True, True])]),
    )

    class FakeModel:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __call__(self, **_kwargs):
            return SimpleNamespace(
                log_probs=torch.tensor([[1.0, 2.0]]),
                entropy=torch.tensor([[3.0, 4.0]]),
            )

    forward = fused_forward_module.fused_forward_model_engine()
    output = forward(
        model=FakeModel(),
        input_ids=_nested([torch.tensor([1, 2])]),
        labels=_nested([torch.tensor([1, 2])]),
        multi_modal_inputs={},
        temperature=1.0,
        calculate_entropy=True,
        pad_token_id=0,
        local_cp_size=2,
        return_dcp_local_token_mask=True,
        dcp_local_output_only=True,
        dcp_compact_output_only=True,
    )

    assert preprocess_local_cp_sizes == [2, 2]
    assert postprocess_calls == [(2, True), (2, True)]
    assert set(output) == {
        "log_probs",
        "entropy",
        "_dcp_local_token_indices",
        "_dcp_full_seq_lens",
        "_dcp_local_token_mask",
    }
