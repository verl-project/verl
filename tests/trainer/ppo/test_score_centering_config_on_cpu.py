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

"""CPU coverage for the score centering config surface: dataclasses, presets and the driver cross-checks."""

import pytest
from omegaconf import OmegaConf

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils.config import _validate_score_centering_config
from verl.workers.config.rollout import RolloutConfig


def test_score_centering_presets_are_bypass_reinforce():
    for cfg in (
        RolloutCorrectionConfig.bypass_pg_sc(),
        RolloutCorrectionConfig.bypass_pg_token_tis_sc(),
        RolloutCorrectionConfig.bypass_pg_token_icepop_sc(),
    ):
        assert cfg.score_centering and cfg.bypass_mode and cfg.loss_type == "reinforce"
    assert RolloutCorrectionConfig.bypass_pg_sc().rollout_is is None
    assert RolloutCorrectionConfig.bypass_pg_token_tis_sc(threshold=3.0).rollout_is_threshold == 3.0
    assert RolloutCorrectionConfig.bypass_pg_token_icepop_sc().rollout_is_threshold == "0.5_5.0"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bypass_mode=False, loss_type="reinforce"),
        dict(bypass_mode=True, loss_type="ppo_clip"),
        dict(bypass_mode=True, loss_type="reinforce", rollout_is="sequence"),
        dict(bypass_mode=True, loss_type="reinforce", rollout_is="token", rollout_is_batch_normalize=True),
    ],
)
def test_score_centering_rejects_unsupported_modes(kwargs):
    with pytest.raises(ValueError, match="score_centering"):
        RolloutCorrectionConfig(score_centering=True, **kwargs)


def test_topk_log_probs_requires_calculate_log_probs_and_full_support():
    with pytest.raises(ValueError, match="calculate_log_probs"):
        RolloutConfig(topk_log_probs=128, calculate_log_probs=False)
    with pytest.raises(ValueError, match="top_p"):
        RolloutConfig(topk_log_probs=128, calculate_log_probs=True, top_p=0.9)
    with pytest.raises(ValueError, match="top_k"):
        RolloutConfig(topk_log_probs=128, calculate_log_probs=True, top_k=50)


def test_topk_log_probs_requires_processed_logprobs():
    with pytest.raises(ValueError, match="processed_logprobs"):
        RolloutConfig(name="vllm", topk_log_probs=128, calculate_log_probs=True, logprobs_mode="raw_logprobs")


def test_topk_log_probs_rejects_negative_head_size():
    with pytest.raises(ValueError, match="must be >= 0"):
        RolloutConfig(name="vllm", topk_log_probs=-1, calculate_log_probs=True)


def test_topk_log_probs_requires_vllm_rollout():
    with pytest.raises(ValueError, match="vLLM"):
        RolloutConfig(name="sglang", topk_log_probs=128, calculate_log_probs=True)


def test_topk_log_probs_raises_vllm_max_logprobs():
    cfg = RolloutConfig(name="vllm", topk_log_probs=128, calculate_log_probs=True)
    assert cfg.engine_kwargs["vllm"]["max_logprobs"] == 128
    cfg = RolloutConfig(
        name="vllm", topk_log_probs=32, calculate_log_probs=True, engine_kwargs={"vllm": {"max_logprobs": 64}}
    )
    assert cfg.engine_kwargs["vllm"]["max_logprobs"] == 64
    with pytest.raises(ValueError, match="max_logprobs"):
        RolloutConfig(
            name="vllm", topk_log_probs=128, calculate_log_probs=True, engine_kwargs={"vllm": {"max_logprobs": 20}}
        )


_RC = {"bypass_mode": True, "loss_type": "reinforce", "rollout_is": "token", "rollout_is_threshold": 2.0}


def _config(
    algorithm_sc=True,
    actor_sc=True,
    loss_mode="bypass_mode",
    topk_log_probs=128,
    use_v1=True,
    algorithm_rc=None,
    actor_rc=None,
    strategy="fsdp",
    use_fused_kernels=False,
    distillation_enabled=False,
):
    policy_loss = {"loss_mode": loss_mode}
    if actor_sc is not None:
        policy_loss["rollout_correction"] = {**_RC, **(actor_rc or {}), "score_centering": actor_sc}
    return OmegaConf.create(
        {
            "algorithm": {"rollout_correction": {**_RC, **(algorithm_rc or {}), "score_centering": algorithm_sc}},
            "actor_rollout_ref": {
                "actor": {"strategy": strategy, "use_fused_kernels": use_fused_kernels, "policy_loss": policy_loss},
                "rollout": {"topk_log_probs": topk_log_probs},
            },
            "distillation": {"enabled": distillation_enabled},
            "trainer": {"use_v1": use_v1},
        }
    )


def test_score_centering_config_accepts_consistent_settings():
    _validate_score_centering_config(_config())


def test_score_centering_disabled_skips_checks():
    _validate_score_centering_config(
        _config(algorithm_sc=False, actor_sc=None, loss_mode="vanilla", topk_log_probs=0, strategy="megatron")
    )


@pytest.mark.parametrize(
    "algorithm_sc, actor_sc",
    [(True, None), (True, False), (False, True)],
)
def test_score_centering_requires_driver_and_actor_flags(algorithm_sc, actor_sc):
    with pytest.raises(ValueError, match="actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering"):
        _validate_score_centering_config(_config(algorithm_sc=algorithm_sc, actor_sc=actor_sc))


@pytest.mark.parametrize("strategy", ["megatron", "veomni"])
def test_score_centering_requires_fsdp_actor(strategy):
    with pytest.raises(ValueError, match="FSDP engine only"):
        _validate_score_centering_config(_config(strategy=strategy))


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_score_centering_accepts_fsdp_strategies(strategy):
    _validate_score_centering_config(_config(strategy=strategy))


def test_score_centering_rejects_fused_kernels():
    with pytest.raises(ValueError, match="actor_rollout_ref.actor.use_fused_kernels"):
        _validate_score_centering_config(_config(use_fused_kernels=True))


def test_score_centering_rejects_distillation():
    with pytest.raises(ValueError, match="distillation.enabled"):
        _validate_score_centering_config(_config(distillation_enabled=True))


def test_score_centering_tolerates_missing_distillation_section():
    config = _config()
    del config["distillation"]

    _validate_score_centering_config(config)


def test_score_centering_requires_bypass_mode_loss():
    with pytest.raises(ValueError, match="actor_rollout_ref.actor.policy_loss.loss_mode"):
        _validate_score_centering_config(_config(loss_mode="vanilla"))


def test_score_centering_requires_rollout_topk_log_probs():
    with pytest.raises(ValueError, match="actor_rollout_ref.rollout.topk_log_probs"):
        _validate_score_centering_config(_config(topk_log_probs=0))


@pytest.mark.parametrize("use_v1", [True, False])
def test_score_centering_accepts_both_trainers(use_v1):
    _validate_score_centering_config(_config(use_v1=use_v1))


def test_score_centering_tolerates_null_rollout_correction():
    config = _config(actor_sc=None)
    config.algorithm.rollout_correction = None
    config.actor_rollout_ref.actor.policy_loss.rollout_correction = None

    _validate_score_centering_config(config)


def test_score_centering_accepts_actor_side_without_is_keys():
    config = _config()
    policy_loss = config.actor_rollout_ref.actor.policy_loss
    policy_loss.rollout_correction = {"bypass_mode": True, "loss_type": "reinforce", "score_centering": True}
    config.algorithm.rollout_correction.rollout_is = None

    _validate_score_centering_config(config)


@pytest.mark.parametrize(
    "actor_rc, match",
    [
        ({"rollout_is": "sequence"}, "actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is"),
        ({"rollout_is_batch_normalize": True}, "rollout_correction.rollout_is_batch_normalize"),
    ],
)
def test_score_centering_rejects_actor_side_is_settings(actor_rc, match):
    with pytest.raises(ValueError, match=match):
        _validate_score_centering_config(_config(actor_rc=actor_rc))


@pytest.mark.parametrize(
    "key, actor_value",
    [
        ("bypass_mode", False),
        ("loss_type", "ppo_clip"),
        ("rollout_is", None),
        ("rollout_is_threshold", 3.0),
    ],
)
def test_score_centering_requires_matching_actor_side(key, actor_value):
    with pytest.raises(ValueError, match=f"rollout_correction.{key}"):
        _validate_score_centering_config(_config(actor_rc={key: actor_value}))


def test_score_centering_reads_missing_actor_side_rollout_is_as_none():
    config = _config()
    del config.actor_rollout_ref.actor.policy_loss.rollout_correction["rollout_is"]

    with pytest.raises(ValueError, match=r"actor_rollout_ref\.actor\.policy_loss\.rollout_correction\.rollout_is "):
        _validate_score_centering_config(config)
