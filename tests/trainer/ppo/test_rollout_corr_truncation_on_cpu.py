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
"""CPU coverage for the truncation guard of rollout correction (issue #8020).

With a truncated sampler (top_p/top_k/min_p) and ``logprobs_mode='processed_logprobs'``,
the rollout returns log-probs renormalized over the truncated support while the actor's
old_log_probs are full-vocabulary, so the correction ratio carries the truncated mass
and k1-style bounds reject every sequence. Covered here: the config-time guard and the
runtime all-rejected warning.
"""

import dataclasses
import logging
import math

import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.rollout_corr_helper as rollout_corr_helper
from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask
from verl.utils.config import _validate_rollout_correction_truncation

# decoupled_seq_is_rs defaults: sequence-level IS + seq_sum_k1 rejection
_SEQ_IS_RS = {
    "rollout_is": "sequence",
    "rollout_is_threshold": 2.0,
    "rollout_rs": "seq_sum_k1",
    "rollout_rs_threshold": "0.5_2.0",
}


def _config(rollout_correction=_SEQ_IS_RS, actor_rollout_correction=False, **rollout_overrides):
    rollout = {"top_p": 1.0, "top_k": -1, "logprobs_mode": "processed_logprobs"}
    rollout.update(rollout_overrides)
    actor_policy_loss = {}
    if actor_rollout_correction is not False:
        actor_policy_loss["rollout_correction"] = actor_rollout_correction
    return OmegaConf.create(
        {
            "algorithm": {"rollout_correction": rollout_correction},
            "actor_rollout_ref": {"actor": {"policy_loss": actor_policy_loss}, "rollout": rollout},
        }
    )


def test_guard_accepts_untruncated_sampler():
    _validate_rollout_correction_truncation(_config())


def test_guard_accepts_truncation_with_raw_logprobs():
    _validate_rollout_correction_truncation(_config(top_p=0.8, logprobs_mode="raw_logprobs"))


def test_guard_skips_when_correction_disabled():
    config = _config(rollout_correction=None, actor_rollout_correction=False, top_p=0.8)
    _validate_rollout_correction_truncation(config)


@pytest.mark.parametrize(
    "rollout_overrides",
    [dict(top_p=0.8), dict(top_k=50), dict(min_p=0.05), dict(top_p=0.95, top_k=20, min_p=0.01)],
)
def test_guard_rejects_truncated_samplers(rollout_overrides):
    with pytest.raises(ValueError, match="rollout_correction"):
        _validate_rollout_correction_truncation(_config(**rollout_overrides))


def test_guard_reads_the_actor_side_mirror_too():
    config = _config(rollout_correction=None, actor_rollout_correction=dict(_SEQ_IS_RS), top_p=0.8)
    with pytest.raises(ValueError, match="rollout_correction"):
        _validate_rollout_correction_truncation(config)


@pytest.mark.parametrize(
    "preset",
    [RolloutCorrectionConfig.decoupled_geo_rs(), RolloutCorrectionConfig.decoupled_seq_is_rs()],
)
def test_guard_rejects_issue_8020_presets(preset):
    # The exact presets measured in issue #8020: with top_p=0.8 both reject every sequence
    # (rollout_rs_seq_masked_fraction=1.0, grad_norm=0.0).
    config = _config(rollout_correction=dataclasses.asdict(preset), top_p=0.8)
    with pytest.raises(ValueError, match="zero gradient"):
        _validate_rollout_correction_truncation(config)


@pytest.fixture(autouse=True)
def _reset_all_rejected_warning():
    rollout_corr_helper._warned_all_rejected = False
    yield
    rollout_corr_helper._warned_all_rejected = False


def _batch(kept_mass: float, batch_size: int = 3, seq_len: int = 8):
    """On-policy log-probs where the rollout's processed log-probs omit `kept_mass` per token.

    log_ratio = old - rollout = log(kept_mass) < 0 on every valid token, exactly the
    truncation offset measured in issue #8020 (kept_mass=0.8 for top_p=0.8).
    """
    old_log_prob = torch.full((batch_size, seq_len), -1.3)
    rollout_log_prob = old_log_prob - math.log(kept_mass)
    response_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    response_mask[:, -1] = False  # some padding so masks are exercised
    return old_log_prob, rollout_log_prob, response_mask


def test_all_rejected_rs_mask_warns(caplog):
    old_log_prob, rollout_log_prob, response_mask = _batch(kept_mass=0.8)
    with caplog.at_level(logging.WARNING, logger=rollout_corr_helper.logger.name):
        _, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
            old_log_prob,
            rollout_log_prob,
            response_mask,
            rollout_rs="seq_mean_k1",  # decoupled_geo_rs: 0.999_1.001
            rollout_rs_threshold="0.999_1.001",
        )
    assert not modified_mask.any()
    assert metrics["rollout_corr/rollout_rs_seq_masked_fraction"] == 1.0
    assert any("rejected or zero-weighted" in record.getMessage() for record in caplog.records)


def test_all_zeroed_icepop_weights_warn(caplog):
    old_log_prob, rollout_log_prob, response_mask = _batch(kept_mass=0.8)
    with caplog.at_level(logging.WARNING, logger=rollout_corr_helper.logger.name):
        weights_proto, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
            old_log_prob,
            rollout_log_prob,
            response_mask,
            rollout_is="sequence",
            rollout_is_threshold="0.5_2.0",
        )
    assert weights_proto is not None
    weights = weights_proto.batch["rollout_is_weights"]
    assert not weights.any()  # 0.8**7 < 0.5 -> IcePop zeroes every sequence
    assert torch.equal(modified_mask, response_mask)  # no RS configured
    assert metrics["rollout_corr/rollout_is_mean"] < 0.01
    assert any("rejected or zero-weighted" in record.getMessage() for record in caplog.records)


def test_matching_logprobs_stay_silent_and_kept(caplog):
    old_log_prob = torch.full((3, 8), -1.3)
    response_mask = torch.ones(3, 8, dtype=torch.bool)
    response_mask[:, -1] = False
    with caplog.at_level(logging.WARNING, logger=rollout_corr_helper.logger.name):
        weights_proto, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
            old_log_prob,
            old_log_prob.clone(),
            response_mask,
            rollout_is="sequence",
            rollout_is_threshold="0.5_2.0",
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold="0.999_1.001",
        )
    assert torch.equal(modified_mask, response_mask)
    assert weights_proto is not None
    weights = weights_proto.batch["rollout_is_weights"]
    assert weights[response_mask].eq(1.0).all()  # on-policy ratio stays inside 0.5_2.0
    assert metrics["rollout_corr/rollout_rs_seq_masked_fraction"] == 0.0
    assert not any("rejected or zero-weighted" in record.getMessage() for record in caplog.records)


def test_all_rejected_warning_fires_once_per_process(caplog):
    old_log_prob, rollout_log_prob, response_mask = _batch(kept_mass=0.8)
    with caplog.at_level(logging.WARNING, logger=rollout_corr_helper.logger.name):
        compute_rollout_correction_and_rejection_mask(
            old_log_prob, rollout_log_prob, response_mask, rollout_rs="seq_mean_k1", rollout_rs_threshold="0.999_1.001"
        )
        compute_rollout_correction_and_rejection_mask(
            old_log_prob, rollout_log_prob, response_mask, rollout_rs="seq_mean_k1", rollout_rs_threshold="0.999_1.001"
        )
    warnings = [record for record in caplog.records if "rejected or zero-weighted" in record.getMessage()]
    assert len(warnings) == 1
