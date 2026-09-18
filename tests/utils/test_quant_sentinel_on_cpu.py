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
"""CPU tests for the quantized-rollout sentinel."""

from types import SimpleNamespace

import pytest

from verl.utils.quant_sentinel import QuantizedRolloutSentinel, check_quantized_rollout_metrics

HEALTHY = {
    "training/rollout_probs_diff_mean": 0.014,
    "training/rollout_probs_diff_valid": 1,
    "rollout_corr/kl": 0.007,
    "response_length/clip_ratio": 0.03,
}


def _cfg(quantization):
    rollout = {"quantization": quantization}
    return SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout))


def test_healthy_metrics_pass():
    s = QuantizedRolloutSentinel("mxfp8")
    for step in range(5):
        s.check(HEALTHY, step)


def test_nan_rollout_probs_raise():
    s = QuantizedRolloutSentinel("mxfp8")
    with pytest.raises(RuntimeError, match="non-finite"):
        s.check({**HEALTHY, "training/rollout_probs_diff_mean": float("nan")}, 1)


def test_nan_probs_with_empty_batch_is_not_flagged():
    # valid == 0 means the batch had no tokens to compare, not that the rollout is broken
    s = QuantizedRolloutSentinel("mxfp8")
    s.check({**HEALTHY, "training/rollout_probs_diff_mean": float("nan"), "training/rollout_probs_diff_valid": 0}, 1)


def test_runaway_kl_raises():
    s = QuantizedRolloutSentinel("mxfp8")
    with pytest.raises(RuntimeError, match="rollout_corr/kl"):
        s.check({**HEALTHY, "rollout_corr/kl": 7.2}, 1)  # the FlashInfer stale-scale signature


def test_all_truncated_needs_consecutive_steps():
    s = QuantizedRolloutSentinel("mxfp8", clip_steps=2)
    s.check({**HEALTHY, "response_length/clip_ratio": 1.0}, 1)  # one step: could be a hard batch
    with pytest.raises(RuntimeError, match="length cap"):
        s.check({**HEALTHY, "response_length/clip_ratio": 1.0}, 2)
    s2 = QuantizedRolloutSentinel("mxfp8", clip_steps=2)
    s2.check({**HEALTHY, "response_length/clip_ratio": 1.0}, 1)
    s2.check({**HEALTHY, "response_length/clip_ratio": 0.4}, 2)  # streak resets
    s2.check({**HEALTHY, "response_length/clip_ratio": 1.0}, 3)


def test_unquantized_rollout_is_never_checked():
    s = QuantizedRolloutSentinel(None)
    s.check({**HEALTHY, "rollout_corr/kl": 50.0, "training/rollout_probs_diff_mean": float("nan")}, 1)


def test_env_disable(monkeypatch):
    monkeypatch.setenv("VERL_QUANT_SENTINEL", "0")
    s = QuantizedRolloutSentinel.from_config(_cfg("mxfp8"))
    s.check({**HEALTHY, "rollout_corr/kl": 50.0}, 1)


def test_trainer_hook_caches_sentinel_and_reads_config(monkeypatch):
    monkeypatch.delenv("VERL_QUANT_SENTINEL", raising=False)
    trainer = SimpleNamespace(config=_cfg("mxfp8"))
    check_quantized_rollout_metrics(trainer, HEALTHY, 1)
    assert isinstance(trainer._quant_sentinel, QuantizedRolloutSentinel) and trainer._quant_sentinel.enabled
    with pytest.raises(RuntimeError):
        check_quantized_rollout_metrics(trainer, {**HEALTHY, "rollout_corr/kl": 3.0}, 2)
    plain = SimpleNamespace(config=_cfg(None))
    check_quantized_rollout_metrics(plain, {**HEALTHY, "rollout_corr/kl": 3.0}, 2)  # bf16 rollout: no-op
