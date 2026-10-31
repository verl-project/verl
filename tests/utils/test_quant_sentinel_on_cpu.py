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

import json
import logging
from types import SimpleNamespace

import pytest

from verl.utils.quant_sentinel import (
    QuantizedRolloutSentinel,
    check_quantized_rollout_metrics,
    rollout_quantization_label,
)

HEALTHY = {
    "training/rollout_probs_diff_mean": 0.014,
    "training/rollout_probs_diff_valid": 1,
    "rollout_corr/kl": 0.007,
    "response_length/clip_ratio": 0.03,
}


def _cfg(quantization, model_path=None, ignore_eos=False, response_length=None):
    rollout = {"quantization": quantization, "ignore_eos": ignore_eos, "response_length": response_length}
    return SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout, model=SimpleNamespace(path=model_path)))


def _structured(**fields):
    # rollout.quantization as the trainer config holds it: always present, all fields null when off.
    return {"linear": None, "moe": None, "ignore": [], "config_file": None, **fields}


def test_healthy_metrics_pass():
    s = QuantizedRolloutSentinel("mxfp8")
    for step in range(5):
        s.check(HEALTHY, step)


def test_nan_rollout_probs_raise():
    s = QuantizedRolloutSentinel("mxfp8", mode="raise")
    with pytest.raises(RuntimeError, match="non-finite"):
        s.check({**HEALTHY, "training/rollout_probs_diff_mean": float("nan")}, 1)


def test_nan_probs_with_empty_batch_is_not_flagged():
    # valid == 0 means the batch had no tokens to compare, not that the rollout is broken
    s = QuantizedRolloutSentinel("mxfp8")
    s.check({**HEALTHY, "training/rollout_probs_diff_mean": float("nan"), "training/rollout_probs_diff_valid": 0}, 1)


def test_runaway_kl_raises():
    s = QuantizedRolloutSentinel("mxfp8", mode="raise")
    with pytest.raises(RuntimeError, match="rollout_corr/kl"):
        s.check({**HEALTHY, "rollout_corr/kl": 7.2}, 1)  # the FlashInfer stale-scale signature


def test_all_truncated_needs_consecutive_steps():
    s = QuantizedRolloutSentinel("mxfp8", mode="raise", clip_steps=2)
    s.check({**HEALTHY, "response_length/clip_ratio": 1.0}, 1)  # one step: could be a hard batch
    with pytest.raises(RuntimeError, match="length cap"):
        s.check({**HEALTHY, "response_length/clip_ratio": 1.0}, 2)
    s2 = QuantizedRolloutSentinel("mxfp8", mode="raise", clip_steps=2)
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
    monkeypatch.setenv("VERL_QUANT_SENTINEL", "raise")
    trainer = SimpleNamespace(config=_cfg("mxfp8"))
    check_quantized_rollout_metrics(trainer, HEALTHY, 1)
    assert isinstance(trainer._quant_sentinel, QuantizedRolloutSentinel) and trainer._quant_sentinel.enabled
    with pytest.raises(RuntimeError):
        check_quantized_rollout_metrics(trainer, {**HEALTHY, "rollout_corr/kl": 3.0}, 2)
    plain = SimpleNamespace(config=_cfg(None))
    check_quantized_rollout_metrics(plain, {**HEALTHY, "rollout_corr/kl": 3.0}, 2)  # bf16 rollout: no-op


def test_default_structured_quantization_is_not_checked():
    # rollout.quantization is an object even when the rollout runs in BF16; that alone must not arm the guard.
    s = QuantizedRolloutSentinel.from_config(_cfg(_structured()))
    assert not s.enabled
    s.check({**HEALTHY, "rollout_corr/kl": 50.0}, 1)


@pytest.mark.parametrize(
    ("fields", "label"),
    [
        ({"linear": "MXFP8", "moe": "MXFP8"}, "linear=MXFP8, moe=MXFP8"),
        ({"linear": "MXFP8"}, "linear=MXFP8"),
        ({"moe": "FP8_BLOCK"}, "moe=FP8_BLOCK"),
        ({"config_file": "/cfg/quant.json"}, "config_file=/cfg/quant.json"),
    ],
)
def test_structured_override_arms_the_guard(monkeypatch, fields, label):
    monkeypatch.setenv("VERL_QUANT_SENTINEL", "raise")
    s = QuantizedRolloutSentinel.from_config(_cfg(_structured(**fields)))
    assert s.enabled and s.quantization == label
    with pytest.raises(RuntimeError, match=label):
        s.check({**HEALTHY, "rollout_corr/kl": 3.0}, 1)


@pytest.mark.parametrize("nested", [False, True])
def test_natively_quantized_model_arms_the_guard(monkeypatch, tmp_path, nested):
    monkeypatch.delenv("VERL_QUANT_SENTINEL", raising=False)
    quant = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    hf_config = {"text_config": {"quantization_config": quant}} if nested else {"quantization_config": quant}
    (tmp_path / "config.json").write_text(json.dumps(hf_config))
    assert rollout_quantization_label(_cfg(_structured(), model_path=str(tmp_path))) == "native fp8"
    assert QuantizedRolloutSentinel.from_config(_cfg(_structured(), model_path=str(tmp_path))).enabled


def test_empty_legacy_override_still_reads_the_model_config(tmp_path):
    # An empty string from a config older than the structured rollout.quantization sets nothing itself.
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    assert rollout_quantization_label(_cfg("", model_path=str(tmp_path))) == "native fp8"
    assert rollout_quantization_label(_cfg("")) is None


def test_bf16_model_or_unreadable_path_is_not_checked(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert rollout_quantization_label(_cfg(_structured(), model_path=str(tmp_path))) is None
    # Remote paths are not fetched from the driver.
    assert rollout_quantization_label(_cfg(_structured(), model_path="hdfs://models/qwen3")) is None
    assert rollout_quantization_label(_cfg(_structured(), model_path=str(tmp_path / "missing"))) is None


def test_by_default_a_finding_is_logged_and_the_run_goes_on(monkeypatch, caplog):
    monkeypatch.delenv("VERL_QUANT_SENTINEL", raising=False)
    s = QuantizedRolloutSentinel.from_config(_cfg("mxfp8"))
    assert s.enabled and s.mode == "warn"
    with caplog.at_level(logging.WARNING, logger="verl.utils.quant_sentinel"):
        s.check({**HEALTHY, "rollout_corr/kl": 7.2}, 1)
    assert "looks broken at step 1" in caplog.text and "VERL_QUANT_SENTINEL=raise" in caplog.text


def test_two_fully_truncated_steps_only_warn_by_default(caplog):
    s = QuantizedRolloutSentinel("mxfp8")
    with caplog.at_level(logging.WARNING, logger="verl.utils.quant_sentinel"):
        s.check({**HEALTHY, "response_length/clip_ratio": 1.0}, 1)
        s.check({**HEALTHY, "response_length/clip_ratio": 1.0}, 2)
    assert "length cap" in caplog.text


def test_ignore_eos_turns_the_clip_rule_off(monkeypatch):
    # Every response runs to the cap by design; the other rules still apply.
    monkeypatch.setenv("VERL_QUANT_SENTINEL", "raise")
    s = QuantizedRolloutSentinel.from_config(_cfg("mxfp8", ignore_eos=True))
    for step in range(1, 4):
        s.check({**HEALTHY, "response_length/clip_ratio": 1.0}, step)
    with pytest.raises(RuntimeError, match="non-finite"):
        s.check({**HEALTHY, "training/rollout_probs_diff_mean": float("nan")}, 4)


def test_zero_clip_steps_turns_the_clip_rule_off():
    s = QuantizedRolloutSentinel("mxfp8", mode="raise", clip_steps=0)
    for step in range(1, 4):
        s.check({**HEALTHY, "response_length/clip_ratio": 1.0}, step)
    s.check(HEALTHY, 4)


def test_clip_rule_needs_the_longest_response_at_the_cap(monkeypatch):
    # The V1 trainer pads responses to the batch's longest, so clip_ratio 1.0 can mean "every response equally
    # short": with rollout.response_length known, a step counts only when the longest response reached it.
    monkeypatch.setenv("VERL_QUANT_SENTINEL", "raise")
    s = QuantizedRolloutSentinel.from_config(_cfg("mxfp8", response_length=256))
    equally_short = {**HEALTHY, "response_length/clip_ratio": 1.0, "response_length/max": 37}
    for step in (1, 2, 3):
        s.check(equally_short, step)
    at_cap = {**HEALTHY, "response_length/clip_ratio": 1.0, "response_length/max": 256}
    s.check(at_cap, 4)
    with pytest.raises(RuntimeError, match="consecutive steps"):
        s.check(at_cap, 5)
