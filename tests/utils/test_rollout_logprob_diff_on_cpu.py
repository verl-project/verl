# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise the refit acceptance decision without distributed workers or inference engines."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def run_refits(monkeypatch, capsys):
    # Import the actual script, isolating its runtime-only dependencies. The
    # temporary module entries are restored before any tests execute main().
    imports = {
        "ray": "init shutdown",
        "hydra": "compose initialize_config_dir",
        "omegaconf": "DictConfig",
        "verl": "",
        "verl.checkpoint_engine": "CheckpointEngineManager",
        "verl.experimental.agent_loop": "AgentLoopManager",
        "verl.experimental.reward_loop": "RewardLoopManager",
        "verl.protocol": "DataProto",
        "verl.single_controller.ray": "RayClassWithInitArgs RayWorkerGroup",
        "verl.single_controller.ray.base": "create_colocated_worker_cls",
        "verl.trainer.ppo.ray_trainer": "ResourcePoolManager Role",
        "verl.trainer.ppo.rollout_corr_helper": "compute_offpolicy_metrics",
        "verl.utils": "omega_conf_to_dataclass tensordict_utils",
        "verl.utils.debug.metrics": "calculate_debug_metrics",
        "verl.utils.device": "get_device_name",
        "verl.workers.engine_workers": "ActorRolloutRefWorker",
        "verl.workers.rollout.llm_server": "LLMServerManager",
        "verl.workers.utils.padding": "left_right_2_no_padding no_padding_2_padding",
    }
    path = Path(__file__).resolve().parents[2] / "scripts/rollout_logprob_diff.py"
    spec = importlib.util.spec_from_file_location("rollout_logprob_diff_under_test", path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as import_patch:
        for name, attributes in imports.items():
            stub = ModuleType(name)
            for attribute in attributes.split():
                setattr(stub, attribute, object)
            import_patch.setitem(sys.modules, name, stub)
        spec.loader.exec_module(script)

    def run(history, threshold=0.01):
        results = []
        for _ in history:
            result = Mock()
            result.meta_info = {"timing": {}}
            result.batch = {
                "old_log_probs": torch.zeros(1, 2),
                "rollout_log_probs": torch.zeros(1, 2),
                "response_mask": torch.ones(1, 2),
                "attention_mask": torch.ones(1, 2),
            }
            result.union.return_value = result
            results.append(result)

        agent_loop = Mock()
        agent_loop.generate_sequences.side_effect = results
        checkpoint = Mock()
        script.ray.init = Mock()
        script.ray.shutdown = Mock()
        monkeypatch.setattr(
            script, "parse_args", lambda: SimpleNamespace(refit_rounds=len(history), max_k3_kl=threshold)
        )
        monkeypatch.setattr(
            script,
            "build_config",
            lambda args: SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=1, temperature=1.0))
            ),
        )
        monkeypatch.setattr(script, "init_workers", lambda config: (agent_loop, Mock(), checkpoint))
        monkeypatch.setattr(script, "DataProto", Mock())
        monkeypatch.setattr(script, "compute_old_log_prob", Mock())
        monkeypatch.setattr(
            script, "compute_offpolicy_metrics", Mock(side_effect=[dict(metrics) for metrics in history])
        )
        monkeypatch.setattr(script, "calculate_debug_metrics", lambda result: {})

        exit_code = script.main()
        output = capsys.readouterr().out
        summary, end = json.JSONDecoder().raw_decode(output)
        # A failed round must not prevent collecting the remaining diagnostics.
        assert agent_loop.generate_sequences.call_count == len(history)
        assert checkpoint.update_weights.call_count == len(history) - 1
        assert checkpoint.sleep_replicas.call_count == len(history)
        script.ray.shutdown.assert_called_once_with()
        return exit_code, summary, output[end:]

    return run


@pytest.mark.parametrize("rounds", [1, 3])
def test_accepts_all_healthy_refits_and_keeps_json_summary(run_refits, rounds):
    history = [{"kl": -0.001, "k3_kl": 0.001, "marker": index} for index in range(rounds)]

    exit_code, summary, verdict = run_refits(history)

    assert exit_code == 0
    assert verdict.strip() == "PASSED"
    assert summary["marker"] == rounds - 1
    assert summary["num_response_tokens"] == 2
    if rounds == 1:
        assert "refit_runs" not in summary
    else:
        assert [run["marker"] for run in summary["refit_runs"]] == list(range(rounds))


@pytest.mark.parametrize("failed_round", [0, 1, 2])
@pytest.mark.parametrize("k3_kl", [0.01, 0.25], ids=["at_threshold", "above_threshold"])
def test_rejects_a_threshold_failure_in_any_round(run_refits, failed_round, k3_kl):
    history = [{"kl": 0.001, "k3_kl": 0.001} for _ in range(3)]
    history[failed_round]["k3_kl"] = k3_kl

    exit_code, summary, verdict = run_refits(history)

    assert exit_code == 1
    assert f"refit round {failed_round + 1}" in verdict
    assert "PASSED" not in verdict
    assert len(summary["refit_runs"]) == 3


@pytest.mark.parametrize("failed_round", [0, 1, 2])
@pytest.mark.parametrize("metric", ["kl", "k3_kl"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "negative_inf"])
def test_rejects_nonfinite_metrics_in_any_round(run_refits, failed_round, metric, value):
    history = [{"kl": 0.001, "k3_kl": 0.001} for _ in range(3)]
    history[failed_round][metric] = value

    exit_code, _, verdict = run_refits(history)

    assert exit_code == 1
    assert f"refit round {failed_round + 1}" in verdict
    assert "PASSED" not in verdict


@pytest.mark.parametrize("k3_kl", [0.25, float("nan")], ids=["above_threshold", "nan"])
def test_still_rejects_a_single_bad_refit(run_refits, k3_kl):
    exit_code, summary, verdict = run_refits([{"kl": 0.001, "k3_kl": k3_kl}])

    assert exit_code == 1
    assert "refit round 1" in verdict
    assert "refit_runs" not in summary


def test_reports_each_failed_refit(run_refits):
    exit_code, _, verdict = run_refits(
        [{"kl": float("nan"), "k3_kl": float("nan")}, {"kl": 0.2, "k3_kl": 0.25}, {"kl": 0.001, "k3_kl": 0.001}]
    )

    assert exit_code == 1
    assert "refit round 1" in verdict
    assert "refit round 2" in verdict
    assert "refit round 3" not in verdict
