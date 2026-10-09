# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
import pytest

from verl.utils.metric.behavior_age import behavior_age_metrics


def test_token_weighted_age_and_coverage_excludes_padding():
    fields = [
        {"behavior_version_segments": [[0, 2], [1, 1]]},
        {"behavior_version_segments": [[2, 7]]},
        {},
    ]
    p = "training/off_policy/"
    result = behavior_age_metrics(fields, [3, 7, 5], [True, True, False], 2)
    assert result[p + "behavior_version/response_coverage"] == result[p + "behavior_version/token_coverage"] == 1
    assert result[p + "token_staleness/mean"] == 0.5
    assert result[p + "token_staleness/stale_fraction"] == 0.3
    assert result[p + "token_staleness/max"] == 2
    assert result[p + "behavior_version/cross_version_response_fraction"] == 0.5
    missing = behavior_age_metrics(fields, [3, 7, 5], [True] * 3, 2)
    assert missing[p + "behavior_version/response_coverage"] == pytest.approx(2 / 3)
    assert missing[p + "behavior_version/token_coverage"] == pytest.approx(10 / 15)
    assert missing[p + "token_staleness/mean"] == 0.5


@pytest.mark.parametrize("segments", [None, [], [[None, 2]], [[3, 2]], [[0, 1]], [[0, -2]], [[True, 2]]])
def test_unknown_or_inconsistent_versions_are_not_reported_as_zero_age(segments):
    result = behavior_age_metrics([{"behavior_version_segments": segments}], [2], [True], 2)
    assert all(value == 0 for value in result.values())
    assert not any("token_staleness" in key for key in result)


@pytest.mark.parametrize("segments", [3, True, "bad", {}, [[0, True]], [[-1, 2]], [[0, 2, 3]]])
def test_malformed_provenance_is_uncovered(segments):
    result = behavior_age_metrics([{"behavior_version_segments": segments}], [2], [True], 0)
    assert result["training/off_policy/behavior_version/response_coverage"] == 0
    assert not any("token_staleness" in key for key in result)


@pytest.mark.parametrize("version", [-1, None, True, 1.5])
def test_invalid_current_version_rejected(version):
    with pytest.raises(ValueError, match="current_version"):
        behavior_age_metrics([], [], [], version)


def test_current_version_zero_and_empty_batch():
    result = behavior_age_metrics([{"behavior_version_segments": [[0, 2]]}], [2], [True], 0)
    assert result["training/off_policy/token_staleness/mean"] == 0
    assert result["training/off_policy/behavior_version/token_coverage"] == 1
    empty = behavior_age_metrics([], [], [], 0)
    assert all(value == 0 for value in empty.values())


@pytest.mark.parametrize("error", ["empty field extra_fields", "Some fields are not ready in all the requested keys!"])
def test_optional_field_reader_preserves_mixed_row_coverage(error):
    from types import SimpleNamespace

    from verl.utils.metric.behavior_age import read_behavior_version_fields

    def get(keys, partition_id, select_fields):
        assert partition_id == "train" and select_fields == ["extra_fields"]
        if "old" in keys:
            raise ValueError(error)
        return {"extra_fields": SimpleNamespace(tolist=lambda: [{"behavior_version_segments": [[1, 2]]}])}

    fields = read_behavior_version_fields(["new", "old"], "train", get)
    result = behavior_age_metrics(fields, [2, 2], [True, True], 2)
    assert result["training/off_policy/behavior_version/response_coverage"] == 0.5
    assert result["training/off_policy/token_staleness/mean"] == 1


def test_optional_reader_propagates_unrelated_errors():
    from verl.utils.metric.behavior_age import read_behavior_version_fields

    def get(**kwargs):
        raise ValueError("partition missing")

    with pytest.raises(ValueError, match="partition missing"):
        read_behavior_version_fields(["x"], "train", get)


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_trainer_metrics_prefix_requests_provenance_only_when_enabled(enabled):
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import torch

    source = Path(__file__).parents[2] / "verl/trainer/ppo/v1/trainer_base.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPOTrainer")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_metrics")
    # Execute the production field collection/age integration before unrelated
    # reward, throughput and GPU metrics require a complete trainer environment.
    end = next(
        i
        for i, node in enumerate(fn.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "global_token_num" for target in node.targets)
    )
    fn.body = fn.body[:end]
    reads = []

    def get(**kwargs):
        assert kwargs["partition_id"] == "training-custom"
        reads.append(kwargs)
        return {}

    def metric_data(**kwargs):
        assert "extra_fields" not in kwargs["fields"]
        return {
            "num_turns": np.array([1, 1]),
            "prompts": SimpleNamespace(offsets=lambda: torch.tensor([0, 1, 2])),
            "responses": SimpleNamespace(offsets=lambda: torch.tensor([0, 2, 4])),
        }

    namespace = {
        "np": np,
        "KVBatchMeta": object,
        "get_metric_data_with_optional_routed_experts": metric_data,
        "tq": SimpleNamespace(kv_batch_get=get),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout={"collect_behavior_version_metrics": enabled})
        ),
        _rollout_moe_lb_metrics_accumulator=None,
    )
    batch = SimpleNamespace(keys=["a", "b"], partition_id="training-custom", tags=[{}, {}])
    metrics = {}
    namespace["_compute_metrics"](trainer, batch, metrics, {}, global_steps=3, epoch=0)
    assert len(reads) == int(enabled)
    if enabled:
        assert metrics["training/off_policy/behavior_version/response_coverage"] == 0
        assert not any("token_staleness" in key for key in metrics)
    else:
        assert metrics == {}
