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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from verl.experimental.teacher_loop.teacher_model import HiddenStateTeacherPool


def request(index, key):
    return {
        "routing_key": key,
        "uid": str(index),
        "batch_index": index,
        "sequence_ids": [10, 11],
        "positions": [0],
        "sequence_hash": str(index),
    }


def pool():
    server = Mock()
    server.compute_teacher_hidden_states.remote.side_effect = lambda **kwargs: kwargs["queue_request"]
    teachers = {key: SimpleNamespace(model_path=f"/model/{key}") for key in ("a", "b")}
    return HiddenStateTeacherPool(teachers, dict.fromkeys(teachers, [server]), weight_swapped=True), server


def test_grouped_execution_returns_original_order():
    shared, server = pool()
    requests = [request(0, "b"), request(1, "a"), request(2, "b")]
    with patch("ray.wait"), patch("ray.get", side_effect=lambda refs: refs):
        receipts = shared.compute(requests)
        same_teacher = shared.compute([request(3, "b")])
    calls = server.compute_teacher_hidden_states.remote.call_args_list
    assert [call.kwargs["queue_request"]["batch_index"] for call in calls] == [1, 0, 2, 3]
    assert all(call.kwargs["prompt_ids"] == call.kwargs["queue_request"]["sequence_ids"] for call in calls)
    assert [receipt["batch_index"] for receipt in receipts] == [0, 1, 2]
    assert [receipt["teacher_weight_version"] for receipt in receipts] == [2, 1, 2]
    assert same_teacher[0]["teacher_weight_version"] == 2
    server.switch_teacher_weights.remote.assert_called_once_with("/model/b")


def test_failed_weight_swap_requires_reload_of_previous_teacher():
    shared, server = pool()
    server.switch_teacher_weights.remote.side_effect = [RuntimeError("reload failed"), None]
    with patch("ray.wait"), patch("ray.get", side_effect=lambda refs: refs):
        with pytest.raises(RuntimeError, match="reload failed"):
            shared.compute([request(0, "b")])
        receipts = shared.compute([request(1, "a")])
    assert receipts[0]["teacher_weight_version"] == 2
    assert [call.args[0] for call in server.switch_teacher_weights.remote.call_args_list] == ["/model/b", "/model/a"]


def test_reject_unknown_teacher_before_submitting_any_requests():
    shared, server = pool()
    with pytest.raises(ValueError, match="Unknown teacher"):
        shared.compute([request(0, "a"), request(1, "unknown")])
    server.compute_teacher_hidden_states.remote.assert_not_called()


@pytest.mark.parametrize(
    "rollout_options",
    [
        {"quantization": {"linear": "FP8_BLOCK"}},
        {"quantization": {"config_file": "/teacher/quantization.json"}},
        {"engine_kwargs": {"vllm": {"hf_overrides": {"quantization_config": {"quant_method": "awq"}}}}},
    ],
)
def test_hidden_opd_rejects_quantized_serving(rollout_options):
    from verl.workers.config import (
        DistillationConfig,
        DistillationLossConfig,
        DistillationTeacherModelConfig,
        RolloutConfig,
        TeacherExecutionConfig,
    )

    from verl.workers.config.rollout import RolloutQuantizationConfig

    if "quantization" in rollout_options:
        rollout_options = {"quantization": RolloutQuantizationConfig(**rollout_options["quantization"])}
    teacher = DistillationTeacherModelConfig(
        model_path="/teacher", inference=RolloutConfig(name="vllm", **rollout_options)
    )
    with pytest.raises(ValueError, match="unquantized"):
        DistillationConfig(
            enabled=True,
            nnodes=1,
            n_gpus_per_node=2,
            teacher_execution=TeacherExecutionConfig(mode="weight_swapped_hidden_state"),
            distillation_loss=DistillationLossConfig(
                loss_mode="forward_kl_hidden_state", use_task_rewards=False, use_policy_gradient=False
            ),
            teacher_models={"teacher_model": teacher},
        )


def test_dedicated_pools_submit_every_teacher_before_wait_and_forward_media():
    teachers = {"small": SimpleNamespace(model_path="/4b"), "large": SimpleNamespace(model_path="/9b")}
    servers = {key: [Mock()] for key in teachers}
    for handles in servers.values():
        handles[0].compute_teacher_hidden_states.remote.side_effect = lambda **kwargs: kwargs["queue_request"]
    scheduler = HiddenStateTeacherPool(teachers, servers, weight_swapped=False)
    image_row = request(0, "large")
    image_row.update(multi_modal_data={"images": ["image"]}, mm_processor_kwargs={"max_pixels": 65536})
    requests = [image_row, request(1, "small"), request(2, "large")]

    with patch("ray.wait") as drain, patch("ray.get", side_effect=lambda refs: refs):
        handles = scheduler.compute(requests)
    assert [handle["teacher_index"] for handle in handles] == [1, 0, 1]
    assert [handle["teacher_weight_version"] for handle in handles] == [1, 1, 1]
    assert [handle["batch_index"] for handle in handles] == [0, 1, 2]
    drain.assert_called_once()
    assert drain.call_args.kwargs["num_returns"] == len(requests)
    call = servers["large"][0].compute_teacher_hidden_states.remote.call_args_list[0].kwargs
    assert call["multi_modal_data"] == {"images": ["image"]}
    assert call["mm_processor_kwargs"] == {"max_pixels": 65536}
    assert "multi_modal_data" not in call["queue_request"]
    for refs in servers.values():
        refs[0].switch_teacher_weights.remote.assert_not_called()


def test_dedicated_hidden_config_allocates_explicit_and_default_replicas():
    from verl.workers.config import (
        DistillationConfig,
        DistillationLossConfig,
        DistillationTeacherModelConfig,
        RolloutConfig,
        TeacherExecutionConfig,
    )

    def teacher(key, tp, replicas=0):
        return DistillationTeacherModelConfig(
            key=key,
            model_path=f"/{key}",
            num_replicas=replicas,
            inference=RolloutConfig(name="vllm", tensor_model_parallel_size=tp),
        )

    cfg = DistillationConfig(
        enabled=True,
        nnodes=1,
        n_gpus_per_node=4,
        teacher_execution=TeacherExecutionConfig(mode="dedicated_hidden_state"),
        distillation_loss=DistillationLossConfig(
            loss_mode="forward_kl_hidden_state", use_task_rewards=False, use_policy_gradient=False
        ),
        teacher_models={
            "teacher_model": teacher("default", 1),
            "small": teacher("small", 1, 2),
            "large": teacher("large", 2),
        },
    )
    assert [(t.num_replicas, t.world_size) for t in cfg.teacher_models.values()] == [(2, 2), (1, 2)]
    assert all(t.inference.engine_kwargs["vllm"]["hidden_state_opd"] for t in cfg.teacher_models.values())
