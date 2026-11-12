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

"""Compare the real Megatron KL and its gradient with a dense CPU reference.

The CP rank/group sizes below describe only the packer's row ownership. Vocabulary
collectives use a real one-rank Gloo group; no KL computation is mocked.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

mpu = pytest.importorskip("megatron.core.parallel_state")

from verl.models.mcore.util import preprocess_thd_engine  # noqa: E402
from verl.trainer.distillation.losses import distillation_ppo_loss  # noqa: E402


@pytest.fixture
def parallel_state(tmp_path, monkeypatch):
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'gloo'}", rank=0, world_size=1)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: dist.group.WORLD)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    yield
    if initialized_here:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_layout", ["contiguous", "zigzag"])
def test_topk_teacher_matches_student_minimum_padding(parallel_state, cp_layout):
    rows = 100
    generator = torch.Generator().manual_seed(1024)
    student = torch.randn(rows, 8, generator=generator)
    teacher = torch.randn(rows, 8, generator=generator).log_softmax(-1).topk(3, dim=-1)

    def nested(value):
        return torch.nested.nested_tensor([value], layout=torch.jagged)

    packing = {
        "min_local_rows": 128,
        "cp_layout": cp_layout,
    }
    packed_student = preprocess_thd_engine(nested(student), **packing)[0].detach().requires_grad_()
    teacher_ids = nested(teacher.indices)
    teacher_logprobs = nested(teacher.values)
    packed_ids = preprocess_thd_engine(teacher_ids, **packing)[0]
    packed_logprobs = preprocess_thd_engine(teacher_logprobs, **packing)[0]
    valid = preprocess_thd_engine(nested(torch.ones(rows, dtype=torch.bool)), **packing)[0]
    assert valid.any()

    payload = {}
    data = {"teacher_ids": teacher_ids, "teacher_logprobs": teacher_logprobs}
    outputs = distillation_ppo_loss(
        config=SimpleNamespace(strategy="megatron"),
        distillation_config=SimpleNamespace(
            distillation_loss=SimpleNamespace(loss_mode="forward_kl_topk", log_prob_min_clamp=-10.0, tail_mass_eps=1e-6)
        ),
        data=data,
        student_logits=packed_student,
        data_format="thd",
        **packing,
        **payload,
    )
    reference_student = packed_student.detach().clone().requires_grad_()
    reference = (
        packed_logprobs.exp() * (packed_logprobs - reference_student.log_softmax(-1).gather(-1, packed_ids))
    ).sum(-1)
    torch.testing.assert_close(outputs["distillation_losses"][valid], reference[valid], atol=2e-6, rtol=2e-5)
    outputs["distillation_losses"][valid].sum().backward()
    reference[valid].sum().backward()
    torch.testing.assert_close(packed_student.grad, reference_student.grad, atol=2e-6, rtol=2e-5)
