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
"""CPU loss-math test for the ``kl_direction`` knob on the top-k distillation loss.

``compute_forward_kl_topk`` historically computed only forward KL,
``KL(teacher || student)``. The ``kl_direction`` knob adds the reverse direction,
``KL(student || teacher)``, a loss-flexibility primitive for on-policy
self-distillation (#6827). This pins both directions against a
from-scratch reference on tiny CPU tensors.
"""

from types import SimpleNamespace

import torch

from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk, kl_divergence


def _nested(rows: torch.Tensor) -> torch.Tensor:
    return torch.nested.nested_tensor([rows], layout=torch.jagged)


def _cfg(direction: str):
    return SimpleNamespace(
        distillation_loss=SimpleNamespace(kl_direction=direction, log_prob_min_clamp=None, use_chunked_topk=False)
    )


def _inputs(seqlen=4, vocab=10, topk=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(1, seqlen, vocab, generator=g)
    teacher_log_probs = torch.log_softmax(torch.randn(seqlen, vocab, generator=g), dim=-1)
    t_lp, t_ids = torch.topk(teacher_log_probs, topk, dim=-1)
    return student_logits, t_lp, t_ids.long()


def test_forward_keeps_current_semantics_reverse_swaps_and_they_differ():
    student_logits, t_lp, t_ids = _inputs()
    student_at_teacher = torch.gather(torch.log_softmax(student_logits, dim=-1), dim=-1, index=t_ids.unsqueeze(0))
    teacher = t_lp.unsqueeze(0)

    fwd = compute_forward_kl_topk(student_logits, _nested(t_lp), _nested(t_ids), _cfg("forward"), "thd")
    rev = compute_forward_kl_topk(student_logits, _nested(t_lp), _nested(t_ids), _cfg("reverse"), "thd")

    # forward == today's behavior: KL(teacher || student)
    torch.testing.assert_close(fwd["distillation_losses"], kl_divergence(log_q=student_at_teacher, log_p=teacher))
    # reverse == KL(student || teacher) over the teacher top-k support
    torch.testing.assert_close(rev["distillation_losses"], kl_divergence(log_q=teacher, log_p=student_at_teacher))
    assert not torch.allclose(fwd["distillation_losses"], rev["distillation_losses"])
