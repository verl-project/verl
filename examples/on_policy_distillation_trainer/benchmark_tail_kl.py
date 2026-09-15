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

"""Compare truncated and tail-aware teacher-top-k KL against a full-vocabulary oracle."""

import argparse

import torch
import torch.nn.functional as F

from verl.trainer.distillation.fsdp.losses import tail_aware_kl_divergence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--topk", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--student-noise", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return F.cosine_similarity(lhs.flatten(), rhs.flatten(), dim=0).item()


def main() -> None:
    args = parse_args()
    if any(k <= 0 or k > args.vocab_size for k in args.topk):
        raise ValueError(f"Every top-k value must be in [1, {args.vocab_size}].")

    torch.manual_seed(args.seed)
    teacher_logits = torch.randn(args.tokens, args.vocab_size, device=args.device)
    student_logits = (teacher_logits + args.student_noise * torch.randn_like(teacher_logits)).requires_grad_(True)
    teacher_log_probs = teacher_logits.log_softmax(dim=-1)
    student_log_probs = student_logits.log_softmax(dim=-1)
    teacher_probs = teacher_log_probs.exp()

    full_loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    full_loss_mean = full_loss.mean()
    full_grad = torch.autograd.grad(full_loss_mean, student_logits, retain_graph=True)[0]

    print(
        "k,full_kl,truncated_kl,clamped_truncated_kl,tail_kl,truncated_relative_error,"
        "tail_relative_error,truncated_gradient_cosine,tail_gradient_cosine,negative_truncated_pct"
    )
    for k in args.topk:
        teacher_topk_log_probs, teacher_topk_ids = torch.topk(teacher_log_probs, k=k, dim=-1)
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
        truncated_loss = (teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1)
        tail_loss, *_ = tail_aware_kl_divergence(
            log_q=student_topk_log_probs,
            log_p=teacher_topk_log_probs,
            tail_mass_eps=1e-7,
        )
        if torch.any(tail_loss < -1e-6) or torch.any(tail_loss > full_loss + 1e-5):
            raise RuntimeError("Tail-aware KL violated its expected [0, full KL] bounds.")
        clamped_truncated_loss_mean = truncated_loss.clamp_min(0).mean()
        tail_loss_mean = tail_loss.mean()
        truncated_grad = torch.autograd.grad(clamped_truncated_loss_mean, student_logits, retain_graph=True)[0]
        tail_grad = torch.autograd.grad(tail_loss_mean, student_logits, retain_graph=True)[0]

        truncated_relative_error = (
            (full_loss_mean - clamped_truncated_loss_mean).abs() / full_loss_mean.clamp_min(1e-12)
        ).item()
        tail_relative_error = ((full_loss_mean - tail_loss_mean).abs() / full_loss_mean.clamp_min(1e-12)).item()
        negative_fraction = (truncated_loss < 0).float().mean().mul(100).item()
        print(
            f"{k},{full_loss_mean.item():.8f},{truncated_loss.mean().item():.8f},"
            f"{clamped_truncated_loss_mean.item():.8f},{tail_loss_mean.item():.8f},"
            f"{truncated_relative_error:.8f},{tail_relative_error:.8f},"
            f"{cosine_similarity(truncated_grad, full_grad):.8f},{cosine_similarity(tail_grad, full_grad):.8f},"
            f"{negative_fraction:.2f}"
        )


if __name__ == "__main__":
    main()
