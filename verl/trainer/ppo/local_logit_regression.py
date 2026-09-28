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
"""LOCAL: Local-Curvature Advantage Logit Regression with a Top-K state-dependent offset.

For a response token ``a`` at state ``s``, LOCAL regresses the displacement of the sampled logit,
centred by the old policy, onto the scaled advantage:

    psi_theta(a) = z_theta(a) - sum_{b in T} pbar(b) z_theta(b)
    delta        = psi_theta(a) - psi_old(a) - eta * A
    loss         = 1/2 * delta^2

``T`` is the Top-K support of the old policy at ``s`` and ``pbar`` its probabilities renormalised on
``T``. Subtracting the old-policy mean removes the state-dependent offset (SDO) of the target logits,
so the residual is measured in the old policy's Fisher geometry. With ``K = |vocab|`` the offset is
the exact optimal SDO ``E_{pi_old}[z_theta - z_old]``.

``z`` are the logits after temperature scaling, i.e. the logits of the sampling distribution.
The loss depends on ``z_theta`` only through differences of logits at one state, so it is invariant
to adding any per-state constant to ``z_theta``.
"""

from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F

LOCAL_LOSS_MODE = "local"

# Old-policy features written by the old-log-prob pass and read by the actor update.
OLD_TOPK_IDS_KEY = "old_local_topk_ids"
OLD_TOPK_LOG_PROBS_KEY = "old_local_topk_log_probs"
OLD_CENTERED_LOGITS_KEY = "old_local_centered_logits"
OLD_FEATURE_KEYS = (OLD_TOPK_IDS_KEY, OLD_TOPK_LOG_PROBS_KEY, OLD_CENTERED_LOGITS_KEY)

# Current-policy feature produced by the engine during the actor update.
CENTERED_LOGITS_KEY = "local_centered_logits"

# Non-tensor flags that switch the engine into the two LOCAL passes.
OLD_FEATURES_FLAG = "local_topk_k"
CURRENT_FEATURES_FLAG = "compute_local_centered_logits"


def is_local_loss(actor_config) -> bool:
    """Return whether the actor config selects the LOCAL policy loss."""
    policy_loss = actor_config.get("policy_loss", None)
    if policy_loss is None:
        return False
    return policy_loss.get("loss_mode", "vanilla") == LOCAL_LOSS_MODE


@torch.no_grad()
def compute_old_topk_features(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: int,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the old-policy Top-K support and the centred logit of each label.

    Rows are processed in chunks so that the float32 copy used for the log-partition function has
    at most ``chunk_size * vocab`` elements.

    Args:
        logits: ``(n, vocab)`` temperature-scaled logits of the old policy.
        labels: ``(n,)`` next-token ids.
        topk: size ``K`` of the support. Clipped to the vocabulary size.
        chunk_size: number of rows per chunk. It changes memory use, not the result.

    Returns:
        topk_ids: ``(n, K)`` int32 token ids of the ``K`` most probable tokens.
        topk_log_probs: ``(n, K)`` float32 full-vocabulary log-probabilities of those tokens.
        centered_logits: ``(n,)`` float32 ``z(label) - sum_k pbar_k z(id_k)``, with ``pbar`` the
            probabilities renormalised on the support.
    """
    num_rows, vocab_size = logits.shape
    k = min(int(topk), vocab_size)
    if k < 1:
        raise ValueError(f"topk must be positive, got {topk}")
    labels = labels.to(device=logits.device, dtype=torch.long)

    topk_ids = torch.empty((num_rows, k), dtype=torch.int32, device=logits.device)
    topk_log_probs = torch.empty((num_rows, k), dtype=torch.float32, device=logits.device)
    centered_logits = torch.empty((num_rows,), dtype=torch.float32, device=logits.device)
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = logits[start:end].float()
        top_logits, top_ids = torch.topk(chunk, k=k, dim=-1)
        log_partition = torch.logsumexp(chunk, dim=-1, keepdim=True)
        top_log_probs = top_logits - log_partition
        # softmax over the support renormalises the old probabilities: pbar = pi / sum_T pi.
        weights = torch.softmax(top_log_probs, dim=-1)
        label_logits = chunk.gather(-1, labels[start:end].unsqueeze(-1)).squeeze(-1)
        topk_ids[start:end] = top_ids.to(torch.int32)
        topk_log_probs[start:end] = top_log_probs
        centered_logits[start:end] = label_logits - (weights * top_logits).sum(dim=-1)
    return topk_ids, topk_log_probs, centered_logits


def compute_topk_centered_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    old_topk_ids: torch.Tensor,
    old_topk_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Centre the current label logit with the stored old-policy Top-K support and weights.

    The result is differentiable in ``logits``. At the old parameters it equals the
    ``centered_logits`` returned by :func:`compute_old_topk_features`.

    Args:
        logits: ``(n, vocab)`` temperature-scaled logits of the current policy.
        labels: ``(n,)`` next-token ids.
        old_topk_ids: ``(n, K)`` old-policy Top-K token ids.
        old_topk_log_probs: ``(n, K)`` old-policy log-probabilities of those tokens.

    Returns:
        ``(n,)`` float32 ``z_theta(label) - sum_k pbar_k z_theta(id_k)``.
    """
    weights = torch.softmax(old_topk_log_probs.detach().float(), dim=-1)
    label_logits = logits.gather(-1, labels.to(dtype=torch.long).unsqueeze(-1)).squeeze(-1).float()
    topk_logits = logits.gather(-1, old_topk_ids.to(dtype=torch.long)).float()
    return label_logits - (weights * topk_logits).sum(dim=-1)


def compute_policy_loss_local(
    centered_logits: torch.Tensor,
    old_centered_logits: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    eta: float,
    loss_agg_mode: str = "token-mean",
    global_batch_info: Optional[dict] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    old_topk_log_probs: Optional[torch.Tensor] = None,
    log_prob: Optional[torch.Tensor] = None,
    old_log_prob: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """LOCAL regression loss ``1/2 (psi_theta - psi_old - eta * A)^2`` aggregated over tokens.

    Args:
        centered_logits: ``(bs, response_length)`` current centred logits ``psi_theta``.
        old_centered_logits: ``(bs, response_length)`` old centred logits ``psi_old``.
        advantages: ``(bs, response_length)`` advantage estimates.
        response_mask: ``(bs, response_length)`` mask of valid response tokens.
        eta: target scale of the logit displacement.
        loss_agg_mode: token aggregation mode passed to ``agg_loss``.
        global_batch_info: global batch statistics passed to ``agg_loss``.
        rollout_is_weights: optional per-token importance weights from rollout correction.
        old_topk_log_probs: optional ``(bs, response_length, K)`` old Top-K log-probabilities, for the
            retained-mass metric.
        log_prob: optional current log-probabilities, for the ``ppo_kl`` metric.
        old_log_prob: optional old log-probabilities, for the ``ppo_kl`` metric.

    Returns:
        The aggregated loss and a dictionary of metrics.
    """
    from verl.trainer.ppo.core_algos import agg_loss

    response_mask = response_mask.to(dtype=torch.bool)
    target = eta * advantages.detach().float()
    feature_delta = centered_logits.float() - old_centered_logits.detach().float()
    residual = feature_delta - target
    loss_mat = 0.5 * residual.square()
    if rollout_is_weights is not None:
        loss_mat = loss_mat * rollout_is_weights.detach()

    pg_loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **(global_batch_info or {}),
    )

    with torch.no_grad():
        metrics = {
            "actor/local_residual_mse": verl_F.masked_mean(residual.square(), response_mask).item(),
            "actor/local_residual_abs": verl_F.masked_mean(residual.abs(), response_mask).item(),
            "actor/local_target_abs": verl_F.masked_mean(target.abs(), response_mask).item(),
            "actor/local_feature_delta_abs": verl_F.masked_mean(feature_delta.abs(), response_mask).item(),
        }
        if old_topk_log_probs is not None:
            topk_mass = old_topk_log_probs.float().exp().sum(dim=-1)
            metrics["actor/local_topk_mass"] = verl_F.masked_mean(topk_mass, response_mask).item()
        if log_prob is not None and old_log_prob is not None:
            metrics["actor/ppo_kl"] = verl_F.masked_mean(old_log_prob - log_prob, response_mask).item()
    return pg_loss, metrics
