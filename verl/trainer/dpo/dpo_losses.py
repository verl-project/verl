# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""
DPO loss functions with a registry pattern for pluggability.

Each loss function takes the same 4 inputs:
    chosen_logps:        log π_θ(y_w | x)   — policy log-prob of chosen response
    rejected_logps:      log π_θ(y_l | x)   — policy log-prob of rejected response
    ref_chosen_logps:    log π_ref(y_w | x)  — reference log-prob of chosen response
    ref_rejected_logps:  log π_ref(y_l | x)  — reference log-prob of rejected response

And returns:
    loss:             scalar loss tensor
    chosen_rewards:   implicit reward for chosen (for metrics, detached)
    rejected_rewards: implicit reward for rejected (for metrics, detached)

Usage:
    from verl.trainer.dpo.dpo_losses import get_dpo_loss_fn

    loss_fn = get_dpo_loss_fn("sigmoid")
    loss, chosen_rewards, rejected_rewards = loss_fn(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps,
        beta=0.1, label_smoothing=0.0
    )
"""

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from verl.base_config import BaseConfig

__all__ = ["DPOLossConfig", "register_dpo_loss", "get_dpo_loss_fn", "DPO_LOSS_REGISTRY"]


@dataclass
class DPOLossConfig(BaseConfig):
    """Hyperparameters of the DPO preference loss.

    Args:
        loss_type: Name of a registered loss, e.g. "sigmoid" (standard DPO), "ipo" or "hinge".
        beta: Strength of the implicit KL penalty toward the reference model.
        label_smoothing: Assumed probability that a preference label is flipped. Used by
            sigmoid, robust, exo_pair, aot_pair and aot; must be below 0.5.
        reference_free: Treat the reference model as uniform (reference log-probs = 0).
    """

    loss_type: str = "sigmoid"
    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError(f"label_smoothing must be in [0, 0.5), got {self.label_smoothing}")


# ─── Registry ────────────────────────────────────────────────────────────

DPO_LOSS_REGISTRY: dict[str, Callable] = {}


def register_dpo_loss(name: str):
    """Decorator to register a DPO loss function.

    Usage:
        @register_dpo_loss("my_loss")
        def compute_my_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, **kwargs):
            ...
            return loss, chosen_rewards, rejected_rewards
    """

    def decorator(fn: Callable) -> Callable:
        if name in DPO_LOSS_REGISTRY and DPO_LOSS_REGISTRY[name] != fn:
            raise ValueError(f"DPO loss '{name}' already registered: {DPO_LOSS_REGISTRY[name]} vs {fn}")
        DPO_LOSS_REGISTRY[name] = fn
        return fn

    return decorator


def get_dpo_loss_fn(name: str) -> Callable:
    """Get a registered DPO loss function by name.

    Args:
        name: Loss name (e.g., "sigmoid", "ipo", "hinge")

    Returns:
        The loss function.

    Raises:
        ValueError: If the loss name is not registered.
    """
    if name not in DPO_LOSS_REGISTRY:
        raise ValueError(f"Unknown DPO loss: '{name}'. Available: {list(DPO_LOSS_REGISTRY.keys())}")
    return DPO_LOSS_REGISTRY[name]


# ─── Helper ──────────────────────────────────────────────────────────────


def _compute_logratios_and_rewards(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    reference_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute log-ratios, logits, and implicit rewards — shared by most losses."""
    ref_scale = 0.0 if reference_free else 1.0

    chosen_logratios = chosen_logps - ref_scale * ref_chosen_logps
    rejected_logratios = rejected_logps - ref_scale * ref_rejected_logps
    logits = chosen_logratios - rejected_logratios

    chosen_rewards = beta * (chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps).detach()

    return chosen_logratios, rejected_logratios, logits, chosen_rewards, rejected_rewards


# ─── Loss Implementations ────────────────────────────────────────────────


@register_dpo_loss("sigmoid")
def dpo_loss_sigmoid(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard DPO loss (Rafailov et al., 2023).

    L = -log σ(β·logits)·(1-ε) - log σ(-β·logits)·ε
    """
    _, _, logits, chosen_rewards, rejected_rewards = _compute_logratios_and_rewards(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, reference_free
    )
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    return losses.mean(), chosen_rewards, rejected_rewards


@register_dpo_loss("robust")
def dpo_loss_robust(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robust DPO with corrected label smoothing for noisy labels."""
    _, _, logits, chosen_rewards, rejected_rewards = _compute_logratios_and_rewards(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, reference_free
    )
    losses = (-F.logsigmoid(beta * logits) * (1 - label_smoothing) + F.logsigmoid(-beta * logits) * label_smoothing) / (
        1 - 2 * label_smoothing
    )
    return losses.mean(), chosen_rewards, rejected_rewards


@register_dpo_loss("ipo")
def dpo_loss_ipo(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    reference_free: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identity Preference Optimization (Azar et al., 2023).

    L = (logits - 1/(2β))²
    """
    _, _, logits, chosen_rewards, rejected_rewards = _compute_logratios_and_rewards(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, reference_free
    )
    losses = (logits - 1 / (2 * beta)) ** 2
    return losses.mean(), chosen_rewards, rejected_rewards


@register_dpo_loss("hinge")
def dpo_loss_hinge(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    reference_free: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hinge (max-margin) DPO loss.

    L = max(0, 1 - β·logits)
    """
    _, _, logits, chosen_rewards, rejected_rewards = _compute_logratios_and_rewards(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, reference_free
    )
    losses = torch.relu(1 - beta * logits)
    return losses.mean(), chosen_rewards, rejected_rewards


@register_dpo_loss("exo_pair")
def dpo_loss_exo_pair(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EXO loss with exploration bonus (Ji et al., 2024).

    Equation 16 from the paper.
    """
    _, _, logits, chosen_rewards, rejected_rewards = _compute_logratios_and_rewards(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, reference_free
    )
    if label_smoothing == 0:
        label_smoothing = 1e-3
    losses = torch.sigmoid(beta * logits) * (
        F.logsigmoid(beta * logits) - math.log(1 - label_smoothing)
    ) + torch.sigmoid(-beta * logits) * (F.logsigmoid(-beta * logits) - math.log(label_smoothing))
    return losses.mean(), chosen_rewards, rejected_rewards


@register_dpo_loss("aot_pair")
def dpo_loss_aot_pair(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aligned Optimal Transport (pair variant).

    Sorts chosen and rejected log-ratios independently before computing DPO loss.
    """
    chosen_logratios, rejected_logratios, _, chosen_rewards, rejected_rewards = _compute_logratios_and_rewards(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, reference_free
    )
    chosen_sorted, _ = torch.sort(chosen_logratios, dim=0)
    rejected_sorted, _ = torch.sort(rejected_logratios, dim=0)
    delta = chosen_sorted - rejected_sorted
    losses = -F.logsigmoid(beta * delta) * (1 - label_smoothing) - F.logsigmoid(-beta * delta) * label_smoothing
    return losses.mean(), chosen_rewards, rejected_rewards


@register_dpo_loss("aot")
def dpo_loss_aot(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aligned Optimal Transport (standard variant).

    Sorts policy and reference log-ratio differences independently.
    """
    _, _, _, chosen_rewards, rejected_rewards = _compute_logratios_and_rewards(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta
    )
    logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logratios_sorted, _ = torch.sort(logratios, dim=0)
    ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
    delta = logratios_sorted - ref_logratios_sorted
    losses = -F.logsigmoid(beta * delta) * (1 - label_smoothing) - F.logsigmoid(-beta * delta) * label_smoothing
    return losses.mean(), chosen_rewards, rejected_rewards
