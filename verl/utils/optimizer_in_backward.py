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
"""Update parameters while the backward pass is still running.

This is the memory optimization requested in
`verl#144 <https://github.com/verl-project/verl/issues/144>`_, modelled after torchtune's
``create_optim_in_bwd_wrapper``
(https://github.com/pytorch/torchtune/blob/main/torchtune/training/memory.py).

The default training loop materializes *every* gradient of a mini-batch before
``optimizer.step()`` runs, so the peak is "all gradients + all optimizer state". Here each
parameter is updated as soon as its accumulated gradient is finalized, which releases that
gradient buffer and grows the optimizer state one parameter at a time.

A global gradient-norm clip needs all gradients, and those only co-exist once the backward
pass is over, so the wrapper runs in one of two modes:

``step_in_hooks=True`` (used when ``clip_grad`` is disabled)
    The real fusion: a ``post_accumulate_grad_hook`` steps the parameter the moment its
    gradient is ready. The reported grad norm is assembled from per-parameter partial norms
    and reduced once, after backward, with a single collective.

``step_in_hooks=False`` (used when ``clip_grad`` is enabled)
    Gradients stay materialized until the last micro-batch's backward returns. The caller
    then computes the grad norm / clip with exactly the same code as the non-fused path and
    :meth:`OptimizerInBackwardWrapper.finalize` steps one parameter at a time, releasing
    each gradient as it goes. Numerically identical to the standard path, but the peak
    gradient memory is not reduced -- clipping and per-parameter release are mutually
    exclusive because the clip coefficient needs every gradient.

The wrapper keeps the wrapped optimizer's ``state`` / ``param_groups`` / ``state_dict()``
authoritative, so checkpoints, LR schedulers and optimizer offload keep working unchanged.
Only :meth:`_step_param` temporarily narrows ``param_groups`` to a single parameter, which
is what lets one parameter be stepped without touching the others.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["OptimizerInBackwardWrapper", "create_optim_in_bwd_wrapper"]


class OptimizerInBackwardWrapper:
    """Per-parameter stepping for a ``torch.optim.Optimizer``.

    Args:
        optimizer: The fully built optimizer. It stays the owner of the optimizer state and
            keeps its own (unmodified) ``param_groups``.
        step_in_hooks: Whether parameters are stepped from ``post_accumulate_grad_hook``
            callbacks. Requires the absence of a global gradient-norm clip, see
            :mod:`verl.utils.optimizer_in_backward`.
        grad_norm_group: Process group that shards each parameter's gradient. The partial
            per-parameter squared norms are reduced over it once, after backward, to rebuild
            the global grad norm. Must be the group whose ranks *split* a parameter's
            gradient (the FSDP mesh), not the data-parallel replicas. ``None`` skips the
            reduction, which is correct for a single rank.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        step_in_hooks: bool = True,
        grad_norm_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.optimizer = optimizer
        self.step_in_hooks = step_in_hooks
        self._grad_norm_group = grad_norm_group

        self._params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]

        self._stepping = False
        self._single_param_groups: dict[torch.nn.Parameter, dict] = {}
        self._sq_norm = None
        self._nonfinite_params = 0

        if step_in_hooks:
            for param in self._params:
                param.register_post_accumulate_grad_hook(self._on_grad_ready)

    # ------------------------------------------------------------------
    # torch.optim.Optimizer compatible surface
    # ------------------------------------------------------------------
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    @property
    def defaults(self):
        return self.optimizer.defaults

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        self._sq_norm = None
        self._nonfinite_params = 0

    # ------------------------------------------------------------------
    # in-backward lifecycle
    # ------------------------------------------------------------------
    def begin_backward(self, *, stepping: bool):
        """Arm or disarm hook stepping for the micro-batch that is about to run.

        Only the last micro-batch of a mini-batch steps: the earlier ones must accumulate
        into ``param.grad`` like the standard path does.
        """
        self._stepping = stepping
        if stepping:
            self._sq_norm = None
            self._nonfinite_params = 0
            self._refresh_single_param_groups()

    def finalize(self, grad_norm: Optional[float] = None, *, step: bool = True) -> float:
        """Finish the in-backward step and return the grad norm to report.

        Args:
            grad_norm: Global grad norm already computed by the caller. ``None`` means the
                caller skipped clipping, so the norm is rebuilt from the partial norms that
                the hooks accumulated (hook mode only).
            step: When False, only the leftover gradients are released -- used when the
                caller decides the update must be skipped, e.g. for a non-finite grad norm.
        """
        self._stepping = False
        if grad_norm is None:
            grad_norm = self._reduce_partial_norm()
        if step:
            self._refresh_single_param_groups()
            for param in self._params:
                if param.grad is not None:
                    self._step_param(param)
        else:
            self.optimizer.zero_grad(set_to_none=True)
        self._sq_norm = None
        return grad_norm

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _on_grad_ready(self, param: torch.nn.Parameter):
        if not self._stepping or param.grad is None:
            return
        grad = param.grad
        self._accumulate_sq_norm(grad)
        if not torch.isfinite(self._local_view(grad)).all():
            # Earlier parameters are already updated by now, so the best we can do is leave
            # this one untouched; the caller reports the non-finite norm.
            self._nonfinite_params += 1
            return
        self._step_param(param)

    def _accumulate_sq_norm(self, grad: torch.Tensor):
        sq = self._local_view(grad).detach().float().pow(2).sum()
        self._sq_norm = sq if self._sq_norm is None else self._sq_norm + sq

    def _reduce_partial_norm(self) -> float:
        if self._sq_norm is None:
            return 0.0
        total = self._sq_norm
        if self._grad_norm_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(total, group=self._grad_norm_group)
        return total.sqrt().item()

    @staticmethod
    def _local_view(grad: torch.Tensor) -> torch.Tensor:
        # A DTensor gradient is sharded across the mesh; only the local shard is available
        # (and only it may be read without a collective inside an autograd hook).
        return grad.to_local() if hasattr(grad, "to_local") else grad

    def _refresh_single_param_groups(self):
        """Snapshot the optimizer's hyper-parameters into one single-parameter group each.

        Rebuilt before every step so LR-scheduler updates (which write into the optimizer's
        own ``param_groups``) are picked up without duplicating the schedule.
        """
        self._single_param_groups = {}
        for group in self.optimizer.param_groups:
            hyper = {k: v for k, v in group.items() if k != "params"}
            for param in group["params"]:
                self._single_param_groups[param] = {**hyper, "params": [param]}

    def _step_param(self, param: torch.nn.Parameter):
        group = self._single_param_groups[param]
        original_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [group]
        try:
            self.optimizer.step()
        finally:
            self.optimizer.param_groups = original_groups
        # Release this parameter's gradient right away: that is the whole point.
        param.grad = None


def create_optim_in_bwd_wrapper(
    optimizer: torch.optim.Optimizer,
    *,
    step_in_hooks: bool = True,
    grad_norm_group: Optional[torch.distributed.ProcessGroup] = None,
) -> OptimizerInBackwardWrapper:
    """Wrap ``optimizer`` so its updates happen during the backward pass."""
    return OptimizerInBackwardWrapper(
        optimizer,
        step_in_hooks=step_in_hooks,
        grad_norm_group=grad_norm_group,
    )
