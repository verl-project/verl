# Copyright 2026 Individual Contributor: yue-zeng-yue
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
"""Correctness-first FSDP2 transport for pinned Dion Muon; imported only on demand."""

import torch
from dion import Muon
from dion.megabatch_base import adjust_lr_rms_norm, adjust_lr_spectral_norm, megabatch_orthogonalize_async
from dion.muon import muon_update_post_orthogonalize, muon_update_pre_orthogonalize
from dion.opt_utils import AsyncTask, as_scalar_tensor, to_local
from torch.distributed.tensor import DTensor


class FSDP2Muon(Muon):
    """Keep Dion's optimizer/state API, replacing only its matrix transport.

    Every rank gathers the BF16 orthogonalizer input using each parameter's
    actual DTensor layout. Dion's *unsharded* shape-batching path then sees the
    same matrix dimensions/batch count as the reference, without padding rows.
    Its momentum, Newton--Schulz, LR scaling, decay and AdamW kernels are reused.

    This replicates full-matrix NS compute on every rank and temporarily holds
    a full same-shape batch. It is intentionally not a performance optimization.
    Momentum and all AdamW state remain in the inherited optimizer.state.
    """

    def _create_ortho_tasks(self, param_groups):
        for group in param_groups:
            shape_groups = {}
            for param in group["params"]:
                if param.grad is not None:
                    shape_groups.setdefault(tuple(param.shape), []).append(param)
            for params in shape_groups.values():
                yield AsyncTask(self._gathered_update(params, group))

    def _gathered_update(self, params, group):
        # Same compiled local momentum update as pinned Dion's native path.
        updates = muon_update_pre_orthogonalize(
            G=to_local([param.grad for param in params]),
            M=to_local([self.state[param]["momentum"] for param in params]),
            momentum=torch.tensor(group["mu"]),
            nesterov=group["nesterov"],
        )
        full_updates = [
            DTensor.from_local(
                update,
                device_mesh=param.device_mesh,
                placements=param.placements,
                run_check=False,
                shape=param.shape,
                stride=param.stride(),
            ).full_tensor()
            for param, update in zip(params, updates, strict=True)
        ]
        # None explicitly selects the local computation path *after* the real
        # parameter mesh has gathered complete matrices. No default PG is used.
        full_updates = yield from megabatch_orthogonalize_async(
            full_updates,
            comm_dim=None,
            device_rank=0,
            world_size=1,
            process_group=None,
            newton_schulz_func=self._newton_schulz_func,
            flatten=False,
            epsilon=torch.tensor(group["epsilon"]),
            global_comm_dim_size=None,
        )
        local_updates = []
        for param, update in zip(params, full_updates, strict=True):
            # Validated 1D Shard(0), contiguous FSDP2 chunking; using the mesh's
            # local rank also handles submeshes that exclude global rank zero.
            rows = (param.shape[0] + param.device_mesh.size() - 1) // param.device_mesh.size()
            start = min(param.device_mesh.get_local_rank() * rows, param.shape[0])
            local_updates.append(update.narrow(0, start, param.to_local().shape[0]).contiguous())
        adjusted_lr = group["lr"]
        if group["adjust_lr"] == "spectral_norm":
            adjusted_lr = adjust_lr_spectral_norm(adjusted_lr, params[0].shape, flatten=False)
        elif group["adjust_lr"] == "rms_norm":
            adjusted_lr = adjust_lr_rms_norm(adjusted_lr, params[0].shape, flatten=False)
        muon_update_post_orthogonalize(
            X=to_local(params),
            U=local_updates,
            base_lr=group["lr"],
            adjusted_lr=adjusted_lr,
            weight_decay=as_scalar_tensor(group["weight_decay"]),
            cautious_wd=False,
        )
