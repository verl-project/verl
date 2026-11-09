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
"""Opt-in, read-only assertions immediately after the real GRPO checkpoint load.

Enable only for the resume experiment with VERL_MUON_VERIFY_RESTORE=1 and
actor_rollout_ref.model.external_lib=tests.special_e2e.fsdp2_muon.restore_probe.
The production loader is called unchanged before inspecting its resulting state.
"""

import functools
import json
import os
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor


def assert_same_state(actual, expected, path="state"):
    """Assert exact, finite tensor and nested metadata equality; count tensors."""
    if not __debug__:
        raise RuntimeError("Restore verification requires assertions enabled; do not use python -O/PYTHONOPTIMIZE")
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        if isinstance(actual, DTensor):
            actual = actual.to_local()
        if isinstance(expected, DTensor):
            expected = expected.to_local()
        actual, expected = actual.detach().cpu(), expected.detach().cpu()
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, path
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), path
        assert torch.equal(actual, expected), path
        return 1
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        return sum(assert_same_state(actual[k], v, f"{path}.{k}") for k, v in expected.items())
    if isinstance(expected, tuple | list):
        assert isinstance(actual, tuple | list) and len(actual) == len(expected), path
        return sum(
            assert_same_state(a, e, f"{path}[{i}]") for i, (a, e) in enumerate(zip(actual, expected, strict=True))
        )
    assert actual == expected, path
    return 0


def install_restore_probe():
    """Wrap only checkpoint loading, with no changes to forward, loss or update."""
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    original = FSDPCheckpointManager.load_checkpoint
    if getattr(original, "_muon_restore_probe", False):
        return

    @functools.wraps(original)
    def checked_load(self, local_path, *args, **kwargs):
        result = original(self, local_path, *args, **kwargs)
        if local_path is None:
            return result
        root = Path(local_path)
        suffix = f"world_size_{self.world_size}_rank_{self.rank}.pt"
        counts = {}
        # FSDP2 state_dict exposes the same DTensor shards used by the loader.
        for name, state in (("model", self.model.state_dict()), ("optim", self.optimizer.state_dict())):
            expected = torch.load(root / f"{name}_{suffix}", map_location="cpu", weights_only=False)
            counts[name] = assert_same_state(state, expected, name)
            del expected
        extra = torch.load(root / f"extra_state_{suffix}", map_location="cpu", weights_only=False)
        assert_same_state(self.lr_scheduler.state_dict(), extra["lr_scheduler"], "scheduler")
        record = {
            "rank": self.rank,
            "checkpoint": str(root),
            "tensor_counts": counts,
            "scheduler_last_epoch": int(self.lr_scheduler.last_epoch),
            "exact_restore": True,
        }
        target = Path(os.environ["VERL_FILE_LOGGER_PATH"]).parent / f"restore_rank_{self.rank}.json"
        target.write_text(json.dumps(record, indent=2) + "\n")
        print(f"MUON_GRPO_RESTORE_EXACT rank={self.rank} counts={counts}", flush=True)
        return result

    checked_load._muon_restore_probe = True
    FSDPCheckpointManager.load_checkpoint = checked_load


if os.environ.get("VERL_MUON_VERIFY_RESTORE") == "1":
    install_restore_probe()
