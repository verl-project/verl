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
"""The opt-in GRPO restore probe must detect state corruption, not just successful loading."""

import pytest
import torch


def test_restore_probe_matches_nested_state():
    """Accept identical nested state and count the tensors checked by the restore probe."""
    from tests.special_e2e.fsdp2_muon.restore_probe import assert_same_state

    state = {"state": {0: {"momentum_buffer": torch.ones(2, 3)}}, "groups": [{"lr": 1e-6}], "step": 15}
    assert assert_same_state(state, state) == 1


@pytest.mark.parametrize("kind", ["tensor", "scheduler", "missing", "nan"])
def test_restore_probe_rejects_changed_state(kind):
    """Reject altered tensors, scheduler metadata, missing keys and nonfinite values."""
    from tests.special_e2e.fsdp2_muon.restore_probe import assert_same_state

    expected = {"momentum": torch.ones(2), "last_epoch": 15}
    actual = {"momentum": torch.ones(2), "last_epoch": 15}
    if kind == "tensor":
        actual["momentum"][0] += 0.01
    elif kind == "scheduler":
        actual["last_epoch"] = 0
    elif kind == "missing":
        actual.pop("momentum")
    else:
        expected["momentum"][0] = float("nan")
        actual["momentum"][0] = float("nan")
    with pytest.raises(AssertionError):
        assert_same_state(actual, expected)


def test_restore_probe_rejects_optimized_python():
    """Fail explicitly when Python optimization would disable restore assertions."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from tests.special_e2e.fsdp2_muon.restore_probe import assert_same_state; assert_same_state(0, 1)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "assertions enabled" in result.stderr


def test_restore_probe_wraps_loader_with_tensor_learning_rate(tmp_path, monkeypatch):
    """Verify the loader wrapper reports exact restoration with a tensor-valued LR."""
    import json
    from types import SimpleNamespace

    from tests.special_e2e.fsdp2_muon.restore_probe import install_restore_probe
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(1e-6))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    suffix = "world_size_1_rank_0.pt"
    torch.save(model.state_dict(), tmp_path / f"model_{suffix}")
    torch.save(optimizer.state_dict(), tmp_path / f"optim_{suffix}")
    torch.save({"lr_scheduler": scheduler.state_dict()}, tmp_path / f"extra_state_{suffix}")
    with torch.no_grad():
        model.weight.zero_()
    calls = []

    def real_load_stub(manager, local_path):
        """Load fixture state files while recording the intercepted checkpoint call."""
        calls.append(local_path)
        manager.model.load_state_dict(torch.load(tmp_path / f"model_{suffix}", weights_only=False))
        manager.optimizer.load_state_dict(torch.load(tmp_path / f"optim_{suffix}", weights_only=False))
        manager.lr_scheduler.load_state_dict(
            torch.load(tmp_path / f"extra_state_{suffix}", weights_only=False)["lr_scheduler"]
        )

    monkeypatch.setattr(FSDPCheckpointManager, "load_checkpoint", real_load_stub)
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(tmp_path / "metrics.jsonl"))
    install_restore_probe()
    manager = SimpleNamespace(model=model, optimizer=optimizer, lr_scheduler=scheduler, rank=0, world_size=1)
    FSDPCheckpointManager.load_checkpoint(manager, str(tmp_path))
    record = json.loads((tmp_path / "restore_rank_0.json").read_text())
    assert calls == [str(tmp_path)]
    assert record["exact_restore"] is True
    assert record["tensor_counts"]["model"] == 2
    assert record["tensor_counts"]["optim"] > 0
