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

"""Diagnostics must remain useful when hardware probes fail, without masking errors."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def diagnostics(monkeypatch, tmp_path):
    # Load the stdlib-only recorder without importing torch or initializing an accelerator.
    path = Path(__file__).resolve().parents[2] / "verl/utils/hccl_diagnostics.py"
    spec = importlib.util.spec_from_file_location("hccl_diagnostics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VERL_HCCL_DIAGNOSTICS", "1")
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(tmp_path / "ascend"))
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-appear")
    # Avoid depending on which accelerator libraries the test runner has imported.
    monkeypatch.setattr(module, "sys", SimpleNamespace(modules={}))
    return module


def records(capsys):
    output = capsys.readouterr().out
    assert "must-not-appear" not in output
    return [json.loads(line.split("] ", 1)[1]) for line in output.splitlines()]


def test_original_exception_survives_failed_probe(diagnostics, monkeypatch, capsys, tmp_path):
    def unavailable():
        raise OSError("device query failed")

    monkeypatch.setattr(diagnostics, "_limits", unavailable)
    original = RuntimeError("hcclCommInitRootInfoConfig: resource unavailable")
    with pytest.raises(RuntimeError) as caught:
        with diagnostics.hccl_diagnostic_stage("fsdp.wrap_model", mesh_ranks=unavailable):
            raise original
    assert caught.value is original
    rows = records(capsys)
    assert [row["status"] for row in rows] == ["begin", "error"]
    assert rows[-1]["exception"] == str(original)
    assert "unavailable" in rows[-1]["limits"]
    assert "unavailable" in rows[-1]["mesh_ranks"]
    persisted = next((tmp_path / "ascend/worker-stages").glob("*.jsonl"))
    assert [json.loads(line) for line in persisted.read_text().splitlines()] == rows


def test_disabled_diagnostics_do_not_probe_or_write(diagnostics, monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("VERL_HCCL_DIAGNOSTICS")

    def forbidden():
        pytest.fail("disabled diagnostics queried hardware")

    monkeypatch.setattr(diagnostics, "_limits", forbidden)
    with diagnostics.hccl_diagnostic_stage("all_gather.init_process_group", tensor=forbidden):
        pass
    original = RuntimeError("original")
    with pytest.raises(RuntimeError) as caught:
        with diagnostics.hccl_diagnostic_stage("all_gather.all_gather_into_tensor"):
            raise original
    assert caught.value is original
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "ascend").exists()


def test_diagnostics_do_not_initialize_device(diagnostics, capsys):
    def forbidden():
        pytest.fail("queried an uninitialized device")

    diagnostics.sys.modules["torch"] = SimpleNamespace(
        __version__="stub",
        distributed=SimpleNamespace(is_initialized=lambda: False),
        npu=SimpleNamespace(is_initialized=lambda: False, current_device=forbidden),
    )
    with diagnostics.hccl_diagnostic_stage("process_group.init"):
        pass
    rows = records(capsys)
    assert [row["status"] for row in rows] == ["begin", "success"]
    assert all("npu" not in row for row in rows)


def test_failed_device_probe_keeps_rank_evidence(diagnostics, capsys):
    def unavailable():
        raise RuntimeError("NPU query failed")

    diagnostics.sys.modules["torch"] = SimpleNamespace(
        distributed=SimpleNamespace(
            is_initialized=lambda: True,
            get_rank=lambda: 7,
            get_world_size=lambda: 16,
            get_backend=lambda: "hccl",
        ),
        npu=SimpleNamespace(
            is_initialized=lambda: True,
            current_device=unavailable,
            memory_allocated=lambda: 1024,
            memory_reserved=lambda: 2048,
        ),
    )
    diagnostics.print_hccl_diagnostics("fsdp.wrap_model", status="error")
    row = records(capsys)[0]
    assert row["distributed"] == {"rank": 7, "world_size": 16, "backend": "hccl"}
    assert "NPU query failed" in row["npu"]["current_device"]


def test_log_report_excludes_old_contents(diagnostics, monkeypatch, tmp_path, capsys):
    log = tmp_path / "worker.log"
    log.write_text("old contents\n", encoding="utf-8")
    monkeypatch.setattr(diagnostics, "_ci_log_files", lambda: iter([log]))
    diagnostics.print_ci_diagnostic_report(baseline=True)
    capsys.readouterr()
    with log.open("a", encoding="utf-8") as stream:
        stream.write("::error::new contents\n")
    diagnostics.print_ci_diagnostic_report()
    output = capsys.readouterr().out
    assert "old contents" not in output
    assert "[VERL_ASCEND_LOG] ::error::new contents" in output
    assert not any(line.startswith("::") for line in output.splitlines())
