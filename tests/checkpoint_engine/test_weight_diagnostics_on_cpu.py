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
import hashlib
import json
import sys

import pytest
import torch

from tests.checkpoint_engine.weight_diagnostics import (
    diagnose_sender,
    diagnostic_env,
    fingerprint,
    mismatch_details,
    record,
    record_export_shards,
    record_loaded_weights,
    summarize,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_fingerprint_chunk_boundaries_and_layout(dtype):
    tensor = torch.arange(24).reshape(4, 6).to(dtype).T
    expected = hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    result = fingerprint(tensor, chunk_bytes=17)
    assert result == {"dtype": str(dtype), "shape": [6, 4], "sha256": expected}
    assert torch.equal(tensor, torch.arange(24).reshape(4, 6).to(dtype).T)


def test_reference_fingerprint_uses_comparison_dtype():
    reference = torch.tensor([0.5, -2.0], dtype=torch.bfloat16)
    assert fingerprint(reference, dtype=torch.float32) == fingerprint(reference.float())


def test_empty_and_scalar_fingerprints():
    assert fingerprint(torch.empty(0))["sha256"] == hashlib.sha256(b"").hexdigest()
    assert fingerprint(torch.tensor(1.0))["shape"] == []


def test_mismatch_details_handles_corruption_and_nonfinite():
    expected = torch.tensor([1.0, 2.0, float("nan"), float("inf")], dtype=torch.bfloat16)
    received = torch.tensor([1.0, 5.0, float("nan"), float("inf")])
    details = mismatch_details(expected, received, chunk_bytes=5)
    assert details["cpu_isclose_mismatches"] == 2
    assert details["nonfinite_mismatches"] == 1
    assert details["max_abs_finite"] == 3.0
    assert details["compared_dtype"] == "torch.float32"
    json.dumps(details, allow_nan=False)


def test_sender_is_lazy_and_preserves_tensor_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR", str(tmp_path))
    tensor = torch.tensor([1.0, 2.0])
    iterator = diagnose_sender(iter([("weight", tensor)]), 2)
    assert not list(tmp_path.glob("*.jsonl"))
    name, observed = next(iterator)
    assert name == "weight" and observed is tensor
    entry = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert entry["stage"] == "sender" and entry["update_index"] == 2


def test_summarize_distinguishes_export_and_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTIC_CASE", "hccl-3072")
    original, changed = torch.tensor([1.0]), torch.tensor([2.0])
    for name, source, received, reference in (
        ("good", original, original, original),
        ("transport", original, changed, original),
        ("export_or_reference", changed, changed, original),
    ):
        for stage, tensor in (("sender", source), ("received", received), ("reference", reference)):
            record(stage, name, 0, **fingerprint(tensor))
    record("received", "missing_sender", 0, **fingerprint(original))
    record("mismatch", "transport", 0, **mismatch_details(original, changed))
    # Case/update boundaries must not join a different sender to this receiver.
    record("sender", "missing_sender", 1, **fingerprint(original))
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTIC_CASE", "hccl-128")
    record("sender", "missing_sender", 0, **fingerprint(original))
    result = summarize(tmp_path)
    assert result["compared"] == 3 and result["incomplete"] == 1
    differences = {entry["name"]: entry for entry in result["differences"]}
    assert not differences["transport"]["sender_equals_received"]
    assert differences["export_or_reference"]["sender_equals_received"]
    assert not differences["export_or_reference"]["received_equals_reference"]
    assert result["assertion_failures"][0]["name"] == "transport"
    assert result["assertion_failures"][0]["cpu_isclose_mismatches"] == 1
    assert json.loads((tmp_path / "comparison.json").read_text()) == result


def test_diagnostic_env_does_not_copy_unrelated_settings(monkeypatch):
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR", "diagnostics")
    monkeypatch.setenv("VERL_CHECKPOINT_WEIGHT_FINGERPRINTS", "1")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-copied")
    assert diagnostic_env("hccl-128") == {
        "VERL_CHECKPOINT_DIAGNOSTICS_DIR": "diagnostics",
        "VERL_CHECKPOINT_WEIGHT_FINGERPRINTS": "1",
        "VERL_CHECKPOINT_DIAGNOSTIC_CASE": "hccl-128",
    }


def test_incomplete_jsonl_is_reported(tmp_path):
    (tmp_path / "interrupted.jsonl").write_text('{"stage":', encoding="utf-8")
    result = summarize(tmp_path)
    assert result["unreadable"] == 1
    assert result["compared"] == 0


@pytest.mark.parametrize("shards", [2, 8])
def test_loaded_shards_include_uneven_and_empty_slices(tmp_path, monkeypatch, shards):
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR", str(tmp_path))
    weight = torch.arange(15).reshape(5, 3)
    record_loaded_weights([("weight", weight)], shards)
    entries = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert entries[0]["stage"] == "loaded"
    local = entries[1:]
    assert len(local) == shards
    expected = list(weight.chunk(shards, dim=0))
    expected.extend([weight[:0]] * (shards - len(expected)))
    for rank, (entry, shard) in enumerate(zip(local, expected, strict=True)):
        assert entry["shard_rank"] == rank
        assert entry["sha256"] == fingerprint(shard)["sha256"]
        assert entry["shape"] == list(shard.shape)


def test_source_audit_separates_loading_sharding_and_gathering(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR", str(tmp_path))
    original, changed = torch.tensor([1.0]), torch.tensor([2.0])
    for name, loaded, shard, sender in (
        ("load", changed, changed, changed),
        ("shard", original, changed, changed),
        ("gather", original, original, changed),
    ):
        record_loaded_weights([(name, loaded)], 1)
        record("export_shard", name, 0, shard_rank=0, **fingerprint(shard))
        for stage, tensor in (("sender", sender), ("received", sender), ("reference", original)):
            record(stage, name, 0, **fingerprint(tensor))
    result = summarize(tmp_path)
    checks = {(item["name"], item["boundary"]): item["equal"] for item in result["source_checks"]}
    assert checks["load", "loaded_to_sender"] is True
    assert checks["load", "loaded_to_reference"] is False
    assert checks["shard", "loaded_to_export_shard"] is False
    assert checks["gather", "loaded_to_export_shard"] is True
    assert checks["gather", "loaded_to_sender"] is False


def test_source_audit_records_real_dtensor(tmp_path, monkeypatch):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    if not dist.is_gloo_available() or sys.platform == "win32":
        pytest.skip("The CPU DTensor integration check requires a Linux Gloo transport")
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR", str(tmp_path / "records"))
    dist.init_process_group("gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1)
    try:
        source = torch.arange(6.0).reshape(2, 3)
        distributed = distribute_tensor(source, init_device_mesh("cpu", (1,)), [Shard(0)])
        record_loaded_weights([("weight", source)], 1)
        record_export_shards({"weight": distributed}, 0, rank=0)
        checks = summarize(tmp_path / "records")["source_checks"]
        assert len(checks) == 1
        assert checks[0]["boundary"] == "loaded_to_export_shard" and checks[0]["equal"] is True
    finally:
        dist.destroy_process_group()
