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

import pytest

from scripts.verify_fp8_trainer_lineage import verify_lineage


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_pair(tmp_path):
    trainer = tmp_path / "trainer"
    rollout = tmp_path / "rollout"
    trainer.mkdir(parents=True)
    rollout.mkdir(parents=True)
    for root, prefix in ((trainer, "trainer"), (rollout, "rollout")):
        config = {"name": prefix}
        if root == rollout:
            config["quantization_config"] = {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        (root / "config.json").write_text(json.dumps(config))
        (root / "model.safetensors.index.json").write_text(prefix + " index")
    manifest = {
        "created_at": "ignored",
        "converter": {"formula": "bf16(fp32(weight) * fp32(scale))"},
        "source": {
            "repository": "org/model-fp8",
            "revision": "abc123",
            "source_manifest_sha256": "manifest",
            "config_sha256": _sha256(rollout / "config.json"),
            "index_sha256": _sha256(rollout / "model.safetensors.index.json"),
        },
        "output": {
            "config_sha256": _sha256(trainer / "config.json"),
            "index_sha256": _sha256(trainer / "model.safetensors.index.json"),
        },
    }
    (trainer / "conversion-manifest.json").write_text(json.dumps(manifest))
    return trainer, rollout


def test_verify_lineage(tmp_path):
    trainer, rollout = _checkpoint_pair(tmp_path)
    result = verify_lineage(trainer, rollout)
    assert result == {
        "conversion_manifest_sha256": result["conversion_manifest_sha256"],
        "converter": {"formula": "bf16(fp32(weight) * fp32(scale))"},
        "output_config_sha256": _sha256(trainer / "config.json"),
        "output_index_sha256": _sha256(trainer / "model.safetensors.index.json"),
        "quant_method": "fp8",
        "quantization_block_size": [128, 128],
        "repository": "org/model-fp8",
        "revision": "abc123",
        "source_manifest_sha256": "manifest",
        "status": "PASS",
    }


def test_verify_lineage_rejects_mismatched_rollout(tmp_path):
    trainer, rollout = _checkpoint_pair(tmp_path)
    (rollout / "config.json").write_text("different")
    with pytest.raises(ValueError, match="rollout_config SHA256 mismatch"):
        verify_lineage(trainer, rollout)


def test_lineage_identity_changes_with_trainer_output(tmp_path):
    first_trainer, first_rollout = _checkpoint_pair(tmp_path / "first")
    second_trainer, second_rollout = _checkpoint_pair(tmp_path / "second")
    first = verify_lineage(first_trainer, first_rollout)
    config_path = second_trainer / "config.json"
    config_path.write_text(json.dumps({"name": "different trainer"}))
    manifest_path = second_trainer / "conversion-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output"]["config_sha256"] = _sha256(config_path)
    manifest_path.write_text(json.dumps(manifest))
    second = verify_lineage(second_trainer, second_rollout)
    assert first["output_config_sha256"] != second["output_config_sha256"]
    assert first["conversion_manifest_sha256"] != second["conversion_manifest_sha256"]


def test_verify_lineage_rejects_non_fp8_rollout(tmp_path):
    trainer, rollout = _checkpoint_pair(tmp_path)
    config_path = rollout / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["quant_method"] = "other"
    config_path.write_text(json.dumps(config))
    manifest_path = trainer / "conversion-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["config_sha256"] = _sha256(config_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rollout checkpoint is not FP8"):
        verify_lineage(trainer, rollout)
