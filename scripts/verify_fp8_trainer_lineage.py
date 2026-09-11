#!/usr/bin/env python3
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

"""Verify that a BF16 trainer checkpoint derives from the FP8 rollout base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    canonical = {key: value for key, value in manifest.items() if key != "created_at"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_lineage(trainer: Path, rollout: Path) -> dict[str, Any]:
    manifest_path = trainer / "conversion-manifest.json"
    with manifest_path.open() as stream:
        manifest = json.load(stream)

    source = manifest["source"]
    output = manifest["output"]
    rollout_config_path = rollout / "config.json"
    checks = {
        "trainer_config": (trainer / "config.json", output["config_sha256"]),
        "trainer_index": (
            trainer / "model.safetensors.index.json",
            output["index_sha256"],
        ),
        "rollout_config": (rollout_config_path, source["config_sha256"]),
        "rollout_index": (
            rollout / "model.safetensors.index.json",
            source["index_sha256"],
        ),
    }
    for name, (path, expected) in checks.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{name} SHA256 mismatch: {actual} != {expected}")

    with rollout_config_path.open() as stream:
        rollout_config = json.load(stream)
    quantization_config = rollout_config.get("quantization_config", {})
    if quantization_config.get("quant_method") != "fp8":
        raise ValueError("rollout checkpoint is not FP8")

    return {
        "conversion_manifest_sha256": canonical_manifest_sha256(manifest),
        "converter": manifest["converter"],
        "output_config_sha256": output["config_sha256"],
        "output_index_sha256": output["index_sha256"],
        "quant_method": quantization_config["quant_method"],
        "quantization_block_size": quantization_config.get("weight_block_size"),
        "repository": source["repository"],
        "revision": source["revision"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_lineage(args.trainer, args.rollout), sort_keys=True))


if __name__ == "__main__":
    main()
