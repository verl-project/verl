# Copyright 2025 Individual Contributor: Steven Zhang
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
"""Summarize per-worker diagnostics without importing accelerator libraries."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(directory):
    """Compare only same-model local transfer pairs, never incompatible TP layouts."""
    result = {"stages": Counter(), "compared": 0, "mismatches": [], "nonfinite": [], "errors": [], "unpaired": 0}
    for path in sorted(Path(directory).glob("*.jsonl")):
        pending = {}
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                stage = event["stage"]
                result["stages"][stage] += 1
                tensors = {t["name"]: t for t in event["tensors"]}
                for name, tensor in tensors.items():
                    if tensor.get("nonfinite", 0):
                        result["nonfinite"].append({"file": path.name, "line": number, "stage": stage, "name": name})
                operation, _, position = stage.partition(".")
                if operation not in ("offload", "reload"):
                    continue
                key = (operation, event["model_id"], event["call"])
                if position == "before":
                    pending[key] = tensors
                elif position == "after":
                    before = pending.pop(key, None)
                    if before is None:
                        result["unpaired"] += 1
                        continue
                    if before.keys() != tensors.keys():
                        result["errors"].append({"file": path.name, "line": number, "tensor_names_changed": True})
                    for name, tensor in tensors.items():
                        old = before.get(name, {})
                        if "sha256" not in old or "sha256" not in tensor:
                            continue
                        if any(old.get(k) != tensor.get(k) for k in ("local_shape", "dtype", "placements")):
                            result["errors"].append({"file": path.name, "line": number, "layout_changed": name})
                            continue
                        result["compared"] += 1
                        if old["sha256"] != tensor["sha256"]:
                            result["mismatches"].append(
                                {"file": path.name, "line": number, "stage": stage, "name": name}
                            )
            except (ValueError, KeyError, TypeError) as exc:
                result["errors"].append({"file": path.name, "line": number, "error": str(exc)})
        result["unpaired"] += len(pending)
    result["interpretation"] = (
        "No tensor values compared; metadata alone cannot establish weight integrity."
        if not result["compared"]
        else "Only recorded same-process local transfer pairs compared; values mode changes synchronization."
    )
    return result


def main():
    """Write the complete summary and print bounded counts for Actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
