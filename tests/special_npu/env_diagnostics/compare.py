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
"""Compare two collect.py reports; prioritize content over installation path/mtime noise."""

import argparse
import json
from pathlib import Path


def fingerprints(report):
    result = {}
    for name, entry in report.get("module_files", {}).items():
        for relative, info in entry.get("files", {}).items():
            result[f"module/{name}/{relative}"] = info.get("sha256", info.get("error", "unhashed"))
    for root, entry in report.get("cann", {}).items():
        for relative, info in entry.get("files", {}).items():
            # Ignore install root, retain subdirectory and every distinct observed binary.
            key = f"cann/{relative}"
            result.setdefault(key, set()).add(info.get("sha256", info.get("error", "unhashed")))
    for path, info in report.get("loaded_libraries", {}).items():
        result.setdefault(f"loaded/{Path(path).name}", set()).add(info.get("sha256", info.get("error", "unhashed")))
    for root, entry in report.get("assets", {}).items():
        for relative, info in entry.get("files", {"": entry}).items():
            key = f"asset/{Path(root).name}/{relative}"
            result.setdefault(key, set()).add(info.get("sha256", info.get("error", "unhashed")))
    return {k: sorted(v) if isinstance(v, set) else v for k, v in result.items()}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}/{key}" if prefix else key))
        return result
    return {prefix: value}


def selected(report):
    modules = report.get("probe", {}).get("modules", {})
    return {
        "image_digest": report.get("image_digest"),
        "collection_issues": report.get("collection_issues"),
        "environment": report.get("environment", {}),
        "packages": {k: v.get("version") for k, v in report.get("packages", {}).items()},
        "module_imports": {
            k: {field: v.get(field) for field in ("version", "git_version", "error")} for k, v in modules.items()
        },
        "python_version": report.get("python_version"),
        "device": report.get("probe", {}).get("device"),
        "torch_config": report.get("probe", {}).get("torch_config"),
        "acl_version": report.get("probe", {}).get("acl_version"),
        "precision_flags": report.get("probe", {}).get("precision_flags"),
        "git": {
            k: {
                "head": v.get("head", {}).get("stdout"),
                "diff_sha256": v.get("diff_sha256"),
                "status": v.get("status", {}).get("stdout"),
            }
            for k, v in report.get("git", {}).items()
        },
        "fingerprints": fingerprints(report),
        "runtime_system_files": report.get("system_files", {}),
        "runtime_commands": {k: v.get("stdout", v.get("error")) for k, v in report.get("commands", {}).items()},
    }


def differences(a, b):
    left, right = flatten(a), flatten(b)
    return [
        {
            "key": key,
            "left_present": key in left,
            "right_present": key in right,
            "left": left.get(key),
            "right": right.get(key),
        }
        for key in sorted(left.keys() | right.keys())
        if key not in left or key not in right or left[key] != right[key]
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path, help="CI report.json")
    p.add_argument("right", type=Path, help="local report.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full", action="store_true", help="另外比较原始报告全部字段，包括路径、时间等噪声")
    args = p.parse_args()
    a, b = json.loads(args.left.read_text()), json.loads(args.right.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    diff = differences(selected(a), selected(b))
    (args.output / "diff.json").write_text(json.dumps(diff, ensure_ascii=False, indent=2) + "\n")
    if args.full:
        (args.output / "full-diff.json").write_text(json.dumps(differences(a, b), ensure_ascii=False, indent=2) + "\n")
    stable = [d for d in diff if not d["key"].startswith("runtime_")]
    runtime = [d for d in diff if d["key"].startswith("runtime_")]
    lines = [
        f"Left (CI): {args.left}",
        f"Right (local): {args.right}",
        f"Software/config/content differences: {len(stable)}",
        f"System/runtime differences: {len(runtime)}",
        "Differences are leads, not proof of causation. Missing/failed probes are not equality.",
    ]
    lines += [
        "CI collection issues: " + json.dumps(a.get("collection_issues", []), ensure_ascii=False),
        "Local collection issues: " + json.dumps(b.get("collection_issues", []), ensure_ascii=False),
    ]
    for title, items in (("SOFTWARE / CONFIG / FILE CONTENT", stable), ("SYSTEM / RUNTIME (includes noise)", runtime)):
        lines.append("\n" + title)
        for d in items:
            lines += [
                "\n" + d["key"],
                "  CI: " + json.dumps(d["left"], ensure_ascii=False),
                "  local: " + json.dumps(d["right"], ensure_ascii=False),
            ]
    (args.output / "diff.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:7]))
    print(f"Details: {args.output / 'diff.txt'}")


if __name__ == "__main__":
    main()
