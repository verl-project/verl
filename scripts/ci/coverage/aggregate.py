# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Bundle job archives with an index keyed by workflow filename; never import torch."""

import argparse
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import yaml


def aggregate(inputs, destination, workflow):
    """Stream the latest available attempt of each job into a namespaced archive."""
    definition = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    results = json.loads(os.environ.get("COVERAGE_NEEDS", "{}"))
    index = {
        "schema_version": 1,
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "source_sha": os.environ["GITHUB_SHA"],
        "selection_ready": False,
        "covstub": "covstub",
        "workflows": {},
    }
    for caller, job in definition["jobs"].items():
        if "uses" not in job:
            continue
        filename = PurePosixPath(job["uses"]).name
        index["workflows"][filename] = {
            "caller": caller,
            "result": results.get(caller, {}).get("result", "unknown"),
            "jobs": {},
        }
    selected = {}
    for archive in sorted(inputs.glob("*/coverage.tar.gz")):
        with tarfile.open(archive, "r:gz") as source:
            manifest = json.load(source.extractfile("manifest.json"))
        if str(manifest["run_id"]) != index["run_id"] or manifest["source_sha"] != index["source_sha"]:
            raise ValueError(f"Mismatched run or source SHA: {archive}")
        filename, job = manifest["workflow"], manifest["job"]
        if filename not in index["workflows"] or not re.fullmatch(r"[\w-]+", job):
            raise ValueError(f"Unexpected workflow/job: {filename}/{job}")
        attempt = int(manifest["run_attempt"])
        if attempt > int(index["run_attempt"]):
            raise ValueError(f"Future attempt: {archive}")
        key = (filename, job)
        previous = selected.get(key)
        if previous and attempt == previous[0]:
            raise ValueError(f"Duplicate job artifact: {filename}/{job}/{attempt}")
        if previous is None or attempt > previous[0]:
            selected[key] = (attempt, archive, manifest)
    if not selected:
        raise ValueError("No job coverage archives were downloaded")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    # Read committed source once, independently of workload environments or reruns.
    snapshot = subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "archive", "--format=tar", "--prefix=covstub/", index["source_sha"], "verl"],
        cwd=Path(__file__).resolve().parents[3],
    )
    with tarfile.open(temporary, "w:gz") as output:
        with tarfile.open(fileobj=io.BytesIO(snapshot), mode="r:") as source:
            for member in source:
                output.addfile(member, source.extractfile(member) if member.isfile() else None)
        for (filename, job), (attempt, archive, manifest) in sorted(selected.items()):
            prefix = f"workflows/{filename}/{job}"
            index["workflows"][filename]["jobs"][job] = {
                "directory": prefix,
                "manifest": f"{prefix}/manifest.json",
                "run_attempt": attempt,
                "collection_complete": manifest["collection_complete"],
                "steps": {
                    entry["id"]: {
                        "name": entry["name"],
                        "directory": f"{prefix}/{entry['directory']}" if entry.get("directory") else None,
                        "status": entry["status"],
                    }
                    for entry in manifest["targets"]
                },
            }
            with tarfile.open(archive, "r:gz") as source:
                seen = set()
                for member in source:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                        raise ValueError(f"Unsafe member: {member.name}")
                    if not (member.isfile() or member.isdir()) or member.name in seen:
                        raise ValueError(f"Unsupported or duplicate member: {member.name}")
                    seen.add(member.name)
                    if path.name == "command.sh" or (path.name.startswith("process-") and path.suffix == ".json"):
                        continue
                    # Older attempts may still contain a redundant source snapshot.
                    if path.parts[0] == "covstub":
                        continue
                    data = source.extractfile(member) if member.isfile() else None
                    member.name = f"{prefix}/{path.as_posix()}"
                    # An inherited PAX path overrides member.name when the tar is read.
                    # Let tarfile regenerate it from the namespaced name when needed.
                    member.pax_headers.pop("path", None)
                    output.addfile(member, data)
        for item in index["workflows"].values():
            item["artifacts_present"] = bool(item["jobs"])
        payload = (json.dumps(index, indent=2) + "\n").encode()
        member = tarfile.TarInfo("index.json")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))
    temporary.replace(destination)
    print(f"Bundled {len(selected)} jobs into {destination}")


def main():
    """Accept artifact download directory and final archive path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workflow", type=Path, default=Path(".github/workflows/coverage-collection-npu.yml"))
    args = parser.parse_args()
    aggregate(args.inputs, args.destination, args.workflow)


if __name__ == "__main__":
    main()
