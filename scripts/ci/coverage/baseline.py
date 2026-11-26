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

"""Publish and select Ascend coverage baselines for PR recommendations."""

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


def expected_jobs():
    workflows = Path(".github/workflows")

    def enabled(jobs):
        # An explicitly paused job is outside the plan. A runtime skip still counts.
        return {name: job for name, job in jobs.items() if job.get("if") not in (False, "${{ false }}")}

    caller = yaml.safe_load((workflows / "coverage-collection-npu.yml").read_bytes())
    expected = {}
    for call in enabled(caller["jobs"]).values():
        if call.get("with", {}).get("coverage_mode") is True:
            path = Path(call["uses"])
            expected[path.name] = set(enabled(yaml.safe_load(path.read_bytes())["jobs"]))
    return expected


def assess(directory, index, expected):
    if not (directory / "covstub/verl/__init__.py").is_file():
        raise ValueError("Missing source snapshot")
    coverage = json.loads((directory / "test_case_map.json").read_bytes())
    covered = set()
    for name, entry in coverage.items():
        files = entry["files"]
        if not isinstance(files, dict) or not all(
            isinstance(lines, list) and all(type(line) is int for line in lines) for lines in files.values()
        ):
            raise ValueError(f"Invalid file coverage for {name}")
        if not all(isinstance(entry[key], int) for key in ("file_count", "line_count")):
            raise ValueError(f"Invalid coverage counts for {name}")
        if any(files.values()):
            covered.add(name)

    covered.intersection_update(expected)
    if not covered:
        raise ValueError("Map covers no enabled workflows")
    missing = []
    for name, jobs in expected.items():
        if name not in covered:
            missing.append(f"{name}: no mapped files")
        archived = index["workflows"].get(name, {}).get("jobs", {})
        for job in sorted(jobs):
            if job not in archived:
                missing.append(f"{name}/{job}: missing archive")
            elif archived[job]["collection_complete"] is not True:
                missing.append(f"{name}/{job}: incomplete collection")
    return {"complete": not missing, "covered_workflows": sorted(covered), "missing": missing}


def api_pages(endpoint, key, *filters):
    output = subprocess.check_output(["gh", "api", "-X", "GET", "--paginate", "--slurp", endpoint, *filters])
    return [item for page in json.loads(output) for item in page[key]]


def candidates(repo):
    api = f"repos/{repo}/actions"
    since = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = api_pages(
        f"{api}/workflows/coverage-collection-npu.yml/runs",
        "workflow_runs",
        "-f",
        f"branch={os.environ['BASE_REF']}",
        "-f",
        "status=completed",
        "-f",
        f"created=>={since}",
    )
    for run in sorted(runs, key=lambda row: (row["created_at"], row["id"]), reverse=True):
        if run["event"] not in ("schedule", "workflow_dispatch"):
            continue
        try:
            artifacts = api_pages(f"{api}/runs/{run['id']}/artifacts?per_page=100", "artifacts")
        except subprocess.CalledProcessError:
            print(f"::warning::Could not list artifacts for run {run['id']}")
            continue
        attempts = []
        for artifact in artifacts:
            match = re.fullmatch(r"ascend-precision-map-(\d+)", artifact["name"])
            if match and not artifact["expired"]:
                attempts.append((int(match[1]), artifact["name"]))
        for attempt, name in sorted(attempts, reverse=True):
            yield run, attempt, name


def select(destination, expected):
    repo = os.environ["GITHUB_REPOSITORY"]
    best_count = 0
    for run, attempt, name in candidates(repo):
        run_id = str(run["id"])
        with TemporaryDirectory(dir=destination.parent) as temp:
            candidate = Path(temp) / "download"
            try:
                subprocess.run(
                    ["gh", "run", "download", run_id, "--repo", repo, "--name", name, "--dir", candidate], check=True
                )
                index = json.loads((candidate / "index.json").read_bytes())
                if (str(index["run_id"]), int(index["run_attempt"]), index["source_sha"]) != (
                    run_id,
                    attempt,
                    run["head_sha"],
                ):
                    raise ValueError("Index does not match the collection run")
                quality = assess(candidate, index, expected)
            except (subprocess.CalledProcessError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                print(f"::warning::Skipping {run_id}/{name}: {error}")
                continue
            count = len(quality["covered_workflows"])
            if quality["complete"] or count > best_count:
                if best_count:
                    shutil.rmtree(destination)
                candidate.rename(destination)
                best_count = count
                best_run = run_id
                print(f"Selected {run_id}/{name}: {count} workflows, complete={quality['complete']}")
            if quality["complete"]:
                break
    if best_count:
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            print(f"run={best_run}", file=output)
    else:
        print("::warning::No usable coverage baseline. Using PR path filters.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["publish", "select"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    expected = expected_jobs()
    if args.operation == "select":
        select(args.directory, expected)
    else:
        path = args.directory / "index.json"
        index = json.loads(path.read_bytes())
        index["baseline_quality"] = assess(args.directory, index, expected)
        print(json.dumps(index["baseline_quality"]))
        path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
