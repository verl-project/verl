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

"""Initialize, execute, and archive Ascend CI coverage; never import torch here."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HOOK_NAME = "verl_ascend_ci_coverage.pth"


def output_dir():
    """Use the unique job directory declared by the coverage setup action."""
    return Path(os.environ["VERL_COVERAGE_ROOT"]).resolve()


def collection_plan(source_workflow, job_id):
    """Read instrumented steps directly from the reusable workflow definition."""
    import yaml

    workflow = yaml.safe_load(source_workflow.read_text(encoding="utf-8"))
    plan = []
    for step in workflow["jobs"][job_id]["steps"]:
        match = re.search(r"scripts/ci/coverage/collect\.py run ([\w-]+) \{0\}", step.get("shell", ""))
        if match is None:
            continue
        plan.append(
            {
                "id": match[1],
                "name": step["name"],
                "entrypoints": sorted(set(re.findall(r"(?:tests|examples)/[\w./-]+\.(?:sh|py)", step["run"]))),
                "requires_verl_coverage": "examples/data_preprocess/" not in step["run"],
            }
        )
    if not plan or len({entry["id"] for entry in plan}) != len(plan):
        raise ValueError(f"Missing or duplicate coverage step identities in {source_workflow.name}/{job_id}")
    return plan


def write_json(path, value):
    """Replace metadata atomically so interruptions leave a readable manifest."""
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize():
    """Snapshot the source and install an inert hook in the workload interpreter."""
    import coverage

    output = output_dir()
    output.mkdir(parents=True, exist_ok=False)
    source_workflow = ROOT / ".github/workflows" / os.environ["VERL_COVERAGE_WORKFLOW"]
    plan = collection_plan(source_workflow, os.environ["VERL_COVERAGE_JOB"])
    manifest = {
        "schema_version": 1,
        "workflow": os.environ["VERL_COVERAGE_WORKFLOW"],
        "job": os.environ["VERL_COVERAGE_JOB"],
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "ref": os.environ.get("GITHUB_REF"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "coverage_version": coverage.__version__,
        "collection_complete": False,
        "worker_coverage_verified": False,
        "selection_ready": False,
        "targets": [{**entry, "status": "pending"} for entry in plan],
    }
    write_json(output / "manifest.json", manifest)
    shutil.copytree(ROOT / "verl", output / "covstub" / "verl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(source_workflow, output / "source-workflow.yml")
    shutil.copyfile(ROOT / "scripts/ci/npu_pytest_suites.json", output / "npu_pytest_suites.json")
    with (output / "packages.txt").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    (output / "coveragerc").write_text(
        "[run]\nbranch = True\nparallel = True\nconcurrency = multiprocessing,thread\nsigterm = True\n"
        f"source = {ROOT / 'verl'}\ndata_file = ${{COVERAGE_FILE?}}\n",
        encoding="utf-8",
    )
    hook = Path(sysconfig.get_path("purelib")) / HOOK_NAME
    # Keep existing sitecustomize and PYTHONPATH intact. Refuse to overwrite another hook.
    startup = str(Path(__file__).with_name("startup.py"))
    with hook.open("x", encoding="utf-8") as stream:
        stream.write(f"import os, runpy; os.environ.get('COVERAGE_PROCESS_START') and runpy.run_path({startup!r})\n")
    manifest["hook"] = str(hook)
    write_json(output / "manifest.json", manifest)


def stop_ray(env):
    """Stop the container-local cluster gracefully before changing attribution."""
    if not shutil.which("ray"):
        return
    # Do not instrument the cleanup CLI itself.
    clean_env = {key: value for key, value in env.items() if not key.startswith("COVERAGE_")}
    subprocess.run(["ray", "stop", "--grace-period=30"], env=clean_env, check=True, timeout=120)


def run_step(step_id, script):
    """Execute an unchanged Actions run block with step-specific coverage identity."""
    if os.environ.get("VERL_COVERAGE_ENABLED", "").lower() != "true":
        # Match Actions' default container shell without loading coverage or PyYAML.
        result = subprocess.run(["sh", "-e", script]).returncode
        return result if result >= 0 else 128 - result
    output = output_dir()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["targets"] if item["id"] == step_id)
    directory = output / "coverage" / step_id
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(script, directory / "command.sh")
    entry.update(status="running", directory=directory.relative_to(output).as_posix())
    write_json(manifest_path, manifest)
    env = os.environ.copy()
    env.update(COVERAGE_PROCESS_START=str(output / "coveragerc"), COVERAGE_FILE=str(directory / "coverage"))
    # Ensures tests import this checkout even in images without an editable install.
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = 1
    try:
        stop_ray(env)
        # Includes any original ray start command, now inside the attribution scope.
        result = subprocess.run(["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", script], env=env).returncode
        entry["exit_code"] = result
        entry["status"] = "passed" if result == 0 else "failed"
    finally:
        try:
            stop_ray(env)
        except (subprocess.SubprocessError, OSError) as error:
            entry["cleanup_error"] = str(error)
            result = result or 1
        write_json(manifest_path, manifest)
    return result if result >= 0 else 128 - result


def package():
    """Preserve partial raw data and report completeness without building a map."""
    import coverage

    output = output_dir()
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Coverage initialization did not produce a manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["targets"]:
        directory = output / "coverage" / entry["id"]
        files = sorted(directory.glob("coverage.*"))
        entry["coverage_files"] = len(files)
        entry["has_branch_data"] = False
        errors = []
        for path in files:
            try:
                data = coverage.CoverageData(basename=str(path))
                data.read()
                entry["has_branch_data"] |= data.has_arcs() and any(data.arcs(name) for name in data.measured_files())
            except Exception as error:
                errors.append(f"{path.name}: {error}")
        processes = []
        for path in directory.glob("process-*.json"):
            try:
                processes.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as error:
                errors.append(f"{path.name}: {error}")
        entry["processes_started"] = len(processes)
        entry["processes_exited"] = sum(item["status"] == "exited" for item in processes)
        entry["hook_errors"] = sum(item["status"] == "hook_failed" for item in processes)
        entry["processes_without_data"] = sum(
            not item.get("data_file") or not Path(item["data_file"]).is_file() for item in processes
        )
        entry["data_errors"] = errors
        entry["data_complete"] = bool(
            entry["status"] == "passed"
            and not entry.get("cleanup_error")
            and not errors
            and not entry["hook_errors"]
            and processes
            and not entry["processes_without_data"]
            and (entry["has_branch_data"] or not entry["requires_verl_coverage"])
        )
        if entry["status"] == "pending":
            entry["status"] = "not_run"
        if not entry["data_complete"]:
            print(f"::warning::Incomplete collection: {entry['name']}")
    # Worker completeness is deliberately separate: real distributed CI inspection is still required.
    manifest["collection_complete"] = all(entry["data_complete"] for entry in manifest["targets"])
    write_json(manifest_path, manifest)
    if manifest.get("hook"):
        Path(manifest["hook"]).unlink(missing_ok=True)
    archive = output.parent / "coverage.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for child in sorted(output.iterdir()):
            stream.add(child, arcname=child.name)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"### {manifest['workflow']} / {manifest['job']}\n\n")
            stream.write("| Step | Status | Raw files | Branch data |\n|---|---|---:|---|\n")
            for entry in manifest["targets"]:
                stream.write(
                    f"| {entry['name']} | {entry['status']} | {entry['coverage_files']} | "
                    f"{entry['has_branch_data']} |\n"
                )
            stream.write("\nWorker attribution is unverified; these artifacts are not selection-ready.\n")
    return 0 if manifest["collection_complete"] else 1


def main():
    """Dispatch the workflow setup, custom shell, or finalizer command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "run", "package"])
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    if args.command == "init":
        initialize()
        return 0
    if args.command == "run":
        return run_step(*args.args)
    return package()


if __name__ == "__main__":
    raise SystemExit(main())
