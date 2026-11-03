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

"""Collect NPU coverage on CI runners; importing this module does not import torch."""

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITES = json.loads(Path(__file__).with_name("npu_pytest_suites.json").read_text())
DISTRIBUTED_TARGET = "tests/special_distributed/test_fsdp2_model_cpu_offload.py"


def run_pytest_suite(name):
    """Run the same pytest arguments as PR CI, without putting them in sys.argv."""
    import pytest

    raise SystemExit(pytest.main(SUITES[name]))


def collect_targets():
    """Collect selected node IDs in a separate, uninstrumented pytest process."""
    import pytest

    class Inventory:
        def pytest_collection_finish(self, session):
            Path(os.environ["NPU_INVENTORY_FILE"]).write_text(
                json.dumps([item.nodeid for item in session.items]), encoding="utf-8"
            )

    args = SUITES[os.environ["NPU_SUITE"]] + ["--collect-only", "-q"]
    raise SystemExit(pytest.main(args, plugins=[Inventory()]))


def pytest_command(args):
    """Keep pytest arguments away from MegatronAdaptor's import-time argv parser."""
    return [sys.executable, "-c", f"import pytest; raise SystemExit(pytest.main({args!r}))"]


def main():
    """Run each test file separately and preserve its raw branch coverage and status."""
    import coverage

    os.chdir(ROOT)
    output = Path(os.environ["NPU_COVERAGE_DIR"]).resolve()
    # Never mix an earlier run's data into this collection.
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "covstub" / "verl"
    shutil.copytree(ROOT / "verl", snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = {
        "schema_version": 1,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "ref": os.environ.get("GITHUB_REF"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "coverage_version": coverage.__version__,
        "scope": "NPU groups in npu_unit_tests.yml (excluding CPU checkpoint tests)",
        "collection_complete": False,
        # This first stage does not certify Ray worker attribution or build a selector map.
        "selection_ready": False,
        "targets": [],
    }

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    with (output / "packages.txt").open("w", encoding="utf-8") as packages:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=packages, check=True)
    shutil.copyfile(Path(__file__).with_name("npu_pytest_suites.json"), output / "npu_pytest_suites.json")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Ignore inherited instrumentation during inventory collection and orchestration.
    env.pop("COVERAGE_PROCESS_START", None)
    env.pop("COVERAGE_FILE", None)
    commands = []
    for suite in SUITES:
        inventory = output / f"inventory-{suite}.json"
        collect_env = {**env, "NPU_SUITE": suite, "NPU_INVENTORY_FILE": str(inventory)}
        subprocess.run(
            [sys.executable, "-c", "from scripts.ci.collect_npu_coverage import collect_targets; collect_targets()"],
            env=collect_env,
            check=True,
        )
        nodeids = json.loads(inventory.read_text())
        if not nodeids:
            raise RuntimeError(f"No tests collected for {suite}")
        # Keep the original -k expression/exclusions. Only replace the target path.
        files = sorted({nodeid.split("::", 1)[0] for nodeid in nodeids})
        for target in files:
            args = [target if arg.startswith("tests/") else arg for arg in SUITES[suite]]
            commands.append((suite, target, pytest_command(args)))
    commands.append(
        (
            "distributed",
            DISTRIBUTED_TARGET,
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", DISTRIBUTED_TARGET],
        )
    )
    config = output / "coveragerc"
    config.write_text(
        "[run]\nbranch = True\nparallel = True\nconcurrency = multiprocessing,thread\nsigterm = True\n"
        f"source = {ROOT / 'verl'}\ndata_file = ${{COVERAGE_FILE?}}\n",
        encoding="utf-8",
    )
    for suite, target, command in commands:
        # Suite is a separate path component; target names retain the tests__ convention.
        directory = Path("coverage") / suite / target.removesuffix(".py").replace("/", "__")
        manifest["targets"].append(
            {
                "suite": suite,
                "target": target,
                "command": command,
                "directory": directory.as_posix(),
                "status": "pending",
            }
        )
    save_manifest()

    # Loaded by child Python interpreters, including torchrun ranks and new Ray workers.
    # The hook is inert outside coverage subprocesses and removed after collection.
    hook = Path(sysconfig.get_path("purelib")) / "verl_ci_coverage.pth"
    with hook.open("x", encoding="utf-8") as stream:
        stream.write(
            "import os; os.environ.get('COVERAGE_PROCESS_START') and __import__('coverage').process_startup()\n"
        )
    try:
        for entry in manifest["targets"]:
            directory = output / entry["directory"]
            directory.mkdir(parents=True)
            run_env = {**env, "COVERAGE_PROCESS_START": str(config), "COVERAGE_FILE": str(directory / "coverage")}
            entry["status"] = "running"
            save_manifest()
            print(f"Running {entry['suite']}: {entry['target']}", flush=True)
            # Continue after a failing target to retain diagnostic data for the remaining suite.
            result = subprocess.run(entry["command"], env=run_env)
            entry["exit_code"] = result.returncode
            files = sorted(directory.glob("coverage.*"))
            has_arcs = False
            for path in files:
                data = coverage.CoverageData(basename=str(path))
                data.read()
                has_arcs |= data.has_arcs() and any(data.arcs(name) for name in data.measured_files())
            entry["coverage_files"] = len(files)
            entry["has_branch_data"] = bool(has_arcs)
            entry["status"] = "passed" if result.returncode == 0 and has_arcs else "failed"
            save_manifest()
        manifest["collection_complete"] = all(entry["status"] == "passed" for entry in manifest["targets"])
        save_manifest()
    finally:
        hook.unlink()
    if not manifest["collection_complete"]:
        raise SystemExit("Coverage collection failed: see manifest.json and the job logs")


if __name__ == "__main__":
    main()
