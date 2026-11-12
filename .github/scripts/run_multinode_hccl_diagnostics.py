"""Run the original training command while collecting diagnostics on each NPU node."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
from pathlib import Path


class NodeDiagnostics:
    """A zero-CPU/zero-NPU Ray actor; its child only reads host resources/logs."""

    def __init__(self, repo, environment):
        self.repo = Path(repo)
        self.environment = {**os.environ, **environment}
        self.collector = self.repo / ".github/scripts/collect_ascend_diagnostics.sh"
        self.monitor = None

    def collect(self, mode):
        result = subprocess.run(
            ["bash", str(self.collector), mode],
            cwd=self.repo,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return (
            f"[VERL_HCCL_NODE] hostname={socket.gethostname()} mode={mode} exit_code={result.returncode}\n"
            + result.stdout
        )

    def start(self):
        # Prove the worker copied the diagnostic checkout rather than a stale cache.
        hashes = {
            name: hashlib.sha256((self.repo / name).read_bytes()).hexdigest()
            for name in (
                "verl/utils/hccl_diagnostics.py",
                "verl/workers/engine/megatron/transformer_impl.py",
                "tests/special_npu/nightly_ci_ascend/run_grpo_glm52_top16_megatron_npu_multinode.sh",
            )
        }
        baseline = self.collect("before")
        self.monitor = subprocess.Popen(
            ["bash", str(self.collector), "monitor"],
            cwd=self.repo,
            env=self.environment,
            start_new_session=True,
            # Inherit the actor's stdout: Ray forwards periodic snapshots to the controller.
        )
        return f"[VERL_HCCL_NODE] source_sha256={json.dumps(hashes, sort_keys=True)}\n" + baseline

    def finish(self):
        if self.monitor is not None and self.monitor.poll() is None:
            os.killpg(self.monitor.pid, signal.SIGTERM)
            try:
                self.monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.monitor.pid, signal.SIGKILL)
                self.monitor.wait()
        return self.collect("after")


def run_training(command, monitors, ray):
    try:
        for monitor in monitors:
            try:
                print(ray.get(monitor.start.remote(), timeout=180), flush=True)
            except Exception as error:
                print(f"[VERL_HCCL_NODE] baseline/monitor startup failed: {error!r}", flush=True)
        print("[VERL_HCCL_NODE] training_command=" + json.dumps(command), flush=True)
        # Preserve the command's stdout, stderr, arguments and exit code.
        return subprocess.call(command)
    finally:
        # Capture both workers even when one is unavailable; never hide a training failure.
        for monitor in monitors:
            try:
                print(ray.get(monitor.finish.remote(), timeout=600), flush=True)
            except Exception as error:
                print(f"[VERL_HCCL_NODE] final collection failed: {error!r}", flush=True)
            finally:
                try:
                    ray.kill(monitor, no_restart=True)
                except Exception as error:
                    print(f"[VERL_HCCL_NODE] monitor cleanup failed: {error!r}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/verl")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a training command is required after --")

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address="auto", log_to_driver=True)
    try:
        nodes = [node for node in ray.nodes() if node.get("Alive") and node.get("Resources", {}).get("CPU", 0) > 0]
        expected = int(os.environ.get("EXPECTED_WORKERS", "2"))
        if len(nodes) != expected:
            raise RuntimeError(f"Expected {expected} NPU worker nodes, found {len(nodes)}")
        environment = {
            key: os.environ[key]
            for key in (
                "VERL_HCCL_DIAGNOSTICS",
                "ASCEND_PROCESS_LOG_PATH",
                "VERL_DIAGNOSTIC_COMMIT",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_JOB",
                "GITHUB_SHA",
                "RUNNER_NAME",
            )
            if key in os.environ
        }
        actor_class = ray.remote(num_cpus=0, num_gpus=0)(NodeDiagnostics)
        monitors = []
        for node in nodes:
            print("[VERL_HCCL_NODE] node=" + json.dumps(node, default=str), flush=True)
            monitors.append(
                actor_class.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                ).remote(args.repo, environment)
            )
        return run_training(command, monitors, ray)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
