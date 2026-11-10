# Copyright (c) 2026 Google LLC. All rights reserved.
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
"""1-chip TPU platform test of the verl TPU CI.

Submitted as a Ray job to the per-test CI RayCluster with the same runtime environment as
examples/tpu/grpo/README.md (verl from the checkout, verl-hardware-plugin as a py_module). Schedules
a single Ray task that reserves exactly one TPU chip, pins the process to it and checks:

1. ``torch_tpu``: host <-> device copy, elementwise ops, a bf16 matmul, autograd, and the same tiny
   MLP forward/backward on TPU and on CPU.
2. verl + verl-hardware-plugin: ``import verl`` loads the plugin (``VERL_USE_EXTERNAL_MODULES``),
   the ``tpu`` platform is selected, and the TPU TorchTitan engine and the TPU weight sync backends
   are registered. The plugin swallows registration errors, so a broken import otherwise only shows
   up later as an unknown engine or backend in the trainer tests.

It is meant to fail fast (minutes) on broken images, libtpu / device plugin problems or a verl /
plugin mismatch before the trainer and RL tests spend time on them.

Usage (inside the Ray cluster):
    python3 tests/special_tpu/tpu_smoke_test.py
"""

import json
import os
import socket
import sys

import ray


def _pin_single_chip_env() -> dict:
    """Restrict this process to the single TPU chip Ray assigned to the task.

    Ray sets TPU_VISIBLE_CHIPS / TPU_CHIPS_PER_HOST_BOUNDS / TPU_HOST_BOUNDS for tasks that
    reserve a subset of a host's chips; the torch_tpu slice-builder variables injected by the
    KubeRay webhook describe the whole host, so override them for a 1x1x1 slice.
    """
    chip_ids = ray.get_runtime_context().get_accelerator_ids().get("TPU", [])
    assert len(chip_ids) == 1, f"expected exactly one assigned TPU chip, got {chip_ids}"
    port = 8471 + int(chip_ids[0])
    env = {
        "TPU_VISIBLE_CHIPS": str(chip_ids[0]),
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "TPU_WORKER_ID": "0",
        "TPU_WORKER_HOSTNAMES": "localhost",
        "CLOUD_TPU_TASK_ID": "0",
        "TORCH_TPU_TOPOLOGY": "1,1,1",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"localhost:{port}",
        "TPU_PROCESS_ADDRESSES": f"localhost:{port}",
        "TPU_PROCESS_PORT": str(port),
    }
    os.environ.update(env)
    return env


@ray.remote(num_cpus=1, resources={"TPU": 1})
def tpu_smoke_task() -> dict:
    env = _pin_single_chip_env()

    import torch
    import torch_tpu  # noqa: F401  (registers the "tpu" device)

    device = torch.device("tpu")
    result = {
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "torch_tpu_version": getattr(torch_tpu, "__version__", "unknown"),
        "pinned_env": env,
    }

    # 1. Host -> device -> host round trip.
    cpu_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    tpu_tensor = cpu_tensor.to(device)
    assert tpu_tensor.device.type == "tpu", f"tensor landed on {tpu_tensor.device}"
    torch.testing.assert_close(tpu_tensor.cpu(), cpu_tensor)

    # 2. Elementwise + reduction.
    torch.testing.assert_close((tpu_tensor * 2 + 1).sum().cpu(), (cpu_tensor * 2 + 1).sum())

    # 3. bf16 matmul (MXU path) against a CPU fp32 reference.
    torch.manual_seed(0)
    a = torch.randn(128, 256)
    b = torch.randn(256, 64)
    out = (a.to(device, torch.bfloat16) @ b.to(device, torch.bfloat16)).float().cpu()
    torch.testing.assert_close(out, a @ b, atol=0.5, rtol=5e-2)

    # 4. Autograd on device.
    w = torch.randn(64, 32, device=device, requires_grad=True)
    x = torch.randn(8, 64, device=device)
    loss = torch.nn.functional.gelu(x @ w).pow(2).mean()
    loss.backward()
    assert w.grad is not None and w.grad.device.type == "tpu"
    grad_norm = w.grad.float().norm().cpu().item()
    assert grad_norm > 0 and grad_norm == grad_norm, f"bad grad norm {grad_norm}"

    result.update({"device": str(tpu_tensor.device), "matmul_max_abs_err": (out - a @ b).abs().max().item()})
    result["grad_norm"] = grad_norm

    # 5. TPU vs CPU: identical tiny MLP (fp32) forward + backward on both devices.
    result["tpu_vs_cpu"] = _compare_mlp_tpu_vs_cpu(device)

    # 6. verl + verl-hardware-plugin on the TPU host.
    result["verl"] = _check_verl_tpu_platform()
    return result


def _compare_mlp_tpu_vs_cpu(device) -> dict:
    """Run the same Linear-GELU-Linear MLP on CPU and TPU; compare output, loss and grads."""
    import torch

    torch.manual_seed(0)
    cpu_model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    tpu_model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    tpu_model.load_state_dict(cpu_model.state_dict())
    tpu_model.to(device)
    x = torch.randn(16, 32)
    target = torch.randn(16, 8)

    def run(model, dev):
        out = model(x.to(dev))
        loss = torch.nn.functional.mse_loss(out, target.to(dev))
        loss.backward()
        grads = {name: p.grad.cpu() for name, p in model.named_parameters()}
        return out.cpu(), loss.cpu(), grads

    cpu_out, cpu_loss, cpu_grads = run(cpu_model, "cpu")
    tpu_out, tpu_loss, tpu_grads = run(tpu_model, device)

    # TPU fp32 matmuls may use reduced-precision MXU passes, hence the loose-ish tolerances.
    tol = {"atol": 2e-2, "rtol": 2e-2}
    torch.testing.assert_close(tpu_out, cpu_out, **tol)
    torch.testing.assert_close(tpu_loss, cpu_loss, **tol)
    for name in cpu_grads:
        torch.testing.assert_close(tpu_grads[name], cpu_grads[name], **tol, msg=lambda m, n=name: f"grad {n}: {m}")

    diffs = {"output": (tpu_out - cpu_out).abs().max().item(), "loss": (tpu_loss - cpu_loss).abs().item()}
    diffs.update({f"grad/{n}": (tpu_grads[n] - cpu_grads[n]).abs().max().item() for n in cpu_grads})
    print("[TPU smoke] TPU vs CPU max |diff|: " + ", ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
    return {"cpu_loss": cpu_loss.item(), "tpu_loss": tpu_loss.item(), "max_abs_diff": diffs}


def _check_verl_tpu_platform() -> dict:
    """Check that verl selects the TPU platform and sees the plugin's TPU engines."""
    assert os.environ.get("VERL_USE_EXTERNAL_MODULES"), "VERL_USE_EXTERNAL_MODULES is not set in the job runtime env"

    import verl  # noqa: F401  (imports VERL_USE_EXTERNAL_MODULES, i.e. verl_hardware_plugin)
    from verl.checkpoint_engine import CheckpointEngineRegistry
    from verl.utils.device import get_device_name, get_resource_name
    from verl.workers.engine import EngineRegistry

    device_name, resource_name = get_device_name(), get_resource_name()
    assert device_name == "tpu", f"verl selected device {device_name!r}, expected 'tpu'"
    assert resource_name == "TPU", f"verl selected Ray resource {resource_name!r}, expected 'TPU'"
    engine_cls = EngineRegistry.get_engine_cls("language_model", "torchtitan")
    assert "TPU" in engine_cls.__name__, (
        f"torchtitan engine on TPU resolved to {engine_cls.__module__}.{engine_cls.__name__}"
    )
    weight_sync = {
        backend: CheckpointEngineRegistry.get(backend).__name__ for backend in ("tpu-sync", "ray-plasma-store")
    }
    info = {
        "device_name": device_name,
        "resource_name": resource_name,
        "torchtitan_engine": f"{engine_cls.__module__}.{engine_cls.__name__}",
        "weight_sync_backends": weight_sync,
    }
    print(f"[TPU smoke] verl TPU platform: {json.dumps(info)}")
    return info


def main() -> int:
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    total_tpus = ray.cluster_resources().get("TPU", 0)
    print(f"[TPU smoke] cluster TPU resources: {total_tpus}")
    assert total_tpus >= 1, "no TPU resources registered in the Ray cluster"

    result = ray.get(tpu_smoke_task.remote(), timeout=900)
    print("[TPU smoke] result:\n" + json.dumps(result, indent=2, default=str))
    print(
        "[TPU smoke] PASSED: torch_tpu import, host<->device copy, bf16 matmul, autograd, TPU-vs-CPU MLP "
        "forward/backward and the verl TPU platform (verl-hardware-plugin) on 1 TPU chip."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
