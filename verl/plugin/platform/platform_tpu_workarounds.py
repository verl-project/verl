# Copyright 2024-2025 BAAI and Google LLC
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
"""Platform-specific utilities and Ray cluster integration workarounds for Google TPU execution."""

import logging
import os

import ray
import ray._private.worker
import torch

from verl.plugin.platform import get_platform

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default concurrency and CPU allocation for Ray TaskRunner actor on TPU worker nodes
DEFAULT_TASK_RUNNER_CPUS = 10
DEFAULT_TASK_RUNNER_CONCURRENCY = 100


def convert_tensors_to_scalars(val):
    """Recursively converts TPU tensors in metrics dictionaries to Python scalars or CPU structures.

    Directly moving a TPU scalar tensor with a special PJRT allocator to the CPU driver process
    via `.cpu()` can trigger allocator assertion errors in containerized multi-host setups.
    Calling `.item()` on singleton tensors or converting non-scalar tensors safely avoids driver process crashes.
    """
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        else:
            try:
                return val.detach().cpu()
            except Exception:
                return val.tolist()
    elif isinstance(val, dict):
        return {k: convert_tensors_to_scalars(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [convert_tensors_to_scalars(v) for v in val]
    elif isinstance(val, tuple):
        return tuple(convert_tensors_to_scalars(v) for v in val)
    elif hasattr(val, "aggregate"):
        try:
            return val.aggregate()
        except Exception:
            return val
    return val


def patch_ray_worker():
    """Patches Ray worker resource lookup to handle containerized TPU device index bounds in GKE.

    In containerized GKE environments where TPU chips are isolated per pod, Raylet physical accelerator
    lookups can raise an `IndexError` when querying host-level accelerator indices.
    """
    try:
        original_func = ray._private.worker.Worker.get_accelerator_ids_for_accelerator_resource

        def patched_func(self, resource_name, resource_regex):
            try:
                return original_func(self, resource_name, resource_regex)
            except IndexError as e:
                import traceback

                print(
                    f"[patch_ray_worker] Intercepted Ray accelerator lookup IndexError for resource '{resource_name}': {e}\n"
                    f"{traceback.format_exc()}"
                )
                return []

        ray._private.worker.Worker.get_accelerator_ids_for_accelerator_resource = patched_func
    except Exception as e:
        logger.warning(f"Failed to apply Ray worker accelerator patch: {e}")


def aggregate_sft_metrics_tpu(metrics):
    """Aggregates metrics returned by multi-host data-parallel training workers on the CPU driver.

    Under distributed TorchTitan training across multiple data-parallel ranks, worker metrics can be
    returned as lists of dictionaries or dictionaries of rank-specific tensors. This helper computes
    their average cleanly.
    """
    if isinstance(metrics, list):
        aggregated_metrics = {}
        if len(metrics) > 0:
            keys = metrics[0].keys()
            for k in keys:
                vals = [m[k] for m in metrics if k in m]
                if len(vals) > 0:
                    try:
                        vals_tensor = torch.tensor(vals, dtype=torch.float32)
                        aggregated_metrics[k] = torch.mean(vals_tensor).item()
                    except Exception:
                        aggregated_metrics[k] = vals[0]
        return aggregated_metrics
    elif isinstance(metrics, dict):
        aggregated_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, list) or (hasattr(v, "shape") and len(v.shape) > 0):
                try:
                    aggregated_metrics[k] = torch.tensor(v, dtype=torch.float32).mean().item()
                except Exception:
                    if isinstance(v, list) and len(v) > 0:
                        aggregated_metrics[k] = v[0]
                    else:
                        aggregated_metrics[k] = v
            else:
                aggregated_metrics[k] = v
        return aggregated_metrics
    else:
        raise TypeError(f"Unexpected metrics structure type: {type(metrics)}")


def extract_validation_loss(metrics) -> float:
    """Extracts scalar validation loss value from worker metrics dictionary or list."""
    if isinstance(metrics, list):
        valid_losses = [m["loss"] for m in metrics if "loss" in m]
        return sum(valid_losses) / len(valid_losses) if valid_losses else 0.0
    elif isinstance(metrics, dict):
        if "loss" in metrics:
            v = metrics["loss"]
            if isinstance(v, list) or (hasattr(v, "shape") and len(v.shape) > 0):
                return torch.tensor(v, dtype=torch.float32).mean().item()
            return float(v)
    elif isinstance(metrics, int | float):
        return float(metrics)
    return float(metrics)


def get_ray_init_kwargs() -> dict:
    """Returns Ray initialization arguments including runtime environment hooks for TPU workers."""
    env_vars = {}
    if "PYTHONPATH" in os.environ:
        env_vars["PYTHONPATH"] = os.environ["PYTHONPATH"]
    if "VERL_PLATFORM" in os.environ:
        env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]
    return {
        "runtime_env": {
            "worker_process_setup_hook": patch_ray_worker,
            "env_vars": env_vars,
        }
    }


def run_trainer_on_tpu(trainer_cls, config):
    """Executes the SFT/PPO trainer inside a remote Ray TaskRunner actor on TPU worker nodes.

    The Ray driver process runs on the head/CPU node where TPU devices are not present.
    Wrapping trainer initialization inside a remote actor ensures that PJRT client initialization
    and device setup run directly on TPU worker nodes.
    """

    @ray.remote(num_cpus=DEFAULT_TASK_RUNNER_CPUS, max_concurrency=DEFAULT_TASK_RUNNER_CONCURRENCY)
    class TaskRunner:
        def run(self, config):
            trainer = trainer_cls(config=config)
            trainer.fit()

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


def get_platform_worker_env_vars(
    resource_pool,
    rank: int,
    world_size: int,
    local_rank: int,
    local_world_size: int,
    name_prefix: str,
    device_name: str,
) -> dict:
    """Generates platform-specific environment variables for worker nodes."""
    env_vars = {}
    if "VERL_PLATFORM" in os.environ:
        env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]
    for var in get_platform().ray_noset_envvars():
        env_vars[var] = "1"
    pgs = resource_pool.get_placement_groups(device_name=device_name)
    tpu_env = get_platform().get_tpu_env_vars(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        name_prefix=name_prefix,
        pgs=pgs,
    )
    env_vars.update(tpu_env)
    return env_vars
