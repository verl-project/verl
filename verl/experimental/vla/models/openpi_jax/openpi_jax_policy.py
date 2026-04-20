"""OpenPI JAX policy wrapper for pi0/pi05 inference.

This module provides a self-contained JAX-based inference path using the official
openpi checkpoint format, as an alternative to the PyTorch conversion path.

Usage:
    policy = OpenPIJaxPolicy.from_checkpoint(
        config_name="pi05_libero",
        checkpoint_dir="/root/data/pi05_libero_absik/checkpoint-30000",
    )
    actions = policy.infer_single(image, wrist_image, state, prompt)
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__file__)

OPENPI_SRC = "/root/openpi/src"
OPENPI_CLIENT_SRC = "/root/openpi/packages/openpi-client/src"


def _ensure_openpi_on_path():
    for p in (OPENPI_SRC, OPENPI_CLIENT_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_jax_cuda_libs():
    """Ensure JAX can find CUDA libraries in the Isaac Sim environment."""
    import os
    nvidia_base = "/workspace/isaaclab/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia"
    cudnn_lib = "/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/nvidia/cudnn/lib"
    extra_paths = []
    if pathlib.Path(nvidia_base).exists():
        for d in sorted(pathlib.Path(nvidia_base).iterdir()):
            lib_dir = d / "lib"
            if lib_dir.is_dir():
                extra_paths.append(str(lib_dir))
    if pathlib.Path(cudnn_lib).exists():
        extra_paths.append(cudnn_lib)
    if extra_paths:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = [p for p in extra_paths if p not in current]
        if new_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + (":" + current if current else "")


def _load_norm_stats(checkpoint_dir: pathlib.Path, asset_id: str | None = None):
    """Load norm_stats directly using openpi.shared.normalize, bypassing checkpoints module."""
    _ensure_openpi_on_path()
    import openpi.shared.normalize as _normalize

    assets_dir = checkpoint_dir / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found at {assets_dir}")

    if asset_id:
        candidate = assets_dir / asset_id
        if candidate.exists() and (candidate / "norm_stats.json").exists():
            norm_stats = _normalize.load(candidate)
            logger.info(f"Loaded norm stats from {candidate}")
            return norm_stats

    for subdir in sorted(assets_dir.iterdir()):
        if subdir.is_dir() and (subdir / "norm_stats.json").exists():
            norm_stats = _normalize.load(subdir)
            logger.info(f"Loaded norm stats from {subdir} (auto-detected)")
            return norm_stats

    raise FileNotFoundError(f"No norm_stats.json found under {assets_dir}")


class OpenPIJaxPolicy:
    """Wraps an openpi JAX model for in-process inference compatible with the verl pipeline."""

    def __init__(self, policy, action_dim: int = 7, action_horizon: int = 10):
        self._policy = policy
        self._action_dim = action_dim
        self._action_horizon = action_horizon

    @classmethod
    def from_checkpoint(
        cls,
        config_name: str = "pi05_libero",
        checkpoint_dir: str = "/root/data/pi05_libero_absik/checkpoint-30000",
        action_dim: int = 7,
        action_horizon: int = 10,
        gpu_id: int = 0,
    ) -> "OpenPIJaxPolicy":
        _ensure_jax_cuda_libs()
        _ensure_openpi_on_path()

        import jax
        import jax.numpy as jnp
        from jax.sharding import SingleDeviceSharding
        import orbax.checkpoint as ocp
        from flax import traverse_util

        from openpi.training import config as _config
        import openpi.models.model as _model
        import openpi.transforms as transforms
        from openpi.policies import policy as _policy_mod
        from openpi.shared import nnx_utils

        logger.info(f"Loading openpi config: {config_name}")
        config = _config.get_config(config_name)

        logger.info(f"Loading JAX checkpoint from: {checkpoint_dir}")
        checkpoint_path = pathlib.Path(checkpoint_dir)
        params_path = (checkpoint_path / "params").resolve()

        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            actual_id = min(gpu_id, len(gpu_devices) - 1)
            device = gpu_devices[actual_id]
            logger.info(f"JAX using GPU device: {device} (requested gpu_id={gpu_id}, available={len(gpu_devices)})")
        else:
            device = jax.devices("cpu")[0]
            logger.warning("No JAX GPU devices found, falling back to CPU")
        sharding = SingleDeviceSharding(device)

        ckptr = ocp.PyTreeCheckpointer()
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}
        restored = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree_util.tree_map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding,
                        restore_type=jax.Array,
                        dtype=jnp.bfloat16,
                    ),
                    item,
                ),
            ),
        )
        params = restored["params"]
        flat = traverse_util.flatten_dict(params)
        if all(kp[-1] == "value" for kp in flat):
            flat = {kp[:-1]: v for kp, v in flat.items()}
            params = traverse_util.unflatten_dict(flat)

        logger.info("Creating JAX model from params...")
        model = config.model.load(params)

        data_config = config.data.create(config.assets_dirs, config.model)
        asset_id = data_config.asset_id
        norm_stats = _load_norm_stats(checkpoint_path, asset_id)

        use_quantile_norm = data_config.use_quantile_norm

        input_transforms = [
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
        output_transforms = [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]

        policy = _policy_mod.Policy(
            model,
            transforms=input_transforms,
            output_transforms=output_transforms,
            sample_kwargs=None,
            metadata=config.policy_metadata,
            is_pytorch=False,
        )

        print("[openpi_jax] JAX policy ready (delta->absolute conversion hardcoded in infer_from_dataproto)", flush=True)
        return cls(policy, action_dim=action_dim, action_horizon=action_horizon)

    def infer_single(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Run inference for a single observation.

        Args:
            image: RGB image (H, W, 3) uint8 or float.
            wrist_image: Wrist camera RGB image (H, W, 3) uint8 or float.
            state: Robot state vector, 7-dim [pos(3) + axisangle(3) + gripper(1)].
            prompt: Language instruction string.

        Returns:
            actions: (action_horizon, action_dim) float32 array of predicted actions.
        """
        obs = {
            "observation/image": np.asarray(image),
            "observation/wrist_image": np.asarray(wrist_image),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": prompt,
        }
        result = self._policy.infer(obs)
        actions = result["actions"]
        return actions[:self._action_horizon, :self._action_dim]

    def infer_batch(
        self,
        images: np.ndarray,
        wrist_images: np.ndarray,
        states: np.ndarray,
        prompts: list[str],
    ) -> np.ndarray:
        """Run inference for a batch of observations (sequentially per-sample).

        Args:
            images: (B, H, W, 3) uint8 or float.
            wrist_images: (B, H, W, 3) uint8 or float.
            states: (B, state_dim) float32.
            prompts: List of B language instruction strings.

        Returns:
            actions: (B, action_horizon, action_dim) float32 array.
        """
        batch_size = images.shape[0]
        all_actions = []
        for i in range(batch_size):
            act = self.infer_single(images[i], wrist_images[i], states[i], prompts[i])
            all_actions.append(act)
        return np.stack(all_actions, axis=0)

    def infer_from_dataproto(self, env_obs) -> dict:
        """Inference entry point compatible with the verl DataProto format.

        Takes the same DataProto as PI0ForActionPrediction.sample_actions() and
        returns a dict with 'action' tensor compatible with the downstream pipeline.

        Args:
            env_obs: verl DataProto with batch keys 'full_image', 'wrist_image', 'state'
                     and non_tensor_batch key 'task_descriptions'.

        Returns:
            dict with:
                'action': torch.Tensor (B, action_horizon, action_dim)
                'full_action': torch.Tensor (B, action_horizon, action_dim)
        """
        full_images = env_obs.batch["full_image"].cpu().numpy()
        wrist_images = env_obs.batch["wrist_image"].cpu().numpy()
        states = env_obs.batch["state"].cpu().numpy()
        prompts = list(env_obs.non_tensor_batch["task_descriptions"])

        if full_images.dtype != np.uint8:
            if full_images.max() <= 1.0:
                full_images = (full_images * 255).astype(np.uint8)
            else:
                full_images = full_images.astype(np.uint8)
        if wrist_images.dtype != np.uint8:
            if wrist_images.max() <= 1.0:
                wrist_images = (wrist_images * 255).astype(np.uint8)
            else:
                wrist_images = wrist_images.astype(np.uint8)

        if full_images.ndim == 4 and full_images.shape[-1] != 3 and full_images.shape[1] == 3:
            full_images = np.transpose(full_images, (0, 2, 3, 1))
        if wrist_images.ndim == 4 and wrist_images.shape[-1] != 3 and wrist_images.shape[1] == 3:
            wrist_images = np.transpose(wrist_images, (0, 2, 3, 1))

        logger.info(
            f"JAX infer input: images={full_images.shape} dtype={full_images.dtype}, "
            f"states={states.shape} state[0]={states[0].tolist()}, prompt={prompts[0][:60]}"
        )

        actions_np = self.infer_batch(full_images, wrist_images, states, prompts)

        # pi05 JAX model outputs delta actions; convert to absolute by adding current state
        # first 6 dims = pos(3) + axisangle(3), dim 7 = gripper (already absolute)
        for i in range(actions_np.shape[0]):
            actions_np[i, :, :6] += states[i, :6]

        print(
            f"[AbsAction] state={states[0, :6].tolist()}, "
            f"absolute={actions_np[0, 0, :7].tolist()}",
            flush=True,
        )

        actions_tensor = torch.from_numpy(actions_np).float()

        return {
            "action": actions_tensor,
            "full_action": actions_tensor,
        }
