# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Minimal checkpoint manager for DeepSpeed engines.
"""

from __future__ import annotations

import logging
import os
import shutil
from types import MethodType
from typing import Optional

import torch
import torch.distributed

from verl.utils.deepspeed_utils import (
    load_deepspeed_checkpoint,
    load_deepspeed_model_to_gpu,
    offload_deepspeed_model_to_cpu,
    save_deepspeed_checkpoint,
)


# Local helper to restore DeepSpeed engine.module before checkpointing.
def _ensure_engine_has_module(engine, module):
    """Make sure engine.__dict__['module'] is set to a real nn.Module."""
    if engine is None:
        return
    # Prefer caller-supplied module; fall back to engine.__dict__ if present.
    fallback = None
    try:
        fallback = engine.__dict__.get("module", None)
    except Exception:
        fallback = None
    if module is None:
        module = fallback
    if module is None:
        try:
            module = object.__getattribute__(engine, "module")
        except Exception:
            module = None
    if module is None:
        return
    try:
        engine.__dict__["module"] = module
    except Exception:
        try:
            object.__setattr__(engine, "module", module)
        except Exception:
            try:
                engine.module = module  # type: ignore
            except Exception:
                pass


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_DS_GETATTR_PATCHED = False


def _patch_deepspeed_getattr():
    """Monkeypatch DeepSpeedEngine.__getattr__ to avoid recursive dir(self) when module is missing."""
    global _DS_GETATTR_PATCHED
    if _DS_GETATTR_PATCHED:
        return
    try:
        from deepspeed import DeepSpeedEngine
    except Exception:
        return

    try:
        def _safe_getattr(self, name):  # type: ignore[override]
            # Only reached when normal lookup failed; avoid dir(self) recursion.
            try:
                d = object.__getattribute__(self, "__dict__")
                mod = d.get("module", None)
            except Exception:
                mod = None
            if mod is not None and hasattr(mod, name):
                return getattr(mod, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        DeepSpeedEngine.__getattr__ = _safe_getattr  # type: ignore[assignment]
        _DS_GETATTR_PATCHED = True
        logger.info("[DeepSpeedCheckpointManager] Patched DeepSpeedEngine.__getattr__ to avoid recursion")
    except Exception as exc:  # pragma: no cover
        logger.warning("[DeepSpeedCheckpointManager] Failed to patch DeepSpeedEngine.__getattr__: %s", exc)


class DeepSpeedCheckpointManager:
    """Lightweight helper around DeepSpeed checkpoint APIs."""

    CKPT_PREFIX = "step_"

    def __init__(self, engine):
        self.engine = engine
        try:
            self.rank = torch.distributed.get_rank()
        except Exception:
            self.rank = 0

    def _get_engine(self):
        """Return the underlying DeepSpeed engine, unwrapping worker containers."""
        eng = self.engine
        for attr in ("actor_engine", "critic_engine", "reward_engine"):
            try:
                candidate = getattr(eng, attr, None)
            except Exception:
                candidate = None
            if candidate is not None:
                return candidate
        return eng

    # ----- internal helpers -----
    def _ckpt_dir(self, root: str, step: int) -> str:
        return os.path.join(root, f"{self.CKPT_PREFIX}{step}")

    def _list_ckpts(self, root: str):
        if not os.path.isdir(root):
            return []
        items = []
        for name in os.listdir(root):
            if not name.startswith(self.CKPT_PREFIX):
                continue
            step_str = name[len(self.CKPT_PREFIX) :]
            if not step_str.isdigit():
                continue
            items.append((int(step_str), os.path.join(root, name)))
        return sorted(items, key=lambda kv: kv[0])

    def _latest(self, root: str):
        ckpts = self._list_ckpts(root)
        return ckpts[-1] if ckpts else None

    def _prune(self, root: str, max_keep: Optional[int]):
        if max_keep is None or max_keep <= 0:
            return
        ckpts = self._list_ckpts(root)
        if len(ckpts) <= max_keep:
            return
        stale = ckpts[:-max_keep]
        for step, path in stale:
            if self.rank == 0:
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info(f"[DeepSpeedCheckpointManager] Pruned checkpoint step={step} path={path}")
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"[DeepSpeedCheckpointManager] Failed to prune {path}: {exc}")

    # ----- public API -----
    def save_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, global_step: int = 0, max_ckpt_to_keep: Optional[int] = None
    ):
        os.makedirs(local_path, exist_ok=True)
        target = self._ckpt_dir(local_path, global_step)
        os.makedirs(target, exist_ok=True)

        ds_engine = self._get_engine()
        # Best-effort: restore module reference before checkpoint to avoid DeepSpeed __getattr__ recursion.
        module = None
        for attr in ("actor_module", "critic_module", "reward_module", "module"):
            try:
                module = getattr(self.engine, attr, None)
            except Exception:
                module = None
            if module is not None:
                break
        if module is None and ds_engine is not None:
            try:
                module = getattr(ds_engine, "module", None)
            except Exception:
                module = None
        _patch_deepspeed_getattr()
        logger.info(
            "[DeepSpeedCheckpointManager] save_checkpoint: engine=%s module_attr=%s ds_engine_module_key=%s",
            type(ds_engine).__name__,
            type(module).__name__ if module is not None else "None",
            type(getattr(ds_engine, '__dict__', {}).get('module', None)).__name__
            if isinstance(getattr(ds_engine, '__dict__', {}), dict) and 'module' in getattr(ds_engine, '__dict__', {})
            else "None",
        )
        _ensure_engine_has_module(ds_engine, module)
        try:
            logger.info(
                "[DeepSpeedCheckpointManager] after_ensure: module_key=%s",
                type(getattr(ds_engine, '__dict__', {}).get('module', None)).__name__
                if isinstance(getattr(ds_engine, '__dict__', {}), dict) and 'module' in getattr(ds_engine, '__dict__', {})
                else "None",
            )
        except Exception:
            pass
        load_deepspeed_model_to_gpu(ds_engine)
        # Wrap module_state_dict to avoid DeepSpeed __getattr__ recursion when module attr is missing.
        orig_module_state = getattr(ds_engine, "module_state_dict", None)
        orig_get_buffer_names = getattr(ds_engine, "_get_buffer_names", None)
        orig_get_zero_frozen = getattr(ds_engine, "_get_zero_frozen_param_attributes", None)
        orig_get_shared_params = getattr(ds_engine, "_get_shared_params", None)
        captured_module = module
        try:
            def _safe_module_state_dict(self, *args, **kwargs):  # type: ignore[override]
                mdl = captured_module
                if mdl is None:
                    try:
                        mdl = object.__getattribute__(self, "module")
                    except Exception:
                        mdl = None
                if mdl is None:
                    raise RuntimeError("DeepSpeed engine.module missing before checkpoint")
                if "exclude_frozen_parameters" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs.pop("exclude_frozen_parameters", None)
                return mdl.state_dict(*args, **kwargs)

            def _safe_get_buffer_names(self):  # type: ignore[override]
                if captured_module is None:
                    return []
                try:
                    return [name for name, _ in captured_module.named_buffers()]
                except Exception:
                    return []

            def _safe_get_zero_frozen_param_attributes(self, attr_func):  # type: ignore[override]
                from collections import OrderedDict

                frozen_param_fragments = OrderedDict()
                mdl = captured_module
                if mdl is None:
                    return frozen_param_fragments
                try:
                    for name, param in mdl.named_parameters():
                        if param.requires_grad:
                            continue
                        try:
                            frozen_param_fragments[name] = attr_func(param)
                        except Exception:
                            continue
                except Exception:
                    pass
                return frozen_param_fragments

            def _safe_get_shared_params(self):  # type: ignore[override]
                # Return empty dict to avoid pickling issues with ProcessGroup/ds_id metadata.
                return {}

            ds_engine.module_state_dict = MethodType(_safe_module_state_dict, ds_engine)  # type: ignore[assignment]
            ds_engine._get_buffer_names = MethodType(_safe_get_buffer_names, ds_engine)  # type: ignore[assignment]
            ds_engine._get_zero_frozen_param_attributes = MethodType(_safe_get_zero_frozen_param_attributes, ds_engine)  # type: ignore[assignment]
            ds_engine._get_shared_params = MethodType(_safe_get_shared_params, ds_engine)  # type: ignore[assignment]
            try:
                save_deepspeed_checkpoint(engine=ds_engine, save_dir=target, tag=str(global_step))
            except (RecursionError, RuntimeError) as exc:
                logger.error("[DeepSpeedCheckpointManager] skip checkpoint due to DeepSpeed save error: %s", exc)
                # Fallback: save consolidated weights only (best effort, rank0)
                self._save_consolidated_fallback(ds_engine, captured_module or module, target, global_step)
        finally:
            if orig_module_state is not None:
                try:
                    ds_engine.module_state_dict = orig_module_state  # type: ignore[assignment]
                except Exception:
                    pass
            if orig_get_buffer_names is not None:
                try:
                    ds_engine._get_buffer_names = orig_get_buffer_names  # type: ignore[assignment]
                except Exception:
                    pass
            if orig_get_zero_frozen is not None:
                try:
                    ds_engine._get_zero_frozen_param_attributes = orig_get_zero_frozen  # type: ignore[assignment]
                except Exception:
                    pass
            if orig_get_shared_params is not None:
                try:
                    ds_engine._get_shared_params = orig_get_shared_params  # type: ignore[assignment]
                except Exception:
                    pass

    def _save_consolidated_fallback(self, ds_engine, module, target: str, global_step: int):
        """Best-effort consolidated save to avoid DeepSpeed recursion."""
        if module is None:
            logger.error("[DeepSpeedCheckpointManager] fallback save skipped: module is None")
            return
        try:
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        except Exception:
            rank, world = 0, 1

        state = None
        if hasattr(ds_engine, "_zero3_consolidated_16bit_state_dict"):
            try:
                logger.info("[DeepSpeedCheckpointManager] fallback: using _zero3_consolidated_16bit_state_dict")
                state = ds_engine._zero3_consolidated_16bit_state_dict()
            except Exception as exc:
                logger.warning("[DeepSpeedCheckpointManager] fallback consolidated failed, will try module.state_dict: %s", exc)

        if state is None:
            try:
                state = module.state_dict()
            except Exception as exc:
                logger.error("[DeepSpeedCheckpointManager] fallback save failed: %s", exc)
                return

        if rank == 0:
            try:
                payload = {"model": state, "step": int(global_step), "tag": str(global_step)}
                path = os.path.join(target, "model_consolidated.pt")
                torch.save(payload, path)
                logger.info("[DeepSpeedCheckpointManager] fallback checkpoint saved at %s", path)
            except Exception as exc:
                logger.error("[DeepSpeedCheckpointManager] fallback save torch.save failed: %s", exc)

        if world > 1:
            try:
                torch.distributed.barrier()
            except Exception:
                pass
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
        offload_deepspeed_model_to_cpu(ds_engine)

        if hdfs_path is not None and self.rank == 0:
            try:
                os.makedirs(hdfs_path, exist_ok=True)
                dst = self._ckpt_dir(hdfs_path, global_step)
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(target, dst)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"[DeepSpeedCheckpointManager] Failed to copy checkpoint to {hdfs_path}: {exc}")

        self._prune(local_path, max_ckpt_to_keep)

    def load_checkpoint(self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True):
        load_dir = local_path
        if os.path.isdir(local_path) and not os.path.basename(local_path).startswith(self.CKPT_PREFIX):
            latest = self._latest(local_path)
            if latest is None:
                logger.warning(f"[DeepSpeedCheckpointManager] No checkpoint found under {local_path}")
                return {}
            load_dir = latest[1]

        if hdfs_path and not os.path.exists(load_dir) and self.rank == 0:
            try:
                os.makedirs(local_path, exist_ok=True)
                shutil.copytree(hdfs_path, load_dir)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"[DeepSpeedCheckpointManager] Failed to copy checkpoint from {hdfs_path}: {exc}")

        ds_engine = self._get_engine()
        module = None
        for attr in ("actor_module", "critic_module", "reward_module", "module"):
            try:
                module = getattr(self.engine, attr, None)
            except Exception:
                module = None
            if module is not None:
                break
        if module is None and ds_engine is not None:
            try:
                module = getattr(ds_engine, "module", None)
            except Exception:
                module = None
        _ensure_engine_has_module(ds_engine, module)
        load_deepspeed_model_to_gpu(ds_engine)
        client_state = load_deepspeed_checkpoint(
            engine=ds_engine,
            load_dir=load_dir,
            tag=None,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
        offload_deepspeed_model_to_cpu(ds_engine)

        if del_local_after_load and os.path.isdir(load_dir):
            try:
                shutil.rmtree(load_dir, ignore_errors=True)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"[DeepSpeedCheckpointManager] Failed to clean up {load_dir}: {exc}")
        return client_state
