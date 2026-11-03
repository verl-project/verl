"""MaxText engine. One process owns the node's GPUs and returns torch tensors."""

from __future__ import annotations

import os

import numpy as np
import torch

from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.maxtext.batch import arrays_from_batch, maxtext_batch


class _MaxTextEngine(BaseEngine):
    """Step-level MaxText trainer. The worker still sees a torch engine."""

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.session = None

    def initialize(self):
        from verl.workers.engine.maxtext.runtime import MaxTextSession

        self.session = MaxTextSession(self.engine_config, self.model_config)
        _pin_torch_device()
        _install_stack_dumps()

    @property
    def module(self):
        """The JAX state is not a torch module. Callers that accept None skip it."""
        return None

    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    def train_mode(self, **kwargs):
        return BaseEngineCtx(self, "train", **kwargs)

    def eval_mode(self, **kwargs):
        return BaseEngineCtx(self, "eval", **kwargs)

    def optimizer_zero_grad(self):
        return None

    def optimizer_step(self):
        return None

    def lr_scheduler_step(self):
        return None

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        return None

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def get_data_parallel_group(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.group.WORLD
        return None

    def is_mp_src_rank_with_outputs(self):
        return True

    def train_batch(self, data, loss_function):
        self.session.set_actor_config(_actor_config(loss_function))
        batch = self._batch(data, with_targets=True)
        metrics = self.session.train(batch)
        return {"loss": [metrics["loss"]], "metrics": {"grad_norm": metrics["grad_norm"]}, "model_output": {}}

    def infer_batch(self, data, loss_function=None):
        del loss_function
        arrays = arrays_from_batch(data)
        batch = self._batch(data, with_targets=False)
        log_probs, entropy = self.session.log_probs_and_entropy(batch)
        width = arrays["token_ids"].shape[1]
        mask = arrays["token_mask"]
        output = {
            "log_probs": _as_nested(torch.as_tensor(log_probs[:, :width], dtype=torch.float32), mask),
            "entropy": _as_nested(torch.as_tensor(entropy[:, :width], dtype=torch.float32), mask),
        }
        return {"loss": [0.0], "metrics": {}, "model_output": output}

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        if forward_only:
            return self.infer_batch(data, loss_function)
        return self.train_batch(data, loss_function)

    def get_per_tensor_param(self, *args, **kwargs):
        del args, kwargs
        _pin_torch_device()
        from verl.workers.engine.maxtext.export import check_layout

        weights = self.session.stacked_weights()
        heads = int(self.engine_config.num_attention_heads)
        kv_heads = int(self.engine_config.num_key_value_heads)
        head_dim = int(self.engine_config.head_dim)
        try:
            check_layout(weights, heads, head_dim)
        except (KeyError, ValueError) as exc:
            sample = ", ".join(list(_paths(weights))[:12])
            raise RuntimeError(
                "MaxText parameters are not in the scanned Qwen3-MoE layout "
                f"the export expects ({exc!r}). First parameter paths: {sample}"
            ) from exc
        device = _export_device()

        def generate():
            if os.environ.get("MAXTEXT_EXPORT_HOST", "0") == "1":
                from verl.workers.engine.maxtext.export import iter_hf_from_stacked

                for name, value in iter_hf_from_stacked(
                    weights, heads, kv_heads, head_dim, fetch=self.session.fetch_bf16
                ):
                    yield name, _to_torch(value, device)
                return
            # Default: gather, transpose and hand over on the GPU. No host copy.
            import jax

            from verl.workers.engine.maxtext.export import iter_hf_from_stacked_device

            target = jax.local_devices()[torch.device(device).index or 0]
            yield from iter_hf_from_stacked_device(weights, heads, kv_heads, head_dim, target)

        return generate(), None

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        """Save the fp32 master weights as HF safetensors. Optimizer state is not saved."""
        del hdfs_path, kwargs
        import time

        import jax
        import jax.numpy as jnp

        from verl.workers.engine.maxtext.checkpoint import prune_checkpoints, write_hf_checkpoint
        from verl.workers.engine.maxtext.export import iter_hf_from_stacked_device

        _pin_torch_device()
        started = time.time()
        weights = self.session.stacked_weights()
        heads = int(self.engine_config.num_attention_heads)
        kv_heads = int(self.engine_config.num_key_value_heads)
        head_dim = int(self.engine_config.head_dim)
        target = jax.local_devices()[torch.device(_export_device()).index or 0]
        tensors = iter_hf_from_stacked_device(weights, heads, kv_heads, head_dim, target, dtype=jnp.float32)
        stats = write_hf_checkpoint(tensors, local_path, source_dir=self._hf_source_dir())
        # local_path is <root>/global_step_N/actor.
        root = os.path.dirname(os.path.dirname(local_path.rstrip("/")))
        removed = prune_checkpoints(root, int(global_step), max_ckpt_to_keep)
        print(
            f"[maxtext-ckpt] saved step {global_step}: {stats['tensors']} tensors, "
            f"{stats['bytes'] / 2**30:.1f} GiB in {time.time() - started:.0f}s; removed {removed}",
            flush=True,
        )

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False, **kwargs):
        """Load weights saved by ``save_checkpoint``. Adam moments start from zero."""
        del hdfs_path, kwargs
        import time

        started = time.time()
        self.session.load_hf_weights(local_path)
        print(f"[maxtext-ckpt] loaded weights from {local_path} in {time.time() - started:.0f}s", flush=True)
        if del_local_after_load:
            import shutil

            shutil.rmtree(local_path, ignore_errors=True)

    def _hf_source_dir(self):
        for candidate in (
            getattr(self.engine_config, "hf_weights_path", None),
            getattr(self.model_config, "local_path", None),
            getattr(self.model_config, "path", None),
        ):
            if candidate and os.path.isdir(str(candidate)):
                return str(candidate)
        return None

    def _batch(self, data, with_targets: bool) -> dict:
        arrays = arrays_from_batch(data)
        batch_size, seq_len = self.session.batch_shape
        return maxtext_batch(
            arrays["token_ids"],
            arrays["token_mask"],
            arrays["response_mask"],
            old_log_probs=arrays["old_log_probs"] if with_targets else None,
            advantages=arrays["advantages"] if with_targets else None,
            batch_size=batch_size,
            seq_len=seq_len,
        )


def _actor_config(loss_function):
    keywords = getattr(loss_function, "keywords", None) or {}
    return keywords.get("config")


def _install_stack_dumps() -> None:
    """Let a hung trainer say where it is: SIGUSR1, or every MAXTEXT_STACK_DUMP_SECONDS."""
    import faulthandler
    import os
    import signal
    import sys

    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    seconds = float(os.environ.get("MAXTEXT_STACK_DUMP_SECONDS", "0") or 0)
    if seconds > 0:
        faulthandler.dump_traceback_later(seconds, repeat=True, file=sys.stderr)


def _pin_torch_device() -> None:
    """Send torch work to GPU 0 for this process.

    The process owns every GPU through JAX, but weights leave through one device
    and the rollout workers each see exactly one GPU as device 0. The worker base
    class may have selected another index, which would make the NCCL barrier key
    differ from the rollout side's.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


def _export_device():
    if torch.cuda.is_available():
        # Match the checkpoint engine's send device; see _checkpoint_device_index.
        return torch.device("cuda", int(os.environ.get("VERL_CHECKPOINT_DEVICE", "0")))
    return torch.device("cpu")


def _to_torch(value: np.ndarray, device) -> torch.Tensor:
    value = np.ascontiguousarray(value)
    if value.dtype.name == "bfloat16":
        return torch.from_numpy(value.view(np.uint16)).view(torch.bfloat16).to(device)
    return torch.from_numpy(value).to(device)


def _as_nested(log_probs: torch.Tensor, token_mask: np.ndarray) -> torch.Tensor:
    sequences = []
    for row, length in enumerate(token_mask.sum(axis=1).astype(int)):
        sequences.append(log_probs[row, :length])
    return torch.nested.as_nested_tensor(sequences, layout=torch.jagged)


def _paths(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _paths(value, f"{prefix}/{key}" if prefix else str(key))
    else:
        yield prefix


@EngineRegistry.register(model_type="language_model", backend="maxtext", device="cuda", vendor="amd")
@EngineRegistry.register(model_type="language_model", backend="maxtext", device="cuda")
class MaxTextEngineWithLMHead(_MaxTextEngine):
    """MaxText language-model engine."""


@EngineRegistry.register(model_type="value_model", backend="maxtext", device="cuda", vendor="amd")
@EngineRegistry.register(model_type="value_model", backend="maxtext", device="cuda")
class MaxTextEngineWithValueHead(_MaxTextEngine):
    """Value head is not available. GRPO does not construct this worker."""

    def initialize(self):
        raise NotImplementedError("MaxTextEngine has no value head; run GRPO with the critic disabled")
