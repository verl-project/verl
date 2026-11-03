"""Read a Hugging Face Qwen3-MoE checkpoint into the scanned MaxText layout.

This is the inverse of ``export.iter_hf_from_stacked``. One leaf at a time is
assembled on the host and placed with the sharding of the parameter it
replaces, so peak host memory is the largest leaf, not the whole model.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

_TOP = {
    ("token_embedder", "embedding"): ("model.embed_tokens.weight", False),
    ("decoder", "decoder_norm", "scale"): ("model.norm.weight", False),
    ("decoder", "logits_dense", "kernel"): ("lm_head.weight", True),
}

# (leaf path, HF suffix under model.layers.N, matrix?)
_PER_LAYER = [
    (("decoder", "layers", "pre_self_attention_layer_norm", "scale"), "input_layernorm.weight", False),
    (("decoder", "layers", "post_self_attention_layer_norm", "scale"), "post_attention_layernorm.weight", False),
    (("decoder", "layers", "self_attention", "query", "kernel"), "self_attn.q_proj.weight", True),
    (("decoder", "layers", "self_attention", "key", "kernel"), "self_attn.k_proj.weight", True),
    (("decoder", "layers", "self_attention", "value", "kernel"), "self_attn.v_proj.weight", True),
    (("decoder", "layers", "self_attention", "out", "kernel"), "self_attn.o_proj.weight", True),
    (("decoder", "layers", "self_attention", "query_norm", "scale"), "self_attn.q_norm.weight", False),
    (("decoder", "layers", "self_attention", "key_norm", "scale"), "self_attn.k_norm.weight", False),
    (("decoder", "layers", "moe_block", "gate", "kernel"), "mlp.gate.weight", True),
]

_EXPERTS = [
    (("decoder", "layers", "moe_block", "wi_0"), "gate_proj"),
    (("decoder", "layers", "moe_block", "wi_1"), "up_proj"),
    (("decoder", "layers", "moe_block", "wo"), "down_proj"),
]


class HFReader:
    """Tensor access by HF name. Returns numpy in the requested dtype."""

    def __init__(self, path: str):
        self.path = path
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as handle:
                self.files = json.load(handle)["weight_map"]
        else:
            single = os.path.join(path, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(f"No safetensors checkpoint under {path}")
            from safetensors import safe_open

            with safe_open(single, framework="pt") as handle:
                self.files = {name: "model.safetensors" for name in handle.keys()}  # noqa: SIM118
        self._local = threading.local()
        self._pool = None

    def get_many(self, names: list, dtype) -> list:
        """Read several tensors concurrently; network filesystems need many requests in flight."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=int(os.environ.get("MAXTEXT_HF_READ_THREADS", "32")))
        return list(self._pool.map(lambda name: self.get(name, dtype), names))

    def get(self, name: str, dtype) -> np.ndarray:
        import torch
        from safetensors import safe_open

        shard = self.files[name]
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = {}
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(self.path, shard), framework="pt")
        tensor = handles[shard].get_tensor(name)
        if np.dtype(dtype).name == "bfloat16":
            import ml_dtypes

            tensor = tensor.to(torch.bfloat16).contiguous()
            return tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        return tensor.to(torch.float32).numpy().astype(dtype)


def _set(tree: dict, path: tuple, value) -> None:
    for key in path[:-1]:
        tree = tree.setdefault(key, {})
    tree[path[-1]] = value


def _get(tree: dict, path: tuple):
    for key in path:
        tree = tree[key]
    return tree


def load_stacked_from_hf(reader: HFReader, like: dict, place: Callable[[object, np.ndarray], object]) -> dict:
    """Build a pytree shaped like ``like`` from HF tensors.

    ``place(old_leaf, host_array)`` returns the replacement leaf, normally a
    ``jax.device_put`` onto ``old_leaf.sharding``.
    """
    out: dict = {}
    for path, (name, matrix) in _TOP.items():
        old = _get(like, path)
        array = reader.get(name, old.dtype)
        array = np.ascontiguousarray(array.T if matrix else array)
        _check(path, array.shape, old.shape)
        _set(out, path, place(old, array))

    sample = _get(like, ("decoder", "layers", "self_attention", "query", "kernel"))
    num_layers = int(sample.shape[1])
    for path, suffix, matrix in _PER_LAYER:
        old = _get(like, path)
        host = np.empty(old.shape, dtype=np.dtype(old.dtype))
        for layer in range(num_layers):
            array = reader.get(f"model.layers.{layer}.{suffix}", old.dtype)
            target = host[:, layer].shape
            host[:, layer] = (array.T if matrix else array).reshape(target)
        _set(out, path, place(old, host))
        del host

    for path, proj in _EXPERTS:
        old = _get(like, path)
        host = np.empty(old.shape, dtype=np.dtype(old.dtype))
        for layer in range(num_layers):
            names = [f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight" for e in range(host.shape[0])]
            for expert, array in enumerate(reader.get_many(names, old.dtype)):
                host[expert, layer] = array.T
        _set(out, path, place(old, host))
        del host
    return out


def _check(path: tuple, got: tuple, want: tuple) -> None:
    if tuple(got) != tuple(want):
        raise ValueError(f"{'/'.join(path)}: checkpoint gives {tuple(got)}, model expects {tuple(want)}")
