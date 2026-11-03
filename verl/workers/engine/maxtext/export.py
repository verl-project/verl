"""Turn MaxText's scanned Qwen3-MoE layout back into Hugging Face tensors.

The layout is the one produced by MaxText's ``convert_qwen3_moe`` after the
final scanned-axis transpose. Arrays are numpy. The checkpoint engine receives
torch tensors from the caller.
"""

from __future__ import annotations

import numpy as np


def check_layout(weights: dict, num_heads: int, head_dim: int) -> tuple[int, int]:
    """Return ``(num_layers, hidden)`` or raise KeyError/ValueError naming the problem."""
    layers = weights["decoder"]["layers"]
    attention = layers["self_attention"]
    layers["moe_block"]
    for leaf in ("key", "value", "out"):
        attention[leaf]["kernel"]
    for name in ("pre_self_attention_layer_norm", "post_self_attention_layer_norm"):
        layers[name]["scale"]
    query = attention["query"]["kernel"]
    if tuple(query.shape[2:]) != (num_heads, head_dim):
        raise ValueError(
            f"query kernel shape {tuple(query.shape)} does not match heads={num_heads} head_dim={head_dim}"
        )
    return int(query.shape[1]), int(query.shape[0])


def iter_hf_from_stacked(weights: dict, num_heads: int, num_kv_heads: int, head_dim: int, fetch=np.asarray):
    """Yield ``(hf_name, numpy_array)`` one layer at a time.

    ``weights`` may hold numpy arrays or sharded device arrays. ``fetch`` turns a
    slice into a host array, so only one layer's slice is on the host at once.
    """
    num_layers, hidden = check_layout(weights, num_heads, head_dim)
    decoder = weights["decoder"]
    layers = decoder["layers"]
    attention = layers["self_attention"]
    moe = layers["moe_block"]

    yield "model.embed_tokens.weight", fetch(weights["token_embedder"]["embedding"])
    yield "model.norm.weight", fetch(decoder["decoder_norm"]["scale"])
    yield "lm_head.weight", np.ascontiguousarray(fetch(decoder["logits_dense"]["kernel"]).T)
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        yield f"{prefix}.input_layernorm.weight", fetch(layers["pre_self_attention_layer_norm"]["scale"][:, layer])
        yield (
            f"{prefix}.post_attention_layernorm.weight",
            fetch(layers["post_self_attention_layer_norm"]["scale"][:, layer]),
        )
        yield (
            f"{prefix}.self_attn.q_proj.weight",
            _dense(fetch(attention["query"]["kernel"][:, layer]), num_heads, head_dim, hidden),
        )
        yield (
            f"{prefix}.self_attn.k_proj.weight",
            _dense(fetch(attention["key"]["kernel"][:, layer]), num_kv_heads, head_dim, hidden),
        )
        yield (
            f"{prefix}.self_attn.v_proj.weight",
            _dense(fetch(attention["value"]["kernel"][:, layer]), num_kv_heads, head_dim, hidden),
        )
        yield f"{prefix}.self_attn.o_proj.weight", _out_proj(fetch(attention["out"]["kernel"][:, layer]), hidden)
        yield f"{prefix}.self_attn.q_norm.weight", fetch(attention["query_norm"]["scale"][:, layer])
        yield f"{prefix}.self_attn.k_norm.weight", fetch(attention["key_norm"]["scale"][:, layer])
        yield f"{prefix}.mlp.gate.weight", np.ascontiguousarray(fetch(moe["gate"]["kernel"][:, layer, :]).T)
        gate_proj = fetch(moe["wi_0"][:, layer])
        up = fetch(moe["wi_1"][:, layer])
        down = fetch(moe["wo"][:, layer])
        for expert in range(gate_proj.shape[0]):
            name = f"{prefix}.mlp.experts.{expert}"
            yield f"{name}.gate_proj.weight", np.ascontiguousarray(gate_proj[expert].T)
            yield f"{name}.up_proj.weight", np.ascontiguousarray(up[expert].T)
            yield f"{name}.down_proj.weight", np.ascontiguousarray(down[expert].T)


def hf_from_stacked(weights: dict, num_heads: int, num_kv_heads: int, head_dim: int) -> dict[str, np.ndarray]:
    """Unstack a converted Qwen3-MoE checkpoint into a dict of Hugging Face tensors."""
    return dict(iter_hf_from_stacked(weights, num_heads, num_kv_heads, head_dim))


def _dense(kernel, heads: int, head_dim: int, hidden: int) -> np.ndarray:
    weight = np.reshape(kernel, (hidden, heads * head_dim))
    return np.ascontiguousarray(weight.T)


def _out_proj(kernel, hidden: int) -> np.ndarray:
    # kernel is (heads, head_dim, hidden), which is the transpose of the HF out projection.
    return np.ascontiguousarray(np.reshape(kernel, (-1, hidden)).T)


def stack_qwen3_for_test(
    hf: dict[str, np.ndarray],
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_experts: int,
) -> dict:
    """Apply the convert_qwen3_moe transforms. Used to test the inverse."""
    hidden = hf["model.embed_tokens.weight"].shape[1]
    intermediate = hf["model.layers.0.mlp.experts.0.gate_proj.weight"].shape[0]

    def zeros(*shape):
        return np.zeros(shape, dtype=np.float32)

    pre = zeros(num_layers, hidden)
    post = zeros(num_layers, hidden)
    query = zeros(num_layers, hidden, num_heads, head_dim)
    key = zeros(num_layers, hidden, num_kv_heads, head_dim)
    value = zeros(num_layers, hidden, num_kv_heads, head_dim)
    out = zeros(num_layers, num_heads, head_dim, hidden)
    q_norm = zeros(num_layers, head_dim)
    k_norm = zeros(num_layers, head_dim)
    gate = zeros(num_layers, hidden, num_experts)
    wi_0 = zeros(num_experts, num_layers, hidden, intermediate)
    wi_1 = zeros(num_experts, num_layers, hidden, intermediate)
    wo = zeros(num_experts, num_layers, intermediate, hidden)
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        pre[layer] = hf[f"{prefix}.input_layernorm.weight"]
        post[layer] = hf[f"{prefix}.post_attention_layernorm.weight"]
        query[layer] = hf[f"{prefix}.self_attn.q_proj.weight"].T.reshape(hidden, num_heads, head_dim)
        key[layer] = hf[f"{prefix}.self_attn.k_proj.weight"].T.reshape(hidden, num_kv_heads, head_dim)
        value[layer] = hf[f"{prefix}.self_attn.v_proj.weight"].T.reshape(hidden, num_kv_heads, head_dim)
        out[layer] = hf[f"{prefix}.self_attn.o_proj.weight"].T.reshape(num_heads, head_dim, hidden)
        q_norm[layer] = hf[f"{prefix}.self_attn.q_norm.weight"]
        k_norm[layer] = hf[f"{prefix}.self_attn.k_norm.weight"]
        gate[layer] = hf[f"{prefix}.mlp.gate.weight"].T
        for expert in range(num_experts):
            name = f"{prefix}.mlp.experts.{expert}"
            wi_0[expert, layer] = hf[f"{name}.gate_proj.weight"].T
            wi_1[expert, layer] = hf[f"{name}.up_proj.weight"].T
            wo[expert, layer] = hf[f"{name}.down_proj.weight"].T
    return {
        "token_embedder": {"embedding": hf["model.embed_tokens.weight"]},
        "decoder": {
            "decoder_norm": {"scale": hf["model.norm.weight"]},
            "logits_dense": {"kernel": hf["lm_head.weight"].T},
            "layers": {
                "pre_self_attention_layer_norm": {"scale": np.transpose(pre, (1, 0))},
                "post_self_attention_layer_norm": {"scale": np.transpose(post, (1, 0))},
                "self_attention": {
                    "query": {"kernel": np.transpose(query, (1, 0, 2, 3))},
                    "key": {"kernel": np.transpose(key, (1, 0, 2, 3))},
                    "value": {"kernel": np.transpose(value, (1, 0, 2, 3))},
                    "out": {"kernel": np.transpose(out, (1, 0, 2, 3))},
                    "query_norm": {"scale": np.transpose(q_norm, (1, 0))},
                    "key_norm": {"scale": np.transpose(k_norm, (1, 0))},
                },
                "moe_block": {"gate": {"kernel": np.transpose(gate, (1, 0, 2))}, "wi_0": wi_0, "wi_1": wi_1, "wo": wo},
            },
        },
    }


def iter_hf_from_stacked_device(
    weights: dict,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device,
    max_layers: int | None = None,
    dtype=None,
):
    """Yield ``(hf_name, torch.Tensor)`` without going through host memory.

    Each slice is cast to bf16, gathered onto the JAX device ``device`` (the one the
    NCCL sender uses), transposed there, and handed to torch with DLPack. A layer is
    yielded as views of a few large tensors. The caller must consume a tensor
    before asking for the next layer's, as the checkpoint engine does.
    ``dtype`` defaults to bfloat16 for the rollout; checkpoints pass float32.
    """
    import jax
    import jax.numpy as jnp
    import torch

    num_layers, hidden = check_layout(weights, num_heads, head_dim)
    if max_layers is not None:
        num_layers = min(num_layers, max_layers)
    decoder = weights["decoder"]
    layers = decoder["layers"]
    attention = layers["self_attention"]
    moe = layers["moe_block"]

    dtype = jnp.bfloat16 if dtype is None else dtype

    def pull(array, transform=None):
        if array.dtype != dtype:
            array = array.astype(dtype)
        array = jax.device_put(array, device)
        if transform is not None:
            array = transform(array)
        array.block_until_ready()
        return torch.from_dlpack(array)

    def dense(array, heads):
        return jnp.reshape(array, (hidden, heads * head_dim)).T

    yield "model.embed_tokens.weight", pull(weights["token_embedder"]["embedding"])
    yield "model.norm.weight", pull(decoder["decoder_norm"]["scale"])
    yield "lm_head.weight", pull(decoder["logits_dense"]["kernel"], lambda a: a.T)
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        yield f"{prefix}.input_layernorm.weight", pull(layers["pre_self_attention_layer_norm"]["scale"][:, layer])
        yield (
            f"{prefix}.post_attention_layernorm.weight",
            pull(layers["post_self_attention_layer_norm"]["scale"][:, layer]),
        )
        yield (
            f"{prefix}.self_attn.q_proj.weight",
            pull(attention["query"]["kernel"][:, layer], lambda a: dense(a, num_heads)),
        )
        yield (
            f"{prefix}.self_attn.k_proj.weight",
            pull(attention["key"]["kernel"][:, layer], lambda a: dense(a, num_kv_heads)),
        )
        yield (
            f"{prefix}.self_attn.v_proj.weight",
            pull(attention["value"]["kernel"][:, layer], lambda a: dense(a, num_kv_heads)),
        )
        yield (
            f"{prefix}.self_attn.o_proj.weight",
            pull(attention["out"]["kernel"][:, layer], lambda a: jnp.reshape(a, (-1, hidden)).T),
        )
        yield f"{prefix}.self_attn.q_norm.weight", pull(attention["query_norm"]["scale"][:, layer])
        yield f"{prefix}.self_attn.k_norm.weight", pull(attention["key_norm"]["scale"][:, layer])
        yield f"{prefix}.mlp.gate.weight", pull(moe["gate"]["kernel"][:, layer, :], lambda a: a.T)
        # (experts, in, out) -> (experts, out, in): the HF layout of each expert matrix.
        swap = lambda a: jnp.swapaxes(a, -1, -2)  # noqa: E731
        gate_proj = pull(moe["wi_0"][:, layer], swap)
        up = pull(moe["wi_1"][:, layer], swap)
        down = pull(moe["wo"][:, layer], swap)
        for expert in range(gate_proj.shape[0]):
            name = f"{prefix}.mlp.experts.{expert}"
            yield f"{name}.gate_proj.weight", gate_proj[expert]
            yield f"{name}.up_proj.weight", up[expert]
            yield f"{name}.down_proj.weight", down[expert]
