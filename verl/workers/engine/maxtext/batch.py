"""Turn a verl sequence batch into the arrays MaxText's train step consumes.

MaxText's data sharding is one spec for every rank-2 array in the batch, and
its loss slices ``array[:micro, :]``. Every array here is ``(batch, seq)``.
``log_probs[i]`` is the log probability of ``token[i + 1]``. The last position
is zero. That is the alignment ``response_from_nested`` slices.
"""

from __future__ import annotations

import numpy as np


def next_token_logprobs(logits, token_ids):
    """Log-prob of ``token_ids[i + 1]`` at index ``i``. ``logits`` is ``(batch, seq, vocab)``."""
    xp = np if isinstance(logits, np.ndarray) else _jax_numpy()
    shifted_logits = logits[:, :-1, :]
    shifted_tokens = token_ids[:, 1:]
    stable = shifted_logits - xp.max(shifted_logits, axis=-1, keepdims=True)
    log_z = xp.log(xp.sum(xp.exp(stable), axis=-1, keepdims=True))
    gathered = xp.take_along_axis(stable, shifted_tokens[..., None], axis=-1)[..., 0]
    body = gathered - log_z[..., 0]
    tail = xp.zeros(token_ids.shape[:1] + (1,), dtype=body.dtype)
    return xp.concatenate([body, tail], axis=1)


def next_token_entropy(logits):
    """Entropy of the next-token distribution at each index, aligned with the log-probs."""
    xp = np if isinstance(logits, np.ndarray) else _jax_numpy()
    shifted = logits[:, :-1, :]
    stable = shifted - xp.max(shifted, axis=-1, keepdims=True)
    weights = xp.exp(stable)
    total = xp.sum(weights, axis=-1, keepdims=True)
    log_z = xp.log(total)
    probs = weights / total
    body = log_z[..., 0] - xp.sum(probs * stable, axis=-1)
    tail = xp.zeros(logits.shape[:1] + (1,), dtype=body.dtype)
    return xp.concatenate([body, tail], axis=1)


def _jax_numpy():
    import jax.numpy as jnp

    return jnp


def response_positions(token_mask: np.ndarray, response_mask: np.ndarray) -> np.ndarray:
    """Mask over the positions whose next token is a response token.

    ``token_mask`` and ``response_mask`` are 0/1 with shape ``(batch, seq)``.
    ``response_mask`` marks the response tokens themselves. The returned mask
    marks the preceding position, which is where that token's log-prob sits.
    """
    if response_mask.shape != token_mask.shape:
        raise ValueError(f"response_mask shape {response_mask.shape} must match the sequence {token_mask.shape}")
    positions = np.zeros_like(response_mask)
    positions[:, :-1] = response_mask[:, 1:]
    return positions * token_mask


def arrays_from_batch(data) -> dict[str, np.ndarray]:
    """Read verl's batch, nested or padded, into dense numpy arrays."""

    token_ids, token_mask = _dense_rows(data["input_ids"])
    response = data.get("response_mask")
    if response is None:
        response = data.get("loss_mask")
    response_mask = _response_mask(token_mask, response)
    return {
        "token_ids": token_ids,
        "token_mask": token_mask,
        "response_mask": response_mask,
        "old_log_probs": _optional_rows(data, "old_log_probs"),
        "advantages": _optional_rows(data, "advantages"),
    }


def _dense_rows(tensor) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if getattr(tensor, "is_nested", False):
        offsets = tensor.offsets().detach().cpu()
        values = tensor.values().detach().cpu()
        lengths = offsets.diff()
        rows = int(lengths.shape[0])
        width = int(lengths.max()) if rows else 0
        ids = torch.zeros(rows, width, dtype=torch.int32)
        mask = torch.zeros(rows, width, dtype=torch.int32)
        for row, (start, end) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True)):
            count = end - start
            ids[row, :count] = values[start:end]
            mask[row, :count] = 1
        return ids.numpy(), mask.numpy()
    ids = tensor.detach().to(dtype=torch.int32).cpu()
    mask = torch.ones_like(ids)
    return ids.numpy(), mask.numpy()


def _response_mask(token_mask: np.ndarray, response) -> np.ndarray:
    import torch

    rows, width = token_mask.shape
    placed = np.zeros((rows, width), dtype=np.float32)
    if response is None:
        return placed
    if getattr(response, "is_nested", False):
        lengths = response.offsets().diff().detach().cpu().tolist()
        seq_lens = token_mask.sum(axis=1).astype(int)
        for row, (resp_len, seq_len) in enumerate(zip(lengths, seq_lens, strict=True)):
            start = max(int(seq_len) - int(resp_len), 0)
            placed[row, start : int(seq_len)] = 1
        return placed
    array = response.detach().cpu().numpy() if isinstance(response, torch.Tensor) else np.asarray(response)
    if array.shape == token_mask.shape:
        return array.astype(np.float32)
    seq_lens = token_mask.sum(axis=1).astype(int)
    for row in range(rows):
        resp_len = int(array.shape[1])
        start = max(int(seq_lens[row]) - resp_len, 0)
        placed[row, start : int(seq_lens[row])] = array[row, : int(seq_lens[row]) - start]
    return placed


def _optional_rows(data, key: str):
    if key not in data.keys():
        return None
    value = data[key]
    if getattr(value, "is_nested", False):
        import torch

        lengths = value.offsets().diff()
        width = int(lengths.max()) if lengths.numel() else 0
        rows = int(lengths.shape[0])
        dense = torch.zeros(rows, width, dtype=torch.float32)
        values = value.values().detach().float().cpu()
        offsets = value.offsets().detach().cpu()
        for row, (start, end) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True)):
            dense[row, : end - start] = values[start:end]
        return dense.numpy()
    return value.detach().float().cpu().numpy()


def maxtext_batch(
    token_ids: np.ndarray,
    token_mask: np.ndarray,
    response_mask: np.ndarray,
    old_log_probs: np.ndarray | None = None,
    advantages: np.ndarray | None = None,
    batch_size: int | None = None,
    seq_len: int | None = None,
) -> dict[str, np.ndarray]:
    """Pad to the compiled ``(batch_size, seq_len)`` and add MaxText's input names.

    ``old_log_probs`` and ``advantages`` are aligned with response tokens. They
    are scattered onto the same positions as ``response_positions``.
    """
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must be (batch, seq), got {token_ids.shape}")
    batch_size = batch_size or token_ids.shape[0]
    seq_len = seq_len or token_ids.shape[1]
    if token_ids.shape[0] > batch_size or token_ids.shape[1] > seq_len:
        raise ValueError(f"batch {token_ids.shape} does not fit the compiled MaxText shape {(batch_size, seq_len)}")
    out_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
    out_mask = np.zeros((batch_size, seq_len), dtype=np.int32)
    rows, cols = token_ids.shape
    out_ids[:rows, :cols] = token_ids
    out_mask[:rows, :cols] = token_mask.astype(np.int32)
    positions = np.arange(seq_len, dtype=np.int32)[None, :].repeat(batch_size, axis=0)
    positions = positions * out_mask

    resp = np.zeros((batch_size, seq_len), dtype=np.float32)
    resp[:rows, :cols] = response_mask.astype(np.float32)
    loss_pos = response_positions(out_mask, resp.astype(np.int32)).astype(np.float32)

    def _scatter(values: np.ndarray | None) -> np.ndarray:
        placed = np.zeros((batch_size, seq_len), dtype=np.float32)
        if values is None:
            return placed
        if values.shape == (rows, cols):
            placed[:rows, :cols] = values
            return placed * loss_pos
        # Response-length values: write them onto the response positions in order.
        for row in range(rows):
            slots = np.flatnonzero(loss_pos[row, :cols] > 0)
            count = min(slots.size, values.shape[1])
            placed[row, slots[:count]] = values[row, :count]
        return placed

    targets = np.zeros_like(out_ids)
    targets[:, :-1] = out_ids[:, 1:]
    target_mask = np.zeros_like(out_mask)
    target_mask[:, :-1] = out_mask[:, 1:]
    return {
        "inputs": out_ids,
        "inputs_position": positions,
        "inputs_segmentation": out_mask,
        "targets": targets,
        "targets_segmentation": target_mask,
        "old_log_probs": _scatter(old_log_probs),
        "advantages": _scatter(advantages),
        "response_mask": loss_pos,
    }
