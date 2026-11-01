# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Helpers for per-pass loss masks in an outer-chunked logical step."""


def reset_chunk_loss_mask_state(engine):
    engine._dsv41_mask_full = None
    engine._dsv41_n0 = None


def chunk_loss_data(engine, model_output, data):
    import torch

    chunk_ctx = getattr(engine, "_dsv41_chunk_ctx", None)
    if chunk_ctx is None:
        return data
    if int(chunk_ctx[1]) <= 1:
        return data

    log_probs = model_output.get("log_probs") if isinstance(model_output, dict) else None
    loss_mask = data.get("loss_mask") if hasattr(data, "get") else None
    if log_probs is None or loss_mask is None or not hasattr(loss_mask, "values"):
        return data

    log_prob_count = int(log_probs.numel())
    mask_values = loss_mask.values()
    if engine._dsv41_mask_full is None:
        engine._dsv41_mask_full = mask_values.clone()
    full_mask = engine._dsv41_mask_full
    if mask_values.numel() != full_mask.numel():
        raise ValueError(
            f"loss_mask changed during a chunked step: expected {full_mask.numel()} values, got {mask_values.numel()}"
        )

    if full_mask.numel() == log_prob_count:
        return data
    if full_mask.numel() < log_prob_count:
        raise ValueError(
            "loss_mask is shorter than log_probs in a chunked pass: "
            f"loss_mask={full_mask.numel()} log_probs={log_prob_count}"
        )

    chunk_index = int(chunk_ctx[0])
    chunk_count = int(chunk_ctx[1])
    if not 0 <= chunk_index < chunk_count:
        raise ValueError(f"chunk index {chunk_index} is outside [0, {chunk_count})")
    if chunk_index == 0:
        engine._dsv41_n0 = log_prob_count
    if engine._dsv41_n0 is None:
        raise RuntimeError("chunk pass state is missing the first pass log_probs length")

    # The window start is derived from pass 0's length, which is only the true cumulative
    # offset when every pass carries the same number of rows. Uneven pass lengths (padding,
    # tail handling) would silently shift the mask against the log-probs - shapes still
    # match, values would not - so refuse that case instead of misplacing it.
    if log_prob_count != engine._dsv41_n0:
        raise ValueError(
            "chunked passes must carry the same number of rows: "
            f"pass 0 has {engine._dsv41_n0}, pass {chunk_index} has {log_prob_count}"
        )
    start = chunk_index * engine._dsv41_n0
    end = start + log_prob_count
    if end > full_mask.numel():
        raise ValueError(
            f"chunk loss_mask window exceeds the original mask: window=[{start}, {end}) mask_length={full_mask.numel()}"
        )
    pass_mask = full_mask[start:end]
    if pass_mask.numel() != log_prob_count:
        raise ValueError(
            f"chunk loss_mask/log_probs lengths differ: loss_mask={pass_mask.numel()} log_probs={log_prob_count}"
        )

    # A shallow copy isolates the mapping while sharing large input tensors.
    pass_data = data.copy()
    pass_data["loss_mask"] = torch.nested.nested_tensor([pass_mask], layout=torch.jagged)
    return pass_data
