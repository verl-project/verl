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

from collections.abc import Iterator
from contextlib import contextmanager

import torch


def _get_output_layer(model):
    from verl.utils.megatron_utils import unwrap_model

    unwrapped = unwrap_model(model)
    language_model = getattr(unwrapped, "language_model", None)
    if language_model is None:
        language_model = unwrapped
    output_layer = getattr(language_model, "output_layer", None)
    if output_layer is None:
        raise RuntimeError("response_only_lm_head requires a Megatron language model with an output_layer")
    return output_layer


@contextmanager
def response_only_output_projection(model, projection_mask: torch.Tensor) -> Iterator[None]:
    """Project only hidden states whose next token contributes to the policy loss.

    ``projection_mask`` is already CP-local and next-token aligned. Megatron's
    sequence-parallel output layer normally gathers hidden states over TP before
    projecting them. Gather explicitly, select active rows, and disable the
    layer's second gather for this invocation.
    """
    if projection_mask.ndim != 2:
        raise ValueError(f"projection_mask must have shape [batch, sequence], got {projection_mask.shape}")
    if projection_mask.dtype != torch.bool:
        projection_mask = projection_mask.to(torch.bool)

    output_layer = _get_output_layer(model)
    sequence_parallel = bool(getattr(output_layer, "sequence_parallel", False))
    if sequence_parallel and not hasattr(output_layer, "disable_grad_reduce"):
        raise RuntimeError(
            "response_only_lm_head requires an output layer that supports disable_grad_reduce "
            "when sequence parallelism is enabled"
        )
    disable_grad_reduce = getattr(output_layer, "disable_grad_reduce", None)

    def select_hidden_states(_module, args, kwargs):
        if args:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("input_")
            if hidden_states is None:
                raise RuntimeError("Could not find the LM-head hidden-state input")

        if sequence_parallel:
            from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=True,
                group=getattr(output_layer, "tp_group", None),
            )

        expected = (projection_mask.shape[1], projection_mask.shape[0])
        if hidden_states.shape[:2] != expected:
            raise RuntimeError(
                "LM-head hidden states and response projection mask are misaligned: "
                f"hidden={tuple(hidden_states.shape)}, mask={tuple(projection_mask.shape)}"
            )

        selected = hidden_states.transpose(0, 1)[projection_mask]
        if selected.numel() == 0:
            # Keep every TP/CP rank in the LM-head backward graph. The matching
            # scalar result is multiplied by zero before it reaches the loss.
            selected = hidden_states[:1, :1, :].reshape(1, hidden_states.shape[-1])
        selected = selected.reshape(-1, 1, hidden_states.shape[-1]).contiguous()

        if args:
            return (selected, *args[1:]), kwargs
        kwargs["input_"] = selected
        return args, kwargs

    handle = output_layer.register_forward_pre_hook(select_hidden_states, with_kwargs=True)
    if sequence_parallel:
        output_layer.sequence_parallel = False
        # The explicit gather above owns the reduce-scatter in backward. Avoid
        # ColumnParallelLinear's non-SP input-gradient all-reduce as well.
        output_layer.disable_grad_reduce = True
    try:
        yield
    finally:
        handle.remove()
        if sequence_parallel:
            output_layer.sequence_parallel = True
            output_layer.disable_grad_reduce = disable_grad_reduce


def select_response_only_inputs(
    label: torch.Tensor,
    temperature: torch.Tensor,
    projection_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Select labels and temperatures aligned with sparse LM-head logits."""
    if projection_mask.dtype != torch.bool:
        projection_mask = projection_mask.to(torch.bool)
    if label.shape != projection_mask.shape or temperature.shape != projection_mask.shape:
        raise ValueError(
            "label, temperature, and projection_mask must have identical shapes; "
            f"got label={label.shape}, temperature={temperature.shape}, mask={projection_mask.shape}"
        )

    sparse_label = label[projection_mask]
    sparse_temperature = temperature[projection_mask]
    num_selected = sparse_label.numel()
    if num_selected == 0:
        sparse_label = label.new_zeros(1)
        sparse_temperature = temperature.new_ones(1)
    return sparse_label.unsqueeze(0), sparse_temperature.unsqueeze(0), num_selected


def restore_response_only_outputs(
    outputs: dict[str, torch.Tensor], projection_mask: torch.Tensor, num_selected: int
) -> dict[str, torch.Tensor]:
    """Scatter sparse per-token outputs back to their CP-local dense layout."""
    projected_tokens = max(num_selected, 1)
    restored = {}
    for name, value in outputs.items():
        if value.shape[:2] != (1, projected_tokens):
            raise ValueError(
                f"Sparse output {name!r} has shape {tuple(value.shape)}; expected prefix (1, {projected_tokens})"
            )
        dense = value.new_zeros((*projection_mask.shape, *value.shape[2:]))
        if num_selected:
            dense[projection_mask] = value[0]
        else:
            # Preserve a zero-gradient path through the LM head on empty CP ranks.
            dense = dense + value.reshape(-1)[:0].sum()
        restored[name] = dense
    return restored
