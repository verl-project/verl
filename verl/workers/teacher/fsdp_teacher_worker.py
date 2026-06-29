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
"""Frozen teacher hosted as a forward-only FSDP worker.

The student-top-K reverse-KL pipeline (verl#6676) needs the teacher's
log-probabilities evaluated *at student-top-K token IDs*, which the existing
vLLM-served teacher cannot produce (its prompt_logprobs path is restricted to
the teacher's own top-K). This module hosts the teacher as a Ray FSDP worker
that runs a single forward pass per train step and chunked-gathers
``log_softmax(teacher_logits)`` at IDs supplied by the trainer.

The worker subclasses :class:`verl.workers.engine_workers.TrainingWorker` to
reuse the standard engine bootstrap (FSDP wrap, dtype, offload, dispatch
collect-info), pinning ``forward_only=True`` so the engine skips optimizer /
gradient setup.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from tensordict import TensorDict

from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils import tensordict_utils as tu
from verl.utils.profiler import DistProfiler
from verl.workers.config import TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker
from verl.workers.teacher.utils import chunked_gather_logprobs

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TeacherFSDPWorker(TrainingWorker):
    """Forward-only FSDP worker that serves teacher log-probabilities.

    Follows the standard :class:`TrainingWorker` flow but force-flips
    ``engine_config.forward_only = True``, so the engine is built without an optimizer.
    Exposes :meth:`compute_logprobs_at_ids`, which runs a teacher forward and gathers
    ``log_softmax(logits)`` at trainer-supplied token IDs without materializing the
    full ``[B, T, V]`` log-softmax tensor.

    Notes:
        - Ulysses sequence parallelism on the teacher is not yet supported; the
          trainer must set ``teacher_fsdp_config.ulysses_sequence_parallel_size = 1``.
        - The trainer must align ``topk_ids`` with the engine's rolled label
          convention (``roll(input_ids_rmpad, -1)``); this worker consumes them as-is.
    """

    def __init__(self, config: TrainingWorkerConfig):
        if config.engine_config is not None:
            # Frozen teacher: no optimizer, no grad, no checkpointing.
            config.engine_config.forward_only = True
            # Disable router replay (only meaningful for stateful training rollouts).
            if hasattr(config.engine_config, "router_replay"):
                config.engine_config.router_replay.mode = "disabled"
        super().__init__(config)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="purple", role="teacher_compute_logprobs_at_ids")
    def compute_logprobs_at_ids(self, data: TensorDict) -> Optional[TensorDict]:
        """Run a teacher forward and gather log-probabilities at given token IDs.

        Expected ``data`` fields (in the rmpad / nested-tensor layout produced by
        the trainer's pre-update step):

        * ``input_ids`` / ``position_ids``: nested tensors, jagged ``(bsz, seq_len_i)``.
        * ``attention_mask``: nested tensor or ``None``.
        * ``topk_ids``: nested ``(bsz, seq_len_i, K)`` int64, *already rolled by 1*
          along the sequence axis so position ``i`` carries the IDs for logits ``i``.
        * The engine's standard knobs (``use_remove_padding``, ``use_fused_kernels``,
          etc.), copied by the trainer from the actor's engine config.

        Returns:
            A :class:`TensorDict` with two nested ``(bsz, seq_len_i, K)`` fields:
            ``teacher_on_student_logp`` (teacher log-probs at the student top-K) and
            ``teacher_topk_ids`` (the teacher's own top-K IDs, for overlap diagnostics).
            Returns ``None`` when the worker is not the output-collecting rank
            (matches the :meth:`TrainingWorker.infer_batch` contract).
        """
        if self.engine.use_ulysses_sp:
            raise NotImplementedError(
                "Ulysses SP is not yet supported by TeacherFSDPWorker.compute_logprobs_at_ids; "
                "set teacher_fsdp_config.ulysses_sequence_parallel_size=1 for now."
            )

        # Inject the same engineering knobs the engine expects (mirrors TrainingWorker.infer_batch).
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=False,  # never use fused kernels: they don't expose logits
        )
        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        chunk_size = int(tu.get_non_tensor_data(data=data, key="teacher_chunk_size", default=1024))

        with self.engine.eval_mode(disable_auto_offload=False), torch.no_grad():
            # Standard input-prep (rmpad / position-id layout), but skip
            # prepare_model_outputs which would force label-conditioned log_probs.
            model_inputs, _ = self.engine.prepare_model_inputs(micro_batch=data)
            output = self.engine.module(**model_inputs)
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, V)

            topk_ids_nested = data["topk_ids"]
            assert topk_ids_nested.is_nested, "topk_ids must be a nested tensor"
            topk_ids_rmpad = topk_ids_nested.values()  # (total_nnz, K)
            assert topk_ids_rmpad.shape[0] == logits_rmpad.shape[0], (
                f"topk_ids rmpad shape {topk_ids_rmpad.shape} does not match logits "
                f"rmpad shape {logits_rmpad.shape}: trainer must align topk_ids with input_ids."
            )

            teacher_logp_rmpad = chunked_gather_logprobs(
                logits=logits_rmpad,
                topk_ids=topk_ids_rmpad,
                chunk_size=chunk_size,
            )

            # Teacher's own top-K IDs at the same width K, for the overlap diagnostics
            # only. Indices only — no gradient, no extra V-sized tensor.
            k = topk_ids_rmpad.shape[-1]
            teacher_topk_ids_rmpad = torch.topk(logits_rmpad, k=k, dim=-1).indices.to(torch.int64)

            cu_seqlens = data["input_ids"].offsets()
            # Move offsets to the same device as values for nested_tensor_from_jagged.
            cu_seqlens = cu_seqlens.to(teacher_logp_rmpad.device)
            teacher_on_student_logp = torch.nested.nested_tensor_from_jagged(
                teacher_logp_rmpad, cu_seqlens
            )
            teacher_topk_ids = torch.nested.nested_tensor_from_jagged(
                teacher_topk_ids_rmpad, cu_seqlens.to(teacher_topk_ids_rmpad.device)
            )

        if not self.engine.is_mp_src_rank_with_outputs():
            return None

        out = TensorDict(
            {
                "teacher_on_student_logp": teacher_on_student_logp,
                "teacher_topk_ids": teacher_topk_ids,
            },
            batch_size=data.batch_size,
        )
        return out.cpu()

    # Lets the trainer's dispatch helpers check whether a worker handle hosts a teacher.
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def is_teacher_worker(self) -> bool:
        return True
