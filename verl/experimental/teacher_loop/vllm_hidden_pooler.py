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

"""Export causal prediction rows from the final vLLM backbone output."""

import json
from bisect import bisect_left

import torch

from verl.experimental.teacher_loop.hidden_state_store import put_hidden_record


class HiddenStatePooler(torch.nn.Module):
    def __init__(self, token_pooler, *, producer: bool, hidden_size: int):
        super().__init__()
        self.token_pooler = token_pooler
        self.producer = producer
        self.hidden_size = hidden_size

    def get_pooling_updates(self, task):
        return self.token_pooler.get_pooling_updates(task)

    def get_supported_tasks(self):
        return self.token_pooler.get_supported_tasks()

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        offsets = cursor.first_token_indices_gpu.tolist()
        ends = cursor.seq_lens_cpu.tolist()
        counts = cursor.num_scheduled_tokens_cpu.tolist()
        finished = cursor.is_finished().tolist()
        outputs = []
        for i, params in enumerate(pooling_metadata.pooling_params):
            request = (params.extra_kwargs or {}).get("opd_record")
            if request is None:
                # Native token poolers use either cursor offsets (0.20) or
                # split the tensor by per-request token counts (0.32).
                row_metadata = pooling_metadata[i : i + 1]
                row_cursor = row_metadata.get_pooling_cursor()
                row_cursor.first_token_indices_gpu = row_cursor.first_token_indices_gpu - offsets[i]
                row_cursor.last_token_indices_gpu = row_cursor.last_token_indices_gpu - offsets[i]
                chunk = hidden_states.narrow(0, offsets[i], counts[i])
                outputs.extend(self.token_pooler(chunk, row_metadata))
                continue
            if not self.producer:
                # The executor only consumes output from rank zero of the last TP group.
                outputs.append(torch.empty(0, dtype=torch.uint8) if finished[i] else None)
                continue
            if hidden_states.ndim != 2 or hidden_states.size(1) != self.hidden_size:
                raise ValueError(f"Expected full teacher hidden width {self.hidden_size}, got {hidden_states.shape}")

            state = pooling_metadata.pooling_states[i]
            start = ends[i] - counts[i]
            if start == 0:
                # A preempted request can restart prefill using its previous pooling state.
                state.clean()
            positions = request["positions"]
            lo, hi = bisect_left(positions, start), bisect_left(positions, ends[i])
            dense = request.get("signal_kind") == "full_logits"
            if lo < hi or (dense and finished[i] and not state.hidden_states_cache):
                indices = torch.tensor(positions[lo:hi], dtype=torch.long, device=hidden_states.device)
                rows = hidden_states.index_select(0, indices + (offsets[i] - start))
                if dense:
                    from verl.trainer.distillation.hidden_state import project_full_logits

                    rows = project_full_logits(
                        request["teacher_model_path"],
                        request["teacher_weight_version"],
                        rows,
                        hidden_states.device,
                        request["projection_chunk_size"],
                    )
                state.hidden_states_cache.append(rows.cpu())
            if not finished[i]:
                outputs.append(None)
                continue
            chunks = state.hidden_states_cache
            if not chunks:
                hidden = hidden_states.new_empty((0, self.hidden_size), device="cpu")
            elif len(chunks) == 1:
                hidden = chunks[0]
            else:
                hidden = torch.cat(chunks)
            handle = put_hidden_record(request, hidden)
            state.clean()
            outputs.append(torch.tensor(list(json.dumps(handle).encode()), dtype=torch.uint8))
        return outputs


def install_hidden_pooler(worker):
    from vllm.distributed import get_tensor_model_parallel_rank

    runner = worker.model_runner
    poolers = runner.get_model().pooler.poolers_by_task
    poolers["token_embed"] = HiddenStatePooler(
        poolers["token_embed"],
        producer=get_tensor_model_parallel_rank() == 0,
        hidden_size=runner.model_config.get_hidden_size(),
    )
