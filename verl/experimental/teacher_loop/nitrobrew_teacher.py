# Copyright 2026 Tilde Research
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

"""Nitrobrew teacher infrastructure.

Replaces the vLLM/SGLang-based teacher server when
distillation.distillation_loss.loss_mode = "nitrobrew".

Instead of top-k logprobs, the teacher returns:
    - lm_head.weight  [V, D_t]  (once at init, distributed to actor ranks)
    - hidden states    [S, D_t]  (per sequence, via Ray remote calls)

The student-side kernel reconstructs teacher logits on-the-fly as z @ W.T
in vocabulary chunks, avoiding the O(N * V) materialisation.
"""

import logging
from typing import Any

import ray
import torch

logger = logging.getLogger(__name__)


@ray.remote
class NitrobrewTeacherWorker:
    """Loads teacher HF model and serves last-layer hidden states."""

    def setup(self, model_path: str, torch_dtype: str = "bfloat16") -> list:
        """Load teacher model, return lm_head.weight as nested list [V, D_t].

        Returns:
            lm_head.weight as Python list for Ray serialisation.
        """
        from transformers import AutoModelForCausalLM

        dtype = getattr(torch, torch_dtype)
        logger.warning(f"NitrobrewTeacherWorker: loading {model_path} (dtype={torch_dtype})")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="balanced",
        )
        self._model.eval()
        self._dtype = dtype

        W = self._model.lm_head.weight.to(dtype).cpu()
        logger.warning(f"NitrobrewTeacherWorker: lm_head.weight {W.shape}")
        return W.tolist()

    def compute_hidden_states(self, sequence_ids: list[int]) -> list:
        """Run teacher forward pass, return last hidden states [S, D_t] as nested list."""
        first_device = next(self._model.parameters()).device
        input_ids = torch.tensor([sequence_ids], dtype=torch.long, device=first_device)

        with torch.no_grad():
            out = self._model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

        h_T = out.hidden_states[-1][0].to(self._dtype)
        return h_T.cpu().tolist()


class NitrobrewAsyncTeacherManager:
    """Async client that round-robins hidden-state requests across NitrobrewTeacherWorker actors."""

    def __init__(self, worker_handles: dict[str, list[Any]]):
        self._worker_handles = worker_handles
        self._counters = {key: 0 for key in worker_handles}

    def _resolve_key(self, routing_key: str | None) -> str:
        if len(self._worker_handles) == 1:
            return next(iter(self._worker_handles))
        if routing_key is None or routing_key not in self._worker_handles:
            raise ValueError(
                f"Routing key {routing_key!r} not found in nitrobrew workers: "
                f"{sorted(self._worker_handles)}"
            )
        return routing_key

    async def compute_teacher_hidden_states_single(
        self,
        sequence_ids: list[int],
        routing_key: str | None = None,
    ) -> torch.Tensor:
        """Request hidden states for a single sequence. Returns [S, D_t] on CPU."""
        key = self._resolve_key(routing_key)
        handles = self._worker_handles[key]
        idx = self._counters[key] % len(handles)
        self._counters[key] += 1

        result: list = await handles[idx].compute_hidden_states.remote(sequence_ids)
        return torch.tensor(result, dtype=torch.bfloat16)


class NitrobrewTeacherModelManager:
    """Manages a pool of NitrobrewTeacherWorker Ray actors.

    Exposes:
        worker_handles: dict[teacher_key, list[ActorHandle]]  -- for AgentLoopWorker
        w_up:           torch.Tensor [V, D_t]                 -- for actor set_teacher_unembed
    """

    def __init__(
        self,
        teacher_model_configs: dict,
        gpus_per_replica: int,
    ):
        self.worker_handles: dict[str, list] = {}
        self.w_up: torch.Tensor | None = None

        for key, teacher_cfg in teacher_model_configs.items():
            num_replicas = teacher_cfg.num_replicas
            model_path = teacher_cfg.model_path
            dtype = getattr(teacher_cfg, "torch_dtype", "bfloat16")

            handles = []
            for _ in range(num_replicas):
                worker = NitrobrewTeacherWorker.options(num_gpus=gpus_per_replica).remote()
                handles.append(worker)
            self.worker_handles[key] = handles

            logger.warning(f"NitrobrewTeacherModelManager: running setup for teacher '{key}'")
            w_list = ray.get(handles[0].setup.remote(model_path, dtype))
            w = torch.tensor(w_list)

            setup_futures = [h.setup.remote(model_path, dtype) for h in handles[1:]]
            ray.get(setup_futures)

            if self.w_up is None:
                self.w_up = w

        assert self.w_up is not None, "No teacher models configured."
