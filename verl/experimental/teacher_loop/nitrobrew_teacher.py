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

"""Nitrobrew teacher infrastructure (vLLM AsyncLLM pooling backend).

Used when ``distillation.distillation_loss.loss_mode == "nitrobrew"``.

Architecture::

                      +----------------------------------+
    AgentLoopWorker   |  NitrobrewAsyncTeacherManager    |
                      |  (round-robin Ray remote calls)  |
                      +-----------------+----------------+
                                        | compute_hidden_states.remote(seq_ids)
                      +-----------------v----------------+
                      |  NitrobrewTeacherWorker (Ray)    |
                      |  - vLLM AsyncLLM (pooling+embed) |
                      |  - PoolingType.ALL -> [S, D]     |
                      |  - on-actor [S, D] @ P_down      |
                      |    -> [S, d_comp]                |
                      +----------------------------------+

Per-teacher SVD: ``W_T (lm_head) ~= W_up @ P_down.T`` with
``W_up [V, d_comp]`` (sent to the actor for student-side logit
reconstruction) and ``P_down [D, d_comp]`` (held by every teacher worker for
on-device projection of hidden states before the Ray RPC boundary).

Loading ``lm_head`` directly from safetensors avoids constructing a full HF
model on the driver just to read a single matrix.

Why this bypasses :class:`~verl.workers.rollout.replica.RolloutReplica` (the
shared rollout/teacher abstraction used by the topk teacher in
:mod:`~verl.experimental.teacher_loop.teacher_model`):

1. vLLM's HTTP server (``/v1/embeddings``) returns one vector per *sequence*,
   not per *token*. The ``runner="pooling"`` + ``convert="embed"`` +
   ``PoolingParams(task="token_embed")`` (ALL pool) path is only reachable
   via :meth:`AsyncLLM.encode` directly, so the standard ``vLLMHttpServer``
   would need a custom RPC method to expose it.
2. The ``P_down @ h`` projection from ``[S, D]`` to ``[S, d_comp]`` must run
   on the actor *before* the network boundary -- otherwise the wire payload
   is ``S * D``, defeating the whole nitrobrew communication-budget claim.
3. Frozen teachers don't need the ``CheckpointEngineWorker`` weight-sync
   stack that ``RolloutReplica.init_colocated`` spawns under the hood.

TODO(nitrobrew): factor into a ``NitrobrewReplica(RolloutReplica)`` once
vLLM exposes per-token embeddings via the HTTP server, so this module can
collapse to a ``launch_servers`` override + a tiny ``NitrobrewServer`` actor.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

import ray
import torch

logger = logging.getLogger(__name__)


def _load_lm_head_weight(model_path: str) -> torch.Tensor:
    """Read the teacher's lm_head (or tied embed_tokens) weight from safetensors.

    Returns a CPU fp32 tensor of shape ``[V, D]``. Avoids constructing a full
    ``AutoModelForCausalLM`` on the driver.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from verl.utils.fs import copy_to_local

    # copy_to_local resolves HDFS paths and returns local paths unchanged. It
    # does not fetch HF Hub IDs (those still need snapshot_download below).
    resolved = copy_to_local(model_path)
    folder = (
        resolved
        if os.path.isdir(resolved)
        else snapshot_download(resolved, allow_patterns=["*.safetensors*", "*.json"])
    )

    cfg = AutoConfig.from_pretrained(folder, trust_remote_code=True)
    target_keys = (
        ["model.embed_tokens.weight", "embed_tokens.weight"] if cfg.tie_word_embeddings else ["lm_head.weight"]
    )

    index_path = os.path.join(folder, "model.safetensors.index.json")
    shard_for_key: dict[str, str] = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            shard_for_key = json.load(f)["weight_map"]

    for key in target_keys:
        if shard_for_key:
            shard = shard_for_key.get(key)
            if shard is None:
                continue
            tensors = load_file(os.path.join(folder, shard))
        else:
            tensors = load_file(os.path.join(folder, "model.safetensors"))
        if key in tensors:
            return tensors[key].float()

    raise ValueError(f"Could not find lm_head/embed_tokens weight in {model_path}")


def _compute_svd(w_t: torch.Tensor, d_comp: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated SVD of the teacher's unembed: ``w_t ~= w_up @ p_down.T``.

    Returns ``(w_up [V, d_comp], p_down [D, d_comp])`` cast to ``dtype``.
    """
    u, sigma, vh = torch.linalg.svd(w_t, full_matrices=False)
    u_r = u[:, :d_comp]
    sigma_r = sigma[:d_comp]
    vh_r = vh[:d_comp, :]
    w_up = (u_r * sigma_r.unsqueeze(0)).to(dtype)
    p_down = vh_r.T.to(dtype)
    return w_up, p_down


@ray.remote
class NitrobrewTeacherWorker:
    """vLLM-backed teacher serving PCA-compressed hidden states.

    Wraps vLLM's :class:`AsyncLLM` in pooling+embed mode with per-token output
    (``PoolingType.ALL``). Hidden states are projected on the actor's GPU to
    ``d_comp`` *before* crossing the Ray RPC boundary, so the network payload
    matches nitrobrew's communication budget regardless of teacher ``D``.
    """

    async def setup(
        self,
        model_path: str,
        d_comp: int,
        p_down_list: list,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_model_len: int,
        enforce_eager: bool,
    ) -> None:
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        torch_dtype = getattr(torch, dtype)
        self._d_comp = d_comp
        self._dtype = torch_dtype
        self._p_down_cpu = torch.tensor(p_down_list, dtype=torch_dtype)  # [D, d_comp]
        self._p_down_dev: dict[torch.device, torch.Tensor] = {}

        logger.warning(
            "NitrobrewTeacherWorker: launching AsyncLLM (model=%s, tp=%d, max_num_seqs=%d, "
            "max_model_len=%d, gmu=%.2f, enforce_eager=%s, dtype=%s)",
            model_path,
            tensor_parallel_size,
            max_num_seqs,
            max_model_len,
            gpu_memory_utilization,
            enforce_eager,
            dtype,
        )

        # ALL pooling cannot resume across steps, so prefill must complete in a
        # single scheduler step. Cap per-step token budget at 4x max_model_len:
        # enough to batch a few full-length sequences per wave without
        # exhausting KV-cache headroom during vLLM's memory profile pass.
        max_num_batched_tokens = min(max_num_seqs, 4) * max_model_len
        engine_args = AsyncEngineArgs(
            model=model_path,
            runner="pooling",
            convert="embed",
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=enforce_eager,
            # ALL pooling rejects partial prefills; chunked prefill and prefix
            # caching can both produce them, so disable both.
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            disable_log_stats=True,
        )
        self._engine = AsyncLLM.from_engine_args(engine_args)
        logger.warning("NitrobrewTeacherWorker: AsyncLLM ready")

    def _p_down_on(self, device: torch.device) -> torch.Tensor:
        cached = self._p_down_dev.get(device)
        if cached is None:
            cached = self._p_down_cpu.to(device, non_blocking=True)
            self._p_down_dev[device] = cached
        return cached

    async def compute_hidden_states(self, sequence_ids: list[int]) -> list:
        """Encode one sequence; return PCA-compressed hidden states ``[S, d_comp]``."""
        from vllm import PoolingParams
        from vllm.inputs import TokensPrompt

        params = PoolingParams(task="token_embed", normalize=False)
        prompt = TokensPrompt(prompt_token_ids=sequence_ids)

        final = None
        async for out in self._engine.encode(
            prompt=prompt,
            pooling_params=params,
            request_id=uuid.uuid4().hex,
        ):
            final = out
        assert final is not None, "AsyncLLM.encode produced no output"

        h = final.outputs.data  # [S, D]
        p = self._p_down_on(h.device)
        z = torch.mm(h.to(self._dtype), p)  # [S, d_comp]
        return z.cpu().tolist()


class NitrobrewAsyncTeacherManager:
    """Async client that round-robins hidden-state requests across workers.

    Mirrors the interface of ``AsyncTeacherLLMServerManager`` but returns
    teacher hidden states instead of ``(teacher_ids, teacher_logprobs)``.
    """

    def __init__(self, worker_handles: dict[str, list[Any]]):
        self._worker_handles = worker_handles
        self._counters = {key: 0 for key in worker_handles}
        self._lock = asyncio.Lock()

    def _resolve_key(self, routing_key: Optional[str]) -> str:
        if len(self._worker_handles) == 1:
            return next(iter(self._worker_handles))
        if routing_key is None or routing_key not in self._worker_handles:
            raise ValueError(
                f"Routing key {routing_key!r} not found in nitrobrew workers: {sorted(self._worker_handles)}"
            )
        return routing_key

    async def compute_teacher_hidden_states_single(
        self,
        sequence_ids: list[int],
        routing_key: Optional[str] = None,
    ) -> torch.Tensor:
        key = self._resolve_key(routing_key)
        handles = self._worker_handles[key]
        async with self._lock:
            idx = self._counters[key] % len(handles)
            self._counters[key] += 1

        result: list = await handles[idx].compute_hidden_states.remote(sequence_ids)
        return torch.tensor(result, dtype=torch.bfloat16)  # [S, d_comp]


class NitrobrewTeacherModelManager:
    """Manage a pool of vLLM-backed :class:`NitrobrewTeacherWorker` Ray actors.

    Exposes:
      - ``worker_handles: dict[teacher_key, list[ActorHandle]]`` for
        :class:`AgentLoopWorker`.
      - ``w_up: torch.Tensor [V, d_comp]`` for actor-side
        ``set_teacher_unembed``.

    ``nitrobrew_d_comp`` and the inference settings (TP / GMU /
    max_num_seqs / max_model_len / enforce_eager / dtype) are read from each
    teacher's :class:`DistillationTeacherModelConfig`.
    """

    def __init__(
        self,
        teacher_model_configs: dict,
        gpus_per_replica: int,
    ):
        self.worker_handles: dict[str, list] = {}
        self.w_up: Optional[torch.Tensor] = None

        for key, teacher_cfg in teacher_model_configs.items():
            d_comp = teacher_cfg.nitrobrew_d_comp
            if d_comp is None:
                raise ValueError(
                    f"teacher_models['{key}'].nitrobrew_d_comp must be set when using the nitrobrew loss mode."
                )

            num_replicas = teacher_cfg.num_replicas
            model_path = teacher_cfg.model_path
            inference = teacher_cfg.inference
            dtype = inference.dtype
            tp = inference.tensor_model_parallel_size
            gmu = inference.gpu_memory_utilization
            max_num_seqs = inference.max_num_seqs
            enforce_eager = inference.enforce_eager
            # validate_and_prepare_for_distillation rewrites prompt_length to
            # (prompt + response) and response_length to 1, so their sum is the
            # required teacher context.
            max_model_len = inference.max_model_len or (inference.prompt_length + inference.response_length)

            torch_dtype = getattr(torch, dtype)

            logger.warning(
                "NitrobrewTeacherModelManager: loading lm_head for '%s' (%s) and computing SVD (d_comp=%d)",
                key,
                model_path,
                d_comp,
            )
            w_t = _load_lm_head_weight(model_path)
            w_up, p_down = _compute_svd(w_t, d_comp, torch_dtype)
            del w_t
            logger.warning(
                "NitrobrewTeacherModelManager: SVD done w_up %s, p_down %s",
                tuple(w_up.shape),
                tuple(p_down.shape),
            )

            handles = []
            for _ in range(num_replicas):
                worker = NitrobrewTeacherWorker.options(
                    num_gpus=gpus_per_replica,
                    max_concurrency=max(max_num_seqs * 2, 256),
                ).remote()
                handles.append(worker)
            self.worker_handles[key] = handles

            p_down_list = p_down.cpu().tolist()
            ray.get(
                [
                    h.setup.remote(
                        model_path,
                        d_comp,
                        p_down_list,
                        dtype,
                        tp,
                        gmu,
                        max_num_seqs,
                        max_model_len,
                        enforce_eager,
                    )
                    for h in handles
                ]
            )

            if self.w_up is None:
                self.w_up = w_up

        assert self.w_up is not None, "No teacher models configured."
