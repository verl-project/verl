# Copyright 2025 Meituan Ltd. and/or its affiliates
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

from dataclasses import dataclass
from functools import reduce
from math import gcd

from omegaconf import OmegaConf, open_dict


@dataclass(frozen=True)
class HybridResPoolLayout:
    trainer_pool: list[int]
    rollout_pool: list[int]
    logical_gpus_per_node: int


def is_hybrid_res_pool_enabled(config) -> bool:
    return bool(OmegaConf.select(config, "async_training.hybrid_res_pool", default=False))


def _parse_partition_chunks(pool_spec: str, *, gpus_per_node: int, field_name: str) -> list[int]:
    chunks = []
    for chunk_text in pool_spec.split("_"):
        if not chunk_text:
            raise ValueError(f"invalid {field_name} partition: empty chunk in {pool_spec!r}")
        try:
            chunk = int(chunk_text)
        except ValueError as exc:
            raise ValueError(f"invalid {field_name} partition: non-integer chunk {chunk_text!r}") from exc
        if chunk <= 0:
            raise ValueError(f"invalid {field_name} partition: chunk {chunk} must be positive")
        if chunk > gpus_per_node:
            raise ValueError(
                f"invalid {field_name} partition: chunk {chunk} exceeds n_gpus_per_node={gpus_per_node}"
            )
        chunks.append(chunk)
    return chunks


def _parse_partition(config) -> tuple[list[int], list[int]]:
    partition = OmegaConf.select(config, "actor_rollout_ref.partition", default=None)
    if not partition:
        raise ValueError("hybrid_res_pool requires actor_rollout_ref.partition")
    if partition.count("-") != 1:
        raise ValueError(f"invalid partition format {partition!r}, expected '<trainer_pool>-<rollout_pool>'")

    trainer_spec, rollout_spec = partition.split("-")
    if not trainer_spec or not rollout_spec:
        raise ValueError(f"invalid partition format {partition!r}, expected non-empty trainer/rollout pools")

    total_nodes = OmegaConf.select(config, "trainer.nnodes", default=None)
    gpus_per_node = OmegaConf.select(config, "trainer.n_gpus_per_node", default=None)
    if total_nodes is None or total_nodes <= 0:
        raise ValueError(f"invalid trainer.nnodes={total_nodes!r}")
    if gpus_per_node is None or gpus_per_node <= 0:
        raise ValueError(f"invalid trainer.n_gpus_per_node={gpus_per_node!r}")

    trainer_chunks = _parse_partition_chunks(trainer_spec, gpus_per_node=gpus_per_node, field_name="trainer")
    rollout_chunks = _parse_partition_chunks(rollout_spec, gpus_per_node=gpus_per_node, field_name="rollout")

    physical_total_gpus = total_nodes * gpus_per_node
    partition_total = sum(trainer_chunks) + sum(rollout_chunks)
    if partition_total != physical_total_gpus:
        raise ValueError(
            f"invalid partition {partition!r}: expected total GPUs {physical_total_gpus}, got {partition_total}"
        )

    return trainer_chunks, rollout_chunks


def _expand_chunks(chunks: list[int], logical_gpus_per_node: int) -> list[int]:
    expanded = []
    for chunk in chunks:
        if chunk % logical_gpus_per_node != 0:
            raise ValueError(
                f"invalid partition chunk {chunk}: not divisible by logical_gpus_per_node={logical_gpus_per_node}"
            )
        expanded.extend([logical_gpus_per_node] * (chunk // logical_gpus_per_node))
    return expanded


def build_hybrid_res_pool_layout(config) -> HybridResPoolLayout:
    if not is_hybrid_res_pool_enabled(config):
        raise ValueError("hybrid_res_pool is disabled")

    rollout_mode = OmegaConf.select(config, "actor_rollout_ref.rollout.mode", default=None)
    if rollout_mode != "async":
        raise ValueError("hybrid_res_pool requires actor_rollout_ref.rollout.mode == 'async'")

    hybrid_engine = OmegaConf.select(config, "actor_rollout_ref.hybrid_engine", default=None)
    if hybrid_engine is not False:
        raise ValueError("hybrid_res_pool requires actor_rollout_ref.hybrid_engine == False")

    trainer_chunks, rollout_chunks = _parse_partition(config)
    logical_gpus_per_node = reduce(gcd, trainer_chunks + rollout_chunks)
    trainer_pool = _expand_chunks(trainer_chunks, logical_gpus_per_node)
    rollout_pool = _expand_chunks(rollout_chunks, logical_gpus_per_node)

    return HybridResPoolLayout(
        trainer_pool=trainer_pool,
        rollout_pool=rollout_pool,
        logical_gpus_per_node=logical_gpus_per_node,
    )


def normalize_hybrid_res_pool_config(config):
    if not is_hybrid_res_pool_enabled(config):
        return None

    layout = build_hybrid_res_pool_layout(config)
    with open_dict(config):
        config.trainer.nnodes = len(layout.trainer_pool)
        config.trainer.n_gpus_per_node = layout.logical_gpus_per_node
        config.rollout.nnodes = len(layout.rollout_pool)
        config.rollout.n_gpus_per_node = layout.logical_gpus_per_node
    return layout
