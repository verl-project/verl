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


def _parse_partition_chunks(pool_spec: str, *, field_name: str) -> list[int]:
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
        chunks.append(chunk)
    return chunks


def _get_positive_int(config, field_path: str) -> int:
    value = OmegaConf.select(config, field_path, default=None)
    if value is None or value <= 0:
        raise ValueError(f"invalid {field_path}={value!r}")
    return value


def _validate_partition_chunks(chunks: list[int], *, gpus_per_node: int, field_name: str) -> None:
    for chunk in chunks:
        if chunk > gpus_per_node:
            raise ValueError(
                f"invalid {field_name} partition: chunk {chunk} exceeds physical_gpus_per_node={gpus_per_node}"
            )


def _parse_partition(config) -> tuple[list[int], list[int]]:
    partition = OmegaConf.select(config, "actor_rollout_ref.partition", default=None)
    if not partition:
        raise ValueError("hybrid_res_pool requires actor_rollout_ref.partition")
    if partition.count("-") != 1:
        raise ValueError(f"invalid partition format {partition!r}, expected '<trainer_pool>-<rollout_pool>'")

    trainer_spec, rollout_spec = partition.split("-")
    if not trainer_spec or not rollout_spec:
        raise ValueError(f"invalid partition format {partition!r}, expected non-empty trainer/rollout pools")

    trainer_chunks = _parse_partition_chunks(trainer_spec, field_name="trainer")
    rollout_chunks = _parse_partition_chunks(rollout_spec, field_name="rollout")

    trainer_nodes = _get_positive_int(config, "trainer.nnodes")
    trainer_gpus_per_node = _get_positive_int(config, "trainer.n_gpus_per_node")
    rollout_nodes = _get_positive_int(config, "rollout.nnodes")
    rollout_gpus_per_node = _get_positive_int(config, "rollout.n_gpus_per_node")

    trainer_total = trainer_nodes * trainer_gpus_per_node
    rollout_total = rollout_nodes * rollout_gpus_per_node
    partition_total = sum(trainer_chunks) + sum(rollout_chunks)

    if partition_total == trainer_total:
        if trainer_nodes == rollout_nodes and trainer_gpus_per_node != rollout_gpus_per_node:
            raise ValueError(
                "hybrid_res_pool raw mode is ambiguous when trainer.nnodes == rollout.nnodes "
                "but trainer.n_gpus_per_node != rollout.n_gpus_per_node; "
                "split-style configs must consume trainer_total + rollout_total"
            )
        _validate_partition_chunks(
            trainer_chunks + rollout_chunks,
            gpus_per_node=trainer_gpus_per_node,
            field_name="combined",
        )
        return trainer_chunks, rollout_chunks

    expected_split_total = trainer_total + rollout_total
    if partition_total != expected_split_total:
        raise ValueError(
            f"invalid partition {partition!r}: expected total GPUs {trainer_total} (raw mode) "
            f"or {expected_split_total} (split mode), got {partition_total}"
        )

    if trainer_nodes != rollout_nodes:
        raise ValueError("hybrid_res_pool split mode requires trainer.nnodes == rollout.nnodes")

    if sum(trainer_chunks) != trainer_total:
        raise ValueError(
            "hybrid_res_pool split mode requires trainer partition total "
            f"{sum(trainer_chunks)} to equal trainer_total={trainer_total}"
        )
    if sum(rollout_chunks) != rollout_total:
        raise ValueError(
            "hybrid_res_pool split mode requires rollout partition total "
            f"{sum(rollout_chunks)} to equal rollout_total={rollout_total}"
        )

    physical_gpus_per_node = trainer_gpus_per_node + rollout_gpus_per_node
    _validate_partition_chunks(
        trainer_chunks + rollout_chunks,
        gpus_per_node=physical_gpus_per_node,
        field_name="combined",
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

    checkpoint_backend = OmegaConf.select(
        config,
        "actor_rollout_ref.rollout.checkpoint_engine.backend",
        default="naive",
    )
    if checkpoint_backend != "naive":
        raise ValueError("hybrid_res_pool requires actor_rollout_ref.rollout.checkpoint_engine.backend == 'naive'")

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
