# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Generic adaptation between transported batches and engine TensorDicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from verl.protocol import BatchData


@dataclass(frozen=True)
class EngineBatchSpec:
    """Fields required by one engine entry point."""

    required_keys: tuple[str, ...] = ()
    optional_keys: tuple[str, ...] = ()
    restore_padding_keys: tuple[str, ...] = ()


@dataclass
class EngineBatchContext:
    """Adapted engine payload and its output restoration callback."""

    payload: Any
    finalize: Callable[[Any], Any]


def run_engine_batch(data: Any, fn: Callable[[Any], Any], spec: EngineBatchSpec) -> Any:
    """Run an engine call through a batch-provided adapter when available."""
    prepare = getattr(data, "prepare_engine_batch", None)
    if prepare is None:
        return fn(data)
    context: EngineBatchContext = prepare(spec)
    return context.finalize(fn(context.payload))


def set_batch_control_fields(data: Any, **kwargs) -> Any:
    return BatchData(data).set_control_fields(**kwargs)


def batch_to_cpu(data: Any) -> Any:
    return BatchData(data).cpu()
