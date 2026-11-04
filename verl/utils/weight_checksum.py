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
"""vLLM API compatibility and rank-qualified digest comparison, without GPU imports."""

from typing import Iterable


class WeightCheckEngineAdapter:
    """Bridge older single-DP vLLM clients and the newer all-DP APIs."""

    def __init__(self, engine, data_parallel_size: int):
        self.engine = engine
        self.data_parallel_size = data_parallel_size

    def validate(self):
        if self.data_parallel_size > 1:
            for name in ("compute_weight_checksums_all", "reset_weights"):
                if not callable(getattr(self.engine, name, None)):
                    raise RuntimeError(f"DP>1 weight checking requires vLLM AsyncLLM.{name}()")
        elif not callable(getattr(self.engine, "collective_rpc", None)):
            raise RuntimeError("weight checking requires vLLM AsyncLLM.collective_rpc()")

    async def checksums(self) -> dict[str, str]:
        self.validate()
        method = getattr(self.engine, "compute_weight_checksums_all", None)
        if not callable(method):
            method = getattr(self.engine, "compute_weight_checksums", None)
        if callable(method):
            result = await method()
        else:
            result = await self.engine.collective_rpc("compute_weight_checksums")
        return merge_worker_checksums([result] if isinstance(result, dict) else result)

    async def reset(self):
        self.validate()
        reset = getattr(self.engine, "reset_weights", None)
        if callable(reset):
            await reset()
        else:
            await self.engine.collective_rpc("reset_weights")


def merge_worker_checksums(per_worker: Iterable[dict[str, str]]) -> dict[str, str]:
    """Merge rank-qualified digests; reject duplicate keys instead of hiding a worker."""
    merged: dict[str, str] = {}
    for worker_checksums in per_worker:
        duplicate_keys = merged.keys() & worker_checksums.keys()
        if duplicate_keys:
            duplicates = ", ".join(sorted(duplicate_keys)[:5])
            raise RuntimeError(f"Duplicate weight checksum keys: {duplicates}")
        merged.update(worker_checksums)
    return merged


def compare_checksums(baseline: dict[str, str], current: dict[str, str]) -> tuple[bool, list[str]]:
    """Return equality and sorted mismatches, including keys missing on either side."""
    mismatches = sorted(key for key in baseline.keys() | current.keys() if baseline.get(key) != current.get(key))
    return not mismatches, mismatches
