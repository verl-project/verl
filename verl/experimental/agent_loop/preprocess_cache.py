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
"""Share the media preprocessing of a prompt across its rollout sessions.

The ``n`` sessions of a GRPO prompt decode the same images/videos and run the processor over the same
prompt. A ``PromptPreprocessCache`` shared by those sessions computes each step once.
"""

import asyncio
from typing import Any, Awaitable, Callable

_PRIMITIVES = (str, bytes, int, float, bool, type(None))
_MISSING = object()


def _snapshot(inputs: Any) -> Any:
    """Copy the containers of ``inputs`` so that callers appending to them later do not change the cache key."""
    if isinstance(inputs, list):
        return [_snapshot(item) for item in inputs]
    if isinstance(inputs, tuple):
        return tuple(_snapshot(item) for item in inputs)
    if isinstance(inputs, dict):
        return {key: _snapshot(value) for key, value in inputs.items()}
    return inputs


def _same_inputs(a: Any, b: Any) -> bool:
    """Containers and primitives compare by value; anything else (images, tensors, arrays) by identity."""
    if a is b:
        return True
    if isinstance(a, list | tuple):
        return type(a) is type(b) and len(a) == len(b) and all(_same_inputs(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same_inputs(v, b[k]) for k, v in a.items())
    if isinstance(a, _PRIMITIVES):
        return type(a) is type(b) and a == b
    return False


class PromptPreprocessCache:
    """Memoizes the preprocessing steps of one prompt so that its rollout sessions compute each once.

    A lookup only hits an entry computed from the same inputs, so a session whose inputs diverge, e.g. a
    multi-turn session that received tool images, computes its own. Concurrent lookups of the same inputs
    wait for the first one. Cached values are shared: callers must copy containers before mutating them.
    """

    def __init__(self):
        self._entries: dict[str, list[tuple[Any, asyncio.Future]]] = {}

    async def get_or_compute(self, step: str, inputs: Any, compute: Callable[[], Awaitable[Any]]) -> Any:
        entries = self._entries.setdefault(step, [])
        for cached_inputs, future in entries:
            if _same_inputs(cached_inputs, inputs):
                value = await asyncio.shield(future)
                if value is _MISSING:
                    return await self.get_or_compute(step, inputs, compute)
                return value

        entry = (_snapshot(inputs), asyncio.get_running_loop().create_future())
        entries.append(entry)
        try:
            value = await compute()
        except BaseException:
            # Waiters retry on their own rather than inheriting this caller's failure or cancellation.
            entries[:] = [cached for cached in entries if cached is not entry]
            entry[1].set_result(_MISSING)
            raise
        entry[1].set_result(value)
        return value
