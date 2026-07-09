# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import logging
from collections import Counter
from typing import Any

from .config import ExtraPrefixCacheConfig
from .epoch import resolve_runtime_model_cache_epoch
from .prefix_provider import get_prefix_provider
from .protocol import PreparedRequest, build_cache_salt, build_request_id

logger = logging.getLogger(__name__)


class ExtraPrefixCacheMetrics:
    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()

    def incr(self, name: str, value: int = 1) -> None:
        self._counters[name] += value

    def snapshot(self) -> dict[str, int]:
        return dict(self._counters)


def compute_store_token_limit(stable_prefix_token_len: int, chunk_size: int = 0) -> int:
    prefix_len = max(int(stable_prefix_token_len or 0), 0)
    chunk_size = max(int(chunk_size or 0), 0)
    if prefix_len <= 0:
        return 0
    if chunk_size <= 0:
        return prefix_len
    return (prefix_len // chunk_size) * chunk_size


def rollout_read_enabled(read_policy: str) -> bool:
    return str(read_policy or "rollout").lower() != "off"


def warmup_write_enabled(write_policy: str) -> bool:
    return str(write_policy or "warmup_only").lower() != "off"


class ExtraPrefixCacheController:
    def __init__(self, config: Any, *, model_path: str | None = None, log: logging.Logger | None = None) -> None:
        self.config = ExtraPrefixCacheConfig.from_any(config)
        self.model_path = model_path
        self.logger = log or logger
        self.metrics = ExtraPrefixCacheMetrics()
        self._warmed_cache_salts: set[str] = set()
        self._prefix_provider = get_prefix_provider(self.config)

    @property
    def enabled(self) -> bool:
        return self.config.enable

    async def prepare(
        self,
        *,
        prompt_ids: list[int],
        metadata: Any,
    ) -> PreparedRequest:
        if not self.config.enable:
            return PreparedRequest(enabled=False)

        runtime_epoch = await resolve_runtime_model_cache_epoch(self.config.__dict__, log=self.logger)
        if runtime_epoch != self.config.epoch:
            self.config = ExtraPrefixCacheConfig.from_any(
                {**self.config.__dict__, "runtime_model_cache_epoch": runtime_epoch}
            )

        prefix = self._prefix_provider.resolve(prompt_ids, metadata)
        if prefix is None:
            self.metrics.incr("invalid_prefix_metadata")
            return PreparedRequest(enabled=False)

        store_token_limit = min(
            compute_store_token_limit(prefix.stable_prefix_token_len, self.config.chunk_size),
            len(prompt_ids),
        )
        if store_token_limit <= 0:
            self.metrics.incr("invalid_prefix_metadata")
            return PreparedRequest(enabled=False)

        cache_salt = build_cache_salt(self.config, prefix, model_path=self.model_path)
        read = rollout_read_enabled(self.config.read_policy)
        backend_request_id = build_request_id(read=read, write=False)

        warmup_prompt_ids = None
        warmup_backend_request_id = None
        if (
            self.config.warmup
            and warmup_write_enabled(self.config.write_policy)
            and cache_salt not in self._warmed_cache_salts
        ):
            warmup_prompt_ids = list(prompt_ids[:store_token_limit])
            warmup_backend_request_id = build_request_id(
                read=False,
                write=True,
                store_token_limit=store_token_limit,
            )
            self._warmed_cache_salts.add(cache_salt)
            self.metrics.incr("warmup_requests")
        else:
            self.metrics.incr("warmup_skipped")

        self.metrics.incr("rollout_requests")
        return PreparedRequest(
            enabled=True,
            cache_salt=cache_salt,
            backend_request_id=backend_request_id,
            stable_prefix_token_len=prefix.stable_prefix_token_len,
            store_token_limit=store_token_limit,
            warmup_prompt_ids=warmup_prompt_ids,
            warmup_backend_request_id=warmup_backend_request_id,
        )
