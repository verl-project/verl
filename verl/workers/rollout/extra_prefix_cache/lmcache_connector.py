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
from typing import Any

from .config import normalize_config
from .protocol import parse_request_id

logger = logging.getLogger(__name__)

try:
    from lmcache.integration.vllm.lmcache_mp_connector import (
        LMCacheMPConnector,
        LMCacheMPConnectorMetadata,
        LMCacheMPRequestMetadata,
    )
    from lmcache.utils import init_logger as lmcache_init_logger

    logger = lmcache_init_logger(__name__)
except Exception:  # pragma: no cover - exercised only when LMCache is installed.
    LMCacheMPConnector = object
    LMCacheMPConnectorMetadata = object
    LMCacheMPRequestMetadata = None


class VerlLMCacheExtraPrefixConnector(LMCacheMPConnector):
    """LMCache connector adapter for verl extra prefix cache.

    vLLM still carries namespace isolation through ``cache_salt``. The adapter
    reads per-request EPC policy from the existing vLLM request id so vLLM and
    LMCache do not need source changes.
    """

    def _allow_untagged(self) -> bool:
        extra_config = getattr(self, "_kv_connector_extra_config", None)
        cfg = normalize_config(extra_config)
        return bool(cfg.get("extra_prefix_cache.allow_untagged", cfg.get("allow_untagged", True)))

    def _policy_from_request_id(self, request_id: str):
        return parse_request_id(request_id, allow_untagged=self._allow_untagged())

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int | None, bool]:
        policy = self._policy_from_request_id(request.request_id)
        if not policy.read:
            tracker = self._get_or_create_request_tracker(request)
            logger.info(
                "ExtraPrefixCache read deny request_id=%s cache_salt=%s",
                request.request_id,
                tracker.cache_salt,
            )
            return 0, False

        ret, will_load = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if ret is not None and policy.tagged:
            tracker = self._get_or_create_request_tracker(request)
            logger.info(
                "ExtraPrefixCache read allow request_id=%s cache_salt=%s new_external_tokens=%s will_load=%s",
                request.request_id,
                tracker.cache_salt,
                ret,
                will_load,
            )
        return ret, will_load

    def _process_new_requests(self, scheduler_output: Any, metadata: LMCacheMPConnectorMetadata) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
        for new_request in scheduler_output.scheduled_new_reqs:
            request_tracker = self._get_request_tracker(new_request.req_id)
            num_new_tokens = scheduler_output.num_scheduled_tokens[new_request.req_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)
            self._maybe_add_store_metadata(request_tracker, lmcache_tokens_per_chunk, metadata)

    def _process_cached_requests(
        self,
        scheduler_output: Any,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._get_request_tracker(request_id)
            new_block_ids = cached_reqs.new_block_ids[idx] or ()
            if request_id not in cached_reqs.resumed_req_ids:
                request_tracker.append_block_ids(new_block_ids)
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)
            self._maybe_add_store_metadata(request_tracker, lmcache_tokens_per_chunk, metadata)

    def _maybe_add_store_metadata(
        self,
        request_tracker: Any,
        lmcache_tokens_per_chunk: int,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        if LMCacheMPRequestMetadata is None:
            return
        r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
            request_tracker,
            lmcache_tokens_per_chunk,
            self._group_tokens_per_block,
        )
        policy = self._policy_from_request_id(request_tracker.request_id)
        if r_meta is None:
            return
        if not policy.write:
            logger.info(
                "ExtraPrefixCache write deny request_id=%s cache_salt=%s range=%d-%d reason=policy",
                request_tracker.request_id,
                request_tracker.cache_salt,
                r_meta.op.start,
                r_meta.op.end,
            )
            return
        limit = policy.store_token_limit
        if limit > 0 and r_meta.op.end > limit:
            logger.info(
                "ExtraPrefixCache write deny request_id=%s cache_salt=%s range=%d-%d limit=%d",
                request_tracker.request_id,
                request_tracker.cache_salt,
                r_meta.op.start,
                r_meta.op.end,
                limit,
            )
            return
        metadata.add_request_metadata(r_meta)
        logger.info(
            "ExtraPrefixCache write allow request_id=%s cache_salt=%s range=%d-%d",
            request_tracker.request_id,
            request_tracker.cache_salt,
            r_meta.op.start,
            r_meta.op.end,
        )
