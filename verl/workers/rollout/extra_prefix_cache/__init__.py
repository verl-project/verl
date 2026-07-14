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

from .config import ExtraPrefixCacheConfig, enabled, normalize_config
from .controller import ExtraPrefixCacheController, compute_store_token_limit
from .epoch import maybe_advance_extra_prefix_cache_epoch, resolve_runtime_model_cache_epoch
from .protocol import PrefixMetadata, PreparedRequest, build_cache_salt, build_request_id, parse_request_id

__all__ = [
    "ExtraPrefixCacheConfig",
    "ExtraPrefixCacheController",
    "PrefixMetadata",
    "PreparedRequest",
    "build_cache_salt",
    "build_request_id",
    "compute_store_token_limit",
    "enabled",
    "maybe_advance_extra_prefix_cache_epoch",
    "normalize_config",
    "parse_request_id",
    "resolve_runtime_model_cache_epoch",
]
