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

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import ExtraPrefixCacheConfig, sanitize_salt

REQUEST_ID_PREFIX = "vepc"
_REQUEST_ID_RE = re.compile(r"^vepc__r(?P<read>[01])__w(?P<write>[01])__l(?P<limit>\d+)__")


@dataclass(frozen=True)
class PrefixMetadata:
    stable_prefix_token_len: int
    stable_prefix_fingerprint: str
    prefix_source: str = "explicit"
    tokenizer_fingerprint: str | None = None
    template_fingerprint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedRequest:
    enabled: bool
    cache_salt: str | None = None
    backend_request_id: str | None = None
    stable_prefix_token_len: int = 0
    store_token_limit: int = 0
    warmup_prompt_ids: list[int] | None = None
    warmup_backend_request_id: str | None = None


@dataclass(frozen=True)
class RequestPolicy:
    read: bool
    write: bool
    store_token_limit: int = 0
    tagged: bool = True


def short_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def build_cache_salt(config: ExtraPrefixCacheConfig, metadata: PrefixMetadata, *, model_path: str | None = None) -> str:
    model_namespace = config.model_namespace or model_path or "model"
    model_hash = short_hash(model_namespace, length=10)
    prefix_fingerprint = sanitize_salt(metadata.stable_prefix_fingerprint)
    return sanitize_salt(f"{config.namespace}:{model_hash}:{config.epoch}:pfx{prefix_fingerprint}")


def build_request_id(
    *,
    read: bool,
    write: bool,
    store_token_limit: int = 0,
    base_request_id: str | None = None,
) -> str:
    base = base_request_id or uuid.uuid4().hex
    read_flag = 1 if read else 0
    write_flag = 1 if write else 0
    limit = max(int(store_token_limit or 0), 0)
    return f"{REQUEST_ID_PREFIX}__r{read_flag}__w{write_flag}__l{limit}__{base}"


def parse_request_id(request_id: str, *, allow_untagged: bool = True) -> RequestPolicy:
    match = _REQUEST_ID_RE.match(request_id or "")
    if match is None:
        return RequestPolicy(
            read=allow_untagged,
            write=allow_untagged,
            store_token_limit=0,
            tagged=False,
        )
    return RequestPolicy(
        read=match.group("read") == "1",
        write=match.group("write") == "1",
        store_token_limit=int(match.group("limit")),
        tagged=True,
    )
