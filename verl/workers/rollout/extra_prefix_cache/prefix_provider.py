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
from typing import Any

from .config import ExtraPrefixCacheConfig
from .protocol import PrefixMetadata, short_hash


class ExplicitPrefixProvider:
    def resolve(self, prompt_ids: list[int], metadata: Any) -> PrefixMetadata | None:
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            return None

        stable_len = metadata.get("stable_prefix_token_len", metadata.get("system_prefix_len"))
        fingerprint = metadata.get("stable_prefix_fingerprint", metadata.get("system_prefix_fingerprint"))
        try:
            stable_len = int(stable_len)
        except (TypeError, ValueError):
            return None
        if stable_len <= 0 or not fingerprint:
            return None
        if stable_len > len(prompt_ids):
            return None

        return PrefixMetadata(
            stable_prefix_token_len=stable_len,
            stable_prefix_fingerprint=str(fingerprint),
            prefix_source=str(metadata.get("prefix_source", "explicit")),
            tokenizer_fingerprint=(
                str(metadata["tokenizer_fingerprint"]) if metadata.get("tokenizer_fingerprint") is not None else None
            ),
            template_fingerprint=(
                str(metadata["template_fingerprint"]) if metadata.get("template_fingerprint") is not None else None
            ),
            extra=dict(metadata),
        )


class HeuristicPrefixProvider:
    def __init__(self, config: ExtraPrefixCacheConfig) -> None:
        self.config = config

    def resolve(self, prompt_ids: list[int], metadata: Any) -> PrefixMetadata | None:
        if not isinstance(metadata, dict):
            return None
        stable_len = metadata.get("stable_prefix_token_len")
        try:
            stable_len = int(stable_len)
        except (TypeError, ValueError):
            return None
        min_len = self.config.heuristic_min_prefix_len
        if stable_len <= 0 or stable_len > len(prompt_ids) or (min_len and stable_len < min_len):
            return None
        fingerprint = short_hash(
            {
                "tokens": hashlib.sha256(bytes(_token_bytes(prompt_ids[:stable_len]))).hexdigest(),
                "tokenizer": self.config.tokenizer_fingerprint,
                "template": self.config.template_fingerprint,
            }
        )
        return PrefixMetadata(
            stable_prefix_token_len=stable_len,
            stable_prefix_fingerprint=fingerprint,
            prefix_source="heuristic",
            tokenizer_fingerprint=self.config.tokenizer_fingerprint,
            template_fingerprint=self.config.template_fingerprint,
            extra=dict(metadata),
        )


def get_prefix_provider(config: ExtraPrefixCacheConfig):
    provider = config.prefix_provider.lower()
    if provider == "heuristic":
        return HeuristicPrefixProvider(config)
    return ExplicitPrefixProvider()


def _token_bytes(tokens: list[int]):
    for token in tokens:
        value = max(int(token), 0)
        yield from value.to_bytes(8, "little", signed=False)
