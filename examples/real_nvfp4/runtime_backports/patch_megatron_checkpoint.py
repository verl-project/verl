# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

"""Backport Megatron-LM #5997 stateless grouped checkpoint handling.

TE 2.18 legitimately returns an empty byte tensor for stateless recipes.
Pinned Core 0.18 decodes it to None and must not index it as an FP8 dict.
Apply only in a disposable runtime build, never in a live environment.
"""

import argparse
import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

BASE_SHA256 = "a463488823796d184746bfce4e9f4a4f41620cc04f5a17de6f9a2e22d6ab61bf"
OLD = """            state = self._decode_extra_state(state)
            extra_states = []
"""
NEW = """            state = self._decode_extra_state(state)
            if state is None:
                return [torch.empty(0, dtype=torch.uint8)] * self.num_gemms
            extra_states = []
"""


def verify_patched_source(source):
    if source.count(NEW) != 1:
        raise RuntimeError("Missing or ambiguous stateless grouped checkpoint guard")
    restored = source.replace(NEW, OLD)
    if hashlib.sha256(restored.encode()).hexdigest() != BASE_SHA256:
        raise RuntimeError("Megatron TE extension differs beyond the approved checkpoint backport")


def patched_source(source):
    if NEW in source:
        verify_patched_source(source)
        return source
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != BASE_SHA256 or source.count(OLD) != 1:
        raise RuntimeError(f"Unsupported Megatron TE extension source: {actual}")
    return source.replace(OLD, NEW)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    target = Path(distribution("megatron-core").locate_file("megatron/core/extensions/transformer_engine.py"))
    original = target.read_text()
    if args.verify_only:
        verify_patched_source(original)
        result = original
    else:
        result = patched_source(original)
        if result != original:
            target.write_text(result)
    print(
        "MEGATRON_STATELESS_CHECKPOINT_GUARD_PASS",
        json.dumps(
            {
                "path": str(target),
                "base_sha256": BASE_SHA256,
                "installed_sha256": hashlib.sha256(result.encode()).hexdigest(),
                "upstream_commit": "7e6a045a2a99659ddd2919fc19013a164c508825",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
