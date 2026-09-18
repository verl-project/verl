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

"""Hash-guarded backport of Megatron-LM #6964 to released Core 0.19.

Upstream: 53abe744b5e8d043a06e54cfaef44cbe23b1e2ca.
FA2 also ships flash_attn.cute; that namespace alone does not establish FA4
availability. Check distribution metadata before importing the optional module.
Never mutate a live environment. Apply only in the disposable image build.
"""

import argparse
import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

BASE_SHA256 = "a019cd00a708a22237383456a7ba85bf522201c14a5b7cb0c1e4f0f12eac9837"
OLD = """try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _get_dist_version

    from flash_attn.cute import flash_attn_varlen_func as flash_attn4_varlen_func
    from packaging.version import Version as _Version

    try:
        HAVE_FA4 = _Version(_get_dist_version("flash-attn-4")) >= _Version(_MIN_FA4_VERSION)
    except PackageNotFoundError:
        HAVE_FA4 = False
except ImportError:
    HAVE_FA4 = False
"""
NEW = """# Backport Megatron-LM #6964: FA2's cute namespace is not an FA4 install.
flash_attn4_varlen_func = None
try:
    from importlib.metadata import version as _fa4_distribution_version
    from packaging.version import Version as _FA4Version

    HAVE_FA4 = _FA4Version(_fa4_distribution_version("flash-attn-4")) >= _FA4Version("4.0.0b20")
    if HAVE_FA4:
        from flash_attn.cute import flash_attn_varlen_func as flash_attn4_varlen_func
except ImportError:
    # PackageNotFoundError is an ImportError subclass. Do not swallow arbitrary
    # failures of an actually installed, supported FA4 implementation.
    HAVE_FA4 = False
"""


def verify_patched_source(source):
    if source.count(NEW) != 1:
        raise RuntimeError("Missing or ambiguous FA4 distribution guard")
    restored = source.replace(NEW, OLD)
    if hashlib.sha256(restored.encode()).hexdigest() != BASE_SHA256:
        raise RuntimeError("Megatron attention differs beyond the approved FA4 backport")


def patched_source(source):
    if NEW in source:
        verify_patched_source(source)
        return source
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != BASE_SHA256 or source.count(OLD) != 1:
        raise RuntimeError(f"Unsupported Megatron attention source: {actual}")
    return source.replace(OLD, NEW)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    target = Path(distribution("megatron-core").locate_file("megatron/core/transformer/attention.py"))
    original = target.read_text()
    if args.verify_only:
        verify_patched_source(original)
        result = original
    else:
        result = patched_source(original)
        if result != original:
            target.write_text(result)
    print(
        "MEGATRON_FA4_DISTRIBUTION_GUARD_PASS",
        json.dumps(
            {
                "path": str(target),
                "base_sha256": BASE_SHA256,
                "installed_sha256": hashlib.sha256(result.encode()).hexdigest(),
                "upstream_commit": "53abe744b5e8d043a06e54cfaef44cbe23b1e2ca",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
