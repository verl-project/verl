# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Hash-guarded runtime backport; run only inside a disposable build container."""

import ast
import hashlib
import json
from importlib.metadata import distribution

BEFORE = "ceb3477473bdb1de36e01687e0b2083ad239d42b1ddc2ec8cec16ba2c477efa6"
AFTER = "e84685635d48e9e6ce1c1022b982d49baca3f9cd2273530c9502df5f2e0147b4"
OLD = """        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )"""
NEW = """        distributed_init_method = _get_mp_distributed_init_method(self.parallel_config)"""
HELPER = """

VERL_SINGLE_RANK_ATOMIC_TCPSTORE = "20260915-v1"


def _get_mp_distributed_init_method(parallel_config):
    # For a single rank the TCPStore is its own only client. Let the actual
    # listener bind port 0 atomically, instead of closing a probe socket and
    # racing with other engines / outgoing ephemeral connections before bind.
    # Never use this for a multi-rank group: peers need a concrete shared port.
    single_rank = (
        parallel_config.world_size == 1
        and parallel_config.data_parallel_size == 1
        and parallel_config.nnodes == 1
        and not parallel_config.enable_elastic_ep
        and os.environ.get("TORCHELASTIC_USE_AGENT_STORE") != "True"
    )
    return get_distributed_init_method(
        get_loopback_ip(), 0 if single_rank else get_open_port()
    )
"""


def patched_source(source):
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual == AFTER:
        return source
    if actual != BEFORE:
        raise RuntimeError(f"Unexpected vLLM executor source: {actual}")
    assert source.count(OLD) == 1
    assert "import os\n" in source
    assert source.count("\nclass MultiprocExecutor(Executor):") == 1
    patched = source.replace(OLD, NEW).replace(
        "\nclass MultiprocExecutor(Executor):",
        HELPER + "\n\nclass MultiprocExecutor(Executor):",
    )
    ast.parse(patched)
    if hashlib.sha256(patched.encode()).hexdigest() != AFTER:
        raise RuntimeError("Atomic TCPStore patch output differs from audited bytes")
    return patched


if __name__ == "__main__":
    target = distribution("vllm").locate_file("vllm/v1/executor/multiproc_executor.py")
    source = target.read_text()
    result = patched_source(source)
    if result != source:
        target.write_text(result)
    print(
        "ATOMIC_TCPSTORE_PATCH",
        json.dumps(
            {
                "path": str(target),
                "before": hashlib.sha256(source.encode()).hexdigest(),
                "changed": result != source,
                "after": hashlib.sha256(result.encode()).hexdigest(),
            }
        ),
        flush=True,
    )
