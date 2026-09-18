"""Hash-guarded runtime backport; run only inside a disposable build container."""

import ast
import hashlib
import json
from importlib.metadata import distribution

BEFORE = "ceb3477473bdb1de36e01687e0b2083ad239d42b1ddc2ec8cec16ba2c477efa6"
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
    assert hashlib.sha256(source.encode()).hexdigest() == BEFORE
    assert source.count(OLD) == 1
    assert "import os\n" in source
    assert source.count("\nclass MultiprocExecutor(Executor):") == 1
    patched = source.replace(OLD, NEW).replace(
        "\nclass MultiprocExecutor(Executor):",
        HELPER + "\n\nclass MultiprocExecutor(Executor):",
    )
    ast.parse(patched)
    return patched


if __name__ == "__main__":
    target = distribution("vllm").locate_file("vllm/v1/executor/multiproc_executor.py")
    result = patched_source(target.read_text())
    target.write_text(result)
    print(
        "ATOMIC_TCPSTORE_PATCH",
        json.dumps(
            {
                "path": str(target),
                "before": BEFORE,
                "after": hashlib.sha256(result.encode()).hexdigest(),
            }
        ),
        flush=True,
    )
