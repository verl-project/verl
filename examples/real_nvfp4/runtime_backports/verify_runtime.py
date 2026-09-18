"""Verify installed bytes and the source-validated refit lifecycle contract."""

import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path

from patch_megatron_fa4 import verify_patched_source
from transformer_engine.pytorch.cpp_extensions import gemm
from transformer_engine.pytorch.module import grouped_linear

from verl.utils.real_nvfp4.vllm_runtime import require_vllm_nvfp4_backports


def main():
    attention_source = distribution("megatron-core").locate_file("megatron/core/transformer/attention.py")
    verify_patched_source(Path(attention_source).read_text())
    checkpoint_source = distribution("megatron-core").locate_file("megatron/core/extensions/transformer_engine.py")
    assert (
        hashlib.sha256(Path(checkpoint_source).read_bytes()).hexdigest()
        == "589453c2a7de977531b292c26fd9c7f934333817229fa9d545b04feafc999f43"
    ), "Core 0.19 release checkpoint source changed"
    expected_versions = {
        "transformer-engine": "2.18.0",
        "transformer-engine-cu13": "2.18.0",
        "transformer-engine-torch": "2.18.0",
        "vllm": "0.27.1",
        "megatron-core": "0.19.0",
        "torch": "2.13.0",
    }
    actual_versions = {name: version(name) for name in expected_versions}
    if actual_versions != expected_versions:
        raise RuntimeError(f"Unexpected dependency versions: {actual_versions}")
    targets = {
        Path(gemm.__file__): "dc3233868f739d67d86a8b6dde9dbfa519a195ae08c40a751c85b08fbe9a931b",
        Path(
            distribution("vllm").locate_file("vllm/v1/executor/multiproc_executor.py")
        ): "e84685635d48e9e6ce1c1022b982d49baca3f9cd2273530c9502df5f2e0147b4",
    }
    hashes = {}
    for path, expected in targets.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unexpected installed file: {path}: {actual}")
        hashes[str(path)] = actual
    if gemm.NVFP4_ROWSCALE_GROUPED_BACKPORT != "20260915-v1":
        raise RuntimeError("Missing grouped rowscale marker")
    if grouped_linear.general_grouped_gemm is not gemm.general_grouped_gemm:
        raise RuntimeError("GroupedLinear did not import the patched implementation")
    require_vllm_nvfp4_backports()
    print(
        "DELIVERY_RUNTIME_GUARD_PASS",
        json.dumps(
            {
                "versions": actual_versions,
                "installed_sha256": hashes,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
