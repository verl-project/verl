"""Offline GPU preflight: single-GPU FAISS must match exact CPU Top-3."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from .single_gpu_index import SingleGpuIndex


def main(output):
    vectors = np.random.default_rng(42).normal(size=(101, 16)).astype("float32")
    queries = np.random.default_rng(7).normal(size=(8, 16)).astype("float32")
    cpu = faiss.IndexFlatIP(16)
    cpu.add(vectors)
    expected_scores, expected_ids = cpu.search(queries, 3)
    index = SingleGpuIndex(cpu, scratch_mib=64)
    try:
        scores, ids = index.search(queries, 3)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(scores, expected_scores, atol=2e-5, rtol=2e-6)
    finally:
        index.close()
    report = {
        "passed": True,
        "gpu_count": faiss.get_num_gpus(),
        "fp32": True,
        "queries": len(queries),
        "topk": 3,
        "ordered_ids_equal": True,
        "max_score_abs_error": float(np.abs(scores - expected_scores).max()),
    }
    Path(output).write_text(json.dumps(report, indent=2))
    print("SINGLE_GPU_FAISS_PREFLIGHT_OK", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    main(parser.parse_args().output)
