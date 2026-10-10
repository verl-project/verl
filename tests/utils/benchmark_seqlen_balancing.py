# Copyright 2026 Individual Contributor: Isaac Li
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

"""CPU-only baseline comparison; pass an uncached seqlen_balancing.py as --baseline.

Example (from the repository root):
    git show 8718ca30:verl/utils/seqlen_balancing.py > /tmp/seqlen_baseline.py
    python tests/utils/benchmark_seqlen_balancing.py --baseline /tmp/seqlen_baseline.py

Times cover the complete karmarkar_karp call, including sorting and merging.
"""

import argparse
import ast
import gc
import heapq
import random
from pathlib import Path
from statistics import median
from time import perf_counter


def _load_partition(path):
    # Compile the unchanged function without importing unrelated framework dependencies.
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "karmarkar_karp")
    namespace = {"heapq": heapq, "__name__": __name__}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["karmarkar_karp"]


def _cast_values(values, dtype):
    if dtype == "int":
        return values
    if dtype == "float":
        return list(map(float, values))
    if dtype.startswith("numpy"):
        import numpy as np

        return list(np.asarray(values, dtype=f"float{dtype[-2:]}"))
    import torch

    return list(torch.tensor(values, dtype=getattr(torch, f"float{dtype[-2:]}"), device="cpu"))


def _time_call(fn, values, k, equal_size, number):
    start = perf_counter()
    for _ in range(number):
        fn(values, k, equal_size)
    return (perf_counter() - start) / number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024, 4096, 8192])
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=["int", "float", "numpy32", "numpy64", "tensor32", "tensor64"], default="int"
    )
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    baseline = _load_partition(args.baseline)
    optimized = _load_partition(Path(__file__).resolve().parents[2] / "verl/utils/seqlen_balancing.py")
    print("n,equal_size,dtype,baseline_ms,optimized_ms,speedup", flush=True)
    for n in args.sizes:
        rng = random.Random(2026 + n)
        values = _cast_values([rng.randrange(1, 32769) for _ in range(n)], args.dtype)
        for equal_size in [False, True]:
            assert baseline(values, args.partitions, equal_size) == optimized(values, args.partitions, equal_size)
            estimate = _time_call(baseline, values, args.partitions, equal_size, 1)
            number = max(1, min(200, int(0.05 / max(estimate, 1e-9))))
            timings = [[], []]
            was_enabled = gc.isenabled()
            gc.disable()
            try:
                for repeat in range(args.repeats):
                    for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
                        fn = [baseline, optimized][index]
                        timings[index].append(_time_call(fn, values, args.partitions, equal_size, number))
            finally:
                if was_enabled:
                    gc.enable()
            before, after = map(median, timings)
            print(
                f"{n},{equal_size},{args.dtype},{before * 1000:.3f},{after * 1000:.3f},{before / after:.3f}", flush=True
            )


if __name__ == "__main__":
    main()
