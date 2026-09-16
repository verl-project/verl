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

"""Fixed-workload routing-storage benchmark, not end-to-end BF16 training.

Run in a scheduled allocation with OMP_NUM_THREADS=1:
    python -m tests.experimental.agent_loop.benchmark_route_storage

Uses the same uint8 source, sequence lengths and selection for both arms.
The reference overrides only storage dtype to reproduce the previous int16
conversion. Both arms use the actual AgentLoopWorker and DataProto methods.
Alternating order and warmups reduce allocator/cache/order bias. This measures
host-side postprocessing and transport only, not generation or Ray networking.
The separate replay test covers BF16 Megatron consumption on CPU and CUDA.
"""

import argparse
import asyncio
import gc
import json
import pickle
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch

from tests.experimental.agent_loop.test_route_storage_on_cpu import worker
from verl.experimental.agent_loop import agent_loop as module
from verl.protocol import DataProto


def run_once(outputs, prompt_length, response_length, compact):
    obj = worker()
    obj.rollout_config.prompt_length = prompt_length
    obj.rollout_config.response_length = response_length
    storage_dtype = module._route_storage_dtype if compact else lambda _routes: torch.int16

    async def postprocess():
        return [await obj._agent_loop_postprocess(output, False, raw_prompt="fixed") for output in outputs]

    gc.collect()
    with patch.object(module, "_route_storage_dtype", storage_dtype):
        start = time.perf_counter()
        internal = asyncio.run(postprocess())
        batch = obj._postprocess(internal)
        postprocess_s = time.perf_counter() - start
    routes = batch.batch["routed_experts"]
    payload_bytes = routes.numel() * routes.element_size()
    del internal
    start = time.perf_counter()
    combined = DataProto.concat([batch, batch])
    selected = combined.select_idxs(list(range(0, len(combined), 2)))
    concat_select_s = time.perf_counter() - start
    del combined, batch, routes
    start = time.perf_counter()
    serialized = pickle.dumps(selected)
    restored = pickle.loads(serialized)
    transport_s = time.perf_counter() - start
    actual = restored.batch["routed_experts"]
    assert actual.dtype == (torch.uint8 if compact else torch.int16)
    # Fingerprints and separate elementwise unit tests guard against losing IDs.
    fingerprint = [int(actual.sum()), int(actual.max()), list(actual.shape)]
    return {
        "arm": "uint8" if compact else "int16",
        "postprocess_s": postprocess_s,
        "concat_select_s": concat_select_s,
        "pickle_roundtrip_s": transport_s,
        "total_s": postprocess_s + concat_select_s + transport_s,
        "route_payload_bytes": payload_bytes,
        "pickle_bytes": len(serialized),
        "fingerprint": fingerprint,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--response-length", type=int, default=20480)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    assert all(value > 0 for value in vars(args).values())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(20260915)
    outputs = []
    for i in range(args.batch_size):
        prompt_len = min(args.prompt_length, 256)
        response_len = max(1, min(args.response_length, (128, 512, 1024, 2048)[i % 4]))
        routes = rng.integers(0, 256, size=(prompt_len + response_len - 1, args.layers, args.topk), dtype=np.uint8)
        outputs.append(
            module.AgentLoopOutput(
                prompt_ids=[1] * prompt_len,
                response_ids=[2] * response_len,
                response_mask=[1] * response_len,
                response_logprobs=[-0.5] * response_len,
                routed_experts=routes,
                reward_score=0.5,
                num_turns=2,
                metrics=module.AgentLoopMetrics(),
            )
        )
    records = []
    for repeat in range(args.repeats + 2):
        pair = []
        for compact in (False, True) if repeat % 2 == 0 else (True, False):
            record = run_once(outputs, args.prompt_length, args.response_length, compact)
            record["repeat"] = repeat - 2
            pair.append(record)
            if repeat >= 2:
                records.append(record)
        assert pair[0]["fingerprint"] == pair[1]["fingerprint"]
        if repeat >= 2:
            print(json.dumps({"paired_samples": pair}), flush=True)
    summary = {}
    for arm in ("int16", "uint8"):
        rows = [row for row in records if row["arm"] == arm]
        summary[arm] = {
            key: statistics.median(row[key] for row in rows)
            for key in ("postprocess_s", "concat_select_s", "pickle_roundtrip_s", "total_s", "route_payload_bytes")
        }
    print(
        json.dumps(
            {
                "scope": "host route pipeline only; no model generation or training",
                "config": vars(args),
                "median": summary,
                "route_pipeline_speedup": summary["int16"]["total_s"] / summary["uint8"]["total_s"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
