# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Check the runtime vocab-parallel top-k path with real two-rank collectives."""

from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")

from verl.utils.megatron.vocab_parallel_topk import topk_logprobs_from_logits  # noqa: E402


def _check_distributed_topk(rank, rendezvous_path):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=60),
    )
    try:
        # Unique, exactly BF16-representable logits avoid ambiguous ordering at
        # ties. Different rows favor different shards and exercise the global
        # normalizer with both close scores and a large dynamic range.
        full = torch.tensor(
            [
                [0.015625, -0.125, 0.1875, 0.25, 0.0625, 0.3125, -0.0625, 0.125, 0.375, 0.0],
                [-1.125, -0.5, 0.25, 0.125, 0.0, 0.5, 1.0, 1.125, 0.75, -0.125],
                [12.0, 11.0, 10.0, 9.0, 8.0, -1.0, -2.0, -3.0, -4.0, -5.0],
            ]
        ).unsqueeze(0)
        for dtype in (torch.float32, torch.bfloat16):
            logits = full.to(dtype)
            local = logits[..., rank * 5 : (rank + 1) * 5].contiguous()
            reference = logits.float().log_softmax(dim=-1)
            for requested_k in (8, 99):
                logps, ids = topk_logprobs_from_logits(local, requested_k, process_group=torch.distributed.group.WORLD)
                expected_logps, expected_ids = reference.topk(min(requested_k, 10), dim=-1)
                assert logps.shape == expected_logps.shape
                assert ids.shape == expected_ids.shape
                assert logps.dtype == torch.float32
                torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
                torch.testing.assert_close(logps, expected_logps, rtol=0, atol=2e-6)
                if requested_k > 10:
                    torch.testing.assert_close(logps.exp().sum(dim=-1), torch.ones_like(logps[..., 0]))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Requires the Gloo backend",
)
def test_vocab_parallel_topk_with_real_collectives(tmp_path):
    torch.multiprocessing.spawn(
        _check_distributed_topk,
        args=(str(tmp_path / "gloo_rendezvous"),),
        nprocs=2,
        join=True,
    )
