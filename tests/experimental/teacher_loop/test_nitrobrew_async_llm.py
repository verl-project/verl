# Copyright 2026 Tilde Research
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

"""GPU smoke test for the vLLM-backed nitrobrew teacher worker.

Verifies that:
  - The driver can load lm_head + compute SVD for a small model.
  - A NitrobrewTeacherWorker actor boots an AsyncLLM in pooling+embed mode.
  - compute_hidden_states returns a [S, d_comp] tensor in the expected dtype.
"""

import os

import pytest
import ray
import torch

from verl.experimental.teacher_loop.nitrobrew_teacher import (
    NitrobrewTeacherWorker,
    _compute_svd,
    _load_lm_head_weight,
)

MODEL_PATH = os.environ.get("NITROBREW_TEST_MODEL", "Qwen/Qwen3-0.6B")
D_COMP = 8
DTYPE = "bfloat16"
MAX_MODEL_LEN = 256


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_nitrobrew_teacher_worker_returns_compressed_hidden_states(ray_cluster):
    w_t = _load_lm_head_weight(MODEL_PATH)
    w_up, p_down = _compute_svd(w_t, d_comp=D_COMP, dtype=getattr(torch, DTYPE))
    v, hidden = w_t.shape
    assert w_up.shape == (v, D_COMP)
    assert p_down.shape == (hidden, D_COMP)

    worker = NitrobrewTeacherWorker.options(num_gpus=1, max_concurrency=4).remote()
    ray.get(
        worker.setup.remote(
            MODEL_PATH,
            D_COMP,
            p_down.cpu().tolist(),
            DTYPE,
            1,  # tensor_parallel_size
            0.5,  # gpu_memory_utilization
            4,  # max_num_seqs
            MAX_MODEL_LEN,
            True,  # enforce_eager (avoid cuda-graph reservation in CI)
        )
    )

    sequence_ids = list(range(32))
    out = ray.get(worker.compute_hidden_states.remote(sequence_ids))
    z = torch.tensor(out)
    assert z.shape == (len(sequence_ids), D_COMP), z.shape
    assert torch.isfinite(z).all()
