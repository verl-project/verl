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
"""Rank-local bake (``rank_local_bake.py``) on a real vLLM TP engine with dense Qwen3/Qwen2; no trainer, no transport.

1. Every TP rank's baked ``slices`` equal the hand-written TP rules (``qwen3_tp_slices``).
2. After zeroing every parameter, feeding the checkpoint the way a ``rank_local_named_tensors`` sync
   does (baked names as their box only, the rest full) through ``_update_weights`` restores every
   parameter bit-exact.

    RANK_LOCAL_TEST_MODEL=<Qwen3 or Qwen2 snapshot dir> RANK_LOCAL_TEST_TP=4 \
        pytest -s tests/workers/rollout/rollout_vllm/test_vllm_rank_local_bake.py
"""

import os

import pytest
import torch

MODEL = os.path.expanduser(os.getenv("RANK_LOCAL_TEST_MODEL", "~/models/Qwen/Qwen3-0.6B"))
TP = int(os.getenv("RANK_LOCAL_TEST_TP", "2"))

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < TP or not os.path.isdir(MODEL), reason=f"needs {TP} GPUs and {MODEL}"
)


def test_rank_local_bake_and_apply(monkeypatch):
    from transformers import AutoConfig
    from vllm import LLM

    from tests.checkpoint_engine.test_utils import qwen3_tp_slices

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # collective_rpc with a function
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
    )
    layouts = llm.collective_rpc("get_weight_ownership")
    hf_config = AutoConfig.from_pretrained(MODEL)
    for r, lay in enumerate(layouts):
        assert lay["tp_rank"] == r and "rank_local_named_tensors" in lay["wire_formats"], lay.keys()
        ref = qwen3_tp_slices(hf_config, TP, r)
        # a tied lm_head is not loaded by vLLM, so it has no box (and is not in every checkpoint)
        assert set(ref) - set(lay["slices"]) <= {"lm_head.weight"}, sorted(set(ref) - set(lay["slices"]))[:5]
        assert lay["slices"] == {k: ref.get(k) for k in lay["slices"]}

    def reload_rank_local(self, model_path):  # in every vLLM worker; self = the worker with verl's extension
        import glob

        from safetensors import safe_open
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        model = self.model_runner.model
        plan = self._get_rank_local_plan()
        device = next(model.parameters()).device
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        for p in model.parameters():
            p.data.zero_()
        for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            weights = []
            with safe_open(f, framework="pt", device=str(device)) as st:
                for k in st.keys():
                    t = st.get_tensor(k).to(torch.bfloat16)
                    if k in plan.slices:
                        t = t[tuple(slice(s, s + n) for s, n in plan.slices[k]["box"])].contiguous()
                    weights.append((k, t))
            self._update_weights(weights, peft_config=None, base_sync_done=False)
        process_weights_after_loading(model, self.model_runner.vllm_config.model_config, device)
        return [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]

    assert llm.collective_rpc(reload_rank_local, args=(MODEL,)) == [[]] * TP
