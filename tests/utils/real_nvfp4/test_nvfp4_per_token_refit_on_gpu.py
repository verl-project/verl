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

"""NVFP4_PER_TOKEN rollout refit on small Qwen3-MoE and DeepSeek-V3 models.

The preset engine refits from the trainer-side export through vLLM's layerwise reload.
Its greedy logprobs must equal vLLM's online nvfp4_per_token method loading the same BF16
weights, and with CUDA graphs a refit must restore them exactly, including after other
weights and after a level-2 sleep.
"""

import gc
import os

import pytest
import torch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
pytest.importorskip("compressed_tensors")
pytest.importorskip("vllm")

from verl.utils.qat.compressed_tensors import (  # noqa: E402
    NVFP4PerTokenWeightQuantizer,
    build_compressed_tensors_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="NVFP4_PER_TOKEN rollout needs a Blackwell (SM100) GPU",
)

NUM_LAYERS = 3
PROMPTS = [[11, 523, 97, 1024, 7], [3, 3, 3, 900, 41, 2049], [1500, 8, 77]]


def _qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=4096,
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=32,
        num_experts_per_tok=8,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        max_position_embeddings=1024,
        torch_dtype="bfloat16",
    )
    return Qwen3MoeForCausalLM(config)


def _deepseek_v3():
    # A dense first layer, MLA attention and a shared expert next to the routed ones.
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    config = DeepseekV3Config(
        vocab_size=4096,
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_hidden_layers=NUM_LAYERS,
        first_k_dense_replace=1,
        num_attention_heads=16,
        num_key_value_heads=16,
        q_lora_rank=768,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=32,
        n_shared_experts=1,
        num_experts_per_tok=8,
        n_group=1,
        # As in DeepSeek-V3's config.json; vLLM keys the router bias on them.
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        topk_group=1,
        norm_topk_prob=True,
        max_position_embeddings=1024,
        torch_dtype="bfloat16",
    )
    return DeepseekV3ForCausalLM(config)


@pytest.fixture(scope="module", params=[_qwen3_moe, _deepseek_v3], ids=["qwen3_moe", "deepseek_v3"])
def model_dir(request, tmp_path_factory):
    torch.manual_seed(0)
    model = request.param().to(torch.bfloat16)
    path = tmp_path_factory.mktemp(request.param.__name__)
    model.save_pretrained(path)
    return str(path)


def _weights(model_dir, scale=1.0):
    from safetensors import safe_open

    out = []
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(model_dir, name), "pt", device="cuda") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    out.append((key, tensor * scale if ".experts." in key else tensor))
    return out


def _engine(model_dir, eager, **kwargs):
    from vllm import LLM

    return LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=4,
        # Small models: several engines run one after another in this process.
        gpu_memory_utilization=0.08,
        enforce_eager=eager,
        compilation_config=None if eager else {"cudagraph_mode": "FULL_DECODE_ONLY"},
        enable_flashinfer_autotune=False,
        seed=0,
        **kwargs,
    )


def _runner(llm):
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    return core.model_executor.driver_worker.worker.model_runner


def _refit(llm, weights):
    from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload

    runner = _runner(llm)
    model = runner.get_model()
    initialize_layerwise_reload(model)
    for start in range(0, len(weights), 32):
        model.load_weights([(name, tensor.clone()) for name, tensor in weights[start : start + 32]])
    finalize_layerwise_reload(model, runner.vllm_config.model_config)


def _logprobs(llm):
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=16, logprobs=0)
    outputs = llm.generate([{"prompt_token_ids": p} for p in PROMPTS], params, use_tqdm=False)
    return [[next(iter(lp.values())).logprob for lp in o.outputs[0].logprobs] for o in outputs]


def _kv_caches(llm):
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    caches = []
    for module in _runner(llm).get_model().modules():
        if isinstance(module, AttentionLayerBase):
            kv = getattr(module, "kv_cache", None)
            caches.extend(t for t in (kv if isinstance(kv, list | tuple) else [kv]) if isinstance(t, torch.Tensor))
    return caches


def _free():
    # Callers drop their engine reference first: shutting an engine down tears down
    # vLLM's process-wide parallel state, which the next engine would otherwise lose.
    gc.collect()
    torch.cuda.empty_cache()


def test_preset_refit_matches_online_quantization(model_dir, request):
    from verl.utils.vllm.vllm_nvfp4_per_token_utils import (
        attest_nvfp4_per_token_runtime,
        clear_kv_cache,
        nvfp4_per_token_quantization_config,
        routed_expert_layers,
    )
    from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches

    carved = [0, NUM_LAYERS - 1]
    # vLLM's online method quantizes the BF16 checkpoint at load. Eager mode keeps the
    # comparison independent of each engine's KV-cache size, which steers graph kernels.
    online = _engine(
        model_dir,
        eager=True,
        quantization="nvfp4_per_token",
        quantization_config={"ignore": [f"model.layers.{i}.mlp.experts" for i in carved]},
        moe_backend="flashinfer_trtllm",
    )
    expected = _logprobs(online)
    del online
    _free()

    apply_vllm_quant_patches()
    ignore = [r"re:.*layers\.(" + "|".join(map(str, carved)) + r")\.mlp\.experts\..*"]
    config = build_compressed_tensors_config(linear=None, moe="NVFP4_PER_TOKEN", ignore=ignore, config_file=None)

    def refit(llm, scale):
        model = _runner(llm).get_model()
        layers = routed_expert_layers(model, config)
        assert sorted(layer for layer, (_, quantized) in layers.items() if quantized) == [1]
        _refit(llm, list(NVFP4PerTokenWeightQuantizer(config)(_weights(model_dir, scale))))
        attest_nvfp4_per_token_runtime(model, config)
        return _logprobs(llm)

    preset = _engine(model_dir, eager=True, load_format="dummy", hf_overrides={"quantization_config": config})
    # The worker finds the per-token config that came in through hf_overrides.
    assert nvfp4_per_token_quantization_config(_runner(preset).vllm_config) is not None
    # Start-up forwards with the dummy weights can leave NaN in the KV cache; the rollout
    # clears it once the engine is up (monkey_patch_model).
    clear_kv_cache(_runner(preset).get_model())
    assert _kv_caches(preset) and all(torch.isfinite(t).all() for t in _kv_caches(preset))
    assert refit(preset, 1.0) == expected
    del preset
    _free()

    # vLLM allows one sleep-mode (CuMem) engine per process, so one model covers the sleep.
    sleep = request.node.callspec.id == "qwen3_moe"
    preset = _engine(
        model_dir,
        eager=False,
        load_format="dummy",
        enable_sleep_mode=sleep,
        hf_overrides={"quantization_config": config},
    )
    first = refit(preset, 1.0)
    assert refit(preset, 1.5) != first
    assert refit(preset, 1.0) == first
    if sleep:
        # A level-2 sleep discards the weights; the next refit must restore every captured tensor.
        preset.sleep(level=2)
        preset.wake_up()
        assert refit(preset, 1.0) == first
    del preset
    _free()
