#!/usr/bin/env python3
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

"""Why is this recipe's NVFP4 rollout not faster than its BF16 rollout?

NeMo-RL reports 1.48x rollout throughput for NVFP4 over BF16 on the same model,
the same vLLM 0.26.0 and the same TE commit. This recipe's production curves
suggest no gain at all -- but those curves are confounded: dynamic sampling
makes the number of generated sequences per step float, and verl does not log
how many generation batches a step used, so wall-clock-per-step cannot be
normalised after the fact.

So measure it directly instead. One variant per process, fixed prompt set,
``ignore_eos`` with a fixed ``max_tokens`` so every variant generates exactly
the same number of tokens, and dummy weights because throughput depends on
shapes and kernels rather than on weight values.

The knobs under test are the ones this recipe added on top of stock vLLM:

* ``enable_pdl=False``, patched into vLLM's two FlashInfer TRTLLM NVFP4 MoE call
  sites to work around an intermittent SM100 startup hang. BF16 goes through
  ``trtllm_bf16_moe.py``, which is unpatched, so only NVFP4 pays for this.
* ``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH`` / ``TRTLLM_DISABLE_FP4_QUANT_FAST_MATH``,
  which likewise only affect FP4 quantization kernels.
* ``gpu_memory_utilization=0.55``, the lowest tier anywhere in this repo, which
  denies NVFP4 the KV-cache headroom its smaller weights would otherwise buy.
* ``cudagraph_mode=FULL_DECODE_ONLY``, which verl forces for real NVFP4.
"""

import json
import os
import sys
import time

# quantization, force_pdl, fp4_fast_math, gpu_mem_util, cudagraph, autotune, n_seqs
#
# v1 of this bench ran on dummy weights and produced six W4A4 variants within 1%
# of each other while both BF16 variants died inside
# `[AutoTuner]: Tuning flashinfer::trtllm_bf16_moe` with an illegal memory
# access -- the same crash seen in the reload repro. With no BF16 number the
# comparison was void, and a sweep where no knob moves the result has to prove
# it can resolve a difference at all before its nulls mean anything. Hence real
# weights, an autotune switch, and `bf16_seq16` as a positive control: dropping
# concurrency 128 -> 16 must show up, otherwise the harness is saturated and
# every other null is meaningless.
VARIANTS = {
    "bf16": (None, None, False, 0.55, "FULL_DECODE_ONLY", False, 128),
    "bf16_seq16": (None, None, False, 0.55, "FULL_DECODE_ONLY", False, 16),
    "bf16_autotune": (None, None, False, 0.55, "FULL_DECODE_ONLY", True, 128),
    "w4a4_prod": ("nvfp4_per_token", False, False, 0.55, "FULL_DECODE_ONLY", False, 128),
    "w4a4_seq16": ("nvfp4_per_token", False, False, 0.55, "FULL_DECODE_ONLY", False, 16),
    "w4a4_pdl_fm": ("nvfp4_per_token", True, True, 0.55, "FULL_DECODE_ONLY", False, 128),
    "w4a4_pdl_fm_m85": ("nvfp4_per_token", True, True, 0.85, "FULL_DECODE_ONLY", False, 128),
    "w4a4_autotune": ("nvfp4_per_token", False, False, 0.55, "FULL_DECODE_ONLY", True, 128),
    # What R3 costs today. verl's capture patch re-runs router.select_experts once
    # per MoE layer per forward because vLLM 0.26 has no native capture; vLLM 0.27
    # exposes supports_routing_replay_capture() and would remove it. Upgrading to
    # 0.27 means torch 2.13, which means rebuilding the apex / flash-attn /
    # megatron-bridge wheelhouse, so the saving has to be measured before it can
    # justify that. `w4a4_prod` is the same run without the wrapper.
    "w4a4_r3wrap": ("nvfp4_per_token", False, False, 0.55, "FULL_DECODE_ONLY", False, 128),
    # Which unquantized MoE backend the BF16 arm should use. verl only sets
    # moe_backend on the real-NVFP4 and delta-sharded paths, so BF16 inherits
    # vLLM's oracle pick -- FLASHINFER_TRTLLM first on CUDA, and vLLM demotes
    # FlashInfer only on SM90, not SM100. That puts the baseline on
    # trtllm_bf16_moe, which has three open upstream bugs (flashinfer#4157, the
    # IMA in the autotune sweep we hit, plus #4919 and #3466) while the FP4 path
    # W4A4 uses has none.
    "bf16_triton": (None, None, False, 0.55, "FULL_DECODE_ONLY", False, 128, "triton"),
    "bf16_triton_at": (None, None, False, 0.55, "FULL_DECODE_ONLY", True, 128, "triton"),
    "bf16_ficutlass": (None, None, False, 0.55, "FULL_DECODE_ONLY", False, 128, "flashinfer_cutlass"),
    "bf16_ficutlass_at": (None, None, False, 0.55, "FULL_DECODE_ONLY", True, 128, "flashinfer_cutlass"),
}


def force_pdl_on() -> None:
    """Undo the image's ``enable_pdl=False`` patch at the FlashInfer boundary.

    The patch edits vLLM's source, so the only way to compare against stock
    behaviour without rebuilding the image is to override the kwarg on the way
    out to FlashInfer.
    """
    import functools

    import flashinfer.fused_moe as fused_moe

    for name in ("trtllm_fp4_block_scale_moe", "trtllm_fp4_block_scale_routed_moe"):
        original = getattr(fused_moe, name, None)
        if original is None:
            print(f"[bench] WARNING: flashinfer has no {name}", flush=True)
            continue

        @functools.wraps(original)
        def with_pdl(*args, __original=original, **kwargs):
            kwargs["enable_pdl"] = True
            return __original(*args, **kwargs)

        setattr(fused_moe, name, with_pdl)
        print(f"[bench] forced enable_pdl=True on {name}", flush=True)


def main() -> int:
    variant = os.environ.get("BENCH_VARIANT", "")
    if variant not in VARIANTS:
        raise SystemExit(f"BENCH_VARIANT must be one of {sorted(VARIANTS)}, got {variant!r}")
    spec = VARIANTS[variant]
    quantization, pdl, fast_math, mem_util, cudagraph, autotune, n_seqs = spec[:7]
    moe_backend = spec[7] if len(spec) > 7 else None

    # These are read when the FP4 kernels are first built, so they must already
    # be set in the environment; the job script does that per variant. Report
    # what actually took effect rather than what was intended.
    env_fast_math = {
        k: os.environ.get(k, "<unset>")
        for k in ("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH")
    }
    print(
        f"[bench] variant={variant} quantization={quantization} force_pdl={pdl} "
        f"fast_math_intended={fast_math} env={env_fast_math} mem={mem_util} cudagraph={cudagraph} "
        f"autotune={autotune} n_seqs={n_seqs} moe_backend={moe_backend or 'auto'}",
        flush=True,
    )

    from vllm import LLM, SamplingParams

    if pdl:
        force_pdl_on()
    if variant == "w4a4_r3wrap":
        from verl.utils.real_nvfp4.r3_monolithic_capture import (
            patch_vllm_monolithic_moe_r3_capture,
        )

        print(f"[bench] R3 capture patch: {patch_vllm_monolithic_moe_r3_capture()}", flush=True)

    out_tokens = int(os.environ.get("BENCH_OUT_TOKENS", "1024"))
    prompt_tokens = int(os.environ.get("BENCH_PROMPT_TOKENS", "512"))

    engine = {
        "model": os.environ["MODEL_PATH"],
        "tensor_parallel_size": 1,
        "enable_expert_parallel": False,
        "max_model_len": 21504,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 128,
        "gpu_memory_utilization": mem_util,
        "enforce_eager": False,
        "dtype": "bfloat16",
        # Real weights: dummy ones drove the BF16 MoE autotuner into an illegal
        # memory access, and online NVFP4 should quantize what production feeds it.
        "load_format": "auto",
        "compilation_config": {"cudagraph_mode": cudagraph},
        "kernel_config": {"enable_flashinfer_autotune": autotune},
        "seed": 0,
    }
    if quantization is not None:
        engine["quantization"] = quantization
    if moe_backend is not None:
        engine["kernel_config"]["moe_backend"] = moe_backend
    llm = LLM(**engine)

    # Same token budget for every variant: ignore_eos removes any dependence on
    # what dummy weights happen to emit.
    prompts = [[(i * 7 + j) % 30000 + 1000 for j in range(prompt_tokens)] for i in range(n_seqs)]
    sampling = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=out_tokens, ignore_eos=True, seed=0)

    from vllm.inputs import TokensPrompt

    payload = [TokensPrompt(prompt_token_ids=p) for p in prompts]

    if variant == "w4a4_r3wrap":
        # The patch only does its extra routing pass when the router carries a
        # capture_fn, which verl installs as part of routing replay. Stand one in
        # so the branch under measurement is actually taken.
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        armed = 0

        def _arm(worker):
            n = 0
            for module in worker.model_runner.get_model().modules():
                if isinstance(module, MoERunner) and getattr(module, "router", None) is not None:
                    module.router.capture_fn = lambda *a, **k: None
                    n += 1
            return n

        armed = sum(llm.collective_rpc(_arm))
        print(f"[bench] armed capture_fn on {armed} MoE runners", flush=True)
        if armed == 0:
            raise SystemExit("R3 variant armed 0 runners; the measurement would be meaningless")

    # Warm up so CUDA-graph capture and autotuning stay out of the measurement.
    llm.generate(payload[: min(8, n_seqs)], SamplingParams(max_tokens=32, ignore_eos=True, seed=0))

    start = time.perf_counter()
    outputs = llm.generate(payload, sampling)
    elapsed = time.perf_counter() - start

    generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    expected = n_seqs * out_tokens
    result = {
        "variant": variant,
        "quantization": quantization,
        "force_pdl": pdl,
        "fp4_fast_math_enabled": fast_math,
        "gpu_memory_utilization": mem_util,
        "cudagraph_mode": cudagraph,
        "moe_backend": moe_backend or "auto",
        "autotune": autotune,
        "n_seqs": n_seqs,
        "out_tokens": out_tokens,
        "generated_tokens": generated,
        "expected_tokens": expected,
        "elapsed_s": round(elapsed, 2),
        "tok_per_s": round(generated / elapsed, 1),
    }
    if generated != expected:
        # Without this the comparison would silently drift between variants.
        result["WARNING"] = f"generated {generated} != expected {expected}"
    print("BENCH_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
