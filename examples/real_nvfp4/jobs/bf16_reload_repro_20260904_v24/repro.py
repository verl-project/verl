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

"""Which unquantized MoE backend can verl's BF16 rollout actually use?

verl's ``PRECISION_MODE=bf16`` rollout comes back with random-looking output
(entropy 6.1, rollout KL 13.9) even though every expert weight is received and
loaded.  This strips verl out of the picture: bring up the real checkpoint,
generate, reload *the same weights from disk*, then generate again.  A correct
setup produces sensible text and a bit-identical reload.

The first run of this script showed the default pick on GB200 is FlashInfer
TRTLLM BF16 -- the one kernel family this image already had to patch (PDL off)
for NVFP4, left unpatched for BF16 -- and it took an illegal memory access
during warmup.  So the script now sweeps one backend per invocation
(``REPRO_VARIANT``) instead of assuming the default is usable.

Runs on one GPU, no Ray and no training.  Read-only w.r.t. the checkpoint.
"""

import os
import sys


def snapshot(worker):
    """Per-parameter fingerprint of the live model, taken inside the worker."""
    model = worker.model_runner.get_model()
    out = {}
    for name, param in model.named_parameters():
        tensor = param.detach()
        out[name] = (
            float(tensor.float().abs().sum()),
            float(tensor.float().mean()),
            tuple(tensor.shape),
            str(tensor.dtype),
        )
    return out


def do_reload(worker):
    worker.model_runner.reload_weights()


def hf_weights_iterator(model_path: str):
    """Yield every checkpoint tensor, the way verl's transport feeds the engine.

    ``reload_weights()`` refuses to source weights from disk when the engine was
    brought up with ``load_format=dummy`` -- which is exactly how verl starts the
    rollout -- so the dummy variants have to hand it an explicit iterator.
    """
    import glob

    from safetensors import safe_open

    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def do_reload_from_disk(worker, model_path: str):
    worker.model_runner.reload_weights(
        weights_iterator=hf_weights_iterator(model_path),
        is_checkpoint_format=True,
    )


def compare(before: dict, after: dict, label: str) -> list[str]:
    drifted = []
    for name, value in before.items():
        if name not in after:
            drifted.append(f"{name}: DISAPPEARED")
            continue
        if after[name][2:] != value[2:]:
            drifted.append(f"{name}: shape/dtype {value[2:]} -> {after[name][2:]}")
        elif after[name][0] != value[0]:
            drifted.append(f"{name}: abs_sum {value[0]:.6g} -> {after[name][0]:.6g}")
    print(f"[{label}] {len(drifted)} of {len(before)} parameters changed", flush=True)
    for line in drifted[:25]:
        print(f"[{label}]   {line}", flush=True)
    if len(drifted) > 25:
        print(f"[{label}]   ... and {len(drifted) - 25} more", flush=True)
    return drifted


# Autotuning is disabled everywhere below: it is what crashed the first run, and
# it is orthogonal to the question of whether a backend computes MoE correctly.
VARIANTS = {
    # Reference: the portable Triton path, no FlashInfer involved.
    "triton": {"kernel_config": {"moe_backend": "triton", "enable_flashinfer_autotune": False}},
    # The GB200 default pick, which is what verl's BF16 rollout was running.
    "trtllm": {"kernel_config": {"moe_backend": "flashinfer_trtllm", "enable_flashinfer_autotune": False}},
    # Same kernel with PDL off, mirroring this image's NVFP4 patch.
    "trtllm_pdl_off": {"kernel_config": {"moe_backend": "flashinfer_trtllm", "enable_flashinfer_autotune": False}},
    "flashinfer_cutlass": {"kernel_config": {"moe_backend": "flashinfer_cutlass", "enable_flashinfer_autotune": False}},
    # verl's actual rollout engine config: CUDA graphs on, autotune left at the
    # vLLM default, long context, max_num_seqs=128.  If this stays coherent then
    # the engine is exonerated and the defect is in verl's BF16 weight export.
    "verl_like": {
        "enforce_eager": False,
        "max_model_len": 20480,
        "max_num_seqs": 128,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
    },
    # R3 asks vLLM to return routed experts, which changes the MoE path.
    "verl_like_r3": {
        "enforce_eager": False,
        "max_model_len": 20480,
        "max_num_seqs": 128,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "enable_return_routed_experts": True,
    },
    # The closest emulation of verl: the engine comes up on dummy weights and
    # only ever sees real weights through a reload, exactly as the rollout does.
    "verl_like_dummy": {
        "enforce_eager": False,
        "max_model_len": 20480,
        "max_num_seqs": 128,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "load_format": "dummy",
    },
    "verl_like_dummy_r3": {
        "enforce_eager": False,
        "max_model_len": 20480,
        "max_num_seqs": 128,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "enable_return_routed_experts": True,
        "load_format": "dummy",
    },
}

# Substrings the base model reliably produces once it holds real weights.
COHERENCE_MARKERS = ("4", "Paris", "2, 3, 5, 7", "x")


def force_pdl_off() -> None:
    """Mirror the image's NVFP4 PDL-off patch onto the BF16 TRTLLM MoE call."""
    import functools

    import flashinfer.fused_moe as fused_moe

    original = fused_moe.trtllm_bf16_moe

    @functools.wraps(original)
    def without_pdl(*args, **kwargs):
        kwargs["enable_pdl"] = False
        return original(*args, **kwargs)

    fused_moe.trtllm_bf16_moe = without_pdl
    print("[repro] forced enable_pdl=False on flashinfer.trtllm_bf16_moe", flush=True)


def main() -> int:
    model_path = os.environ["MODEL_PATH"]
    variant = os.environ.get("REPRO_VARIANT", "triton")
    if variant not in VARIANTS:
        raise SystemExit(f"unknown REPRO_VARIANT={variant!r}; expected {sorted(VARIANTS)}")
    overrides = VARIANTS[variant]
    print(f"[repro] variant={variant} overrides={overrides}", flush=True)

    from vllm import LLM, SamplingParams

    if variant == "trtllm_pdl_off":
        force_pdl_off()

    engine_kwargs = {
        "model": model_path,
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "gpu_memory_utilization": float(os.environ.get("REPRO_GPU_UTIL", "0.55")),
        "enforce_eager": True,
        "seed": 0,
        **overrides,
    }
    llm = LLM(**engine_kwargs)
    # Greedy so any output difference is a weight difference, not sampling noise.
    sampling = SamplingParams(temperature=0.0, max_tokens=48, seed=0)
    prompts = [
        "Question: What is 2+2?\nAnswer:",
        "The capital of France is",
        "List the first five prime numbers:",
        "Solve for x: 3x + 6 = 21. x =",
    ]

    text_before = [o.outputs[0].text for o in llm.generate(prompts, sampling)]
    fingerprint_before = llm.collective_rpc(snapshot)[0]

    if overrides.get("load_format") == "dummy":
        llm.collective_rpc(do_reload_from_disk, args=(model_path,))
    else:
        llm.collective_rpc(do_reload)

    fingerprint_after = llm.collective_rpc(snapshot)[0]
    text_after = [o.outputs[0].text for o in llm.generate(prompts, sampling)]

    print("=" * 72, flush=True)
    for index, (before, after) in enumerate(zip(text_before, text_after, strict=True)):
        verdict = "SAME" if before == after else "DIFFERENT"
        print(f"[gen {index}] {verdict}", flush=True)
        print(f"[gen {index}]   before: {before!r}", flush=True)
        print(f"[gen {index}]   after : {after!r}", flush=True)

    drifted = compare(fingerprint_before, fingerprint_after, "weights")
    text_changed = text_before != text_after

    print("=" * 72, flush=True)
    if overrides.get("load_format") == "dummy":
        # Starting from dummy weights, the reload is supposed to change
        # everything; what matters is that the model is coherent afterwards.
        incoherent = [
            index
            for index, (marker, after) in enumerate(zip(COHERENCE_MARKERS, text_after, strict=True))
            if marker not in after
        ]
        if not incoherent:
            print(
                f"VERL_BF16_RELOAD_REPRO PASS variant={variant} reload=dummy_to_real changed_params={len(drifted)}",
                flush=True,
            )
            return 0
        print(
            f"VERL_BF16_RELOAD_REPRO FAIL variant={variant} reload=dummy_to_real incoherent_prompts={incoherent}",
            flush=True,
        )
        return 1
    if not drifted and not text_changed:
        print(f"VERL_BF16_RELOAD_REPRO PASS variant={variant} reload=identity", flush=True)
        return 0
    print(
        f"VERL_BF16_RELOAD_REPRO FAIL variant={variant} changed_params={len(drifted)} text_changed={int(text_changed)}",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
