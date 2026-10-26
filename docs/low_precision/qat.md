# Low-Precision Rollout and Quantization-Aware Training

Last updated: 10/25/2026

In RL training, rollout usually dominates the step time, and for large MoE models it is bound by memory capacity and bandwidth. Running rollout in low precision:

- shrinks weight memory, leaving more room for KV cache and higher concurrency;
- reduces memory traffic during decode, improving throughput;
- uses FP8/FP4 Tensor Cores on Hopper/Blackwell to speed up prefill.

The cost is a numerical gap between the high-precision trainer and the low-precision rollout (training-inference mismatch). It makes importance sampling and off-policy correction less accurate and, in bad cases, destabilizes training. Quantization-Aware Training (QAT) narrows this gap by training the model against the same quantization the rollout uses.

## Background: model weight formats

HuggingFace checkpoints come in two flavors:

1. **High precision**: BF16 weights, no `quantization_config` in `config.json`.
2. **Low precision**: `config.json` carries a `quantization_config` describing how each layer is quantized. Models often use different formats for linear layers (attention, dense MLP, shared experts) and MoE (routed experts).

| Model | Architecture | `quant_method` | Linear | MoE (routed experts) | Not quantized |
|---|---|---|---|---|---|
| [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json) | 92 layers, 512 experts, top-10 | none | BF16 | BF16 | - |
| [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) | 78 layers, 256 experts, top-8, first 3 dense | none | BF16 | BF16 | - |
| [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) | same as GLM-5.2 | `fp8` | FP8 E4M3, 128x128 block, FP32 scale (`weight_scale_inv`), dynamic activation | same as linear | router `mlp.gate`, `indexers_proj`, `embed_tokens`, `lm_head`, MTP `eh_proj`, norms (`modules_to_not_convert`) |
| [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) | 93 layers, 896 experts, first layer dense | `compressed-tensors` | BF16 | MXFP4 W4A16: FP4 E2M1, 1x32 group, UE8M0 scale (`weight_packed` + `weight_scale`) | `self_attn`, `shared_experts`, dense `mlp`, `lm_head`, `vision_tower`, `mm_projector` (`ignore` regex) |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) | 40 layers, 384 experts, top-6 | `fp8` | FP8 E4M3, 32x32 block (`wo_a` may be 128x128), UE8M0 scale (`scale_fmt: ue8m0`) | FP4 E2M1, 1x32 group, UE8M0 scale (`expert_dtype: fp4`), same layout as MXFP4 | router `ffn.gate`, indexer `wk`/`weights_proj`, `compressor` |

- GLM-5.3 uses the standard HF FineGrainedFP8 format (same as DeepSeek-V3), which both vLLM and SGLang parse.
- Kimi-K3 uses compressed-tensors: a `Linear`-targeted config group plus an `ignore` list that excludes non-expert layers yields "linear BF16 + MoE MXFP4".
- DeepSeek-V4.1-Flash's `fp8` config has private fields (`scale_fmt`, `expert_dtype`) and needs dedicated engine support.

The trainer always trains in high precision; low-precision checkpoints are dequantized before training. The rollout uses the model's own `quantization_config` if it has one; a BF16 model can run rollout in BF16 or, for higher throughput, in low precision through `rollout.quantization`.

## How it works

- **Trainer**: always trains in high precision (BF16). With QAT enabled, the forward pass uses fake-quantized weights (quantize then dequantize) and gradients flow through a straight-through estimator.
- **Rollout**: the inference engine starts with dummy weights and either the model's native
  `quantization_config` or a `rollout.quantization` override. It allocates parameters as if
  it were loading that low-precision checkpoint (for example FP8 `weight` +
  `weight_scale_inv`).
- **Weight refit export**: the training backend first exports weights under canonical
  HuggingFace names. A trainer-side `WeightQuantizer`, selected from the effective rollout
  `quantization_config`, converts each sync bucket into exactly the checkpoint tensors
  expected by the rollout engine. Standard native FP8 is quantized to the `Fp8Config`
  layout; native and overridden compressed-tensors configs use the same
  compressed-tensors exporter. DeepSeek-V4 is already quantized by Megatron-Bridge, so its
  exporter only normalizes the packed byte dtype. The rollout worker therefore loads
  pre-quantized tensors and is not responsible for quantization.

![weight_quant](https://github.com/verl-project/verl-data/blob/main/images/weight_quantizer_refit.png?raw=true)

This exporter/loader boundary is intended to decouple the two sides. The target architecture
is the Cartesian product of training backends `[Megatron, VeOmni, TorchTitan]` and rollout
engines `[vLLM, SGLang]`: each trainer exports canonical named tensors, each
`WeightQuantizer` defines a checkpoint wire format, and each rollout engine only implements
loading/refit for that format. The support matrix below records the combinations currently
validated; it does not imply that the full target matrix is implemented yet.

For an override-based rollout, the HuggingFace checkpoint stays in BF16; no offline
quantization or calibration step is needed. The override format is described using the open
[compressed-tensors](https://github.com/neuralmagic/compressed-tensors) schema, which can
express different schemes for linear and routed MoE layers.

## Support matrix

The trainer must be Megatron (FSDP is not yet supported for this feature). The rollout engine can be vLLM or SGLang; TRT-LLM raises an error if `rollout.quantization` is set. The engine columns list the compressed-tensors scheme each engine implements.

verl defaults supported FP8, MXFP8 and MXFP4 Linear/MoE kernels to DeepGEMM.
The training path is moving to the same backend, so this default avoids
backend-dependent activation quantization, scale layouts and fused-MoE
epilogues drifting between training and inference. It also prevents an engine's
`auto` policy from changing kernels across versions or GPU architectures.
Explicit rollout engine kwargs take precedence and can select Linear and MoE
backends independently. The exceptions are vLLM MXFP8 Linear, for which vLLM
0.29 has no DeepGEMM kernel, and NVFP4, for which neither engine provides a
DeepGEMM implementation.

| Precision | Preset | QAT mode | GPU | vLLM Linear | vLLM MoE | SGLang Linear | SGLang MoE | vLLM status in verl | SGLang status in verl |
|---|---|---|---|---|---|---|---|---|---|
| FP8 block (E4M3, 128x128 weight blocks, dynamic 1x128 activations) | `FP8_BLOCK` | `fp8` | Hopper / Blackwell | `W8A8Fp8`<br>`linear_backend=deep_gemm`² | `W8A8Fp8MoE`<br>`moe_backend=deep_gemm` | `W8A8Fp8`<br>`fp8_gemm_runner_backend=deep_gemm` | `W8A8Fp8MoE`<br>`moe_runner_backend=deep_gemm` | ✅ Hopper / Blackwell² | ✅ Hopper |
| MXFP8 | `MXFP8` | `mxfp8` | Blackwell | `W8A8Mxfp8`<br>FlashInfer CuteDSL¹ | `W8A8Mxfp8MoE`<br>`moe_backend=deep_gemm` | native `quant_method=mxfp8`<br>`fp8_gemm_runner_backend=deep_gemm` | native `quant_method=mxfp8`<br>`moe_runner_backend=deep_gemm` | ✅ Blackwell | ✅ Blackwell |
| MXFP4 (MoE only) | `MXFP4` | `mxfp4` / `mxfp4_mxfp8` | Blackwell | ❌ (BF16 dense) | MXFP4×FP8<br>`moe_backend=deep_gemm` | ❌ (BF16 dense) | MXFP4×FP8<br>`moe_runner_backend=deep_gemm` | ✅ Blackwell (MoE only) | ✅ Blackwell (MoE only) |
| NVFP4 (MoE only) | `NVFP4` | `nvfp4` | Blackwell | ❌ | W4A4<br>`(moe_backend=flashinfer_cutlass)` | ❌ (BF16 dense) | W4A4<br>`auto` selects FlashInfer CUTLASS | ✅ Blackwell (MoE only) | ✅ Blackwell (MoE only) |

MXFP8, MXFP4 and NVFP4 need Blackwell (SM100) for the validated paths above. On Hopper, vLLM may fall back to Marlin weight-only kernels, whose numerics differ from Blackwell W8A8 / W4A4; those paths are outside this support and validation matrix. The `MXFP4` preset has no such fallback: its MXFP4-weight × FP8-activation experts are served only by vLLM's DeepGEMM FP4 MoE kernel (SM100/SM120), and Marlin and Triton reject the FP8 1x128 activation scheme, so on Hopper the rollout fails at engine start.

¹ verl passes `linear_backend=deep_gemm`, but vLLM 0.29 has no DeepGEMM
MXFP8 Linear kernel and deliberately falls back to its MXFP8 auto-selection.
On Blackwell this selects FlashInfer CuteDSL. The routed MoE path does use
DeepGEMM.

² On Hopper, verl sets `VLLM_USE_DEEP_GEMM_E8M0=0` for FP32 block-scale
configs so DeepGEMM preserves their scale values. On other architectures verl
does not override the option: vLLM's default selects packed E8M0 on Blackwell.

MXFP4 and NVFP4 support is scoped to routed experts. Both engines support BF16 linear + FP4 MoE; vLLM additionally supports MXFP8 linear + MXFP4 MoE with QAT mode `mxfp4_mxfp8`. SGLang does not support that mixed override. FP4 linear QAT, export and refit are out of scope. verl's `MXFP4` checkpoint preset stores MXFP4 expert weights with group size 32 and E8M0 scales. DeepGEMM dynamically quantizes their expert activations to FP8 with group size 128; the training-side MXFP4 mode uses the same runtime recipe. This differs from compressed-tensors' `MXFP4` preset, which is W4A4, and from vLLM's native `Mxfp4Config`, whose checkpoint format is nominally W4A16.

## Configuration

### Rollout quantization

```yaml
actor_rollout_ref:
  rollout:
    quantization:
      linear: FP8_BLOCK   # attention, dense MLP, shared experts
      moe: FP8_BLOCK      # routed experts
      ignore: []          # extra layers kept in BF16: exact names or "re:<regex>"
      config_file: null   # path to a compressed-tensors quantization_config JSON
```

- When `linear`, `moe` and `config_file` are all `null` (the default), the rollout runs in the model's native precision.
- `linear` and `moe` can be set independently. Setting one of them to `null` keeps those layers in BF16, e.g. `linear: null, moe: FP8_BLOCK` quantizes only the routed experts.
- `MXFP4` and `NVFP4` are valid only for `moe`. Use `linear: null` for BF16 dense layers. vLLM also supports `linear: MXFP8, moe: MXFP4`; that mixed override is not supported by SGLang.
- The following layers are always kept in BF16: `lm_head`, embeddings, MoE routers (`mlp.gate`, `shared_expert_gate`), vision towers and MTP layers.
- `ignore` adds more layers on top of these, using HuggingFace layer names. Layers incompatible with the selected scheme are not detected automatically: list them explicitly. For example, MXFP8 on Qwen3.5 requires `["re:.*linear_attn\\.in_proj_[ab]$"]` because these projections are too small for the vLLM kernel.
- For an all-MXFP8 SGLang rollout, regex ignores must have a known SGLang module-name translation. Use an exact SGLang module name for other ignores; unsupported regex patterns are rejected before launch.

From the command line:

```bash
actor_rollout_ref.rollout.quantization.linear=FP8_BLOCK \
actor_rollout_ref.rollout.quantization.moe=FP8_BLOCK
```

For Qwen3.5 MXFP8:

```bash
actor_rollout_ref.rollout.quantization.linear=MXFP8 \
actor_rollout_ref.rollout.quantization.moe=MXFP8 \
'actor_rollout_ref.rollout.quantization.ignore=["re:.*linear_attn\\.in_proj_[ab]$"]'
```

### Custom `quantization_config`

For full control, point `config_file` at a compressed-tensors `quantization_config`. It can be either the `quantization_config` object itself or a HuggingFace `config.json` that contains one. `config_file` cannot be combined with `linear`, `moe` or `ignore`, and it is used as-is (the default ignore list is not added). Every config group must numerically match a supported preset.

Example equivalent to `linear: FP8_BLOCK, moe: FP8_BLOCK` for Qwen3.5:

```json
{
  "quant_method": "compressed-tensors",
  "format": "float-quantized",
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "weights": {"num_bits": 8, "type": "float", "symmetric": true,
                  "strategy": "block", "block_structure": [128, 128], "dynamic": false},
      "input_activations": {"num_bits": 8, "type": "float", "symmetric": true,
                            "strategy": "group", "group_size": 128, "dynamic": true},
      "format": "float-quantized"
    }
  },
  "ignore": [
    "re:.*lm_head$", "re:.*embed_tokens$", "re:.*visual\\..*", "re:^mtp\\..*",
    "re:.*mlp\\.gate$", "re:.*shared_expert_gate$",
    "re:.*linear_attn\\.in_proj_a$", "re:.*linear_attn\\.in_proj_b$"
  ]
}
```

### QAT (Megatron)

QAT is optional. Enable it on the actor and pick the mode that matches the rollout preset:

```yaml
actor_rollout_ref:
  actor:
    megatron:
      qat:
        enable: true
        mode: fp8   # FP8_BLOCK -> fp8, MXFP8 -> mxfp8, MXFP4 -> mxfp4, NVFP4 -> nvfp4
```

QAT requires a quantized rollout: set `rollout.quantization` or use a checkpoint with a native `quantization_config` that verl supports for export and refit. verl checks at startup that the QAT mode matches the effective rollout configuration. Megatron QAT uses [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer) for fake quantization. MXFP8 uses ModelOpt's native `MXFP8_DEFAULT_CFG`; FP8_BLOCK uses a verl backend because ModelOpt's `FP8_DEFAULT_CFG` is per-tensor and its 2D-block preset is weight-only. MXFP4 and NVFP4 use compressed-tensors-compatible QDQ backends so fake quantization is bit-exact with the exported checkpoint format. Both FP4 modes apply only to routed experts. Use `mxfp4_mxfp8` for the vLLM combination where dense linear layers also use MXFP8.
Its `ignore_patterns` use Megatron module names and `fnmatch` syntax. Keep them aligned with the rollout ignore list; for Qwen3.5 MXFP8, add `*linear_attn.in_proj_a` and `*linear_attn.in_proj_b`.

### Engine notes

- **vLLM**: quantized FP8/MXFP8/MXFP4 rollouts default to
  `linear_backend=deep_gemm` and `moe_backend=deep_gemm` when those options are
  omitted. Set either option under `engine_kwargs.vllm` to override it
  independently. MXFP8 Linear falls back as noted above, while the MXFP4 preset
  has BF16 dense layers.
  Hopper FP8 block configs with FP32 scales explicitly disable DeepGEMM E8M0;
  all other cases retain vLLM's architecture-specific default.
- **SGLang**: quantized FP8/MXFP8/MXFP4 rollouts similarly default
  `fp8_gemm_runner_backend` and `moe_runner_backend` to `deep_gemm`. Explicit
  values under `engine_kwargs.sglang` take precedence.
- **vLLM MXFP4**: vLLM 0.29's compressed-tensors path otherwise ignores the
  requested expert activation scheme. verl routes MXFP4 experts through the
  DeepGEMM FP4 backend, whose dynamic FP8 activations use group size 128.
- **vLLM NVFP4**: verl supports only W4A4 routed experts and forces FlashInfer CUTLASS. W4A16, dense NVFP4 and other MoE backends are rejected.

## Example

Qwen3.5-35B-A3B on 8 GPUs with FP8 block rollout and FP8 QAT:

```bash
python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    actor_rollout_ref.model.path=Qwen/Qwen3.5-35B-A3B \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.qat.enable=True \
    actor_rollout_ref.actor.megatron.qat.mode=fp8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.quantization.linear=FP8_BLOCK \
    actor_rollout_ref.rollout.quantization.moe=FP8_BLOCK \
    ...
```

## Measuring the mismatch

`scripts/rollout_logprob_diff.py` runs one rollout, recomputes the log-probs with the Megatron actor, and reports the gap (`kl`, `k3_kl`, absolute log-prob difference, etc.). Use it to check a configuration before a long run:

```bash
python scripts/rollout_logprob_diff.py --model-path ~/models/Qwen3.5-35B-A3B --ep 8 \
    --quant-linear FP8_BLOCK --quant-moe FP8_BLOCK \
    --quant-ignore 're:.*linear_attn\.in_proj_[ab]$' \
    --qat-mode fp8 \
    --qat-ignore '*linear_attn.in_proj_a' --qat-ignore '*linear_attn.in_proj_b' \
    --r3-router-replay
```

Pass `--engine sglang` for SGLang, and any extra Hydra overrides as trailing arguments. Reference `k3_kl` on
Qwen3.5-35B-A3B (EP=8; vLLM and Blackwell SGLang rollout TP=2; Blackwell
SGLang uses the default CUDA Graph configuration):

| GPU | Engine | R3 | Rollout | Backend | k3_kl |
|---|---|---|---|---|---|
| Hopper | vLLM | ✅ | BF16 | - | 0.00031 |
| Hopper | SGLang | ✅ | BF16 | - | 0.00031 |
| Hopper | vLLM | ✅ | FP8_BLOCK | DeepGEMM | 0.00225 |
| Hopper | SGLang | ✅ | FP8_BLOCK | DeepGEMM | 0.00231 |
| Hopper | vLLM | ✅ | FP8_BLOCK + FP8 QAT | DeepGEMM | 0.00187 |
| Hopper | SGLang | ✅ | FP8_BLOCK + FP8 QAT | DeepGEMM | 0.00188 |
| Blackwell | vLLM | ❌ | MXFP8 linear + MoE | FlashInfer CuteDSL / DeepGEMM | 0.00387 |
| Blackwell | vLLM | ❌ | MXFP8 linear + MoE + MXFP8 QAT | FlashInfer CuteDSL / DeepGEMM | 0.00325 |
| Blackwell | SGLang | ❌ | MXFP8 linear + MoE | DeepGEMM (Linear + MoE) | 0.00324 |
| Blackwell | SGLang | ❌ | MXFP8 linear + MoE + MXFP8 QAT | DeepGEMM (Linear + MoE) | 0.00327 |
| Blackwell | vLLM | ❌ | MXFP4 MoE | DeepGEMM (MoE; BF16 Linear) | 0.00790 |
| Blackwell | vLLM | ❌ | MXFP4 MoE + MXFP4 QAT | DeepGEMM (MoE; BF16 Linear) | 0.00140 |
| Blackwell | SGLang | ❌ | MXFP4 MoE | DeepGEMM (MoE; BF16 Linear) | 0.00571 |
| Blackwell | SGLang | ❌ | MXFP4 MoE + MXFP4 QAT | DeepGEMM (MoE; BF16 Linear) | 0.00200 |
| Blackwell | vLLM | ❌ | NVFP4 MoE | FlashInfer CUTLASS (MoE; BF16 Linear) | 0.00808 |
| Blackwell | vLLM | ❌ | NVFP4 MoE + NVFP4 QAT | FlashInfer CUTLASS (MoE; BF16 Linear) | 0.00722 |
| Blackwell | SGLang | ❌ | NVFP4 MoE | FlashInfer CUTLASS (MoE; BF16 Linear) | 0.00946 |
| Blackwell | SGLang | ❌ | NVFP4 MoE + NVFP4 QAT | FlashInfer CUTLASS (MoE; BF16 Linear) | 0.00653 |

## Limitations

- SGLang MXFP8, MXFP4 and NVFP4 custom refit requires the default bucketed
  weight sync so all derived kernel layouts can be rebuilt after the final
  bucket. The checkpoint engine's `delta_flush` wire format is not supported.
- Native vLLM `Fp8Config` rollout is supported without a rollout override. Standard
  128x128 block-FP8 models are quantized by the trainer-side exporter; DeepSeek-V4
  keeps the private mixed FP8/FP4 layout already exported by Megatron-Bridge.
- Megatron QAT and the rollout use separate ignore lists and naming conventions. Users must keep them aligned.
