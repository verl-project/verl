# RFC: Multimodal Continuous Token for VL Model Families

Last updated: 06/21/2026.

> Extends [PR #6779](https://github.com/volcengine/verl/pull/6779) (text-only Continuous Token) to support vision-language models in agentic multi-turn rollout.

## Motivation

PR #6779 established the Continuous Token (CT) framework for text-only multi-turn agentic rollout, ensuring the TITO (Token-In-Token-Out) invariant: the token sequence at turn N-1 is a bit-perfect prefix of the prompt at turn N. This avoids BPE re-encoding artifacts and enables single-trajectory training with correct loss masks.

However, VL models (Qwen2.5-VL, Qwen3-VL, MiMo-VL) introduce additional complexity:

1. **Processor dependency** — VL models require a HuggingFace processor (not just a tokenizer) to produce `pixel_values` and `image_grid_thw` alongside token IDs.
2. **Image placeholder expansion** — A single `<|image_pad|>` in the chat template gets expanded into N pad tokens by the processor, where N depends on image resolution.
3. **Incremental image token handling** — In multi-turn rollout, new images appear in later turns. CT needs only the new turn's expanded image placeholder token IDs to keep the runtime prefix correct.
4. **Final tensor reconstruction** — Training still needs full `pixel_values` / `image_grid_thw`, so the agent loop rebuilds multimodal tensors from the final text and complete image list during postprocessing.

Without multimodal CT, VL models in agentic rollout must either (a) re-encode the entire history including all images every turn, or (b) fall back to the legacy non-CT path, losing the TITO guarantee.

## Design

### Architecture

```
ContinuousTokenBuilder (base)
├── QwenContinuousTokenBuilder (text, ChatML boundary: newline after <|im_end|>)
│   ├── QwenVLContinuousTokenBuilder (adds processor-backed vision token rendering)
│   └── MiMoVLContinuousTokenBuilder (adds processor-backed rendering + content flattening)
└── (other text builders from PR #6779: MiniMax, GLM, Gemma4, GptOss)
```

VL builders **inherit** from `QwenContinuousTokenBuilder`, reusing its ChatML boundary logic while adding vision-specific methods.

### MergeResult Contract

```python
@dataclass(frozen=True)
class MergeResult:
    token_ids: list[int]
    appended_token_count: int
    kind: MergeKind = "non_assistant"
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0
```

`MergeResult` stays token-only. VL builders use the multimodal processor to render the correct token IDs, but they do not return delta `pixel_values` / `image_grid_thw` through the merge result. The agent loop keeps the original image objects in `multi_modal_data`, and final training tensors are rebuilt from the full image list during postprocessing.

### VL Builder Core Methods

| Method | Purpose |
|--------|---------|
| `supports_multimodal()` | Class-level flag; controls gate logic in agent_loop |
| `render_tokens_with_mm(messages, images)` | Full processor render → token_ids |
| `build_initial_tokens(messages)` | Turn 1: processor render for image placeholder expansion |
| `merge_tokens(prev, updated, runtime_ids)` | If new images: synthetic prefix + trim render for incremental token IDs. If no new images: text-only incremental merge |
| `extract_vision_placeholders(token_ids)` | Finds `<|vision_start|>...<|vision_end|>` spans |
| `count_vision_tokens(grid_row)` | Computes merged token count: `t*(h//merge)*(w//merge)` |

### Image Tensor Handling

When a new image appears at turn N, `merge_tokens` calls the multimodal processor on the appended messages with a synthetic prefix, then trims the synthetic prefix token IDs. This gives CT the correct expanded image placeholder tokens without re-processing old images for rollout token construction.

The processor's pixel tensors from this incremental call are intentionally discarded. The agent loop accumulates original image objects (`agent_data.image_data`) and `_compute_multi_modal_inputs()` later rebuilds full `pixel_values` / `image_grid_thw` from the final token sequence and complete image list.

### Agent Loop Integration

Three gates were flipped to enable VL builders:

1. **`agent_loop.py`** — Init no longer hard-fails when `processor is None`; only fails if the resolved builder is VL AND processor is missing.
2. **`tool_agent_loop.py`** — CT merge path runs when `not new_images OR builder.supports_multimodal()`.
3. **`single_turn_agent_loop.py`** — Same gate pattern.

A `_cap_text_prompt_length` safety guard skips truncation for VL builders to avoid splitting vision token spans.

## Model Support

### VL Families

| Family | Models | Status |
|--------|--------|--------|
| `qwen25vl` / `qwen3vl` | Qwen2.5-VL-*, Qwen3-VL-* | Verified on H100 |
| `mimovl` | MiMo-VL-* | Verified on H100 (with template flattening) |
| `glm4v` | GLM-4.5V, GLM-4.1V | Verified on H100 |
| `kimivl` | Kimi-VL-* | Verified on H100 |

### Future

- DeepSeek-VL2 (non-standard processor, complex tiling)

## Correctness Verification

### CT vs Legacy Comparison

Tested on NVIDIA H100 80GB with real model weights:

| Model | Scenario | CT tokens | Legacy tokens | Result |
|-------|----------|-----------|---------------|--------|
| Qwen2.5-VL-7B | single_image | 89 | 89 | **MATCH** |
| Qwen2.5-VL-7B | multi_turn_new_image | 136 | 136 | **MATCH** |
| Qwen2.5-VL-7B | text_after_image | 109 | 109 | **MATCH** |
| MiMo-VL-7B | single_image | 95 | 95 | **MATCH** |
| MiMo-VL-7B | multi_turn_new_image | 138 | 138 | **MATCH** |
| MiMo-VL-7B | text_after_image | 112 | 112 | **MATCH** |

**6/6 scenarios produce byte-identical token sequences.** Zero mismatch cases.

### Unit Tests

Targeted unit tests cover:
- MergeResult construction, backward compat, immutability
- All text subclass boundary behaviors
- VL builder: build, merge, placeholder extraction
- Wiring/factory: family resolution, auto-inference, VL processor gating

## Backward Compatibility

- `MergeResult` remains token-only → existing text-only constructors work unchanged
- CT is disabled by default → no behavior change unless `continuous_token.enable = True`
- VL gates are `supports_multimodal()` checks → text-only builders never enter VL code paths
- Full multimodal tensors are still produced in the existing postprocessing path

## Limitations

1. **Processor call on new images** — When a new image appears, CT still needs a processor call for the appended messages so image placeholders expand to the correct number of tokens.
2. **No GPU training validation yet** — Token-level correctness is verified, but a full GRPO training run (Qwen2.5-VL + geo3k, 50 steps) has not been performed.
3. **QwenVL / MiMoVL code duplication** — The two VL builders share ~200 lines of identical logic. A mixin extraction is planned as follow-up.

## References

- [LMSYS Blog: No Token Left Behind](https://blog.lmsys.org/2025-05-15-no-token-left-behind/)
- [verl RFC Issue #6719](https://github.com/verl-project/verl/issues/6719) — CT design rationale
- [verl PR #6779](https://github.com/volcengine/verl/pull/6779) — Text-only CT (base)
- [slime PR #1141](https://github.com/THUDM/slime/pull/1141) — Reference multimodal multi-turn implementation
