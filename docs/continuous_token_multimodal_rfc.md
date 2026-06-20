# RFC: Multimodal Continuous Token for VL Model Families

> Extends [PR #6779](https://github.com/volcengine/verl/pull/6779) (text-only Continuous Token) to support vision-language models in agentic multi-turn rollout.

## Motivation

PR #6779 established the Continuous Token (CT) framework for text-only multi-turn agentic rollout, ensuring the TITO (Token-In-Token-Out) invariant: the token sequence at turn N-1 is a bit-perfect prefix of the prompt at turn N. This avoids BPE re-encoding artifacts and enables single-trajectory training with correct loss masks.

However, VL models (Qwen2.5-VL, Qwen3-VL, MiMo-VL) introduce additional complexity:

1. **Processor dependency** — VL models require a HuggingFace processor (not just a tokenizer) to produce `pixel_values` and `image_grid_thw` alongside token IDs.
2. **Image placeholder expansion** — A single `<|image_pad|>` in the chat template gets expanded into N pad tokens by the processor, where N depends on image resolution.
3. **Incremental image handling** — In multi-turn rollout, new images appear in later turns. Re-processing all previous images every turn is wasteful; only the delta should be computed.
4. **Pixel tensor indexing** — `pixel_values` dim0 is indexed by raw patches (`t*h*w`), not by merged vision tokens (`t*(h//merge)*(w//merge)`). Slicing at the wrong granularity corrupts image data.

Without multimodal CT, VL models in agentic rollout must either (a) re-encode the entire history including all images every turn, or (b) fall back to the legacy non-CT path, losing the TITO guarantee.

## Design

### Architecture

```
ContinuousTokenBuilder (base)
├── QwenContinuousTokenBuilder (text, ChatML boundary: newline after <|im_end|>)
│   ├── QwenVLContinuousTokenBuilder (adds vision rendering + delta slicing)
│   └── MiMoVLContinuousTokenBuilder (adds vision rendering + content flattening)
└── (other text builders from PR #6779: MiniMax, GLM, Gemma4, GptOss)
```

VL builders **inherit** from `QwenContinuousTokenBuilder`, reusing its ChatML boundary logic while adding vision-specific methods.

### MergeResult Extension

```python
@dataclass(frozen=True)
class MergeResult:
    # Existing text fields (unchanged)
    token_ids: list[int]
    appended_token_count: int
    kind: MergeKind = "non_assistant"
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0

    # New MM fields (all have safe defaults → backward compatible)
    pixel_values: Any = None
    image_grid_thw: list[tuple[int, int, int]] = field(default_factory=list)
    image_token_spans: list[tuple[int, int]] = field(default_factory=list)
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
```

Text-only code continues to construct `MergeResult(token_ids=..., appended_token_count=...)` without change.

### VL Builder Core Methods

| Method | Purpose |
|--------|---------|
| `supports_multimodal()` | Class-level flag; controls gate logic in agent_loop |
| `render_tokens_with_mm(messages, images)` | Full processor render → (token_ids, mm_extras) |
| `build_initial_tokens(messages)` | Turn 1: processor render, caches `_last_mm_extras` |
| `merge_tokens(prev, updated, runtime_ids)` | If new images: full re-render + delta slice. If no new images: text-only incremental merge |
| `_slice_mm_delta(prev_count, full_extras)` | Slices `pixel_values` at raw patch boundary (`t*h*w`) and `image_grid_thw` |
| `extract_vision_placeholders(token_ids)` | Finds `<|vision_start|>...<|vision_end|>` spans |
| `count_vision_tokens(grid_row)` | Computes merged token count: `t*(h//merge)*(w//merge)` |

### Delta Slicing (Key Insight)

When a new image appears at turn N, `merge_tokens` re-renders the full message history through the processor (because VL processors are not incrementally callable), then slices:

```python
# pixel_values is indexed by RAW patches (t*h*w), not merged tokens
prev_patch_count = sum(row[0] * row[1] * row[2] for row in grid_thw[:prev_image_count])
delta_pixel_values = pixel_values[prev_patch_count:]
delta_grid_thw = grid_thw[prev_image_count:]
```

The MergeResult contains only the **delta** (new images' data), not the full history.

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

116 tests pass (76 CPU text + 40 MM-specific), covering:
- MergeResult construction, backward compat, immutability
- All text subclass boundary behaviors
- VL builder: build, merge, delta slice, placeholder extraction
- Wiring/factory: family resolution, auto-inference, VL processor gating

## Backward Compatibility

- `MergeResult` MM fields all have `default_factory` → existing text-only constructors work unchanged
- CT is disabled by default → no behavior change unless `continuous_token.enable = True`
- VL gates are `supports_multimodal()` checks → text-only builders never enter VL code paths
- No modifications to base PR #6779's existing code lines (only additions)

## Limitations

1. **Full re-render on new images** — When a new image appears, the entire message history is re-rendered through the processor. This is O(N) in history length per merge with new images, because HuggingFace VL processors are not incrementally callable.
2. **No GPU training validation yet** — Token-level correctness is verified, but a full GRPO training run (Qwen2.5-VL + geo3k, 50 steps) has not been performed.
3. **QwenVL / MiMoVL code duplication** — The two VL builders share ~200 lines of identical logic. A mixin extraction is planned as follow-up.

## References

- [LMSYS Blog: No Token Left Behind](https://blog.lmsys.org/2025-05-15-no-token-left-behind/)
- [verl RFC Issue #6719](https://github.com/verl-project/verl/issues/6719) — CT design rationale
- [verl PR #6779](https://github.com/volcengine/verl/pull/6779) — Text-only CT (base)
- [slime PR #1141](https://github.com/THUDM/slime/pull/1141) — Reference multimodal multi-turn implementation
