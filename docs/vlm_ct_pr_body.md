### What does this PR do?

Last updated: 07/02/2026

This PR depends on https://github.com/verl-project/verl/pull/6779 (Continuous Token base infrastructure). It extends `ContinuousTokenBuilder` to `VLContinuousTokenBuilder` and integrates it with `ToolAgentLoop`.

What's added is that it now enables the **processor's** encoding of text with multimodal info. Multimodal info is **not** concatenated turn-by-turn; instead `AgentLoopWorker` postprocesses it by running the processor over the *full* message history to recover all multimodal tensors (e.g. `pixel_values`, `image_grid_thw`). The Continuous Token (CT) path only needs the processor to expand image placeholders into the correct number of pad tokens so the runtime token prefix stays bit-perfect (TITO), while the heavy pixel tensors produced by those incremental calls are discarded and rebuilt once at the end.

What's unchanged is that each VL model family can still inherit its text model family's specific behavior (especially the `merge_token_id` / turn-boundary behavior of Qwen and GLM) on text. VL builders are composed via Python MRO from a `VLContinuousTokenMixin` (processor-backed rendering) plus a text-family builder (boundary handling), so e.g. `QwenVLContinuousTokenBuilder` reuses Qwen's ChatML newline reinsertion and `GLM46VContinuousTokenBuilder` reuses GLM's `<|observation|>` / `<|user|>` trim.

### Test

> For changes that can not be tested by CI (e.g., algorithm implementation, new model support), validate by experiment(s) and show results like training curve plots, evaluation results, etc.

Because the correctness of multimodal CT is fundamentally about **token-level equivalence** (does incremental CT rendering produce the same token/pad-token stream as a single full re-encode?), it is validated with real model processors/tokenizers rather than a training run. No GPU or model weights are required — generation is mocked and only the processor/tokenizer is loaded.

Three comparison axes are used per VL model:

- **CT vs Full** — CT incremental concatenation vs. a single full-trajectory processor re-encode. This is the ground-truth check.
- **CT vs Legacy** — CT vs. the `ToolAgentLoop` path with CT disabled.
- **Multimodal encode correctness** — number of expanded image pad tokens in CT must equal that of the full re-encode.

Currently, multimodal CT officially supports **Qwen2.5-VL, Qwen3-VL, GLM-4V, and MiMo-VL**. Other multimodal models are **not supported yet and will be supported soon**.

Results for the supported models (`tests/experimental/agent_loop/continuous_token/vl_models_ct_report.md`):

| Model | Resolved builder | Image pad expanded | CT vs Legacy | CT vs Full | MM encode correct |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | `QwenVLContinuousTokenBuilder` | ✅ | ⚠️ DIFFER (expected) | ✅ EQUAL | ✅ |
| `Qwen/Qwen3-VL-2B-Instruct` | `QwenVLContinuousTokenBuilder` | ✅ | ⚠️ DIFFER (expected) | ✅ EQUAL | ✅ |
| `zai-org/GLM-4.1V-9B-Thinking` | `GLM46VContinuousTokenBuilder` | ✅ | ⚠️ DIFFER (expected) | ✅ EQUAL | ✅ |
| `XiaomiMiMo/MiMo-VL-7B-RL` | `MiMoVLContinuousTokenBuilder` | ✅ | ⚠️ DIFFER (expected) | ✅ EQUAL | ✅ |

The `CT vs Legacy` **DIFFER** cases are **expected and correct**: legacy renders each turn in isolation and drops/duplicates turn-boundary tokens (Qwen/MiMo miss the `<|im_end|>\n` newline; GLM double-counts the `<|observation|>` boundary token). `CT vs Full` being **EQUAL** shows that **CT is the side that matches canonical full encoding**, and the inserted/trimmed boundary tokens carry `loss_mask=0` / `logprob=0.0`. See `tests/experimental/agent_loop/continuous_token/vl_ct_vs_legacy_comparison_result.md` for the per-token diffs.

Unit tests: `tests/utils/test_continuous_token_on_cpu.py` covers builder construction, family resolution / auto-inference, VL processor gating, placeholder expansion, and merge alignment.

**Known test gap (TODO):** the current mock trajectories never exercise a non-empty `reasoning_content` on assistant turns — every mocked assistant message has empty/absent reasoning. A follow-up should add a case with non-empty `reasoning_content` to verify CT rendering / boundary handling stays token-equivalent when reasoning is present.

### API and Usage Example

> Demonstrate how the API changes if any, and provide usage example(s) if possible.

No user-facing config change is required beyond the existing CT switch. When `rollout.multi_turn.continuous_token.enable=True` and a multimodal `processor` is available, the agent loop automatically resolves and instantiates the correct VL builder (`model_family=auto`), or you may pin a family explicitly. Currently supported VL families are `qwen25vl`, `qwen3vl`, `glm4v`, and `mimovl`; other multimodal families are not supported yet and will be supported soon.

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      continuous_token:
        enable: true
        model_family: auto   # or an explicit VL family, e.g. qwen25vl
```

```python
# The factory picks a VL builder when a multimodal processor is supplied.
from verl.utils.continuous_token_wiring import create_continuous_token_builder

builder = create_continuous_token_builder(
    tokenizer,
    model_family="auto",           # -> QwenVLContinuousTokenBuilder for Qwen2.5-VL
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    processor=processor,           # REQUIRED for VL families
    mm_processor_kwargs={"min_pixels": ..., "max_pixels": ...},  # builder-level constant
)

# Turn 1: expand image placeholders through the processor. Processor kwargs are the
# builder-level constant captured above, not passed per render.
prompt_ids = builder.build_initial_tokens(
    messages,
    tools=tool_schemas,
    images=images,                 # list of PIL images / refs
)

# Later turns: incremental merge keeps the runtime prefix bit-perfect,
# including newly appended tool-response images.
merge_result = builder.merge_non_assistant_tokens(
    previous_messages, updated_messages, runtime_token_ids, tools=tool_schemas,
)
```

### Design & Code Changes

> Demonstrate the high-level design if this PR is complex, and list the specific changes.

High-level design: VL builders keep the `MergeResult` contract **token-only**. The processor is used solely to render the correct token IDs (with image placeholders expanded); pixel tensors from incremental merges are intentionally discarded. `AgentLoopWorker` accumulates the original image objects and, during postprocessing (`_compute_multi_modal_inputs`), re-runs the processor over the full final text + complete image list to produce training-ready `pixel_values` / `image_grid_thw`. This avoids re-encoding all images every turn while still guaranteeing the TITO invariant.

#### 1. Main code changes

- **`verl/utils/continuous_token.py`**
  - Added multimodal hooks to the base `ContinuousTokenBuilder`: `supports_multimodal()` (class flag, default `False`), `render_tokens_with_mm(...)` (default raises `NotImplementedError`), and extended `build_initial_tokens(...)` to accept and ignore `images`/`videos`/`audios`/`mm_processor_kwargs` for text builders.
  - Added `VLContinuousTokenMixin`: processor-backed rendering (`render_tokens_with_mm`, `_render_tokens`, `build_initial_tokens`), media extraction from OpenAI-style content blocks, and `supports_multimodal() -> True`. Rendering goes through the **processor** chat template (not the tokenizer template) because some VL processors ship a different template (e.g. MiMo-VL's tokenizer template cannot render list-of-blocks content).
  - Added concrete VL builders composed via MRO: `VLContinuousTokenBuilder` (generic), `QwenVLContinuousTokenBuilder`, `MiMoVLContinuousTokenBuilder`, `MiniMaxVLContinuousTokenBuilder`, `Gemma4VLContinuousTokenBuilder`, `GLM46VContinuousTokenBuilder`, `KimiVLContinuousTokenBuilder`, and `DeepSeekVL2ContinuousTokenBuilder`.
- **`verl/utils/continuous_token_wiring.py`**
  - Added VL families to `ContinuousTokenModelFamily` and the builder registry; added name-based auto-inference for VL models (matched **before** their text families).
  - Added the `_TEXT_TO_VL_FAMILY` upgrade path: a unified text+vision checkpoint whose name has no `vl` marker (generic `default`, or `gemma4`) is upgraded to its VL counterpart when a multimodal processor is supplied.
  - `create_continuous_token_builder(...)` now accepts `processor=` and requires it for VL builders (`_is_multimodal_processor` checks for `image_processor`).
- **`verl/experimental/agent_loop/agent_loop.py`**
  - CT init no longer hard-fails when a processor is present; it now resolves the family first and only fails if the resolved builder is VL **and** the processor is missing. The processor is threaded into the builder.
  - `ct_build_initial_tokens(...)` forwards `images`/`videos`/`audios`/`mm_processor_kwargs` to the builder; renamed merge entrypoints to `merge_non_assistant_tokens` / `merge_assistant_tokens`.
  - `_cap_text_prompt_length` skips left-truncation for VL builders to avoid splitting vision token spans.
  - `_compute_multi_modal_inputs` rebuilds full multimodal tensors from the final decoded text and the complete accumulated image list.
- **`verl/experimental/agent_loop/tool_agent_loop.py`**
  - Passes multimodal inputs to `ct_build_initial_tokens`.
  - Tool-response images are carried as real objects in message content (`{"type": "image", "image": img}`) instead of bare `{"type": "image"}` placeholders.
  - The CT merge path now runs for VL builders **even when new images arrive this turn** (`not new_images OR builder.supports_multimodal()`), and appends the new image objects to `agent_data.image_data` for final postprocessing.
- **`verl/experimental/agent_loop/single_turn_agent_loop.py`**
  - Same gating update so VL CT is used for single-turn multimodal prompts, forwarding media to `ct_build_initial_tokens`.

#### 2. VL Continuous Token model support

Officially supported in this PR:

| Family | Builder | Models | Text-boundary behavior inherited |
|---|---|---|---|
| `qwen25vl` | `QwenVLContinuousTokenBuilder` | Qwen2.5-VL | Qwen ChatML `<|im_end|>\n` newline reinsertion |
| `qwen3vl` | `QwenVLContinuousTokenBuilder` | Qwen3-VL / Qwen3-VL-MoE | Qwen ChatML `<|im_end|>\n` newline reinsertion |
| `glm4v` | `GLM46VContinuousTokenBuilder` | GLM-4V (e.g. GLM-4.1V) | GLM `<|observation|>` / `<|user|>` trim |
| `mimovl` | `MiMoVLContinuousTokenBuilder` | MiMo-VL (Qwen2.5-VL arch) | Qwen ChatML boundary |

All other multimodal models are **not supported yet and will be supported soon.** This includes builders that already have scaffolding in the registry but are not yet validated end-to-end (`minimaxvl` / MiniMax-VL-01, `kimivl` / Kimi-VL, `gemma4vl` / Gemma4, `deepseekvl2` / DeepSeek-VL2, and the generic `vldefault`), as well as models with non-standard processors (see below).

#### 3. Nemotron VL is not supported

`nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` uses an InternVL-style custom multimodal pipeline. Its `AutoProcessor` does **not** expose a standard HF `image_processor`, so `_is_multimodal_processor` returns `False` and wiring falls back to the text `ContinuousTokenBuilder` (there is also no Nemotron-VL builder in the registry). The standard VL CT path therefore does not cover this non-standard processor interface.

#### 4. Kimi-VL multimodal encode limitation (not yet fixed)

Kimi-VL is **not yet supported**. Although its clean turn boundaries make the token sequences match in the harness, its **multimodal encode has a known problem that has not been fixed yet**: the Kimi-VL processor's image-placeholder expansion does not integrate cleanly with the incremental CT merge path in all cases. Kimi-VL multimodal CT support will be added once this is resolved.
