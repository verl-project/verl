Last updated: 07/05/2026.

### What does this PR do?

This PR extends `ContinuousTokenBuilder` to `VLContinuousTokenBuilder` and integrates it with `AgentLoop (SingleTurnAgentLoop + ToolAgentLoop)`.

What's added is that it now enables the **processor's** encoding of text with multimodal info. Multimodal info is **not** concatenated turn-by-turn; instead `AgentLoopWorker` postprocesses it by running the processor over the *full* message history to recover all multimodal tensors (e.g. `pixel_values`, `image_grid_thw`). The Continuous Token (CT) path only needs the processor to expand image placeholders into the correct number of pad tokens, while the heavy pixel tensors produced by those incremental calls are discarded and rebuilt once at the end. (This might not be the most efficient way and open to optimize, because this involve re-processing of multimodal info. Collect pixel tensors produced by those incremental calls and concate each turn manually avoid re-processing.)

What's unchanged is that each VL model family can still inherit its text model family's specific behavior (especially the `merge_token_id` / turn-boundary behavior of Qwen and GLM) on text. VL builders are composed via Python MRO from a `VLContinuousTokenMixin` (processor-backed rendering) plus a text-family builder (boundary handling), so e.g. `QwenVLContinuousTokenBuilder` reuses Qwen's ChatML newline reinsertion and `GLM46VContinuousTokenBuilder` reuses GLM's `<|observation|>` / `<|user|>` trim.

### Test
There are two types of test

1. Unit tests: `tests/utils/test_continuous_token_on_cpu.py` covers builder construction, family resolution / auto-inference, VL processor gating, placeholder expansion, and merge alignment.
2. A [script](https://github.com/gxlvera/verl/blob/97e1e1c705df96a585a3ce3f53f63bd75f05cadf/tests/experimental/agent_loop/continuous_token/compare_vl_agentloop_ct_vs_legacy.py) that tests the difference in the output (including response mask and logprob) between agentloop with CT enabled and disabled (legacy path) is as below. A full comparison result is in this [doc](https://github.com/gxlvera/verl/blob/97e1e1c705df96a585a3ce3f53f63bd75f05cadf/tests/experimental/agent_loop/continuous_token/vl_ct_vs_legacy_comparison_result.md).

```text
Family      Models  Runs  Pass  Mismatch  Error  Notes
Qwen VL     3       9     3     6         0      Instruct + Thinking; single-turn passes, tool cases mismatch because legacy misses the <|im_end|>\n newline. CT is correct.
MiMo VL     1       3     1     2         0      Shares the Qwen ChatML boundary; single-turn passes, tool cases mismatch (CT correct).
GLM VL      3       5     3     2         0      GLM-4.6V runs all (tool cases mismatch on a duplicated <|observation|>, CT correct); GLM-4V/4.5V single-turn only, single-turn passes.
Gemma-4 VL  1       3     1     2         0      Run explicitly (not a default model); single-turn passes, tool cases mismatch because the legacy gemma4 tool path drops image pads. CT is correct.
```

Every `CT vs Legacy` **mismatch** above is **expected and correct**: legacy renders each turn in isolation and drops/duplicates turn-boundary tokens (Qwen/MiMo miss the `<|im_end|>\n` newline; GLM double-counts the `<|observation|>` boundary token; the legacy gemma4 tool path drops tool-response image pads). `CT vs Full` is **EQUAL** in all cases, showing **CT is the side that matches canonical full encoding**; the inserted/trimmed boundary tokens carry `loss_mask=0` / `logprob=0.0`. See the linked result doc for the per-token diffs.

### API and Usage Example

> Demonstrate how the API changes if any, and provide usage example(s) if possible.

Continuous Token is now the **default and only** tokenization path for the agent loop — there is no `enable`/`disable` switch to set. The model family (boundary handling) and VL/text builder are auto-inferred from the model / tokenizer path; when a multimodal `processor` is available the agent loop resolves and instantiates the correct VL builder automatically. If a request carries multimodal inputs (images/videos/audios) but the resolved builder does **not** support multimodal **or** no processor is present, the agent loop raises loudly instead of silently falling back to a legacy path. Currently supported VL families are `qwen25vl`, `qwen3vl`, `glm4v` (GLM-4.6V full; GLM-4V/4.5V single-turn only), `gemma4vl`, and `mimovl`. `kimivl`, `minimaxvl`, and `deepseekvl2` have builders in the registry but are blocked by verl model loading (not a CTB limitation).

No config toggle is required (the former `data.continuous_token.enable` field has been removed). Multimodal support is decided purely by the resolved builder plus the presence of a processor:

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

- **`verl/utils/continuous_token.py`** — Added multimodal hooks to the base builder (`supports_multimodal()`, `render_tokens_with_mm(...)`, media-aware `build_initial_tokens(...)`), the `VLContinuousTokenMixin` (processor-backed rendering via the **processor** chat template, since some VL processors ship a different template), and the concrete VL builders (`QwenVL`, `MiMoVL`, `MiniMaxVL`, `Gemma4VL`, `GLM46V`, `KimiVL`, `DeepSeekVL2`, plus generic `VLContinuousTokenBuilder`).
- **`verl/utils/continuous_token_wiring.py`** — Registered the VL families with name-based auto-inference (matched before text families) and the `_TEXT_TO_VL_FAMILY` upgrade path; `create_continuous_token_builder(...)` now takes `processor=` and requires it for VL builders.
- **`verl/experimental/agent_loop/agent_loop.py`** — CT is always constructed (family auto-resolved); a shared `_assert_mm_supported(...)` guard raises whenever multimodal inputs are present but the builder is not VL-capable or the processor is missing; forwards media to `ct_build_initial_tokens(...)`; skips left-truncation for VL builders; and rebuilds full multimodal tensors from the final text + accumulated images in `_compute_multi_modal_inputs`. The legacy `apply_chat_template` instance method and system-prompt handling were removed.
- **`verl/experimental/agent_loop/tool_agent_loop.py`** — Carries tool-response images as real objects (`{"type": "image", "image": img}`), and always runs the CT merge path; new tool-response images trigger the `_assert_mm_supported(...)` guard before any state mutation. The legacy gpt-oss/gemma4/text fallbacks were removed.
- **`verl/experimental/agent_loop/single_turn_agent_loop.py`** — Same guard so VL CT covers single-turn multimodal prompts; the legacy branch was removed.

#### 2. VL Continuous Token model support

Officially supported in this PR:

| Family | Builder | Models |
|---|---|---|
| `qwen25vl` | `QwenVLContinuousTokenBuilder` | Qwen2.5-VL |
| `qwen3vl` | `QwenVLContinuousTokenBuilder` | Qwen3-VL / Qwen3-VL-MoE (incl. Thinking) |
| `mimovl` | `MiMoVLContinuousTokenBuilder` | MiMo-VL (Qwen2.5-VL arch) |
| `glm4v` | `GLM46VContinuousTokenBuilder` | GLM-4.6V (full tool loop); GLM-4V/GLM-4.1V, GLM-4.5V (single-turn only) |
| `gemma4vl` | `Gemma4VLContinuousTokenBuilder` | Gemma-4 (non-unified `gemma4` variant) |

The remaining VL families have CTB builders/scaffolding in the registry (`minimaxvl` / MiniMax-VL, `kimivl` / Kimi-VL, `deepseekvl2` / DeepSeek-VL2, and the generic `vldefault`) but are **not usable end-to-end yet because verl itself cannot load these models/processors — this is a verl loading limitation, not a CTB one.** They will be enabled once verl supports loading them. Models with non-standard processors are covered separately below.

**Nemotron VL is not supported**

`nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` uses an InternVL-style custom multimodal pipeline. Its `AutoProcessor` does **not** expose a standard HF `image_processor`, so `_is_multimodal_processor` returns `False` and wiring falls back to the text `ContinuousTokenBuilder` (there is also no Nemotron-VL builder in the registry). The standard VL CT path therefore does not cover this non-standard processor interface.

**Kimi-VL / MiniMax-VL / DeepSeek-VL2 are blocked by verl model loading**

Kimi-VL, MiniMax-VL, and DeepSeek-VL2 are **not supported yet, and the blocker is verl model loading — not the CT builder.** Each already has a builder in the registry (`kimivl` / `minimaxvl` / `deepseekvl2`), but verl itself currently cannot load these models/processors end-to-end, so they cannot be instantiated, validated in the harness, or run in a rollout. Once verl can load them, they can be enabled through the same auto-inference path with no CTB changes.
