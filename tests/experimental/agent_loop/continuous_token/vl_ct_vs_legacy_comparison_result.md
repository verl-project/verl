# VL Continuous Token (CT) vs Legacy Agent Loop Comparison Results

This file records the current CT-vs-legacy end-to-end alignment results for the
vision-language (VL) model set used by
`compare_vl_agentloop_ct_vs_legacy.py`. Like its text sibling
(`compare_agentloop_ct_vs_legacy.py`), this is not a unit test but a script that
compares the trajectory and metadata (`prompt_ids`, `response_ids`, `loss_mask`,
`logprobs`) produced by the real agent loop with the VL Continuous Token builder
enabled (CT) versus disabled (legacy).

The difference from the text harness is that every trajectory embeds real
images — both in the initial **user prompt** and inside **tool responses** — and
every render goes through the model's multimodal `processor` (not the bare
tokenizer), so vision placeholders expand into the same per-image pad-token
spans the rollout backend consumes.

## Setup

Default trajectories:

```text
vl_singleturnchat        SingleTurnAgentLoop   image in user prompt
vl_multiturnsingletool   ToolAgentLoop         image in user prompt + 1 tool image
vl_multiturnmultitool    ToolAgentLoop         image in user prompt + 2 parallel tool images
```

The trajectories live in the shared fixtures module
`scripts/chat_template_mock_trajectories.py` (`build_vl_trajectories`, alongside
the text mock trajectories) and this script imports them. They are kept as a
separate VL set (not in the text `TRAJECTORIES` tuple) because the text
chat-template checker renders trajectories with a bare tokenizer and cannot
expand image content; images are also built lazily so text-only imports never
require Pillow. Small solid-color PIL images are used (84×84 prompt image, 56×56
tool crops) so token counts stay small and CPU image processing stays fast.

A deterministic server is used as the inference backend for the AgentLoop
instead of calling SGLang/vLLM. The assistant output token ids are generated
from the mock trajectory using the tested model's own **processor** chat
template. For each assistant turn:

1. render `prefix_messages` with `add_generation_prompt=True` through the
   processor (with the images available so far);
2. render `prefix_messages + [assistant_message]` with
   `add_generation_prompt=False` through the processor;
3. take the token-id suffix of the full render after the prompt render;
4. to mimic the model stopping at EOS, truncate after the final EOS token so the
   template whitespace that follows it (e.g. a trailing `\n`) is not emitted as
   part of the generated response. GLM is the exception: its template does not
   emit an EOS at the assistant turn boundary, so a tool-call turn is terminated
   with the canonical `<|observation|>` stop token (matching real GLM rollout).

The image placeholders live in the shared prefix of both renders, so they cancel
in the suffix diff; the suffix is the pure assistant text/tool-call token
stream.

For **thinking** editions (e.g. Qwen3-VL-*-Thinking) the template opens a
`<think>` block at the generation prompt. A content-only assistant message then
renders an *empty* `<think>\n\n</think>\n\n` block whose `\n\n` tokenizes
differently from the generation-prompt `<think>\n`, so a naive suffix diff cannot
align. Rather than injecting synthetic `<think>` content into the mock message,
the assistant message is rendered **normally** and the extraction is reconciled
purely on the prompt side: the empty think block is folded into the prompt render
so the suffix diff realigns (`_replace_qwen_generation_think_open_with_empty_block`,
the Qwen analogue of the existing GLM/Nemotron/MiniMax prompt rewrites). The
extracted assistant output is therefore the plain content/tool-call stream, and
no fabricated reasoning is added.

VL chat templates commonly render assistant `content` verbatim but **drop** a
structured `tool_calls` field (Qwen2.5-VL, Qwen3-VL, MiMo-VL all do).
So a tool-call turn is mocked by embedding the parser's raw tool-call text as
assistant `content` (Hermes `<tool_call>…</tool_call>`, GLM `<tool_call>name
<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`). This is what a VL
model actually emits at rollout time and lets the real tool parser extract the
call so the loop executes the deterministic image-returning tool.

**Gemma-4 is the exception to the raw-text embedding.** Its template renders a
tool response *only* via a forward-scan launched from an assistant carrying a
*structured* `tool_calls` field, and has no standalone `role="tool"` branch. A
bare tool message whose preceding assistant is raw text is silently dropped, so
the raw-text scheme above cannot build a faithful Gemma prefix. For Gemma the
ground-truth builder therefore mirrors production `ToolAgentLoop`: the running
history uses a structured assistant (`content` = reasoning + `tool_calls`) and
tool messages stamped with `tool_call_id`/`name`, while the per-turn *output*
delta is still taken from the raw-text assistant the model emits
(`_gemma_structured_assistant` / `_gemma_stamped_tool_messages`).

Tool responses are produced by deterministic `FunctionTool` instances that
return a `ToolResponse` with `image=[...]` + text; the real AgentLoop state
machine appends them exactly as it would in production.

Pass criteria:

```text
Flow is complete, and CT/legacy have identical prompt_ids, response_ids,
loss_mask, and logprobs.
```

## Default Models

```python
DEFAULT_MODELS = [
    # Qwen VL
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-2B-Thinking",
    # MiMo VL (Qwen2.5-VL architecture)
    "XiaomiMiMo/MiMo-VL-7B-RL",
    # GLM vision. GLM-4.6V runs all trajectories; GLM-4V (4.1V) and GLM-4.5V are
    # single-turn only (their templates mishandle tool-role images) and route to
    # the same GLM46VContinuousTokenBuilder.
    "zai-org/GLM-4.1V-9B-Thinking",
    "zai-org/GLM-4.5V",
    "zai-org/GLM-4.6V",
]
```

GLM-4V (GLM-4.1V) and GLM-4.5V are restricted to `vl_singleturnchat` by the
harness (see the GLM VL section); GLM-4.6V runs all three trajectories. So the
default run is `4×3 + 1 + 1 + 3 = 17` cases, not `7×3`.

## Summary

```text
default_models: 7
default_trajectory_runs: 17
pass: 7
mismatch: 10
error: 0
```

## Results

```text
Family      Models  Runs  Pass  Mismatch  Error  Notes
Qwen VL     3       9     3     6         0      Instruct + Thinking; single-turn passes, tool cases mismatch because legacy misses the newline after assistant EOS. CT is correct.
MiMo VL     1       3     1     2         0      Same as Qwen VL: shares the Qwen ChatML boundary; single-turn passes, tool cases mismatch (CT correct).
GLM VL      3       5     3     2         0      GLM-4.6V runs all (tool cases mismatch on a duplicated <|observation|> boundary token, CT correct); GLM-4V/4.5V single-turn only (tool loop unsupported), single-turn passes.
```

## Qwen VL

Models covered:

```text
Qwen/Qwen2.5-VL-3B-Instruct
Qwen/Qwen3-VL-2B-Instruct
Qwen/Qwen3-VL-2B-Thinking
```

Result:

```text
total: 9
pass: 3
mismatch: 6
error: 0
```

```text
Model group        vl_singleturnchat  vl_multiturnsingletool  vl_multiturnmultitool
Qwen2.5-VL         pass               mismatch                mismatch
Qwen3-VL-Instruct  pass               mismatch                mismatch
Qwen3-VL-Thinking  pass               mismatch                mismatch
```

The Thinking edition produces the exact same boundary result as Instruct. Its
chat template opens a `<think>` block at the generation prompt, so a
content-only assistant message renders an empty `<think>\n\n</think>\n\n` block
whose `\n\n` tokenizes differently from the generation-prompt `<think>\n` and
breaks a naive suffix diff. The harness renders the assistant message normally
and reconciles this on the prompt side (folds the empty think block into the
prompt render, the Qwen analogue of the GLM/Nemotron/MiniMax prompt rewrites);
no synthetic `<think>` content is injected. The CT-vs-legacy comparison then
exercises the same Qwen ChatML boundary as Instruct (see below).

Qwen VL uses `tool_parser=hermes` and the shared `QwenVLContinuousTokenBuilder`
(Qwen ChatML boundary handling + VL processor rendering). The single-image chat
case passes: `build_initial_tokens` renders the initial image prompt through the
processor identically to the legacy processor render.

The tool-agent mismatches are the expected Qwen ChatML boundary. After the
assistant turn ends at `<|im_end|>` (id `151645`), the full `apply_chat_template`
render of the next (tool-response) turn is:

```text
<|im_end|>\n<|im_start|>user\n<tool_response>...
```

Legacy renders the tool-response turn in isolation and appends it directly after
the assistant `<|im_end|>`, missing the newline:

```text
<|im_end|><|im_start|>user\n<tool_response>...
```

CT (`QwenVLContinuousTokenBuilder._merge_non_assistant_token_ids`) restores the
canonical `\n` (id `198`) at the boundary before the tool-response header, so CT
is one token longer and matches the full render. The inserted newline carries
`loss_mask=0` and `logprob=0.0`, and the tool-response image pad tokens are
identical on both sides. **CT is the correct side.**

This is the same class of mismatch documented for the text Qwen tool cases; it
is present here for tool responses that additionally carry images, confirming
the image pad-token spans do not shift the boundary correction.

## MiMo VL

Models covered:

```text
XiaomiMiMo/MiMo-VL-7B-RL
```

Result:

```text
total: 3
pass: 1
mismatch: 2
error: 0
```

```text
Model group  vl_singleturnchat  vl_multiturnsingletool  vl_multiturnmultitool
MiMo-VL      pass               mismatch                mismatch
```

MiMo-VL shares the Qwen2.5-VL architecture and ChatML template, so it uses
`tool_parser=hermes` and `MiMoVLContinuousTokenBuilder` (Qwen boundary handling
via the processor chat template). Results are identical in character to Qwen VL:
single-turn passes, and the tool cases show the same `<|im_end|>\n<|im_start|>`
newline boundary that legacy drops and CT restores (`loss_mask=0`,
`logprob=0.0`). **CT is the correct side.**

## GLM VL

Models covered:

```text
zai-org/GLM-4.6V                                 all trajectories
zai-org/GLM-4.1V-9B-Thinking   (GLM-4V family)   single-turn only
zai-org/GLM-4.5V                                 single-turn only
```

Result:

```text
total: 5
pass: 3
mismatch: 2
error: 0
```

```text
Model group  vl_singleturnchat  vl_multiturnsingletool  vl_multiturnmultitool
GLM-4.6V     pass               mismatch                mismatch
GLM-4.1V     pass               n/a (unsupported)       n/a (unsupported)
GLM-4.5V     pass               n/a (unsupported)       n/a (unsupported)
```

All three GLM vision models route to `tool_parser=glm` and the same
`GLM46VContinuousTokenBuilder` (GLM observation/user boundary trim + VL processor
rendering), and all three load through verl's `hf_processor` (GLM-4.1V and
GLM-4.5V use `Glm4vProcessor`; GLM-4.6V uses `Glm46VProcessor`). Every
single-image chat case passes: `build_initial_tokens` renders the initial image
prompt through the processor identically to the legacy processor render (CT ==
legacy == full, prompt image expands to the same `<|image|>` pad span on all
sides).

> **Tool-agent-loop support.** Only **GLM-4.6V** can run the tool agent loop —
> it renders tool-role images correctly
> (`<|begin_of_image|>…<|image|>…<|end_of_image|>`). **GLM-4V (GLM-4.1V-9B-Thinking)
> and GLM-4.5V are single-turn only**: their templates mishandle tool-role images
> (GLM-4.1V drops `role="tool"` messages entirely; GLM-4.5V serializes the tool
> message's multimodal `content` list as a Python string instead of expanding the
> image). They are still fully supported for the **single-turn agent loop** via
> the same builder, so the harness restricts them to `vl_singleturnchat` (see
> `SINGLE_TURN_ONLY_MODEL_MARKERS` in the script).

GLM-4.6V correctly expands tool-role images and runs all three trajectories. The
single-turn case passes; both tool cases mismatch, and this is exactly the
boundary `GLM46VContinuousTokenBuilder` is designed to correct.

GLM does not terminate assistant turns with an EOS token in the template render;
after a tool call the model canonically stops at `<|observation|>` (id `151338`),
and the following tool-response turn's own render *also* begins with
`<|observation|>`. So the raw assistant generation ends with the boundary token:

```text
...</tool_call><|observation|>
```

Legacy attributes that trailing `<|observation|>` to the assistant response with
`loss_mask=1`, then appends the tool-response turn which re-emits its own
`<|observation|>` header — the boundary token is duplicated and mis-attributed as
generated:

```text
# vl_multiturnsingletool
response_ids: legacy_len=63 ct_len=62 first_mismatch=24 legacy_value=151338 ct_value=198
loss_mask:    legacy_len=63 ct_len=62 first_mismatch=23 legacy_value=1      ct_value=0
logprobs:     legacy_len=63 ct_len=62 first_mismatch=23 legacy_value=-0.0105 ct_value=0.0

# vl_multiturnmultitool
response_ids: legacy_len=93 ct_len=92 first_mismatch=33 legacy_value=151338 ct_value=198
loss_mask:    legacy_len=93 ct_len=92 first_mismatch=32 legacy_value=1      ct_value=0
logprobs:     legacy_len=93 ct_len=92 first_mismatch=32 legacy_value=-0.0114 ct_value=0.0
```

CT (`GLM46VContinuousTokenBuilder._merge_non_assistant_token_ids`) trims the
trailing `<|observation|>` from the assistant output before appending the
canonical tool-response block, so the boundary token appears exactly once and is
never counted as a generated token. CT is therefore one token shorter, and at
the boundary the next token is the tool-response `\n` (id `198`) rather than the
duplicated `<|observation|>`. The trimmed token carried `loss_mask=1` in legacy
and is correctly dropped by CT.

**CT matches the canonical one-shot render.** Rendering the entire trajectory in a
single `processor.apply_chat_template(..., tools=trajectory.tool_schemas,
add_generation_prompt=False)` call (with image pads expanded) and comparing it to
the CT sequence (`prompt_ids + response_ids`) is **bitwise identical**:

```text
Trajectory               CT len / Full len   equal   <|observation|>   <|image|>
vl_multiturnsingletool   261 / 261           True    1 / 1             18 / 18
vl_multiturnmultitool    358 / 358           True    1 / 1             27 / 27
```

The full render contains exactly **one** `<|observation|>` per tool round; CT
reproduces it exactly, while legacy carries a second (duplicated) one — hence the
CT-vs-legacy mismatch. **CT is the correct side.** The image pad-token spans from
the tool responses are identical on both sides.

## Gemma-4 VL

Models covered:

```text
google/gemma-4-E4B-it    all trajectories (run explicitly; not in DEFAULT_MODELS)
```

Only the non-unified ``gemma4`` variant is supported: its ``AutoProcessor``
resolves to a real ``Gemma4Processor`` (with a ``Gemma4ImageProcessor``). The
``gemma4_unified`` variant (e.g. ``gemma-4-12B-it``) needs a
``Gemma4UnifiedProcessor`` that the installed transformers does not ship, so it
is not supported here.

Result:

```text
total: 3
pass: 1
mismatch: 2
error: 0
```

```text
Model group  vl_singleturnchat  vl_multiturnsingletool  vl_multiturnmultitool
Gemma-4      pass               mismatch                mismatch
```

Gemma-4 uses `tool_parser=gemma4` and the `Gemma4VLContinuousTokenBuilder`
(inherits the Gemma text builder's dummy-assistant + incremental tool-group
rendering, plus VL processor rendering). The single-image chat case passes:
`build_initial_tokens` renders the initial image prompt through the processor
identically to the legacy processor render.

The two tool cases mismatch, and **CT is the correct side**. Gemma renders a
tool response only when it follows a structured `tool_calls` assistant
(forward-scan embed, template lines 262–314). The CT builder satisfies this via
its dummy synthetic assistant, so it expands each tool-response image into the
full `<|image>` + 256×`<|image|>` + `<image|>` pad span. The **legacy** path does
not: `ToolAgentLoop`'s `gemma4` branch manually formats tool responses as
text-only and **drops the image pads entirely**
(`tool_agent_loop.py`, the `elif self.tool_parser_name == "gemma4"` branch only
joins `type == "text"` parts). So legacy emits the tool-response text and jumps
straight to the next turn, while CT inserts the image block:

```text
# vl_multiturnsingletool (one tool image)
response_ids: legacy_len=58  ct_len=316  first_mismatch=45  legacy_value=818('The')            ct_value=255999('<|image>')

# vl_multiturnmultitool (two parallel tool images)
response_ids: legacy_len=89  ct_len=605  first_mismatch=56  legacy_value=50('<|tool_response>') ct_value=255999('<|image>')
```

For the single tool image the gap is exactly one 56×56 crop
(`316 − 58 = 258` = `<|image>` + 256 pads + `<image|>`); for two parallel tool
images it is two such spans (`605 − 89 = 516`). At the first divergence CT begins
the tool-response image block (`<|image>`) while legacy has already skipped it —
continuing to the final answer (`The…`) in the single case, or to the second
tool's text response (`<|tool_response>`) in the parallel case. The dropped image
pads carry `loss_mask=0` / `logprob=0.0` on the CT side.

This mismatch is **expected**: the `Gemma4VLContinuousTokenBuilder` faithfully
renders tool-response images (matching the model's own multimodal chat template),
whereas the legacy `gemma4` tool path is text-only and lossy for image-bearing
tool responses. Gemma-4 is therefore *not* in `SINGLE_TURN_ONLY_MODEL_MARKERS`
(unlike GLM-4V/4.5V, whose templates cannot render tool-role images at all and
which the CT builder cannot work around).

## Reproducing

```bash
# Requires the model processors/tokenizers (auto-downloaded from HF cache).
# No GPU or model weights are needed: generation is mocked and only the
# processor/tokenizer are loaded.
python tests/experimental/agent_loop/continuous_token/compare_vl_agentloop_ct_vs_legacy.py \
    --allow-download \
    --ledger /tmp/vl_ct_ledger.json \
    --mismatch-jsonl /tmp/vl_ct_mismatch.jsonl

# A single model / trajectory:
python tests/experimental/agent_loop/continuous_token/compare_vl_agentloop_ct_vs_legacy.py \
    --allow-download \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --trajectory vl_multiturnsingletool

# Gemma-4 (not a default model; run explicitly; non-unified gemma4 variant only):
python tests/experimental/agent_loop/continuous_token/compare_vl_agentloop_ct_vs_legacy.py \
    --allow-download \
    --model google/gemma-4-E4B-it
```
