# Continuous Token (CT) vs Legacy Agent Loop Comparison Results

This file records the current CT-vs-legacy end-to-end tokenizer alignment
results for the default model set used by
`compare_agentloop_ct_vs_legacy.py`. Note that the `compare_agentloop_ct_vs_legacy.py` is not a unit test but a script that compares the trajecotory and metadata (response mask, logprob) between CT and legacy behavior.

## Setup

Default trajectories:

```text
singleturnchat          SingleTurnAgentLoop
multiturnsingletool     ToolAgentLoop
multiturnmultitool      ToolAgentLoop
```
A deterministic server is used as inference backend for AgentLoop for this comparison instead of
calling SGLang/vllm. The assistant output token ids are generated from the above mock trajectories. The trajectories are stored in messages format, so the tested model's own tokenizer and chat template are used to render the **token-id** for each assistant turn. Tool responses are produced
by deterministic tools and are appended by the real AgentLoop state machine.

Pass criteria:

```text
Flow is complete, and CT/legacy have identical prompt_ids, response_ids,
loss_mask, and logprobs.
```

## Default Models

```python
DEFAULT_MODELS = [
    # Qwen
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-0.8B",
    # GLM
    "zai-org/GLM-4.7",
    "zai-org/GLM-4.7-Flash",
    "zai-org/GLM-5",
    "zai-org/GLM-5.1",
    # Kimi.
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.5",
    "moonshotai/Kimi-K2.6",
    # ByteDance Seed.
    "ByteDance-Seed/Seed-OSS-36B-Instruct",
    # MiniMax
    "MiniMaxAI/MiniMax-M2",
    "MiniMaxAI/MiniMax-M2.5",
    "MiniMaxAI/MiniMax-M2.7",
    # MiMo
    "XiaomiMiMo/MiMo-7B-SFT",
    "XiaomiMiMo/MiMo-7B-RL",
    # Nemotron
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    # Gemma-4 (non-unified ``gemma4`` variant only; ``gemma4_unified`` is unsupported)
    "google/gemma-4-E4B-it",
    # GPT-OSS
    "openai/gpt-oss-20b"
]
```

DeepSeek is still WIP and is intentionally not in `DEFAULT_MODELS`. The checked
DeepSeek tokenizer family does not expose a Jinja `chat_template` compatible
with this `apply_chat_template`-based e2e harness. To support it, verl's
`apply_chat_template` wrapper needs to recognize DeepSeek models and call the
official DeepSeek `encoding` script to render/tokenize messages, instead of
relying on `tokenizer.apply_chat_template`.

## Summary

```text
default_models: 28
default_trajectory_runs: 84
pass: 40
mismatch: 38
error: 6
```

## Results

```text
Family        Models  Runs  Pass  Mismatch  Error  Notes
Qwen          10      30    10    20        0      Single-turn passes; tool cases mismatch because legacy misses the newline after assistant EOS. CT is correct.
GLM           4       12    4     8         0      Single-turn CT-vs-legacy passes; tool cases mismatch on the observation boundary. CT is correct.
Kimi          3       9     9     0         0      All default trajectories pass with tool_parser=kimi. 
Seed-OSS      1       3     3     0         0      All default trajectories pass with tool_parser=seed. 
MiniMax       3       9     3     0         6      Single-turn passes; tool cases are legacy-only TemplateError. 
MiMo          2       6     2     4         0      Single-turn passes; tool cases mismatch on the same post-EOS newline boundary as Qwen. CT is correct.
Nemotron      3       9     3     6         0      Single-turn passes; tool cases mismatch because legacy misses the tool-response user header. CT is correct.
Gemma-4       1       3     3     0         0      All default trajectories pass with tool_parser=gemma4 (prompt-side thought-channel trim + <turn|> / <|tool_response> boundary reconciliation). CT matches legacy.
GPT-OSS       1       3     3     0         0      All pass with tool_parser=gpt-oss. Requires harmony append-only reconstruction (keep prior-turn CoT) and emitting tool calls in the model's real layout (recipient in channel + <|constrain|>json) so the parser can extract them. CT matches legacy.
```

## Qwen

Models covered:

```text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-72B-Instruct
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
Qwen/Qwen3-4B
Qwen/Qwen3-4B-Instruct-2507
Qwen/Qwen3-4B-Thinking-2507
Qwen/Qwen3-8B
Qwen/Qwen3.5-0.8B
```

Result:

```text
total: 30
pass: 10
mismatch: 20
error: 0
```

```text
Model group       singleturnchat  multiturnsingletool  multiturnmultitool
Qwen2.5           pass            mismatch             mismatch
Qwen3             pass            mismatch             mismatch
Qwen3-Thinking    pass            mismatch             mismatch
Qwen3.5           pass            mismatch             mismatch
```

Qwen2.5 uses `tool_parser=hermes`; Qwen3.5 uses the existing `qwen3_coder`
parser. Qwen2.5, Qwen3, and Qwen3.5 all use the shared Qwen CT builder because
their ChatML templates share the same boundary behavior. The tool-agent
mismatches are expected:
after assistant EOS, full `apply_chat_template` contains:

```text
<|im_end|>\n<|im_start|>
```

Legacy appends the next rendered turn directly after `<|im_end|>`, so it misses
the newline boundary. CT restores that canonical newline. The inserted newline
has `loss_mask=0` and `logprob=0.0`.

### Qwen3 Thinking template (`<think>` prompt-side reconciliation)

`Qwen3-4B-Thinking-2507` is the dedicated thinking edition (unlike the dense
`Qwen3-*` / `Qwen3-4B-Instruct-2507`, which do **not** open a think block at the
generation prompt). Its template appends `<think>\n` to the generation prompt,
while a plain (reasoning-free) assistant message renders an *empty* block
`<think>\n\n</think>\n\n`. The generation-prompt `\n` (token `198`) and the empty
block's `\n\n` (token `271`) tokenize differently, so the assistant-output suffix
diff cannot align directly. Rather than injecting synthetic `<think>` content
into the mock message, the assistant turn is rendered **normally** and the
extraction is reconciled purely on the prompt side: the trailing `<think>\n` is
folded into the empty block `<think>\n\n</think>\n\n`
(`_replace_qwen_generation_think_open_with_empty_block`, the Qwen analogue of the
existing GLM/Nemotron/MiniMax prompt rewrites). The extracted assistant output is
therefore the plain content/tool-call stream (no fabricated reasoning), and the
CT-vs-legacy tool-case mismatch is the same post-EOS newline boundary as the rest
of the Qwen family — CT is the correct side.

## GLM

Models covered:

```text
zai-org/GLM-4.7
zai-org/GLM-4.7-Flash
zai-org/GLM-5
zai-org/GLM-5.1
```

Result:

```text
total: 12
pass: 4
mismatch: 8
error: 0
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
GLM          pass            mismatch             mismatch
```

GLM uses `tool_parser=glm`. 

Current GLM comparison result:

```text
CT vs legacy:
match: 4 / 12
mismatch: 8 / 12
error: 0 / 12
```

For CT-vs-legacy, the tool-case mismatch is the observation boundary. Full
`apply_chat_template` renders:

```text
</tool_call><|observation|><tool_response>
```

Legacy keeps the assistant output's stop observation and then appends another
observation from the rendered tool response:

```text
</tool_call><|observation|><|observation|><tool_response>
```

CT removes the ambiguous duplicate boundary and matches the canonical
observation boundary. The retained CT observation boundary tokens have `loss_mask=0` and
`logprob=0.0`.

## Kimi

Models covered:

```text
moonshotai/Kimi-K2-Instruct
moonshotai/Kimi-K2.5
moonshotai/Kimi-K2.6
```

Result:

```text
total: 9
pass: 9
mismatch: 0
error: 0
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
Kimi         pass            pass                 pass
```

All pass with `tool_parser=kimi`

## ByteDance Seed

Models covered:

```text
ByteDance-Seed/Seed-OSS-36B-Instruct
```

Result:

```text
total: 3
pass: 3
mismatch: 0
error: 0
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
Seed-OSS     pass            pass                 pass
```

All pass with `tool_parser=seed`. 

## MiniMax

Models covered:

```text
MiniMaxAI/MiniMax-M2
MiniMaxAI/MiniMax-M2.5
MiniMaxAI/MiniMax-M2.7
```

Result:

```text
total: 9
pass: 3
mismatch: 0
error: 6
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
MiniMax      pass            legacy-only error    legacy-only error
```

MiniMax tool-agent cases are legacy-only errors. The MiniMax template requires
each `role=tool` message to be preceded by an assistant message with
`tool_calls`. Legacy renders isolated `add_messages` containing only tool
messages, so it raises:

```text
Message has tool role, but there was no previous assistant message with a tool call!
```

CT is the correct side for tool responses because it renders the tool message
with a synthetic assistant tool-call prefix, preserving the template context.

## MiMo

Models covered:

```text
XiaomiMiMo/MiMo-7B-SFT
XiaomiMiMo/MiMo-7B-RL
```

Result:

```text
total: 6
pass: 2
mismatch: 4
error: 0
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
MiMo         pass            mismatch             mismatch
```

MiMo uses the Hermes parser. MiMo shares the Qwen ChatML boundary behavior, so
its tool-agent cases mismatch for the same reason as the Qwen family: after
assistant EOS, full `apply_chat_template` contains

```text
<|im_end|>\n<|im_start|>
```

Legacy appends the next rendered turn directly after `<|im_end|>` (response value
`151644` = `<|im_start|>`), while CT restores the canonical newline (value `198`
= `\n`) before it. The inserted newline has `loss_mask=0` and `logprob=0.0`, and
CT is the correct side.

## Nvidia Nemotron

Models covered:

```text
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
```

Result:

```text
total: 9
pass: 3
mismatch: 6
error: 0
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
Nemotron     pass            mismatch             mismatch
```

The tool-case mismatch is a tool-response boundary mismatch where CT is the
correct side. Full `apply_chat_template` renders:

```text
</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>
...
```

Legacy isolated tool-message rendering omits the `<|im_start|>user` header:

```text
</tool_call>
<|im_end|><tool_response>
...
```

CT renders tool messages with the preceding synthetic assistant tool-call
context, so it emits the same `<|im_start|>user\n<tool_response>` boundary as
full `apply_chat_template`. The legacy is wrong.

## Gemma-4

Models covered:

```text
google/gemma-4-E4B-it
```

Only the non-unified ``gemma4`` variant is supported. The ``gemma4_unified``
variant (e.g. ``gemma-4-12B-it``) needs a ``Gemma4UnifiedProcessor`` that the
installed transformers does not ship, so it is not part of the default set.

Result:

```text
total: 3
pass: 3
mismatch: 0
error: 0
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
Gemma-4      pass            pass                 pass
```

Gemma-4 uses `tool_parser=gemma4` (auto-resolved from the model name). All three
default trajectories pass — CT and legacy produce identical `prompt_ids`,
`response_ids`, `loss_mask`, and `logprobs`.

Gemma-4 needs three prompt-side / boundary reconciliations for the assistant
output to be extracted correctly (none change the mock message, only how the
suffix is diffed):

- The generation prompt opens a reasoning channel
  (`<|channel>thought\n<channel|>`) that a plain reasoning-free assistant message
  omits. The trailing opener is trimmed on the prompt side
  (`_trim_gemma_generation_thought_channel`) so the prompt stays a clean prefix.
- Gemma-4 ends an assistant turn with `<turn|>` (token `106`) rather than the
  tokenizer eos, so `<turn|>` is registered as an assistant-end token
  (`_gemma_assistant_end_token_ids`).
- An assistant tool-call turn ends with a trailing `<|tool_response>` opener
  (scaffolding for the following tool message). The model stops before it and the
  loop appends the tool response separately, so that trailing opener is dropped.

## GPT-OSS

Models covered:

```text
openai/gpt-oss-20b
```

Result:

```text
total: 3
pass: 3
mismatch: 0
error: 0
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
GPT-OSS      pass            pass                 pass
```

GPT-OSS uses `tool_parser=gpt-oss` (auto-resolved from the model name). All three
runs pass — CT and legacy produce **identical** `prompt_ids`, `response_ids`,
`loss_mask`, and `logprobs`, and the tool-agent flow completes. GPT-OSS ends a
final answer with `<|return|>` (the tokenizer eos) and a tool-call turn with
`<|call|>`; `<|call|>` is the model's real stop token for a tool call, so it is
registered as an assistant-end token (`_gptoss_assistant_end_token_ids`) and kept.

GPT-OSS needs two harmony-specific handlings in the harness; neither touches
business code (`_gptoss_assistant_generation_ids`):

### 1. Append-only reconstruction (keep prior-turn CoT)

The harmony chat template is **not append-only** across turns: it intentionally
**drops the analysis (CoT) channels of prior tool-call turns once a later final
answer exists** (the template's own comment: "CoT is dropped during all previous
turns, so we never render it for inference"). So re-rendering the whole message
list (`prefix + assistant`) removes the earlier `<|channel|>analysis...<|end|>`
spans, while rendering the prefix alone with `add_generation_prompt=True` keeps
them — the generation prompt is no longer a token-prefix of the full render.

In RL rollout the sequence is built by **appending** generated tokens and prior
CoT is never re-dropped, so the append-only truth is
`prompt (keeps prior CoT) + this turn's own generation`. The harness constructs
each assistant turn's own generation directly rather than via a lossy whole-list
re-render.

### 2. Tool calls in the model's real harmony layout

The recipient in a harmony tool-call header is spec-compliant in two positions;
the official library renders one layout when templating prompts, but **the model
almost always generates the other layout**, and verl's `GptOssToolParser` (like
sglang's detector) targets the layout the model generates:

```text
<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>ARGS<|call|>
```

This checkpoint's chat template instead renders the recipient in the role header
and omits `<|constrain|>` (`... to=functions.NAME<|channel|>commentary json ...`),
which the parser cannot match — so a template-rendered tool call would stall the
tool-agent flow. The harness therefore emits each tool call in the model's real
layout (recipient in the channel header + `<|constrain|>json`), matching what the
model emits at rollout and what the parser expects. This is a harness fidelity
fix, not a verl bug: in production the model generates this layout and the parser
handles it, and verl already bypasses the template for gpt-oss tool responses
(`build_gpt_oss_tool_response_text`).

With both handlings, all three trajectories complete and CT matches legacy on
every token field.
