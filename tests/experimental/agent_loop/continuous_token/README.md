# Continuous Token Agent Loop Test Results

This file records the current CT-vs-legacy end-to-end tokenizer alignment
results for the default model set used by
`compare_agentloop_ct_vs_legacy.py`.

## Test Setup

Default trajectories:

```text
singleturnchat          SingleTurnAgentLoop
multiturnsingletool     ToolAgentLoop
multiturnmultitool      ToolAgentLoop
```

Pass criteria:

```text
Flow is complete, and CT/legacy have identical prompt_ids, response_ids,
loss_mask, and logprobs.
```

For tool-agent cases, the deterministic server uses the tested model's own
tokenizer and chat template to render structured assistant messages, then
returns the token-id suffix for each assistant turn. Tool responses are produced
by deterministic tools and are appended by the real AgentLoop state machine.

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
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
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
default_models: 25
default_trajectory_runs: 75
pass: 37
mismatch: 32
error: 6
```

## Results

```text
Family        Models  Runs  Pass  Mismatch  Error  Notes
Qwen          9       27    9     18        0      Single-turn passes; tool cases mismatch because legacy misses the newline after assistant EOS.
GLM           4       12    4     8         0      Single-turn CT-vs-legacy passes; tool cases mismatch on the observation boundary. GLM-5.1 matches full template; GLM-4.7/5 have acceptable think-boundary CT-vs-full raw mismatches.
Kimi          3       9     9     0         0      All default trajectories pass with tool_parser=kimi. CT-vs-full-template is WIP.
Seed-OSS      1       3     3     0         0      All default trajectories pass with tool_parser=seed. CT-vs-full-template is WIP.
MiniMax       3       9     3     0         6      Single-turn passes; tool cases are legacy-only TemplateError. CT-vs-full-template is WIP.
MiMo          2       6     6     0         0      All default trajectories pass with Hermes parser. CT-vs-full-template is WIP.
Nemotron      3       9     3     6         0      Single-turn passes; tool cases mismatch because legacy misses the tool-response user header. CT-vs-full-template is WIP.
```

Unless stated otherwise in a family section, the pass/mismatch/error counts are
strict CT-vs-legacy results. CT-vs-full-template has been reviewed and
classified for Qwen and GLM. For Kimi, Seed-OSS, MiniMax, MiMo, and Nemotron,
CT-vs-full-template is still WIP and should not be treated as a finalized
alignment result.

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
Qwen/Qwen3-8B
Qwen/Qwen3.5-0.8B
```

Result:

```text
total: 27
pass: 9
mismatch: 18
error: 0
```

```text
Model group       singleturnchat  multiturnsingletool  multiturnmultitool
Qwen2.5           pass            mismatch             mismatch
Qwen3             pass            mismatch             mismatch
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

The harness reports CT-vs-legacy and CT-vs-full-template separately. Each
section uses the metric that matches what it is comparing. CT-vs-legacy is a
strict token/metadata comparison, so a mismatch is simply a mismatch. There is
no separate material-mismatch concept for CT-vs-legacy.

CT-vs-full-template compares decoded CT text with a complete post-hoc
`apply_chat_template` render. For that comparison, raw mismatch is an exact
text mismatch. Material mismatch means the meaningful generated trajectory
differs: visible assistant content, tool calls, tool responses, or non-empty
reasoning content are different. Pure template bookkeeping differences are
reported as acceptable mismatches and do not count as material.

For CT-vs-full-template, two differences are currently acceptable.

1. Final trailing newline after the last assistant EOS. Full
   `apply_chat_template` may render:

   ```text
   ...<|im_end|>\n
   ```

   CT decodes the generated runtime stream as:

   ```text
   ...<|im_end|>
   ```

   This is not material because the model generation stops at `<|im_end|>` and
   does not emit the template-only newline after the final EOS.

2. CT-only empty Qwen3 think block at assistant tool-call turns. Per-turn
   generation may produce:

   ```text
   <|im_start|>assistant
   <think>

   </think>

   I will check Seattle first.
   <tool_call>...</tool_call><|im_end|>
   ```

   A complete post-hoc trajectory render may suppress the empty think block and
   produce:

   ```text
   <|im_start|>assistant
   I will check Seattle first.
   <tool_call>...</tool_call><|im_end|>
   ```

   This is not material because the dropped block is exactly empty and appears
   only as Qwen3 generation-time formatting immediately after the assistant
   header. Removing it leaves the same visible content and the same tool call.
   A non-empty `<think>...</think>` difference is not ignored and still counts
   as a material mismatch.

Current Qwen split comparison result:

```text
CT vs legacy:
match: 9 / 27
mismatch: 18 / 27
error: 0 / 27

CT vs full template:
raw mismatch: 27 / 27
acceptable mismatch: 27 / 27
material mismatch: 0 / 27
```

All Qwen CT-vs-full-template raw mismatches are acceptable mismatches. There
are no material CT-vs-full-template mismatches for the current Qwen result.

Qwen2.5, Qwen3-4B-Instruct-2507, Qwen3.5, and Qwen3 base single-turn cases
only have acceptable final trailing-newline differences: 19 cases total. Qwen3
base tool-agent cases (`Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`)
additionally have acceptable CT-only empty think blocks at assistant tool-call
turns: 8 cases total.


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

Current GLM split comparison result:

```text
CT vs legacy:
match: 4 / 12
mismatch: 8 / 12
error: 0 / 12

CT vs full template:
raw mismatch: 9 / 12
acceptable mismatch: 9 / 12
material mismatch: 0 / 12
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

CT removes the ambiguous duplicate boundary and matches the full-template
boundary. The retained CT observation boundary tokens have `loss_mask=0` and
`logprob=0.0`.

For CT-vs-full-template, GLM-5.1 matches exactly on all three trajectories.
GLM-4.7, GLM-4.7-Flash, and GLM-5 have acceptable raw mismatches on all three
trajectories. The mismatch is the assistant thinking boundary. The CT runtime
stream contains the generation-time thinking opener:

```text
<|assistant|><think>Yes, move the Friday demo ...
```

The full post-hoc `apply_chat_template` render contains the corresponding
close boundary before assistant content when the assistant message has no
separate `reasoning_content`:

```text
<|assistant|></think>Yes, move the Friday demo ...
```

Tool-call turns show the same pattern:

```text
CT:   <|assistant|><think>I will check Seattle first ...<tool_call>...
Full: <|assistant|></think>I will check Seattle first ...<tool_call>...
```

This is not counted as material. For these dumped cases, replacing each CT
`<|assistant|><think>` boundary with the full-template
`<|assistant|></think>` boundary makes the decoded CT trajectory exactly equal
to the full post-hoc render. The visible assistant content, tool calls, and tool
responses are unchanged; only the template boundary used to connect a
generation-time assistant stream with a no-reasoning post-hoc assistant message
differs.

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

CT-vs-full-template status: WIP. The result above is CT-vs-legacy only.

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

CT-vs-full-template status: WIP. The result above is CT-vs-legacy only.

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

CT-vs-full-template status: WIP. The result above is CT-vs-legacy only.

## MiMo

Models covered:

```text
XiaomiMiMo/MiMo-7B-SFT
XiaomiMiMo/MiMo-7B-RL
```

Result:

```text
total: 6
pass: 6
mismatch: 0
error: 0
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
MiMo         pass            pass                 pass
```

MiMo passes all default trajectories with the Hermes parser.

CT-vs-full-template status: WIP. The result above is CT-vs-legacy only.

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

CT-vs-full-template status: WIP. The result above is CT-vs-legacy only.
