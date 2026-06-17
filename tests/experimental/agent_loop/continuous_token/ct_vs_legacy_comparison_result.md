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
Qwen          9       27    9     18        0      Single-turn passes; tool cases mismatch because legacy misses the newline after assistant EOS. CT is correct.
GLM           4       12    4     8         0      Single-turn CT-vs-legacy passes; tool cases mismatch on the observation boundary. CT is correct.
Kimi          3       9     9     0         0      All default trajectories pass with tool_parser=kimi. 
Seed-OSS      1       3     3     0         0      All default trajectories pass with tool_parser=seed. 
MiniMax       3       9     3     0         6      Single-turn passes; tool cases are legacy-only TemplateError. 
MiMo          2       6     6     0         0      All default trajectories pass with Hermes parser. 
Nemotron      3       9     3     6         0      Single-turn passes; tool cases mismatch because legacy misses the tool-response user header. CT is correct.
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

The harness reports CT-vs-legacy as a strict token/metadata comparison, so a
mismatch is simply a mismatch.

```text
CT vs legacy:
match: 9 / 27
mismatch: 18 / 27
error: 0 / 27
```

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
pass: 6
mismatch: 0
error: 0
```

```text
Model group  singleturnchat  multiturnsingletool  multiturnmultitool
MiMo         pass            pass                 pass
```

MiMo passes all default trajectories with the Hermes parser.

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
