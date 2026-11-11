# Code execution in Runta sandboxes

Lets the policy run the code it writes inside a [Runta](https://pypi.org/project/runta-sdk/) runtime during
multi-turn rollout. Each trajectory gets its own runtime, so files written in one turn are
still there in the next, and trajectories cannot see each other.

| File | Purpose |
| --- | --- |
| `runta_tool.py` | `RuntaSandboxTool`, a `BaseTool` that exposes `code_interpreter(code)` |
| `runta_agent_loop.py` | `RuntaToolAgentLoop` (`runta_tool_agent`), a `ToolAgentLoop` that deletes the runtime when the trajectory ends |
| `config/runta_tool_config.yaml` | tool config for `multi_turn.tool_config_path` |
| `config/runta_agent_loop_config.yaml` | agent loop config for `agent.agent_loop_config_path` |

## Setup

The Runta SDK is an optional dependency; verl does not install it.

```bash
pip install "runta-sdk>=0.2.6,<0.3"
export RUNTA_TOKEN=...   # or log in once so that ~/.config/runta/config.toml exists
```

The SDK reads the credentials itself; nothing goes into the YAML files. Set `RUNTA_ENDPOINT`
only if you use a non-default endpoint. On a multi-node Ray cluster the SDK and the
credentials must be available on every node that runs agent loop workers.

## Usage

Run from the repository root (the configs refer to `examples.runta_sandbox.*`) and add
these overrides to any agent-loop training command, for example the ones in
[Agentic RL Training](../../docs/start/agentic_rl.rst):

```bash
python3 -m verl.trainer.main_ppo \
    ... \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/runta_sandbox/config/runta_tool_config.yaml \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/runta_sandbox/config/runta_agent_loop_config.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=runta_tool_agent
```

The dataset must prompt the model to call `code_interpreter`.
`examples/data_preprocess/dapo_multiturn_w_tool.py` produces one.

A dataset `agent_name` column overrides `default_agent_loop`. If your dataset sets
`agent_name: tool_agent`, change it to `runta_tool_agent`: the stock `tool_agent` runs the tool
too, but never deletes the runtimes (see below).

## Runtime lifecycle

`ToolAgentLoop` calls `create -> execute -> release` around every tool call, so those hooks
cannot own a sandbox that has to outlive a single call. Instead:

1. The first `code_interpreter` call of a trajectory creates a runtime, keyed by
   `agent_data.request_id`. A trajectory that never calls the tool never creates one.
2. Later calls of the same trajectory reuse it. Every call starts a new interpreter process:
   files and installed packages persist, variables do not.
3. If a call fails and Runta reports the runtime as shut down, crashed or deleted (for example
   because it reached `auto_shutdown_after_secs`), the tool deletes it, logs a warning, creates
   a fresh runtime and runs that call once more on it. Files from earlier turns are gone.
4. `RuntaToolAgentLoop.run()` deletes the runtime the trajectory ends with in a `finally` block,
   whether the trajectory finished, failed or was cancelled.

If the trainer process is killed, step 4 cannot run. Every runtime is therefore created with
`auto_shutdown_after_secs` / `auto_delete_after_secs`, so Runta reclaims it on its own. Leftovers
are easy to spot by their name prefix:

```bash
runta ps -a | grep verl-sandbox
```

## Configuration

Keys under `config:` in `config/runta_tool_config.yaml`:

| Key | Default | Meaning |
| --- | --- | --- |
| `language` | `python` | `python` or `bash` |
| `default_timeout` | `30` | seconds per execution; the process is killed afterwards |
| `max_output_bytes` | `65536` | bytes kept per output stream |
| `workdir` | runtime default | working directory inside the runtime |
| `name_prefix` | `verl-sandbox` | runtimes are named `<name_prefix>-<8 hex chars>` |
| `vcpus` / `memory_mib` | `1` / `1024` | size of a runtime created from the base runtime; ignored (with a warning) when `checkpoint_id` is set |
| `checkpoint_id` | `null` | start from a Runta checkpoint instead of the base runtime; the runtime keeps the checkpoint's size |
| `create_timeout` | `240` | seconds to wait for a new runtime to become ready |
| `auto_shutdown_after_secs` | `3600` | server-side limit on a runtime's running time, counted from its start, not from its last use |
| `auto_delete_after_secs` | `300` | delete this long after shutdown |

Things to size before a real run:

- **Concurrency.** One runtime is alive for every in-flight trajectory that has called the tool,
  up to `train_batch_size * rollout.n` at once. Check this against your Runta quota.
- **Dependencies.** The base runtime is Ubuntu with Python 3 and pip. Runta runtimes are not
  created from container images, so to preinstall packages, set one runtime up by hand, take a
  checkpoint and pass its id as `checkpoint_id`. Size that runtime as the trajectories need it:
  `vcpus` / `memory_mib` do not apply to runtimes created from a checkpoint.
- **`auto_shutdown_after_secs`** must exceed the longest trajectory you expect. A trajectory
  that outlives it continues in a fresh runtime, without its earlier files.

## What the model sees

| Outcome | Tool response |
| --- | --- |
| success | stdout followed by stderr, or `(no output)` |
| non-zero exit | stdout, stderr, then `[exit code N]` |
| the program ran into `default_timeout` | stdout and stderr so far, then `Execution timed out after N seconds.` |
| no answer from Runta within `default_timeout` + 10 s | `Sandbox error: no response from the sandbox within N seconds. ...` |
| Runta API/transport error during execution | `Sandbox error: <error>` |
| runtime could not be created | `Error executing tool 'code_interpreter': <error>` (from `ToolAgentLoop`) |

Runta reports a command it killed with exit code 124, which a program can also return itself
(`sys.exit(124)`, bash `timeout 5 cmd`). The tool only calls it a timeout when the command ran
for about `default_timeout`; otherwise it is an ordinary `[exit code 124]`.

Code wrapped in a markdown fence is unwrapped only when the whole argument is one fenced block.
Anything else, including code that contains a fence in a string, runs unchanged.

## Limitation: infrastructure failures do not stop training

The last three rows are not the model's fault, but `ToolAgentLoop` has no way to fail or skip a
trajectory from inside a tool: the error text goes to the model, the step reward is 0 and
training continues. With an expired token, an exhausted quota or a Runta outage, every tool
call of the run fails this way, and the policy is trained on those trajectories.

Watch the logs of the agent loop workers. Authentication and quota failures are logged at
`ERROR` with the cause, e.g. `Runta authentication failed: the token is invalid or expired`;
other sandbox failures and replaced runtimes are logged at `WARNING`.

## Tests

```bash
pytest tests/tools/test_runta_sandbox_tool_on_cpu.py
```

The tests replace the `runta` module with a fake, so they need neither the SDK nor an account.
