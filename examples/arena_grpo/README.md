# GRPO with the OpenAgora Sandbox Agent Loop

This example trains a model with GRPO while the agent executes inside
[OpenAgora](https://github.com/albert-lv/OpenAgora) sandboxes (Docker
containers) instead of calling tools in the trainer process:

- **Sandboxed execution**: agents run inside Docker containers orchestrated by
  the OpenAgora server.
- **Active LLM proxy**: the OpenAgora proxy injects sampling parameters and
  records per-token logprobs for every LLM call the agent makes.
- **Decoupled verification**: rewards are computed by OpenAgora's independent
  verification plane, not by the agent or a trainer-side reward function.
- **RL-grade trajectories**: every proxy request/response pair is stored in a
  structured trajectory log and converted back into verl's `AgentLoopOutput`.

## Prerequisites

1. Install verl as usual, plus the OpenAgora SDK in the same environment:

   ```bash
   pip install openagora-sdk
   # or from source:
   # pip install git+https://github.com/albert-lv/OpenAgora.git#subdirectory=python/openagora-sdk
   ```

2. Start the OpenAgora server (see the OpenAgora repository for installation):

   ```bash
   openagora-server --sandbox=docker --grpc :9090 --http :9093
   ```

3. Start an OpenAI-compatible LLM backend that the OpenAgora proxy forwards
   agent requests to, e.g. vLLM:

   ```bash
   vllm serve Qwen/Qwen2.5-0.5B-Instruct \
     --port 8001 --dtype bfloat16 --enforce-eager --max-model-len 2048
   ```

4. Build the agent sandbox image on the host (default
   `openagora-agent-minimal:latest`; see the OpenAgora repository).

## Dataset format

Standard verl RL parquet dataset (`prompt`/`reward_model`/... columns) with an
`extra_info` struct column. `extra_info` may carry these optional keys:

- `openagora_verify`: per-sample verification command executed by the
  OpenAgora verification plane (e.g. `pytest -q /tests`). Takes precedence over
  the `ARENA_VERIFY_COMMAND` environment variable.
- `task_file`: per-sample task definition that replaces the default
  prompt-based task payload sent to the sandbox.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ARENA_ENDPOINT` | `localhost:9090` | gRPC endpoint of the OpenAgora server |
| `ARENA_AGENT_IMAGE` | `openagora-agent-minimal:latest` | Sandbox image for the agent |
| `ARENA_LLM_BACKEND` | `http://localhost:8001/v1` | LLM backend URL the proxy forwards to |
| `ARENA_VERIFY_COMMAND` | `true` | Fallback verify command (per-sample `openagora_verify` wins) |
| `ARENA_TIMEOUT_SECONDS` | `600` | Rollout timeout in seconds |

## Launch

```bash
bash examples/arena_grpo/run_qwen2_5_0_5b_fsdp.sh
```

The script sets the `ARENA_*` environment variables, then runs
`train_grpo_arena.py`, which imports `verl.experimental.agent_loop.arena_agent_loop`
to register the `arena_agent` loop and delegates to `verl.trainer.main_ppo`.
Ray workers that do not inherit the driver's imports register the loop through
`arena_agent_loop.yaml` (passed via
`actor_rollout_ref.rollout.agent.agent_loop_config_path`).

The equivalent minimal Python invocation is:

```bash
export ARENA_ENDPOINT=localhost:9090
export ARENA_AGENT_IMAGE=openagora-agent-minimal:latest
export ARENA_LLM_BACKEND=http://localhost:8001/v1

python3 examples/arena_grpo/train_grpo_arena.py \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.rollout.agent.default_agent_loop=arena_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/arena_grpo/arena_agent_loop.yaml \
  ...
```
