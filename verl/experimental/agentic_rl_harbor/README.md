# Recipe: Agentic RL with Harbor

**Author:** `https://github.com/myjlyjly`

Last updated: 05/21/2026.

This recipe wires VeRL's `AgentLoopBase` to [Laude Institute's Harbor](https://github.com/laude-institute/b-harbor) framework so each training sample is a full Harbor `Trial` running a multi-turn agent against a remote sandbox (Daytona / Modal).
Harbor's per-turn rollout details are flattened back into the linear `(prompt_ids, response_ids, response_mask)` format VeRL's PPO/GRPO trainers expect, and the loop plugs straight into [`fully_async_policy`](../fully_async_policy/README.md) for async training.

## Layout

```
verl/experimental/agentic_rl_harbor/
├── harbor_agent_loop.py                     # @register("harbor_agent") -- one Trial per sample
├── harbor_dataset.py                        # HarborTaskDataset: enumerates task directories
├── prepare_harbor_dataset.py                # Pull a Harbor dataset from HF Hub and unpack it
├── cleanup_daytona.py                       # List / delete leaked Daytona sandboxes (preflight)
├── config/harbor_agent.yaml                 # agent_loop registry entry + TrialConfig template
├── templates/
│   └── qwen3_acc_thinking.jinja2            # Qwen3 chat template w/ accumulate-thinking
└── shell/
    └── run_qwen3_8b_harbor_fully_async.sh   # full demo (fully_async + GRPO + Harbor)
```

## Prereqs

```bash
# Python deps (versions this recipe has been tested with)
pip install 'harbor[daytona]==0.6.1' 'litellm==1.82.6'

# Sandbox credentials -- Harbor runs each trial in a remote sandbox. Pick one:
#   Daytona (default https://app.daytona.io/api):
export DAYTONA_API_KEY=...
export DAYTONA_API_URL=...
#   Modal:
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
```

## Prepare data

Harbor expects each sample to be a directory containing `instruction.md` (plus
optional verifier and starter files).
`prepare_harbor_dataset.py` pulls a HF dataset whose parquet files have `path`
and `task_binary` columns and unpacks each `task_binary` tarball under `~/data/harbor/<repo>/`:

```bash
# Default (Qwen3-8B can get non-zero reward signal here):
python verl/experimental/agentic_rl_harbor/prepare_harbor_dataset.py \
    --dataset DCAgent/exp_rpt_curriculum-easy
python verl/experimental/agentic_rl_harbor/prepare_harbor_dataset.py \
    --dataset laion/exp_rpt_exercism-python-v2
```

## Quick start

```bash
# Optional preflight -- delete any Daytona sandboxes leaked by a previous crash
# (ray stop --force / OOM / SIGKILL skip Harbor's _stop_sandbox teardown, leaving
# sandboxes alive on the Daytona side and consuming your CPU quota until the
# 30 min auto_stop_interval elapses). See "Sandbox concurrency cap" below.
python verl/experimental/agentic_rl_harbor/cleanup_daytona.py            # list only
python verl/experimental/agentic_rl_harbor/cleanup_daytona.py --delete   # delete all

# Full demo: GRPO + fully_async + Harbor on Qwen3-8B (8 GPUs, 4 rollout + 4 train).
# All knobs are env-var overridable.
ray stop --force && bash verl/experimental/agentic_rl_harbor/shell/run_qwen3_8b_harbor_fully_async.sh
```

## How it fits together

* `data.custom_cls` points at `HarborTaskDataset`, which emits one row per task
  directory. Each row carries `task_path` and `agent_name="harbor_agent"`.
* `actor_rollout_ref.rollout.agent.agent_loop_config_path` points at
  `config/harbor_agent.yaml`, which registers `HarborAgentLoop` and supplies the
  `harbor_trial_config` template (Harbor `TrialConfig`: trials_dir, agent.name,
  max_turns, environment.type, etc.).
* `HarborAgentLoop.run()` picks a VeRL rollout server, builds a per-trial
  `TrialConfig` (sets `task.path` / `agent.kwargs.api_base` /
  `agent.kwargs.session_id`), runs the trial via Harbor, and converts
  `rollout_details` to the VeRL agent-loop output via
  `_build_step_wise` + `_merge_stepwise`.

## Key configuration notes

* **`served_model_name`.** vLLM advertises the model under
  `actor_rollout_ref.rollout.prometheus.served_model_name` (basename'd if it
  contains a path separator). HarborAgentLoop reuses that value to build the
  LiteLLM target `hosted_vllm/<served_model_name>`. Set
  `prometheus.enable=True` and `prometheus.served_model_name=<basename>` (the
  shell scripts do this for you) to keep both sides in sync.
* **Chat template.** The default Qwen3 chat template strips `<think>...</think>`
  blocks from non-last assistant messages on every re-render, breaking the
  prefix invariant `_merge_stepwise` relies on. Pass
  `templates/qwen3_acc_thinking.jinja2` to vLLM via
  `+actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template=...` (the shell
  scripts do this for you). The merge logic now tolerates prefix divergence
  (it flushes a new merge group and randomly picks one for the policy gradient
  update), but every divergence costs you a smaller training group from that
  trajectory — keeping the prefix invariant gives one big group per trajectory.
* **`harbor_trial_config` is read straight from the YAML** — its sub-fields
  (`trials_dir`, `agent.name`, `agent.kwargs.max_turns`, `environment.type`,
  ...) are not exposed through hydra's main config tree, so they cannot be
  overridden on the CLI. To change them, edit `config/harbor_agent.yaml` or
  copy it and set `actor_rollout_ref.rollout.agent.agent_loop_config_path` to
  the copy.
* **`partial_rollout` is not supported.** Harbor trials run in a remote sandbox
  (daytona/modal) whose container/shell-session state cannot be checkpointed,
  so `async_training.partial_rollout=False` is mandatory. This caps the loop
  at fully_async Mode 3 (async stream pipeline w/ stale samples); see
  [`fully_async_policy/README.md`](../fully_async_policy/README.md) for the
  Mode 1/2/3/4 taxonomy.
* **Sandbox concurrency cap (Daytona).** Every trial allocates one Daytona
  sandbox sized by `override_cpus` in `config/harbor_agent.yaml` (default 1).
  Daytona's free tier caps total in-flight CPU at 10 — exceeding it surfaces as:
  ```
  Failed to create sandbox: Total CPU limit exceeded. Maximum allowed: 10.
  ```
  Concurrent trials follow `fully_async_rollouter.py:set_max_required_samples`:
  ```
  max_required_samples   = ppo_mini_batch_size * require_batches
                           * (staleness_threshold + 1) * trigger_parameter_sync_step
  max_concurrent_samples = min(replicas * 16, max_required_samples)
  concurrent_trials      ~= max_concurrent_samples * N_RESP_PER_PROMPT
  ```
  The shipped defaults (`ppo_mini=2, require=1, trigger=2, n=2, staleness=0.1`)
  evaluate to `2*1*1.1*2 = 4` samples → 8 concurrent trials (2 CPU of headroom
  under the 10-cap). Raising `STALENESS_THRESHOLD` back to `1.0` to fully
  exercise Mode 3 requires upgrading the Daytona tier
  (<https://app.daytona.io/dashboard/limits>) or switching to a local sandbox.
  The shell script has a detailed trade-off table in its `Async knobs` block.
* **Leaked sandboxes after a crash.** `ray stop --force`, OOM kills, or any
  abrupt termination skips `DaytonaEnvironment._stop_sandbox`, leaving live
  sandboxes behind that continue to consume the account's CPU quota until
  `auto_stop_interval_mins` elapses (30 min in the shipped yaml). Run
  `cleanup_daytona.py --delete` as a preflight before re-launching.
* **`min/max_global_steps` is stamped 0.** Standard agent loops get the rollout
  server's weight version through `extra_fields["global_steps"]`; Harbor goes
  through LiteLLM/HTTP and bypasses that path, so we stamp 0 instead. This
  makes fully_async's `param_version_diversity` log metric always read 1, but
  staleness control / partial-rollout stats are unaffected (they stay correct
  because we never enable partial_rollout).

## Tests

CPU-only unit tests for the prefix-aware merge live at
[`tests/experimental/agentic_rl_harbor/test_harbor_merge_on_cpu.py`](../../../tests/experimental/agentic_rl_harbor/test_harbor_merge_on_cpu.py).
They cover: clean-prefix collapse to a single group, prefix-divergence flush
into multiple groups, malformed `rollout_details` rejection, and the
short-prompt edge case.

## Acknowledgements

* Harbor framework: <https://github.com/laude-institute/b-harbor>
* SkyRL (Apache 2.0): <https://github.com/NovaSky-AI/SkyRL> — the following
  pieces are adapted from SkyRL's Harbor integration (file headers carry the
  per-file provenance URLs):
  * `_build_step_wise` / `_merge_stepwise` follow `harbor_generator.py`'s
    two-stage layout.
  * `prepare_harbor_dataset.py` is near-verbatim from
    `examples/train_integrations/harbor/prepare_harbor_dataset.py`.
  * The `harbor_trial_config` block in `config/harbor_agent.yaml` is
    verbatim from `examples/train_integrations/harbor/harbor_trial_config/default.yaml`.
  * `templates/qwen3_acc_thinking.jinja2` is verbatim from
    `skyrl/train/utils/templates/qwen3_acc_thinking.jinja2`.
