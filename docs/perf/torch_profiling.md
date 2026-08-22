# PyTorch Profiling in verl

Last updated: 08/04/2026.

This guide explains how to use the native [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) for profiling verl training runs.

## Configuration

Profiling in verl can be configured through parameters in the trainer configuration file (e.g., `ppo_trainer.yaml`).

### Global Profiling Control

In `global_profiler`, you can control when and how profiling occurs globally:

* **`global_profiler.steps`**: List of step numbers to profile. E.g., `[1, 2, 5]` profiles steps 1, 2, and 5. Set to `null` to disable.
* **`global_profiler.save_path`**: Directory to save the profiling results. Default is `outputs/profile`.

### Role Profiling Control

Each RL role (Actor, Critic, etc.) has its own `profiler` configuration:

* **`enable`**: Whether to enable profiling for this role.
* **`all_ranks`**: If `True`, profiles all ranks.
* **`ranks`**: List of specific ranks to profile if `all_ranks` is `False`.
* **`tool_config.torch`**: Configuration specific to the PyTorch Profiler.

#### PyTorch Profiler Options (`tool_config.torch`)

You can customize the PyTorch Profiler behavior using the following fields under `tool_config.torch`:

* **`contents`**: List of contents to profile. An empty list (the default) collects everything.
    *   **`cpu`**: Profile CPU activities. Collected whether or not you list it: operator names and
        verl's per-stage markers are CPU-side events, so a device-only trace would be bare kernels
        with no way to tell which stage they belong to. Listing it is therefore redundant, and the
        rest of `contents` is honored as written.
    *   **`cuda`**: Profile CUDA activities.
    *   **`memory`**: Track tensor memory allocation/free.
    *   **`shapes`**: Record shapes of operator inputs.
    *   **`stack`**: Record source code file and line number.
* **`profile_token_start`**: Effective only for the rollout role; defines the start response-token index for rollout decoding collection. It is applied only when valid (0-based, `profile_token_end > profile_token_start`, and within response length).
* **`profile_token_end`**: Effective only for the rollout role; defines the stop response-token index (exclusive) for rollout decoding collection. It is applied only when valid (0-based, `profile_token_end > profile_token_start`, and within response length).
* **`schedule`**: Optional [`torch.profiler.schedule`](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule) over the **update loop's mini-batches**, so an update stage with many identical mini-batches does not bloat its trace. `verl` advances it once per update mini-batch; the log-prob forwards and rollout are never sub-sampled. Enabled only when `active > 0`. The full `skip_first`/`wait`/`warmup`/`active`/`repeat` selects which update mini-batches are captured in both modes. In **continuous mode** (`discrete: False`) skipping ahead to a later mini-batch also drops the earlier stages (rollout, log-prob) from that one shared trace; leave `skip_first`/`wait`/`warmup` at 0 to keep them. In **`discrete: True` mode** the update stage is isolated in its own trace, so the other stages keep their own full traces regardless. It never emits torch's `ProfilerStep#<n>` rows. See [Scheduling the update loop's mini-batches](#3-scheduling-the-update-loops-mini-batches).


## Examples

### 1. Whole-Step Collection

Collects one trace file per profiled RL step per process, holding everything that process ran
during the step. Note that this is the whole step *as seen by one worker*, not the whole RL
system: see [What one RL step looks like on disk](#what-one-rl-step-looks-like-on-disk).

```yaml
global_profiler:
  steps: [1, 2, 5]
  save_path: ./outputs/profile

actor_rollout_ref:
  actor:
    profiler:
      enable: True
      all_ranks: True
      tool_config:
        torch:
          discrete: False
          contents: [cpu, cuda]
  # rollout & ref follow actor settings
```

### 2. Discrete Mode Collection

Discrete mode saves a separate trace file per stage within each profiled step. This is useful for detailed analysis and is **mandatory** when using Agent Loop.

**Configuration Example**

This configuration supports profiling both Training (Actor) and Inference (Rollout). You can enable/disable them independently.

```yaml
actor_rollout_ref:
  actor:
    profiler:
      enable: True # Set to True to profile training
      all_ranks: False
      ranks: [0] # Global Rank 0
      tool_config:
        torch:
          discrete: True
          contents: [cpu, cuda]
  rollout:
    profiler:
      enable: True # Set to True to profile inference
      all_ranks: False
      ranks: [0] # Global GPU rank(s); each is mapped to the replica that owns it
      tool_config:
        torch:
          discrete: True # REQUIRED 
          # Optional response-token window for rollout engine side collection.
          # If start/stop are not set, the entire rollout stage is collected.
          # Collect tokens in [12, 46), i.e. token index 12~45.
          profile_token_start: 12
          profile_token_end: 46
  # ref follow actor settings
```

**Agent Loop Mode Description**

When Rollout runs in [Agent Loop](../advance/agent_loop.rst) mode, performance data for the Rollout phase **must be collected using discrete mode**. In this case, the Profiler is triggered by the inference engine backend.

1. Rank Definition: `ranks` in the Rollout configuration are global GPU ranks, the same as in the
   training roles, and you get traces for exactly those GPUs. A rollout replica drives one inference
   engine spanning `world_size = tensor_model_parallel_size * data_parallel_size *
   pipeline_model_parallel_size` GPUs, and replica `r` owns global ranks `[r*world_size,
   (r+1)*world_size)`. Each listed rank is mapped to the replica that owns it
   (`replica = rank // world_size`), and that replica's engine is profiled. A tensor-parallel group
   cannot be profiled a GPU at a time, so the engine traces every GPU in the replica, but only the
   traces for the ranks you listed are surfaced (see Trace Location) -- the rest are left aside. So
   with `tp=2` `ranks: [0, 8]` gives you exactly global GPUs 0 and 8 (from replicas 0 and 4), not
   their tp-mates 1 and 9. Ranks that land on a replica that does not exist are silently ignored.
   `all_ranks: True` profiles and keeps every replica; leaving `ranks` empty profiles the replica
   that owns global rank 0 and keeps all of its GPUs.

2. Inference Engine Support: Currently, vLLM and SGLang engines are supported without additional settings. Specific details are as follows:

   *   **vLLM Engine**: Automatically collects AsyncLLM scheduling stacks and inference process performance data.
   *   **SGLang Engine**: Automatically collects inference process performance data. Does not support the memory option in contents.

3. Collection Window: rollout replicas are profiled for the whole training step. Generation is
   decoupled from the step in the V1 trainer -- prompts are served asynchronously and consumed from
   the replay buffer -- so there is no single generation call to wrap.

4. Trace Location: the engines take an output directory rather than a file name, so each replica
   writes into its own `<save_path>/agent_loop_rollout_replica_<n>/` on the node that hosts it.
   Set `global_profiler.relocate_results: True` to have the requested GPUs' traces moved up into
   `save_path` itself when the step finishes; that keeps every trace of a step in the one directory
   you configured, which is what post-processing that does not walk sub-directories needs. Relocated
   files are named `rollout-replica<n>-globalrank<g>_<engine's own file name>`, where `<n>` is the
   replica index and `<g>` is the file's absolute global GPU rank
   (`replica * world_size + tp_rank`), so the names line up with the global `ranks` you configured
   -- e.g. with `tp=2`, `ranks: [0, 8]` puts just `rollout-replica0-globalrank0_...` and
   `rollout-replica4-globalrank8_...` in `save_path`. The tp-mates the engine also traced (global
   GPUs 1 and 9 here) are left in the per-replica sub-directory, so `save_path` shows exactly the
   ranks you asked for. This works by reading the per-GPU tensor-parallel rank out of the engine's
   own file name -- vLLM writes `...-rank-<k>...` and SGLang writes `...-TP-<k>...`. When that rank
   cannot be read unambiguously -- an engine that names traces by host/pid, or SGLang using extra
   parallel dimensions (`-DP-`/`-PP-`/`-EP-`), where a GPU's linear offset is layout-dependent -- the
   trace is kept (never dropped) and named just `rollout-replica<n>_<engine's own file name>`, so you
   may see the replica's other GPUs too. The rollout engine does not run `finish_hook_cmd` itself:
   the colocated training worker's single end-of-run upload of `save_path` (which now includes these
   relocated traces) is what moves them off the node.

### 3. Scheduling the update loop's mini-batches

The unit `global_profiler.steps` selects is one RL step, and by default a profiled step is
collected whole: everything the worker runs lands in one trace, with each stage and each update
mini-batch annotated inside it:

* the update loop wraps each iteration in a `mini_batch<i>` row nested under `update_actor` /
  `critic_update`, numbered from `0` within the step; the engine splits each mini-batch into
  gradient-accumulation micro-batches, shown as `micro_batch<j>` rows nested inside;
* a forward-only stage (`compute_log_prob`, `compute_ref_log_prob`, critic values) is one row named
  after the stage, and the engine splits its batch into `micro_batch<j>` rows inside -- one per
  forward. If you see a single `micro_batch0`, the batch fit in one micro-batch (common when the
  inference token budget `log_prob_max_token_len_per_gpu` / `*_micro_batch_size_per_gpu` is large);
  lower it to split the forward into more micro-batches.

When the update loop has many identical mini-batches, recording all of them bloats the trace.
`tool_config.torch.schedule` sub-samples them: it only ever narrows the update stage's mini-batches
(`prof.step()` is advanced there only, never on the log-prob forwards or rollout) and is active only
when `active > 0`. It never writes torch's `ProfilerStep#<n>` rows -- a step() here is a single
mini-batch, not a meaningful RL step, so those rows would only add noise. What the schedule
sub-samples depends on `discrete`:

* **`discrete: False`** (one continuous trace per process). The schedule selects which update
  mini-batches are kept: `skip_first`/`wait`/`warmup` skip ahead to a later mini-batch and `active`
  sets how many are recorded. Because `step()` is advanced once per mini-batch, a window boundary
  always lands between mini-batches, so the captured ones stay intact. The catch is that torch only
  persists a window ending in `RECORD_AND_SAVE`, so skipping ahead also drops the rollout/log-prob
  stages that run before the active window. Leave `skip_first`/`wait`/`warmup` at 0 (the default) to
  record from the start, keeping every earlier stage in full plus the first `active` update
  mini-batches. Leave the schedule unset (or `active: 0`) to keep every mini-batch.

  ```yaml
  actor_rollout_ref:
    actor:
      profiler:
        enable: True
        all_ranks: True
        tool_config:
          torch:
            discrete: False   # one continuous trace per process
            schedule:
              active: 1       # keep every earlier stage in full, plus mini_batch0 of the update loop
              # skip_first: 1 # ...or skip ahead: capture only mini_batch1 (earlier stages dropped)
  ```

* **`discrete: True`** (one trace per stage). The update stage is profiled in isolation, so the
  full schedule applies to its mini-batches: `skip_first`/`wait`/`warmup` drop the leading ones and
  only the `active` window is kept, repeated `repeat` times (`0` = until the loop ends). The
  log-prob, rollout and ref stages each get their own full trace, untouched by the schedule. The
  update-stage files are tagged with the mini-batch window they hold, e.g. `..._mb3-4`.

  ```yaml
  actor_rollout_ref:
    actor:
      profiler:
        enable: True
        all_ranks: True
        tool_config:
          torch:
            discrete: True
            schedule:
              skip_first: 1
              wait: 1
              warmup: 1
              active: 2   # actor_update trace holds only mini_batch3..4; log-prob/ref traces stay full
              repeat: 0
  ```

Either way the trace carries no `ProfilerStep#<n>` rows: navigate it by the stage rows and the
`mini_batch<i>` rows inside the update loop. In `discrete: False` a schedule keeps `active` update
mini-batches starting at the one `skip_first`/`wait`/`warmup` point to (skipping ahead drops the
earlier stages); leave those at 0 to keep every earlier stage in full, or unset the schedule to keep
every mini-batch. With `global_profiler.profile_continuous_steps: True` a run of consecutive profiled
steps shares one file.

If a step's trace is too large to be workable, narrow it with `discrete: True` (one file per stage)
and sub-sample its update loop with `schedule`, or profile fewer ranks (`ranks`/`all_ranks`).

## Output file naming

Because profiling runs in every training process, each trace file is named so it can be
attributed to a specific process without opening it. The stem is:

```
[<role>_][<scope>_][step<S>_]rank<r>[-of-<world>][_tp<..>-pp<..>-dp<..>-cp<..>]_pid<pid>_<timestamp>.json.gz
```

* **`role`**: the worker role (e.g. `actor`, `ref`, `value-model` for the critic), so results
  from different roles at the same rank are distinguishable. A colocated hybrid worker reports
  its combined role, `actor-rollout-ref` (underscores in labels become hyphens, since underscore
  separates the fields).
* **`scope`**: the profiled region passed to `start_profile`/`annotate` -- `train` for a training
  worker's whole-step window, or a stage name such as `actor-update` in discrete mode.
* **`step<S>`**: the RL step (`global_steps`) being profiled, i.e. one of `global_profiler.steps`.
  With `profile_continuous_steps` this is the first step of the run of steps the file holds.
* **`rank`/`world`**: the global `torch.distributed` rank and world size.
* **`tp/pp/dp/cp`**: tensor/pipeline/data/context parallel ranks, included when Megatron's
  parallel state is initialized (plain FSDP data parallelism only reports `rank`).

### What one RL step looks like on disk

No single trace file covers a whole RL step end to end, because the step does not run in a
single process. A profiled step leaves you with:

* **Training-side traces** (`scope` = `train`), written by the actor/ref/critic workers. These
  hold the work those workers actually run: the log-prob forwards and the actor update's
  forward/backward/optimizer. This is why the scope is called `train` and not `e2e`.
* **Rollout traces**, written by the inference engines themselves into
  `<save_path>/agent_loop_rollout_replica_<n>/` (or into `save_path` with
  `relocate_results: True`), on the node hosting each replica. Generation
  never appears in a training-side trace: with Agent Loop the engines run in their own
  processes, and in the V1 trainer generation is decoupled from the step entirely (the trainer
  samples already-generated data from the replay buffer), so the actor process is simply idle
  while the rollout happens.

To reason about the full step, read the timings that the trainer logs (`gen`, `old_log_prob`,
`ref`, `update_actor`, ...) and open the per-process traces for whichever part you're drilling
into. Trace timestamps are wall clock, so training and rollout traces from the same step can be
lined up side by side in Perfetto.

### Telling stages apart

Within a training-side trace, `discrete` decides whether the stage shows up in the file name or
inside the trace:

* **`discrete: True`** writes one file per stage, and the stage lands in `scope`:
  `actor-rollout-ref_actor-update_step2_rank0-of-8_pid123_<ts>.json.gz`,
  plus siblings for `actor-compute-log-prob`, `ref-compute-log-prob`, `train-batch` and so on.
  Use this when you want to attribute time to a stage from the file name alone.
* **`discrete: False`** (the default) collects the worker's whole step into one trace, so
  `scope` is `train` and cannot name a single stage. The stages are still separated *inside* the
  trace: each one is wrapped in a `torch.profiler.record_function` carrying its stage label, which
  names the role and the function together -- `actor_compute_log_prob`, `ref_compute_log_prob`,
  `actor_update` -- so searching for it in Perfetto/Chrome tracing gives that stage's window.
  Stages that declare no role, such as the inner engine's `train_batch`, appear under the method
  name, and each iteration of the update loop adds a `mini_batch<i>` row.

Note that verl asks the profiler to write exactly `<stem>.json.gz`. If your files carry an
extra segment (e.g. `<stem>.json.1785391501.gz`), it was added after the fact by whatever
post-processes or uploads them, such as a `finish_hook_cmd`.

### Missing roles or stages

Seeing a single `actor...` file per rank, with no separate reference/critic file and no
`compute_log_prob` anywhere, is usually one of the following rather than a lost trace:

* **One file per process, not per role.** The PyTorch profiler is process-global, so a colocated
  hybrid worker records actor *and* reference work into one trace named after the combined role
  (`actor-rollout-ref`). A separate file appears only for a role that runs in its own process,
  e.g. a critic (`value-model`), or a reference model that is not colocated.
* **The hybrid worker follows `actor.profiler`.** It builds its profiler from
  `actor_rollout_ref.actor.profiler`, so `ref.profiler.enable: True` on its own profiles nothing,
  and turning the actor's profiler off also drops the reference stages that share the process.
* **The role may not exist.** There is no critic unless the algorithm uses a value model (GRPO
  and friends do not), and no reference model unless a KL term needs one.
* **A schedule can only sub-sample the update loop, never the log-prob forwards.**
  `tool_config.torch.schedule` narrows the update stage's mini-batches only (`step()` is advanced
  there). In `discrete: False` it selects which update mini-batches to keep: with
  `skip_first`/`wait`/`warmup` at 0 it records from the start so every earlier stage (rollout,
  log-prob) is kept, but skipping ahead to a later mini-batch drops those earlier stages (torch only
  persists a window ending in `RECORD_AND_SAVE`). In `discrete: True` the schedule applies only to
  the isolated update stage, and the log-prob/rollout stages keep their own full traces. So if a
  continuous trace shows only update forward/backward and no log-prob forwards, check first whether
  the schedule skipped ahead; otherwise confirm a log-prob stage actually ran (e.g. that a reference
  model exists) and that
  CPU activity was collected (see the next bullet).
* **Traces collected before CPU activity became unconditional.** A device-only run
  (`contents: [cuda]` on an older verl) has no `record_function` ranges and no operator names, so
  no stage can be located in it even though the kernels of every stage are there, and a log-prob
  forward looks just like the forward half of the update.
* **`compute_log_prob` can legitimately be skipped.** With
  `algorithm.rollout_correction.bypass_mode=True` the trainer reuses the rollout's log probs
  instead of recomputing them, so the actor forward never runs. With LoRA
  (`ref_in_actor`) the reference forward is served by `compute_log_prob` on the actor worker, so
  it appears under that name instead of `compute_ref_log_prob`.

## Traces with no GPU kernels

If a trace only contains CPU operators and no CUDA kernels, the profiler's CUPTI subscription
most likely lost a race. CUPTI accepts a single subscriber per process, and some CUDA images
install a startup hook that points `NVTX_INJECTION64_PATH` at `libcupti.so`. The first NVTX range
in the process then loads libcupti as the NVTX handler and takes that slot, after which Kineto
fails with `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` and drops every CUDA activity. verl
emits NVTX ranges itself, so `NCCL_NVTX_DISABLE=1` does not avoid it.

verl therefore points `NVTX_INJECTION64_PATH` at an unloadable path for all workers when
`global_profiler.tool=torch`, and logs a warning when it does. Set `VERL_KEEP_NVTX_INJECTION=1`
to keep the inherited value, e.g. when you rely on that injection for another tool and accept
traces without CUDA activity.

## Visualization

Collected trace files (usually `.json` or `.json.gz`) are stored flat in the configured
`save_path`: every role, rank and scope writes there directly, since the naming scheme above
already keeps the files unique and self-describing. Rollout engine traces are the one exception,
because the engines are given a directory to write to; `relocate_results: True` pulls them into
`save_path` as well.

To ship the traces off the node, set the hook once on `global_profiler` (every role inherits it). It
runs **once, after the last profiled step**, on each selected rank -- not once per step.

`save_path` is one flat directory that *accumulates* every profiled step's traces (backend `stop()`
and `relocate_results` still run every step, so by the last step everything is there). Because the
command runs a single time rather than once per step, a command that uploads the whole directory
sends each trace exactly once:

```bash
    global_profiler.save_path="$PROFILE_SAVE_PATH" \
    global_profiler.relocate_results=True \
    global_profiler.finish_hook_cmd='my-upload-tool "$VERL_PROFILE_SAVE_PATH"'
```

Running once is also what avoids the "N copies of the same trace" problem. If the command instead
ran every profiled step, a directory upload would re-send step 1's file at step 2, again at step 3,
and so on -- verl writes each trace once, but the upload re-sends the older ones -- and an uploader
that versions by upload time then stores that one trace as several `*.gz` differing only in the
trailing timestamp. A single upload at the end sends each trace once, so there are no duplicate
versions.

`save_path` is usually node-local, so the command runs on every selected rank/node. Choose the
finish-hook ranks so exactly one rank per node uploads that node's directory (otherwise several ranks
sharing a node's directory would each upload it). When unset, the finish-hook ranks default to the
profiled ranks; narrow them with `finish_hook_ranks=[...]` (see
[Nsight Systems profiling](nsight_profiling.md) for choosing ranks).

The command runs through the shell, so the context variables `VERL_PROFILE_SAVE_PATH`,
`VERL_PROFILE_TOOL`, `VERL_PROFILE_RANK`, `VERL_PROFILE_PID` and `VERL_PROFILE_ROLE` are all available
(quote them in single quotes so your shell does not expand them early). Hydra's override parser
rejects a value that contains its own quotes, so a command that needs quoting belongs in a small
script that the hook calls.

The hook prints the command, its output and its exit code to the worker's log. Rollout replicas do
not run the command themselves: they relocate their traces into `save_path` (with
`relocate_results`), and the colocated training worker's single end-of-run upload covers them too.
See [Nsight Systems profiling](nsight_profiling.md) for the full description of the hook, including
how to choose which ranks run it.

You can visualize them using:

1.  **Chrome Tracing**: Open `chrome://tracing` in a Chrome browser and load the JSON file.
2.  **Perfetto**: Open [ui.perfetto.dev](https://ui.perfetto.dev/) and load the file (recommended for large traces).
3.  **TensorBoard**: If using the TensorBoard plugin for PyTorch Profiler.
