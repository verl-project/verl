# HSDP hang repro on Intel XPU — handoff for isolated K8s testing

## Goal

Reproduce/diagnose a reported HSDP (Hybrid Sharded Data Parallel) hang when running
verl GRPO training (Qwen2.5-3B-Instruct) on 4x Intel XPU, and evaluate whether
[kahlun/verl#25](https://github.com/kahlun/verl/pull/25) (Ray accelerator device-pinning fix)
resolves it.

This round of testing ran in a single Docker container (`verl-intel-gpu:latest`) on a
bare-metal box with 4 physical GPUs directly visible to every process. Results were
confounded by another, unrelated job sharing the same host (see "Confound" below) —
that's the main reason to re-run on isolated K8s-scheduled GPU allocations, where each
pod only sees the GPU(s) it was assigned.

## Environment (source machine)

- 4x Intel(R) Arc(TM) Pro B60 Graphics, 24480 MiB (~24GB) each, driver `1.15.38308+1`,
  oneAPI Level-Zero V2 backend
- `torch==2.13.0+xpu`, `ray==2.58.0`, `vllm==0.29.0`
- `intel_extension_for_pytorch` (ipex) **not installed** — some code paths may fall back
  to slower/generic implementations
- Host had 231GiB RAM total; no container memory cgroup limit was set (`unlimited`)
- verl repo at `/opt/verl`, branch `intel-xpu-plugin-e2e-vllm` (fork: `kahlun/verl`)

## Patch applied: kahlun/verl#25

**What it fixes:** Ray's device-pinning logic used a hardcoded list of
`RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES` env var names to decide whether to
explicitly pin `LOCAL_RANK` to a physical accelerator. That list is stale for Intel
GPU — Ray renamed `ONEAPI_DEVICE_SELECTOR` to `ZE_AFFINITY_MASK`
(`ray-project/ray#64440`) and the NOSET name moved with it. If unaccounted for, this can
make every rank in a job silently default to physical device 0, i.e. every rank runs on
the same GPU — a collision that manifests as OOM or as collectives falling back to
paths meant for co-located ranks, not as an explicit device error.

**Fix:** resolves the physical accelerator id Ray assigns against what the process can
actually see (`visible_devices_envvar`), instead of a fixed env var name list. Also adds
a `ray_device_index()` platform hook so non-standard runtimes can override the mapping.

**Files touched:**
- `verl/plugin/platform/platform_base.py` — new `ray_device_index()` default impl
- `verl/single_controller/base/worker.py` — always pin from Ray's own accelerator
  assignment (previously only pinned when a specific NOSET flag was set)
- `verl/utils/device.py` — new `get_ray_device_index()` helper
- `verl/utils/ray_utils.py` — derive the NOSET var name from the platform's own visible-devices
  keyword instead of a fixed list
- `tests/plugin/test_platform_abstraction.py` — new unit tests

**Status here:** applied cleanly (`git apply`, no conflicts) on top of
`intel-xpu-plugin-e2e-vllm` @ `742390ff`. All 21 tests in
`tests/plugin/test_platform_abstraction.py` pass, including the new
`TestRayDeviceIndex` / `TestRayNosetDetection` classes.

**Not yet verified:** whether the real Intel XPU platform + Ray combination actually
pins 4 distinct physical devices at runtime (only checked via unit tests with a mocked
platform). Worth confirming on K8s by dumping `LOCAL_RANK` / `ZE_AFFINITY_MASK` from
`/proc/<pid>/environ` for each live `WorkerDict` rank during init.

## Repro command (baseline, `optimizer_offload=True`)

See `run_hsdp_hang_repro.sh`. Key config: `fsdp_size=2` (HSDP, 2-way shard within
groups of 2 across 4 ranks — deliberately not full 4-way sharding), `param_offload=True`,
`optimizer_offload=True`, vLLM colocated with `gpu_memory_utilization=0.4`,
`enable_sleep_mode=True`.

**Result:** got through weight loading, then a *slow* (~40 min, not a real deadlock —
confirmed via repeated `py-spy dump`, CPU stayed active) `torch.distributed.broadcast`
inside `set_model_state_dict(broadcast_from_rank0=True)` /
`fsdp2_load_full_state_dict` (`verl/utils/fsdp_utils.py:513`). After that it proceeded
normally: vLLM rollout server spawned, ref log-prob computation ran. ~1hr in, it crashed:

```
RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
  ... offload_fsdp2_model_to_cpu (verl/utils/fsdp_utils.py:211)
  ... get_torch_device().empty_cache() -> torch._C._xpu_emptyCache()
```
during CPU-offload cleanup after `WorkerDict.actor_rollout_ref_compute_log_prob`.

`DEVICE_LOST` is a Level-Zero/driver-level fault, not a Python/Ray-level bug — Ray's
device-pinning logic (what PR #25 touches) can't prevent the driver dying underneath an
already-correctly-pinned process.

Checked host `dmesg`/`journalctl -k` for a GPU engine reset at the exact crash time
(10:17–11:20 on the day of the run) — **found none**. However the same host has a
chronic history of `xe` driver "exec queue reset detected" / "Engine reset" / "Timedout
job" / coredump events spanning multiple unrelated days (at least 2 weeks of
history checked), so the underlying driver stack is generally fragile under memory
pressure on this box, independent of this repro.

## Repro command (variant, `optimizer_offload=False`)

See `run_hsdp_no_optim_offload.sh`. Same as baseline except
`actor_rollout_ref.actor.fsdp_config.optimizer_offload=False` (kept `param_offload=True`
and `enable_sleep_mode=True`).

**Motivation:** back-of-envelope VRAM math for this config — 3.09B params, fp32 AdamW
optimizer state (~12 bytes/param unsharded), HSDP `fsdp_size=2` (only 2-way shard, not
4-way) — puts optimizer state alone at ~18GB/rank before params/gradients/activations,
on a 24GB card also colocating a vLLM rollout engine reserving 40% of device memory.
Wanted to see if removing optimizer offload produces a clean, diagnosable
`torch.OutOfMemoryError` (confirming the math) vs. success vs. the same `DEVICE_LOST`.

**Result — none of the above cleanly.** GPU memory telemetry right after FSDP init:

```
After FSDP, memory allocated (GB): 5.82, memory reserved (GB): 7.36, device memory used/total (GB): 23.89/23.91
```

i.e. **~20MB of headroom left on a 24GB card** — consistent with the VRAM math, but it
didn't outright OOM at that instant. It proceeded further than the math suggested it
would: vLLM rollout spawned (weights synced via `/dev/shm` shared memory — "IPC is not
supported on your devices, falling back to shared memory for weight transfer"), reward
loop workers initialized, and it reached `"all initialize finished, ready to fit"`
(i.e., past all init, about to start the actual training step). Then:

```
(raylet) node_manager.cc:3496: 10 Workers (tasks / actors) killed due to memory
pressure (OOM), 0 Workers crashed due to other reasons
```

This is a **host RAM** OOM (Ray's own memory monitor), not GPU VRAM — host was at
202GiB/231GiB used with **swap (8GB) fully exhausted** at the time.

### Confound: this host was not exclusively running our job

`ps aux` on the *host* (not just inside the container) showed a concurrent, unrelated,
memory/compute-heavy job the entire time both tests ran:

```
sdp  ...  python train.py --repo-id-or-model-path openai/gpt-oss-20b --qlora \
          --lora-rank 8 ... --device xpu   (rl_qlora_grpo_biomedical_qa, PubMedQA)
sdp  ...  torch._inductor.compile_worker (32 workers, parent=that job)
```

A QLoRA GRPO run on a 20B model, also targeting `--device xpu`, with 32-way inductor
compile workers. This is a strong confound for **both** failures seen in this round:

- The host RAM OOM: that job's compile workers + model state compete for the same RAM
  pool as our CPU-offloaded FSDP state and vLLM's `/dev/shm` weight-transfer buffers
  (`/dev/shm/verl_weights_*`, ~2GB × 4 ranks = 8GB at the time of inspection).
- Possibly also the earlier `DEVICE_LOST` crash — two independent multi-process jobs
  contending for the same physical GPUs outside of Ray's/verl's awareness could plausibly
  destabilize the shared Level-Zero/driver state.

**No container-level memory limit was in effect** (`docker inspect` showed
`Memory=0`, i.e. unlimited; cgroup `memory.max` = `max`) — the constraint was true host
RAM exhaustion shared across tenants, not a cgroup ceiling.

## Open questions to resolve on isolated K8s GPU allocations

1. **Does PR #25 actually prevent device collisions at runtime** on the real Intel XPU
   platform + Ray, or only in the unit-tested mock? Verify by dumping
   `LOCAL_RANK`/`ZE_AFFINITY_MASK`/`ONEAPI_DEVICE_SELECTOR` from each live worker's
   `/proc/<pid>/environ` during `init_model`, and confirm 4 distinct physical ids.
2. **Does `optimizer_offload=False` complete a full training step cleanly** with no
   other tenant on the node? GPU margin was ~20MB free even without contention — worth
   watching specifically for a GPU-side `OutOfMemoryError` once host RAM contention is
   ruled out, since that margin is razor-thin regardless of who else is on the box.
3. **Does the `DEVICE_LOST` crash reproduce in isolation**, or was it purely a
   contention artifact from the other job? If it reproduces cleanly on an isolated K8s
   GPU allocation, that points back at something in verl/oneCCL/driver rather than
   host-sharing.
4. **K8s device-plugin GPU exposure differs materially from this bare Docker
   container.** Here, all 4 GPUs were visible to every process and isolation relied
   entirely on Ray's own env-var-based masking (the exact mechanism PR #25 patches). On
   K8s, the device plugin typically restricts each pod to only its assigned GPU(s) via
   cgroup device isolation — a different code path. This is arguably the *more*
   realistic environment for validating whether PR #25's fix is even reachable/needed,
   since some of the failure modes it targets (e.g., stale NOSET env var names) are
   specific to how Ray infers masking in a given deployment.

## Recommendation going in

- Keep `param_offload=True` — GPU memory margin without any offload is not comfortable
  for this HSDP (`fsdp_size=2`) + colocated-vLLM config on 24GB cards.
- Treat `optimizer_offload` as a speed/margin trade-off, not settled either way yet —
  re-test in isolation before dropping it.
- `enable_sleep_mode=True` was left on throughout (not implicated in either failure).
- Confirm no other GPU/RAM-heavy job is scheduled on the same node before attributing
  any new failure to verl/PR #25 itself.
