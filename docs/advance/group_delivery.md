# Synchronous GRPO with group delivery

Last updated: 10/27/2026.

`trainer.v1.trainer_mode=group_delivery` runs rollout and Megatron training on
separate GPU pools. It submits one fixed batch of prompts under policy version
`k`. Once all responses and rewards for a prompt are available, that complete
GRPO group runs forward/backward while other prompts are still generating.
Actor parameters remain unchanged until **every group in the original batch**
has contributed gradients. Then there is one optimizer step, one scheduler step,
and one weight synchronization before the next batch is submitted.

This differs from `separate_async`, which updates the actor per mini-batch and
can consume older policy versions. `group_delivery` does not prefetch next-step
prompts, refill failed groups, drop slow groups, or resume partially generated
trajectories across policy versions.

## Configuration

Use the standard `verl.trainer.main_ppo` entry point with these overrides:

```bash
model_engine=megatron \
trainer.v1.trainer_mode=group_delivery \
actor_rollout_ref.hybrid_engine=False \
algorithm.adv_estimator=grpo \
data.train_batch_size=64 \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_epochs=1 \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
actor_rollout_ref.rollout.nnodes=1 \
actor_rollout_ref.rollout.n_gpus_per_node=8 \
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
```

Set `trainer.nnodes` and `trainer.n_gpus_per_node` for the training pool, and
configure the dataset, model, rollout backend and parallel topology as usual.
The rollout pool is additional to the training pool. Polling for completed
groups defaults to 50 ms (`trainer.v1.group_delivery.poll_interval`).

`tests/special_e2e/run_v1_group_delivery.sh` is a three-GPU, two-step ALFWorld
launcher (two training GPUs and one rollout GPU). It takes `MODEL_PATH` and
`DATA_DIR` from the environment. See the [A6000 handoff](group_delivery_handoff.md)
for environment setup, the measured long-tail workload and acceptance commands.

For an overlap ablation, `trainer.v1.group_delivery.wait_for_full_batch=True`
holds delivery until every original group is ready, then uses the same worker
and per-group gradient accumulation. It is a measurement control, not another
training algorithm. Default: `False`.

## Gradient and failure semantics

The controller forwards one complete prompt group to all actor ranks at a time.
DP padding uses synthetic sequences with zero response masks. This allows a
single group to be delivered even when its response count is smaller than DP
size. All ranks execute the same collective sequence; they do not consume
independent queues. Small groups can still incur poor pipeline utilization.

Megatron's pipeline schedule finalizes its gradients after each call, including
DP reduce-scatter and, in some configurations, per-token gradient scaling.
Keeping old gradients in those buffers would process them again on the next
call. Instead, the worker copies each group's finalized gradients to a separate
FP32 accumulator and resets the schedule buffers. After all groups arrive, it
restores the weighted global mean directly to the native buffers and invokes the
optimizer. Gradient clipping and the scheduler run only at this final boundary.

For `token-mean`, group gradients are weighted by the group's valid response-token
count and divided by the whole batch's token count. For sequence means, the
weights are valid sequence counts. Consequently, short groups do not receive the
same weight as long groups under a token-mean objective. Padding is excluded from
both normalizers. The implementation costs one additional FP32 gradient buffer
and performs gradient communication per group; it prioritizes correctness over
minimizing communication.

If rollout or backward fails, the batch fails without an optimizer update. A
failed group is not replaced by a new prompt. Foreign, duplicate, incomplete,
and wrong-version groups are rejected. At training step `k`, each trajectory
must report prompt step `k` and both rollout weight-version bounds equal to `k-1`.
Checkpoints are taken only after the full update; mid-accumulation recovery is not supported. Restore from the previous
completed checkpoint after a failed run. An async checkpoint containing pending
prompt groups cannot be resumed in this mode.

## Supported scope

The initial implementation supports standard GRPO with vanilla PPO loss, one
epoch and one optimizer step per rollout batch. Token means, sequence means,
entropy regularization, reference-policy KL loss, TP, PP, static CP and native
Megatron DDP/distributed optimizer use the existing engine collectives. These
GPU paths have regression scripts but have not been run on the CPU development
machine; they require the acceptance matrix in the handoff. Rollout log probabilities or
recomputed old log probabilities can be used; actor weights are fixed for both.

The following combinations are rejected rather than silently changing the
objective:

- Non-Megatron backends, FP16, non-FP32 gradient buffers, Megatron FSDP,
  precision-aware optimizers, quantized training, parameter/optimizer offload, dynamic CP and VPP.
- MTP, nonzero MoE auxiliary/z losses, expert-bias updates, quantile balancing and Sinkhorn routing.
  MoE models with these features disabled can use ordinary expert parallelism.
- Critic training, distillation, KL-in-reward, batch-dependent policy losses,
  rollout importance/rejection correction, score centering, dynamic group
  filtering, custom samplers and rollout skipping.
- Variable `rollout.n` and multiple output trajectories per agent session.
  Ordinary multi-turn sessions returning one final trajectory are supported.

`seq-mean-token-sum-norm` additionally requires a positive, fixed
`actor_rollout_ref.actor.loss_scale_factor`. Group-local maximum sequence lengths
must not change the normalization of this objective.

## Verification

```bash
pytest -q tests/workers/test_group_gradient_accumulator_on_cpu.py \
  tests/workers/test_group_delivery_worker_on_cpu.py \
  tests/trainer/ppo/v1/test_group_delivery_on_cpu.py \
  tests/special_standalone/test_group_delivery_alfworld_on_cpu.py
```

The numerical tests compare a single full-batch GRPO-style update with streaming
groups of unequal response lengths and different completion orders. They check
the accumulated gradient, clipped gradient norm and AdamW parameter update.
Controller tests check that a ready group is consumed while another is still
running and that failures cannot commit a partial batch.

GPU acceptance includes `tests/models/test_group_delivery_megatron.py`, which
uses a tiny locally generated Qwen3 model, fixed responses, the real Megatron
optimizer and two consecutive steps. It compares owned FP32 gradient shards,
clipped gradient norms and optimizer updates for whole-batch versus delivered
groups. The matrix includes DP, TP, PP, CP, combinations and groups smaller than
DP. Cases requiring more GPUs than are visible are explicitly skipped.

`group_delivery_event` log records mark batch begin, group backward begin/end,
optimizer begin/end and weight sync begin/end. A backward begin with
`pending_groups > 0` is evidence of actual overlap. The metric
`training/groups_backward_started_during_rollout` counts such groups per batch.

The Megatron per-token loss path also preserves the remainder when distributing
real token counts across static CP ranks. Truncating every rank's share would
make the final gradient divisor depend on group/micro-batch boundaries.

## Relationship to existing work

This mode reuses the upstream Tinker split primitives, standalone rollout setup,
TransferQueue, GRPO advantage computation, token packing and Megatron schedules.
Its addition is the fixed-batch delivery/commit protocol and finalized-gradient
accumulator. The public PR audit and remaining GPU validation are recorded in
[the handoff](group_delivery_handoff.md).
