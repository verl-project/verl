# Search-R1 multi-message recipe, matched to sd-search inputs

This recipe uses verl v0.9.1 (`1876b06`) and Qwen2.5-3B Base, with a BaseTool
HTTP search client and separate assistant/tool messages. It remains a Search-R1
control: no SD hindsight or self-distillation loss is enabled.

For the experimental three-teacher skill-bank training task, see
[SKILL_TEACHERS.md](SKILL_TEACHERS.md) and `run_skill_grpo.sh`. The baseline
`run_grpo.sh` remains unchanged.

## Alignment reference

The numerical reference is `sd-search/configs/paper_3b.yaml`, composed by its
four-GPU launcher. The concrete reference is
`sd-search/outputs/sd-train-4gpu-resumable-20261122-r5/config.yaml`; the existing
`search-r1-matched-4gpu-20261122-r1/config.yaml` has the same common parameters
and disables the SD term. Reports and a reference copy are in
`../artifacts/sd-search-alignment/`.

| Setting | This recipe |
| --- | --- |
| Training questions per step | 256 |
| Trajectories per question | 5 |
| Trajectories per rollout batch | 1280 |
| New verl PPO mini-batch setting | 256 questions = 1280 trajectories |
| Actor micro-batch | Fixed 32 trajectories per GPU; dynamic batching disabled; sd-search reference uses 1 |
| Old-policy / reference log-prob micro-batch | 32 / 32 per GPU |
| Loss reduction | Sequence mean of each sequence's mean token loss |
| Actor learning rate | 1e-6, constant schedule, no warmup |
| PPO clip / KL coefficient | 0.2 / 0.001 (`low_var_kl`) |
| Entropy coefficient / PPO epochs | 0 / 1 |
| Actor parameter/optimizer CPU offload | Disabled |
| Reference parameter CPU offload | Enabled |
| Gradient checkpointing / remove padding | Enabled / disabled |
| Formal steps / epochs | 200 / 1 |
| Save / validate every | 50 / 50 steps; no validation before training |
| Validation batch | 128 questions |
| Temperature / top-p / top-k / seed | 1 / 1 / -1 / 42 |
| Initial prompt / single assistant turn | 1024 / 512 tokens |
| Observation budget | 512 tokens including `<documents>` wrapper |
| Search budget | Up to 3 searches, plus final assistant action |
| Top-k passages | 3 |
| vLLM TP / max sequences / batched tokens | 1 / 1024 / 8192 |
| vLLM eager execution / free cache | Enabled / enabled |

Old sd-search `ppo_mini_batch_size=1280` counts trajectories. New verl V1
multiplies the configured mini-batch by rollout.n, so copying 1280 would request
6400 trajectories. Setting 256 provides one optimizer mini-batch per rollout
batch, with 320 local trajectories on each of four DP ranks. Each rank performs
fixed-size micro-batches with gradient accumulation. At micro-batch 32, each rank
accumulates ten micro-batches per optimizer update. Old-policy and reference
log-prob micro-batches are independently configurable and also default to 32.
These training-side micro-batches do not control vLLM request concurrency.

There is one vLLM replica per GPU with TP=1. `max_num_seqs=1024` is a scheduler
ceiling, not a promise of 1024 simultaneously active sequences. A batch contains
1280 trajectories total, averaging 320 per replica; routing, tool waits and the
8192-token scheduler budget determine actual concurrency. AgentLoop workers are
CPU orchestrators and are not additional GPU training tasks.

## Explicit differences retained by the user

- Training reward remains **EM**, as does validation. sd-search uses F1 for
  training and EM for validation. Reward alignment was explicitly deferred;
  F1-labelled inputs are rejected before generation. There is no format bonus,
  search reward or neural reward model. The final assistant answer is scored
  against the gold aliases; retrieved text is never used as a prediction.
- vLLM memory utilization remains **0.60** as requested; the sd-search reference
  uses 0.45. This is the total vLLM memory budget, not a pure KV-cache fraction.
  The matched five-step run `search-r1-5step-20261122-131439` completed on four
  H200s with actor micro-batch 1 and memory budget 0.60. The subsequent micro-batch
  4 experiment retains that budget and all other training hyperparameters.
- Actor, old-policy log-prob and reference log-prob micro-batches are now all 32,
  as requested for the next throughput experiment. The preceding profiled run
  `search-r1-5step-20261122-134506` used 4 / 1 / 1; the sd-search reference and
  first matched GPU run use 1 / 1 / 1. Override `PPO_MICRO_BATCH_SIZE`,
  `OLD_LOG_PROB_MICRO_BATCH_SIZE` and `REF_LOG_PROB_MICRO_BATCH_SIZE` to reproduce
  an earlier setting. Each rank still processes 320 trajectories per global
  update, using ten accumulated actor micro-batches when the setting is 32.
- Entropy monitoring is disabled for this experiment (`trainer.log_old_policy_entropy=False`
  and `actor.calculate_entropy=False`). The entropy coefficient remains zero.
  This removes a monitoring-only dense softmax that exhausted H200 memory at
  micro-batch 32; old-policy/reference log-probabilities and the training objective
  are retained. Earlier timing comparisons included the entropy monitoring cost.
- Context uses genuine assistant/tool messages. The initial user prompt is
  preserved byte-for-byte from sd-search. Qwen's template renders tool messages
  as separate user turns containing `<tool_response>`; evidence is wrapped in
  `<documents>`. This differs from sd-search's single-assistant continuation.
- All history is retained. Generation stops if its input exceeds 4096 tokens;
  it does not copy the old engine's left-truncation of history. Initial prompts
  longer than 1024 tokens are rejected, not silently truncated. The whole
  trajectory storage budget is 4096 tokens and engine max_model_len is 5120,
  allowing for message boundaries and the final generation. Length fields across
  the two frameworks do not have identical meanings.
- The newer engine and library versions can change numerical behavior. Matching
  shared hyperparameters and data does not establish identical optimization.

## Data

The default directory is `../data/search_r1_sd_search/`. Files are generated from
`sd-search/data/processed/`, preserving exact source prompt text, questions,
answer aliases, source labels, split labels, original indices and sample order.
Only `agent_name` and extra AgentLoop metadata are added. Train contains 169615
questions; monitor contains the same 512 held-out questions used by sd-search.
`eval_all.parquet` contains its full 51713-question report set; all seven
individual evaluation files and smoke files are preserved as well.

From the verl root, prepare and verify these copies offline:

```bash
python3 -m examples.search_r1.prepare_sd_search_data
```

`datasets.json` records ordered payload hashes of source and adapted files.
Existing files must pass the same fingerprint and EM-routing checks. Sources
are never modified. Earlier author-aligned inputs remain under
`../data/search_r1_official/`; they are not the current launcher's default.

## Runtime

Use the mirrored image:

```text
registry.h.pjlab.org.cn/ailab-llmagent-llmagent_gpu_pool/verl-search:vllm024.dev2
sha256:476b81c1eeddca6ac68247e56840fac24a75cf20ab01803cec3d5e791ee92feb
```

The original allocation is 4 GPUs, 80 CPUs, 880 GiB host RAM (`--memory 901120`).
Set `N_GPUS=8` (or preparation helper `--gpus 8`) to train on eight GPUs while
keeping the same CPU/RAM allocation. The existing FP32 FAISS service uses four
GPUs selected by `RETRIEVAL_GPU_INDICES` (default `0,1,2,3`), shared with training.
The launcher resolves these indices to GPU UUIDs and scopes CUDA visibility to
the retrieval subprocess; training and telemetry retain the full allocation.
Retrieval uses the existing
sd-search Python/CUDA environment. The recipe's `retrieval_service` reuses its
corpus reader and four-GPU index, with explicit FP32 E5 parameters, no autocast,
and TF32 disabled. E5 defaults to `RETRIEVAL_ENCODER_DEVICE=cuda:0`, the first
GPU in the retrieval subprocess's UUID mask. Set `RETRIEVAL_ENCODER_DEVICE=cpu`
for the CPU control. Its FAISS overlay is
scoped to the retrieval subprocess. Training uses the image's Python and this
checkout on PYTHONPATH. The worker performs no downloads or package installs.

After preparing inputs and verifying the runtime, from the verl root:

```bash
bash examples/search_r1/run_grpo.sh
```

For a five-step training check with full-sized batches and no validation:

```bash
TOTAL_STEPS=5 bash examples/search_r1/run_grpo.sh \
  trainer.test_freq=-1 trainer.save_freq=5
```

The launcher starts its retrieval subprocess, waits for `/health`, and closes
that subprocess on exit. Set `START_RETRIEVER=0` only to use an already verified
service in the same network namespace. `RUN_DIR`, `SD_ROOT`, `MODEL_PATH`,
`DATA_DIR`, `VAL_FILE`, `TRAIN_PYTHON`, `RETRIEVER_PYTHON`, `TRAIN_BATCH_SIZE`,
`PPO_MINI_BATCH_SIZE`, `ROLLOUT_N`, `TOTAL_STEPS`, `N_GPUS`,
`RETRIEVAL_GPU_INDICES`, `RETRIEVAL_ENCODER_DEVICE` and `ROLLOUT_GPU_MEMORY` are
explicit overrides. `DRY_RUN=1` saves command.json without starting GPU work.
`ROLLOUT_ENFORCE_EAGER=False` enables the engine's CUDA graph path, while
`ROLLOUT_MAX_BATCHED_TOKENS=16384` increases the scheduler's token budget.
The five-step preparation helper accepts `--cuda-graphs
--max-num-batched-tokens 16384` for this comparison. Keep the 0.60 memory budget
and compare steady-step timings separately from graph compilation/capture time.

Console, retrieval startup/timing reports, rollout dumps and checkpoints are
saved under RUN_DIR. The rjob preparation helper applies five-step overrides
but inherits the matched batch and mini-batch defaults.

The AgentLoop validates ChatML boundary tokens once per worker tokenizer. A
bounded cache shares initial prompt tokenization between trajectories, including
concurrent requests for the same prompt. Keys include messages, template and
template options; each trajectory receives its own token list. Histories after
the initial prompt still use the same continuous-token builder and masks.

SearchTool batches up to 64 queued queries per HTTP request with a 2 ms
collection window and at most four requests in flight per worker. Responses are
mapped back by query position, preserving passage formatting and EM rewards.
Configure `batch_size: 1` and `batch_wait_ms: 0` to disable batching. Server
profiles attached to tool metrics describe the whole HTTP batch; they must not
be summed once per trajectory as independent service work.

From the parent directory, `scripts/prepare_five_step_job.py --telemetry
--rollout-verification --comparison-baseline <previous-run>` also prepares a
fixed-response AgentLoop equivalence benchmark and a real retrieval comparison
before training. It checks ordered passages and similarity scores for single
query versus batch-16 requests. Preparing the job does not submit it.

## Verification

From the directory above verl:

```bash
PYTHONPATH="$PWD/verl" .envs/search-r1-tests/bin/python -m pytest \
  verl/examples/search_r1/tests -q -o addopts=''
```

Tests use the real Qwen tokenizer and Continuous Token builder with fake
inference and local HTTP retrieval. They cover history boundaries, masks,
logprobs, budgets, reward provenance, deferred F1 routing, prompt preservation
and data order. CPU checks do not validate GPU memory or throughput.

### Adjacent search-decision hidden-state loss

Opt in independently of static skill teachers (disabled unless explicitly enabled):

```bash
bash examples/search_r1/run_skill_grpo.sh \
  +turn_hidden_diversity=search_r1 \
  turn_hidden_diversity.enabled=true \
  turn_hidden_diversity.margin=0.1 \
  turn_hidden_diversity.coefficient=0.001
```

The same options work with `run_grpo.sh` without skill teachers. During actor
updates, the final normalized hidden vector entering the LM head is selected at
`</think>` in each adjacent pair of valid search turns. Endpoints are located in
the actual sampled token IDs; no response is retokenized. Only endpoints ending
before the action tag are accepted, including a whitespace suffix on the closing
tag token. Answer/malformed/truncated turns break adjacency. Tools and padding
are excluded. Current-turn states are compared with detached previous-turn
states, using `relu(margin - (1 - cosine_similarity))` in FP32. The pair mean,
weighted by `coefficient`, is added to the existing PPO and optional skill losses.
Counts are taken across the optimizer mini-batch and DP group before splitting
micro-batches. A micro-batch with no pairs contributes a differentiable zero.

This first implementation supports the synchronous V1 FSDP/FSDP2 recipe, with
`use_remove_padding=False`, unfused kernels and sequence parallel size 1.
It captures only selected vectors during the existing actor forward; rollout,
old-policy/reference/teacher scoring do not capture hidden states. It does not
request all layers' hidden states. The previous endpoint is detached only in the
distance branch; current states can still backpropagate through causal attention.

Logged under `actor/turn_hidden/`: `loss` (unweighted pair mean), `weighted_loss`,
`cosine_mean`, `distance_mean`, `active_fraction`, `pairs`, `margin`, and
`coefficient`. `turn_hidden/prepared_pairs` counts usable pairs in the sampled
batch; `timing_s/turn_hidden_prepare` measures mask preparation. Actor metrics
aggregate over gradient accumulation and DP; repeated PPO epochs/mini-batches
follow the trainer's normal aggregation, so pair counts there describe optimizer
visits. Zero pairs yield zero means; consult pair counts when interpreting them.

Representation distance does not establish semantic progress or evidence gain.
A repeated query may already have a different state, and exact identical vectors
are a stationary point of cosine similarity. Monitor repeated queries, EM, the
active fraction, stage timings and memory in addition to this auxiliary loss.

The turn-hidden launcher now defaults to FP32 E5 on `cuda:0` (the first
retrieval-visible GPU), and teacher micro-batch 32 per GPU. Set
`RETRIEVAL_ENCODER_DEVICE=cpu` for a CPU control or
`TEACHER_MICRO_BATCH_SIZE=64` to test a larger teacher batch. Final Hydra CLI
overrides take precedence. FAISS remains sharded over four GPUs. GPU encoder
memory remains resident while teacher/actor run, so capacity checks must include
it. With 256 prompts, five rollouts and DP=4, the current fixed batch splitter
requires the teacher micro-batch to divide 320; 64, 80, 160 and 320 pass that
arithmetic constraint, whereas 128 does not. These are legal sizes, not verified
GPU capacity limits. Both 32 and 80 completed five-step runs; 32 scored faster,
while GPU E5 plus 80 reached about 139.95 GiB of device memory. The full-run
recipe therefore defaults to GPU E5 plus teacher micro-batch 32.
