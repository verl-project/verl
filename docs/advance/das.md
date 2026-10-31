# DAS: Distribution-Aware Speculative Decoding for Rollouts

**Author**: `https://github.com/kfswain`

Last updated: 10/20/2026

DAS ([arXiv 2511.13841](https://arxiv.org/abs/2511.13841)) speeds up RL rollouts with suffix-tree
speculative decoding that knows about the training loop: the same prompts recur every epoch and GRPO
samples several completions per prompt, so a per-problem suffix tree built from *earlier* completions of
the same problem predicts the next ones well. DAS is lossless (the sampled distribution is unchanged) and
only affects wall-clock time.

verl's implementation lives in `verl/experimental/das/` and rides vLLM's built-in suffix drafter
(`speculative_config.method=suffix`, backed by `arctic-inference`, which vLLM already requires for that
method). It is off by default and gated by one switch, `actor_rollout_ref.rollout.das.enable`.

> **Status: experimental.** vLLM only, one data-parallel shard per engine
> (`rollout.data_parallel_size=1`; multiple engine replicas are fine). Supported trainers: the v1
> TransferQueue agent loop (`main_ppo` default) and `verl/experimental/fully_async_policy`.

## 1. What it does

Two things on top of stock suffix decoding:

1. **Per-problem suffix trees.** A problem is identified by the hash of its first-turn prompt token ids.
   Every finished rollout is collected into a central, job-scoped store and redistributed to all engines
   as versioned deltas; each TP-rank worker keeps one small suffix tree per problem next to vLLM's own
   global cache. When drafting, the per-problem tree and the stock cache compete and the higher-scoring
   draft wins. Same-batch siblings (GRPO) see each other's tokens immediately through a transient
   self-insert.
2. **Length-aware draft budgets.** The store tracks each problem's generation length and classifies it
   Short / Medium / Long; the worker drafts up to `budget_short` / `budget_medium` / `budget_long` tokens
   per round (upgrade-only at runtime as the response grows). `budget_short: 0` means Short problems skip
   speculation entirely while their data is still collected.

## 2. Enabling

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    data_parallel_size: 1
    engine_kwargs:
      vllm:
        speculative_config:
          method: suffix
          num_speculative_tokens: 24          # >= das.budget_long
          suffix_decoding_max_tree_depth: 24
          suffix_decoding_max_spec_factor: 2.0
          suffix_decoding_min_token_prob: 0.1
          suffix_decoding_max_cached_requests: 2000
    das:
      enable: true
```

Command-line form:

```bash
python3 -m verl.trainer.main_ppo ... \
  actor_rollout_ref.rollout.das.enable=true \
  '+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config={method:suffix,num_speculative_tokens:24,suffix_decoding_max_tree_depth:24,suffix_decoding_max_spec_factor:2.0,suffix_decoding_min_token_prob:0.1}'
```

The draft cap, tree depth, spec factor and token-probability floor all come from vLLM's
`speculative_config`; `rollout.das.*` only configures the store, the budgets and the push path.

## 3. Configuration reference (`rollout.das`)

| key | default | meaning |
|---|---|---|
| `enable` | `false` | master switch |
| `window_iterations` | 8 | training iterations of trajectories kept in the store |
| `fresh_iterations` | 2 | newest iterations also carry a 2x-count shadow copy (recency weighting); 0 disables |
| `poll_interval_s` | 2.0 | engine delta-pump poll cadence |
| `max_delta_batch_tokens` | 500000 | cap per delta batch shipped to an engine |
| `budget_long` / `budget_medium` / `budget_short` | 24 / 8 / 0 | draft tokens per round by length class (also capped by `num_speculative_tokens`) |
| `short_max_tokens` / `long_min_tokens` | 256 / 1024 | length-class thresholds on the problem's smoothed generation length |
| `max_total_tokens` / `max_problems` / `max_seqs_per_problem` | 64M / 8192 / 512 | store caps (oldest iterations, LRU problems, oldest sequences) |
| `push_batch_size` / `push_flush_interval_s` / `push_queue_capacity` | 64 / 0.5 / 4096 | agent-loop worker -> store push batching |
| `service_name` | `verl_das_suffix_service` | Ray actor name of the store (job-scoped) |

## 4. How it attaches

| process | DAS code | what it does |
|---|---|---|
| trainer / rollouter | `DASAgentLoopManagerTQ`, `DASFullyAsyncAgentLoopManager` (`agent_loop_tq.py`, `agent_loop.py`) | creates the store (`SuffixDataService`, a named Ray actor owned by the job) and, in the TQ trainer, calls `begin_iteration(global_steps)` once per step |
| agent-loop workers | `DASClientMixin` via `wrap_llm_client` (`client.py`) | hashes the first-turn prompt into a problem id, names engine requests `dasp{hash}-{uuid}`, pushes response tokens to the store after `generate`; eval batches are gated off |
| vLLM server actor | `DeltaPump` (`pump.py`) | polls the store and applies delta batches to every TP rank with `collective_rpc("das_apply_deltas")` |
| TP-rank workers | `vLLMDASColocateWorkerExtension` (`worker_extension.py`), `proposer_patch.py`, `drafter_state.py` | installs the in-place patch of vLLM's `SuffixDecodingProposer` before the drafter exists; holds the per-problem trees and budgets |

**Step boundaries.** The store withholds the in-progress iteration: deltas only become servable when the
iteration advances, so applies land once per training step, right after the weights were synced and before
the new rollout decodes (the TQ manager calls `begin_iteration(global_steps)`; in the fully-async trainer
the first response carrying a new weight version opens the next iteration). Engines therefore never apply
trees mid-step, and all TP ranks stay byte-identical.

**Identity.** Problem ids are prefix-matched out of the request id by the proposer; any request without the
prefix (or under `rollout.full_determinism=true`, which keeps verl's stable request ids) goes through stock
suffix decoding unchanged.

## 5. Observability

- Store metrics: `SuffixDataService.get_metrics()` (`das_iteration`, `das_problem_count`, `das_total_tokens`,
  `das_withheld_entries`, ...), reachable with `ray.get_actor(service_name)`.
- vLLM's `vllm:spec_decode_*` metrics report drafts and acceptance as usual.
- Logs (with `VERL_LOGGING_LEVEL=INFO`): in each TP-rank worker `DAS: SuffixDecodingProposer patched in place`,
  `DAS: suffix proposer initialized with per-problem drafter state`, and one
  `DAS: applied delta batch -> store version V (iteration I, trees=T, drafts_emitted=D)` per training step; in the
  trainer process `DAS: store iteration I -> problems=P total_tokens=N log_version=V` at every step boundary.
  (The vLLM server actor's own INFO lines, including the pump's, are not surfaced by verl's logging setup.)

## 6. Limits and expectations

- Gains depend on acceptance, which depends on how repetitive the policy's outputs are under sampling. In
  our measurements per-problem trees mattered most when problems recur across epochs and when each engine
  runs below its KV limit; at an engine's concurrency limit speculation can be net negative. Measure on your
  workload with `das.enable` on and off; the reward curve must be identical.
- Host memory: about 300 bytes per stored token per TP-rank process plus the store itself
  (`max_total_tokens` bounds it).
- vLLM only (SGLang has no suffix drafter), `data_parallel_size=1` per engine, no persistence across jobs.
