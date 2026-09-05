# Evidence-driven communication phase autotuning

Last updated: 09/04/2026.

`scripts/autotune_communication_phase.py` turns semantic communication traces
into a conservative policy recommendation. It is an offline analysis tool: it
does not add sleeps, timers, or scheduling behavior to a training process.

The recommendation is keyed by framework, topology, operation pair, transport,
message sizes, and observed world size. Results from different workload keys
are never pooled.

## Required trace evidence

Pass every rank shard from every candidate run. Each JSONL record must contain:

- `framework`, `run_id`, `rank`, `operation`, `message_bytes`, `transport`, and
  `topology_class`;
- `world_size`, so an accidentally omitted highest-rank shard is detected
  directly;
- `gpu_start_timestamp_ns` and `gpu_end_timestamp_ns` from a common monotonic
  time domain across ranks;
- `gpu_timestamp_semantics=kernel-observed`, the shared `timestamp_domain`, and
  a measured non-negative `clock_sync_error_bound_us` for that domain;
- `consumer_timestamp_ns` for operation B;
- optionally, `critical_path_duration_us` on both records for the enclosing
  step's slowest-rank critical-path duration;
- `requested_offset_us` for an offset experiment, and optionally `policy`;
- semantic context fields that pair operation A and B, such as `iteration`,
  `microbatch`, and `layer`;
- stable `process_group_id` and non-negative `communicator_sequence_id` values
  for both operations;
- `metadata.completion_observed=true` for asynchronous operations;
- `sequence_consistent=false` when launch-order validation detects divergence.

The default pairing fields also cover weight-version and bucket traces. Override
them with `--pair-by` when a framework uses a different semantic boundary.
Every selected run must contain a complete, contiguous rank set beginning at
rank zero. The implementation has no topology-specific rank-count branch.
CUDA-event eligibility/completion brackets may be inspected with the same tool,
but they are rejected as policy evidence and produce `insufficient_evidence`.
Likewise, evidence whose measured clock error exceeds
`--max-clock-sync-error-us` cannot select or refine a policy.

## Decision rules

For each global A/B trial, the tuner derives:

```text
realized offset = gpu_start(B) - gpu_start(A)
consumer slack = consumer(B) - gpu_end(B)
pair completion = max_rank(end(A), end(B)) - min_rank(start(A), start(B))
rank skew = max launch/finish skew over A and B
```

When `critical_path_duration_us` is present on every rank and candidate, it is
the optimization objective. Otherwise the tuner uses pair completion as the
explicitly labelled fallback. Candidates using different objective sources are
not compared.

Negative consumer slack means B was still incomplete when its consumer arrived.
A candidate is rejected if its lower-tail slack is worse than the baseline
deadline guard, if added rank skew exceeds the remaining measured slack, if
logical sequence validation failed, if the requested and realized directions
disagree, or if it has too few complete trials.

Among the remaining candidates, the objective is lexicographic: minimize p95
critical path, then consumer wait, then rank skew. A non-baseline policy is
recommended only when a deterministic bootstrap confidence interval says its
critical-path improvement is positive. Trials with a unique shared semantic
context use paired mean improvement; otherwise the reported fallback is an
independent p95 comparison. This makes the tool fail closed under ambiguous or
noisy evidence.

## Adaptive refinement without a fixed delay

The tuner may propose another requested offset. Every proposal is the midpoint
of an already measured request interval. Its interpolated consumer slack must
remain inside the baseline-derived deadline guard, and the tuner never
extrapolates beyond the measured envelope. Run the proposed points, append the
new trace shards, and invoke the tuner again.

No default millisecond offset exists. The scale comes entirely from the
observed workload, GPU realization, and consumer deadline.

## Usage

```bash
python scripts/autotune_communication_phase.py \
  --trace-jsonl traces/candidate-*/rank-*.jsonl \
  --operation-a ep_dispatch_tokens \
  --operation-b dp_grad_reduce_scatter \
  --output-json phase-policy.json
```

The output includes all candidate summaries and rejection reasons, the baseline
and recommended settings, the realized offset of the recommendation, and any
data-derived refinement points with their interpolated realized offset and
deadline slack. A `keep_baseline` or `insufficient_evidence`
decision is a valid result and must not be converted into a phase shift by the
caller.

## Topology policy cells

Every recommendation also contains a content-addressed `topology_cell_id` and
its canonical `topology_fingerprint`. The fingerprint includes:

- single-node or multi-node scope;
- local PCIe, NVLink, or unknown fabric;
- world size and the exact rank-to-node placement, without embedding ephemeral
  hostnames;
- accelerator model by rank;
- an optional opaque topology inventory signature.

Single-node PCIe, single-node NVLink, and multi-node measurements therefore
land in separate cells. Multi-node evidence must include a consistent
`topology_signature`; node count alone is not sufficient to identify a cluster
topology. Runs with missing ranks, partially missing node identities, or a
declared class that disagrees with observed host placement are rejected.

Use the selector before applying a stored policy to a new run:

```bash
python scripts/communication_topology_policy.py \
  --policy-json phase-policy.json \
  --target-trace-jsonl current-run/rank-*.jsonl \
  --output-json compatible-policy.json
```

Selection is exact: every fingerprint field must match. There is no nearest
topology, model-family fallback, or cross-topology extrapolation. If no cell
matches, the command exits unsuccessfully and the workload must collect a new
baseline and candidate traces for its own cell.

NIC traffic classes, rail assignment, and route selection are outside this
policy layer. `topology_signature` is only an opaque compatibility identity; it
does not configure or infer any NIC behavior.

An aggregate phase-sweep summary may record isolated/contended latency and
GPU-realized offsets, but it has no consumer timestamp. Feed semantic trace
records to this tool; do not infer consumer slack from aggregate collective
latency.
