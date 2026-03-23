---
name: spec-metrics
description: Extract and summarize decoupled speculation metrics from logs, traces, or experiment notes. Invoke explicitly with `/spec-metrics` when the user wants metric extraction, experiment summaries, or rollout performance breakdowns.
disable-model-invocation: true
---

# Spec Metrics

Use this skill only when explicitly invoked as `/spec-metrics`.

## Purpose

Extract and summarize metrics for decoupled speculation experiments.

## Inputs

Typical inputs include:

- pasted logs
- experiment output
- benchmark notes
- metric tables

## Metrics To Look For

- accept rate
- average accepted draft tokens
- fallback frequency
- verifier latency
- draft latency
- end-to-end throughput
- graph hit or miss indicators

## Output Format

1. Metric summary table
2. Observed anomalies
3. Most likely performance bottlenecks
4. Suggested next measurements

## Rules

- If a metric is inferred rather than directly logged, label it as inferred.
- If the logs are insufficient, say what extra fields should be logged next run.
