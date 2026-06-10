from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

DEFAULT_STEADY_WARMUP_STEPS = 5

PLOT_SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "overview": (
        ("scale/rollout_trainer_ratio", "steady/throughput_per_node_median"),
        ("scale/rollout_trainer_ratio", "steady/throughput_median"),
        ("scale/rollout_trainer_ratio", "steady/idle_imbalance_mean"),
        ("scale/rollout_trainer_ratio", "steady/update_to_gen_ratio_median"),
        ("scale/total_nodes", "steady/throughput_per_node_median"),
    ),
    "ratio": (
        ("scale/rollout_trainer_ratio", "steady/throughput_median"),
        ("scale/rollout_trainer_ratio", "steady/throughput_per_node_median"),
        ("scale/rollout_trainer_ratio", "steady/update_to_gen_ratio_median"),
        ("scale/rollout_trainer_ratio", "steady/trainer_idle_mean"),
        ("scale/rollout_trainer_ratio", "steady/rollouter_idle_mean"),
        ("scale/rollout_trainer_ratio", "steady/idle_imbalance_mean"),
        ("scale/rollout_trainer_ratio", "steady/param_sync_frac_median"),
        ("scale/rollout_trainer_ratio", "steady/queue_fill_frac_mean"),
        ("scale/rollout_trainer_ratio", "steady/active_tasks_fill_frac_mean"),
        ("scale/rollout_trainer_ratio", "steady/dropped_stale_frac"),
        ("scale/rollout_trainer_ratio", "steady/dropped_stale_samples_delta"),
    ),
    "scale": (
        ("scale/total_nodes", "steady/throughput_median"),
        ("scale/total_nodes", "steady/throughput_per_node_median"),
        ("scale/total_nodes", "steady/param_sync_frac_median"),
        ("scale/total_nodes", "steady/update_to_gen_ratio_median"),
        ("scale/total_nodes", "steady/trainer_idle_mean"),
        ("scale/total_nodes", "steady/rollouter_idle_mean"),
        ("scale/total_nodes", "steady/idle_imbalance_mean"),
        ("scale/total_nodes", "steady/queue_fill_frac_mean"),
        ("scale/total_nodes", "steady/active_tasks_fill_frac_mean"),
        ("scale/total_nodes", "steady/dropped_stale_frac"),
        ("scale/total_nodes", "steady/dropped_stale_samples_delta"),
    ),
}

STEADY_REDUCERS: tuple[tuple[str, str, Any], ...] = (
    ("steady/throughput_median", "perf/throughput", median),
    ("steady/throughput_per_node_median", "scale/throughput_per_node", median),
    ("steady/update_to_gen_ratio_median", "scale/update_to_gen_ratio", median),
    ("steady/param_sync_frac_median", "scale/param_sync_frac", median),
    ("steady/trainer_idle_mean", "fully_async/trainer/idle_ratio", mean),
    ("steady/rollouter_idle_mean", "fully_async/rollouter/idle_ratio", mean),
    ("steady/idle_imbalance_mean", "scale/idle_imbalance", mean),
    ("steady/queue_fill_frac_mean", "scale/queue_fill_frac", mean),
    ("steady/active_tasks_fill_frac_mean", "scale/active_tasks_fill_frac", mean),
    ("steady/pending_queue_mean", "fully_async/monitor/queue/pending_queue_size", mean),
)


def _get(config: Any, path: str) -> Any:
    value = config
    for part in path.split("."):
        if value is None:
            return None
        if hasattr(value, "get"):
            try:
                value = value.get(part, None)
                continue
            except TypeError:
                pass
        value = getattr(value, part, None)
    return value


def _num(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _div(numerator: Any, denominator: Any) -> float | None:
    numerator, denominator = _num(numerator), _num(denominator)
    if numerator is None or denominator is None or math.isclose(denominator, 0.0, abs_tol=1e-12):
        return None
    return numerator / denominator


def _compact(metrics: Mapping[str, float | None]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if value is not None}


def _first_metric(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics.get(name)
    return None


def iter_plot_pairs(
    sections: Mapping[str, Sequence[tuple[str, str]]] | None = None,
) -> list[tuple[str, str]]:
    sections = PLOT_SECTIONS if sections is None else sections
    return [pair for pairs in sections.values() for pair in pairs]


def compute_scale_config_metrics(config: Any) -> dict[str, float]:
    rollout_nodes = _num(_get(config, "actor_rollout_ref.rollout.nnodes"))
    trainer_nodes = _num(_get(config, "trainer.nnodes"))
    if rollout_nodes is None:
        rollout_nodes = _num(_get(config, "rollout.nnodes"))
    total_nodes = None if rollout_nodes is None or trainer_nodes is None else rollout_nodes + trainer_nodes
    return _compact(
        {
            "scale/rollout_nodes": rollout_nodes,
            "scale/trainer_nodes": trainer_nodes,
            "scale/total_nodes": total_nodes,
            "scale/rollout_trainer_ratio": _div(rollout_nodes, trainer_nodes),
        }
    )


def compute_scale_step_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    update_actor_s = _num(metrics.get("timing_s/update_actor"))
    param_sync_s = _num(
        _first_metric(
            metrics,
            "timing_s/param_sync",
            "timing_s/timing_s/param_sync",
        )
    )
    update_plus_sync_s = None if update_actor_s is None or param_sync_s is None else update_actor_s + param_sync_s
    trainer_idle = _num(metrics.get("fully_async/trainer/idle_ratio"))
    rollouter_idle = _num(metrics.get("fully_async/rollouter/idle_ratio"))
    idle_delta = None if trainer_idle is None or rollouter_idle is None else trainer_idle - rollouter_idle

    return _compact(
        {
            "scale/throughput_per_node": _div(metrics.get("perf/throughput"), metrics.get("scale/total_nodes")),
            "scale/update_plus_sync_s": update_plus_sync_s,
            "scale/update_to_gen_ratio": _div(update_plus_sync_s, metrics.get("timing_s/gen")),
            "scale/param_sync_frac": _div(param_sync_s, metrics.get("timing_s/step")),
            "scale/gen_frac": _div(metrics.get("timing_s/gen"), metrics.get("timing_s/step")),
            "scale/update_frac": _div(update_actor_s, metrics.get("timing_s/step")),
            "scale/idle_imbalance": None if idle_delta is None else abs(idle_delta),
            "scale/trainer_idle_minus_rollouter_idle": idle_delta,
            "scale/pending_queue_per_rollout_node": _div(
                metrics.get("fully_async/monitor/queue/pending_queue_size"),
                metrics.get("scale/rollout_nodes"),
            ),
            "scale/active_tasks_per_rollout_node": _div(
                metrics.get("fully_async/monitor/active_tasks_size"),
                metrics.get("scale/rollout_nodes"),
            ),
            "scale/queue_fill_frac": _div(
                metrics.get("fully_async/monitor/queue/pending_queue_size"),
                metrics.get("fully_async/static/max_queue_size"),
            ),
            "scale/active_tasks_fill_frac": _div(
                metrics.get("fully_async/monitor/active_tasks_size"),
                metrics.get("fully_async/static/max_concurrent_samples"),
            ),
        }
    )


def _series(history: Sequence[Mapping[str, Any]], metric_name: str) -> list[float]:
    return [
        value
        for row in history
        if (value := _num(row.get(metric_name))) is not None
    ]


def compute_steady_summary(
    history: Sequence[Mapping[str, Any]],
    warmup_steps: int = DEFAULT_STEADY_WARMUP_STEPS,
) -> dict[str, float]:
    steady_history = history[max(0, warmup_steps) :]
    if not steady_history:
        return {}

    summary = {
        output_name: reducer(values)
        for output_name, input_name, reducer in STEADY_REDUCERS
        if (values := _series(steady_history, input_name))
    }
    if dropped_stale := _series(steady_history, "fully_async/count/dropped_stale_samples"):
        summary["steady/dropped_stale_samples_delta"] = dropped_stale[-1] - dropped_stale[0]
        if total_generated := _series(steady_history, "fully_async/count/total_generated_samples"):
            generated_delta = total_generated[-1] - total_generated[0]
            if generated_delta > 0:
                summary["steady/dropped_stale_frac"] = (
                    summary["steady/dropped_stale_samples_delta"] / generated_delta
                )
    return summary
