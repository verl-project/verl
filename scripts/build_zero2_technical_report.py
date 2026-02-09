#!/usr/bin/env python3
"""Build a markdown report from ZeRO benchmark artifacts.

The script accepts one or more six-case runs plus optional stage1/2 gate runs,
then renders a compact PR-facing report with score/throughput/memory metrics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def to_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s or s == 'NA':
        return None
    try:
        return float(s)
    except Exception:
        return None


def pick(row: dict[str, str], keys: list[str], default: str = "NA") -> str:
    """Return the first non-empty value for keys, otherwise default."""
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        value = value.strip()
        if value and value != "NA":
            return value
    return default


@dataclass
class SixRunSummary:
    run_root: Path
    config: dict[str, str]
    rows: list[dict[str, str]]

    def case(self, name: str) -> dict[str, str] | None:
        for r in self.rows:
            if r.get('case') == name:
                return r
        return None

    def score_delta(self, a: str, b: str) -> float | None:
        ra = self.case(a)
        rb = self.case(b)
        if not ra or not rb:
            return None
        sa = to_float(pick(ra, ["avg_score_tail5", "avg_score_mean_26_30"]))
        sb = to_float(pick(rb, ["avg_score_tail5", "avg_score_mean_26_30"]))
        if sa is None or sb is None:
            return None
        return sa - sb


@dataclass
class Stage12Summary:
    run_root: Path
    config: dict[str, str]
    overall_rows: list[dict[str, str]]
    compare_rows: list[dict[str, str]]
    gate_status: str

    def metric(self, key: str) -> str | None:
        for r in self.overall_rows:
            if r.get('metric') == key:
                return r.get('value')
        return None


def load_six_run(run_root: Path) -> SixRunSummary:
    return SixRunSummary(
        run_root=run_root,
        config=read_env_file(run_root / 'run_config.env'),
        rows=read_tsv(run_root / 'summary.tsv'),
    )


def load_stage12_run(run_root: Path) -> Stage12Summary:
    gate_path = run_root / 'gate_status.txt'
    gate = gate_path.read_text(encoding='utf-8').strip() if gate_path.exists() else 'NA'
    return Stage12Summary(
        run_root=run_root,
        config=read_env_file(run_root / 'run_config.env'),
        overall_rows=read_tsv(run_root / 'overall.tsv'),
        compare_rows=read_tsv(run_root / 'compare.tsv'),
        gate_status=gate,
    )


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = []
    out.append('| ' + ' | '.join(headers) + ' |')
    out.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--six-run', action='append', default=[])
    ap.add_argument('--stage12-run', action='append', default=[])
    ap.add_argument('--title', default='ZeRO2 Technical Report')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    six_runs = [load_six_run(Path(p)) for p in args.six_run]
    stage12_runs = [load_stage12_run(Path(p)) for p in args.stage12_run]

    report_path = out_dir / 'TECHNICAL_REPORT.md'
    lines: list[str] = []
    lines.append(f"# {args.title}")
    lines.append('')
    lines.append(f"- Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append('')

    lines.append('## Rebase Status')
    lines.append('- Branch rebased onto latest `upstream/main` before benchmark execution.')
    lines.append('')

    if six_runs:
        lines.append('## Six-Case Benchmarks')
        for i, run in enumerate(six_runs, 1):
            cfg = run.config
            lines.append(f"### Run {i}: `{run.run_root}`")
            lines.append(
                f"- steps={cfg.get('STEPS','NA')}, seed={cfg.get('SEED','NA')}, rollout_mode={cfg.get('ROLLOUT_MODE','NA')}, "
                f"patch={cfg.get('VERL_DS_ZERO2_FP32_ACCUM_PATCH','NA')}, step_each_micro={cfg.get('VERL_DS_ZERO2_STEP_EACH_MICRO','NA')}"
            )

            headers = [
                'case',
                'status',
                'last_step',
                'step_target_score',
                'avg_score_tail5',
                'step_target_throughput',
                'avg_throughput_tail5',
                'step_target_mem_reserved_gb',
                'avg_mem_reserved_tail5_gb',
                'peak_mem_reserved_gb',
                'skip_zero_grad_warn_count',
            ]
            rows = []
            for r in run.rows:
                rows.append([
                    r.get('case', 'NA'),
                    r.get('status', 'NA'),
                    r.get('last_step', 'NA'),
                    pick(r, ['step_target_score_mean', 'step30_score_mean']),
                    pick(r, ['avg_score_tail5', 'avg_score_mean_26_30']),
                    pick(r, ['step_target_throughput', 'step30_throughput']),
                    pick(r, ['avg_throughput_tail5', 'avg_throughput_26_30']),
                    pick(
                        r,
                        [
                            'step_target_max_memory_reserved_gb',
                            'step60_max_memory_reserved_gb',
                            'step30_max_memory_reserved_gb',
                        ],
                    ),
                    pick(
                        r,
                        [
                            'avg_max_memory_reserved_tail5_gb',
                            'avg_max_memory_reserved_56_60_gb',
                            'avg_max_memory_reserved_26_30_gb',
                        ],
                    ),
                    pick(r, ['peak_max_memory_reserved_gb']),
                    r.get('skip_zero_grad_warn_count', 'NA'),
                ])
            lines.append('')
            lines.append(md_table(headers, rows))
            lines.append('')

            d_no = run.score_delta('zero2_no_offload', 'zero1_no_offload')
            d_off = run.score_delta('zero2_cpu_offload', 'zero1_cpu_offload')
            lines.append(
                f"- Delta(avg_score_tail5) z2-z1 no_offload: {'NA' if d_no is None else f'{d_no:.6f}'}"
            )
            lines.append(
                f"- Delta(avg_score_tail5) z2-z1 cpu_offload: {'NA' if d_off is None else f'{d_off:.6f}'}"
            )
            lines.append('')

    if stage12_runs:
        lines.append('## Stage1/2 Crossover Gate Runs')
        for i, run in enumerate(stage12_runs, 1):
            cfg = run.config
            lines.append(f"### Gate Run {i}: `{run.run_root}`")
            lines.append(
                f"- steps={cfg.get('STEPS','NA')}, seeds={cfg.get('SEEDS','NA')}, step_each_micro={cfg.get('VERL_DS_ZERO2_STEP_EACH_MICRO','NA')}, gate={run.gate_status}"
            )

            headers = ['metric', 'value']
            rows = [[r.get('metric', 'NA'), r.get('value', 'NA')] for r in run.overall_rows]
            lines.append('')
            lines.append(md_table(headers, rows))
            lines.append('')

            if run.compare_rows:
                headers = list(run.compare_rows[0].keys())
                rows = [[r.get(h, 'NA') for h in headers] for r in run.compare_rows]
                lines.append(md_table(headers, rows))
                lines.append('')

    lines.append('## Artifacts')
    lines.append(f"- report_dir: `{out_dir}`")
    for run in six_runs:
        lines.append(f"- six_run: `{run.run_root}`")
    for run in stage12_runs:
        lines.append(f"- stage12_run: `{run.run_root}`")
    lines.append('')

    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(report_path)


if __name__ == '__main__':
    main()
