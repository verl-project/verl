#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plot per-replica KV-cache + MFU signals on one time-aligned figure.

Each metric is a :class:`Panel` subclass owning its own parsing, series transform,
axis style, and summary line; :func:`build_panels` returns the ordered list and the
plot/summary loops just iterate it — adding a panel is writing one subclass and
appending an instance. Eighteen panels share the time x-axis; the same replica colour
repeats down the figure so the eye tracks one replica vertically:

  1. KV Load (retained)        — kv_cache_load = retained_blocks / num_gpu_blocks
  2. vLLM usage_perc           — running-only fraction (1 - free/total)
  3. MFU                       — vLLM estimated_flops / peak_flops (realtime
                                 60s window average). --peak-tflops sets the
                                 denominator (default 560 = Atlas 800I A3 / NPU)
  4. running requests          — num_requests_running
  5. waiting requests          — num_requests_waiting
  6. cumulative gpu evictions  — retained_blocks drops between tally snapshots
  7. gpu prefix hit %          — windowed local prefix-cache hit rate
  8. prefill recompute tokens  — cache-miss prefill tokens over the window
  9. external-hit rate         — cross-replica (mooncake) hits / prefix lookups
 10. dispatched samples        — requests dispatched in the last 5 min (per-replica)
 11. completed samples         — requests completed in the last 5 min (per-replica)
 12. cumul. completed          — lifetime completed-request total per replica
                                 (raw cumulative counter, not a window delta)
 13. avg turn                  — avg dispatch-turn of samples dispatched in the
                                 last 5 min (per-replica) — re-dispatch / retry churn
 14. RPM                       — completed requests / min over the last 5 min (per-replica)
 15. avg prompt len            — avg dispatched prompt length over the last 5 min
                                 (per-replica) — request-size signal seen at dispatch
 16. route latency             — balancer route() scoring latency per dispatch
                                 (global single line; one point per acquire_server)
 17. route load (scored)       — per-replica load that drove the combined-score
                                 dispatch (all replicas; one point per dispatch)
 18. load (sticky overload chk)— load of the sticky-bound replica evaluated in
                                 is_overloaded (one replica per sticky dispatch)

Panels 1–15 are per-replica (one line each); panels 17–18 are also per-replica
(instantaneous, one point per dispatch). Panel 16 (route latency) is a single
global line — route() is a balancer-wide call, not per-replica. Panels 10–15
derive from the ``router-dispatch`` loguru line; panels 17–18 derive from
``route-load loads={...}`` and ``is-overload replica=... load=...`` lines
emitted by the kvcaware strategy — absent on sticky-win / least-inflight
dispatches, so the panels are skipped when the log has no such lines.

The panel hierarchy captures the reusable shapes under the leaf panels:
:class:`SlidingPanel` (a windowed per-interval rate — :class:`MFUPanel` is its
throughput instance), :class:`CumulativePanel` (a running total folded from
tally history — :class:`EvictPanel` is its drop-counter instance),
:class:`TrailingDeltaPanel` (a trailing-window aggregate of the per-replica
dispatch counters — dispatched/completed/avg-turn/RPM/avg-prompt-len are its
instances), and :class:`LoadPanel` (per-replica instantaneous load from
route-load / is-overload lines — panels 17–18 are its instances).

Sources are the ``vllm-evidence ...``, ``kv-events tally: ... retained_blocks/replica=...``,
``router-dispatch replica=... ...``, ``... routed to server=... route=<ms> ...``,
``route-load loads={...}``, and ``is-overload replica=... load=...``
loguru lines emitted by the kvcaware collector/balancer/strategy (parsed by :class:`LogParser`).
``usage=`` and ``flops=`` are optional
(older logs parse fine; those panels stay empty). LLM decode is bandwidth-bound,
so MFU is naturally low (10-40%) in decode-heavy phases — expected, not idle.

Usage:
    python plot_metrics.py LOG [LOG ...]
    python plot_metrics.py D.log --frac 0.3 --max-points 1500 --peak-tflops 750

The output image is written next to the first log as ``<log-name>.png``.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

# ---- fixed defaults (not exposed as CLI flags) -----------------------------
_YMAX = 1.05  # top of the 0-1 panels (load, usage)
_MFU_WINDOW_S = 60.0  # realtime MFU sliding-window length (last 1 min)
# Trailing-window length for the request-dispatch panels (dispatched /
# completed / avg-turn / RPM). 5 minutes, per the requirement.
_DISPATCH_WINDOW_S = 300.0
_TITLE = (
    "KV Load / usage / MFU / run / wait / evictions / prefix-hit / prefill-recompute / "
    "external-hit / dispatched / completed / cumul-completed / avg-turn / RPM / "
    "avg-prompt-len / route-latency / route-load-scored / sticky-overload-load (time-aligned)"
)


def _fmt_walltime(t_min, t_max) -> str:
    """Format the run walltime (log_t_max - log_t_min) as ``Xh Ym Zs``.

    Returns ``"-"`` if either bound is missing (no parseable timestamp at all).
    """
    if t_min is None or t_max is None:
        return "-"
    total = int((t_max - t_min).total_seconds())
    if total < 0:
        return "-"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


# ---- pipeline downsampling (generic, not metric-specific) -------------------
def _downsample(points: list, max_n: int) -> list:
    n = len(points)
    if max_n <= 0 or n <= max_n:
        return points
    idx = sorted(set(round(i) for i in (j * (n - 1) / (max_n - 1) for j in range(max_n))))
    return [points[i] for i in idx]


# ---- log parsing ------------------------------------------------------------
class LogParser:
    """Parses kvcaware-collector loguru lines into ``(ts, kind, fields)``.

    ``kind`` is ``'evidence'`` / ``'tally'`` / ``'dispatch'`` / ``'route'`` /
    ``'route_load'`` / ``'is_overload'``; ``fields`` is the regex groupdict;
    ``ts`` may be None (the caller skips it to keep the time axis honest).
    """

    # vllm-evidence replica=X kv=0.021 usage=0.450 run=240 wait=12 | TTFT=.. .. |
    #   prefill=3365 cached=41472 (hit=92.5%) decode=4175 external=0 flops=8400000000000000
    # `usage=` and `flops=` are optional so older logs still parse.
    _EVIDENCE = re.compile(
        r"vllm-evidence\s+replica=(?P<rep>\S+)\s+kv=(?P<kv>\S+)"
        r"(?:\s+usage=(?P<usage>\S+))?"
        r"\s+run=(?P<run>\S+)\s+wait=(?P<wait>\S+)"
        r".*?prefill=(?P<pre>\d+)\s+cached=(?P<cac>\d+)\s+\(hit=(?P<hit>\S+?)\)"
        r"\s+decode=\d+\s+external=(?P<ext>\d+)(?:\s+flops=(?P<flops>-?\d+))?"
    )
    # Anchor on `retained_blocks/replica=` so the earlier events={...} dict isn't captured.
    _TALLY = re.compile(r"retained_blocks/replica=(?P<d>\{[^}]*\})")
    # router-dispatch replica=<id> dispatched=<cumul> completed=<cumul> turn_sum=<cumul>
    #   prompt_len_sum=<cumul> ...
    # Carries per-replica cumulative counters; the plot derives trailing-window
    # panels from their per-replica deltas. The trailing [dispatch #N] is ignored.
    # prompt_len_sum is optional so older logs (pre-request-length tracking) still
    # parse — it defaults to 0 and the avg-prompt-len panel stays flat/empty.
    _DISPATCH = re.compile(
        r"router-dispatch\s+replica=(?P<rep>\S+)\s+dispatched=(?P<dispatched>\d+)"
        r"\s+completed=(?P<completed>\d+)\s+turn_sum=(?P<turn_sum>\d+)"
        r"(?:\s+prompt_len_sum=(?P<prompt_len_sum>\d+))?"
    )
    # request=<uuid> routed to server=<addr> (ranking=.., pool=.., route=<ms>, strategy=[..])
    # route= is the per-dispatch route() scoring latency in ms; one line per acquire_server.
    _ROUTE = re.compile(r"routed to server=(?P<srv>\S+).*?route=(?P<route>[\d.]+)ms")
    # route-load loads={'s0': 0.4213, 's1': 0.1822} — per-replica load (all replicas) that drove
    # a combined-score dispatch, emitted once per such dispatch from score(). dict form mirrors
    # the tally line (ast.literal_eval). Absent on sticky-win / least-inflight dispatches.
    _ROUTE_LOAD = re.compile(r"route-load\s+loads=(?P<d>\{[^}]*\})")
    # is-overload replica=<id> load=<val> — the load of the sticky-bound replica that the
    # overload check (is_overloaded) evaluated, one replica per sticky-check dispatch.
    _IS_OVERLOAD = re.compile(r"is-overload\s+replica=(?P<rep>\S+)\s+load=(?P<load>[\d.]+)")
    _TS_LOGURU = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,.]\d+)?")
    _TS_VLLM = re.compile(r"INFO\s+(?P<ts>\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    _ASSUMED_YEAR = 2026  # vLLM engine logs omit the year; router (loguru) logs carry a full date.
    _ANCHORS = (
        "vllm-evidence",
        "retained_blocks/replica=",
        "router-dispatch",
        "routed to server",
        "route-load loads=",
        "is-overload replica=",
    )

    @classmethod
    def is_signal(cls, line: str) -> bool:
        """Cheap pre-filter so noise lines skip the regex scan entirely."""
        return any(a in line for a in cls._ANCHORS)

    @classmethod
    def parse(cls, line: str):
        m = cls._EVIDENCE.search(line)
        if m:
            return cls._ts(line), "evidence", m.groupdict()
        m = cls._TALLY.search(line)
        if m:
            return cls._ts(line), "tally", m.groupdict()
        m = cls._DISPATCH.search(line)
        if m:
            return cls._ts(line), "dispatch", m.groupdict()
        m = cls._ROUTE.search(line)
        if m:
            return cls._ts(line), "route", m.groupdict()
        m = cls._ROUTE_LOAD.search(line)
        if m:
            return cls._ts(line), "route_load", m.groupdict()
        m = cls._IS_OVERLOAD.search(line)
        if m:
            return cls._ts(line), "is_overload", m.groupdict()
        return None

    @classmethod
    def _ts(cls, line: str):
        m = cls._TS_LOGURU.search(line)
        if m:
            return datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
        m = cls._TS_VLLM.search(line)
        if m:
            return datetime.strptime(f"{cls._ASSUMED_YEAR} {m.group('ts')}", "%Y %m-%d %H:%M:%S")
        return None


# ---- panels -----------------------------------------------------------------
class Panel:
    """One time-aligned subplot.

    Subclass to customise behaviour; the base draws one solid line per replica
    from ``data[self.key]`` and leaves the series unchanged.

    - :meth:`extract`   — value from a parsed evidence line (None to skip / not evidence-derived).
    - :meth:`transform` — post-process the per-replica series after truncation, before downsample.
    - :meth:`derive`    — build the series from non-evidence input (e.g. tally history); default None.
    - :meth:`draw` / :meth:`summarize` — render and report.
    """

    def __init__(
        self,
        key: str,
        ylabel: str,
        *,
        height: float = 2.0,
        ylim_top: float | None = None,
        hline: float | None = None,
        summary=None,
    ):
        self.key = key
        self.ylabel = ylabel
        self.height = height
        self.ylim_top = ylim_top
        self.hline = hline
        self._summary = summary  # callable(points) -> str

    def points_for(self, data: dict, rep: str) -> list:
        return data.get(self.key, {}).get(rep, [])

    # -- per-line / per-series behaviour (override in subclasses) --
    def extract(self, fields: dict):
        return None

    def transform(self, points: list, peak_flops: float) -> list:
        return points

    def derive(self, retained: dict):
        return None

    def derive_dispatch(self, dispatch: dict):
        """Build the per-replica series from the dispatch buffer; default None.

        Parallel to :meth:`derive` (which feeds off the per-replica retained
        history): dispatch panels override this to fold each replica's
        ``[(ts, dispatched, completed, turn_sum), ...]`` buffer into a
        ``{replica: [(ts, val), ...]}`` series (one line per replica).
        """
        return None

    def derive_route(self, route_lat):
        """Build the series from the global route-latency buffer; default None.

        Parallel to :meth:`derive_dispatch`: route panels override this to fold
        the flat global ``[(ts, ms), ...]`` buffer into a series. Default None
        so the main loop can call it uniformly on every panel.
        """
        return None

    def derive_loads(self, route_load, is_overload):
        """Build the series from the route-load / is-overload buffers; default None.

        Both buffers are ``{replica: [(ts, load), ...]}``. Load panels override
        this, picking one buffer via ``self.source``. Default None so the main
        loop can call it uniformly on every panel.
        """
        return None

    # -- render --
    def draw(self, ax, data, colors, order) -> None:
        for rep in order:
            pts = self.points_for(data, rep)
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    color=colors[rep],
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.9,
                )
        if self.hline is not None:
            ax.axhline(self.hline, color="red", linestyle=":", linewidth=1.0, alpha=0.5)
        ax.set_ylabel(self.ylabel)
        if self.ylim_top is not None:
            ax.set_ylim(top=self.ylim_top)
        ax.grid(True, alpha=0.3)

    def summarize(self, data: dict, rep: str) -> str:
        return self._summary(self.points_for(data, rep)) if self._summary else ""

    # -- numeric helpers over a [(ts, val)] series --
    @staticmethod
    def to_float(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def last_val(pts):
        return pts[-1][1] if pts else float("nan")

    @staticmethod
    def minimum(pts):
        return min((p[1] for p in pts), default=float("nan"))

    @staticmethod
    def maximum(pts):
        return max((p[1] for p in pts), default=float("nan"))

    # -- summary-formatter factories (per-instance summary strategies) --
    @staticmethod
    def rng(name: str, fmt: str):
        return lambda b: (
            f"{name}[min={Panel.minimum(b):{fmt}} max={Panel.maximum(b):{fmt}} last={Panel.last_val(b):{fmt}}]"
        )

    @staticmethod
    def last(name: str, fmt: str, suffix: str = ""):
        return lambda b: f"{name}[last={Panel.last_val(b):{fmt}}{suffix}]"

    @staticmethod
    def cum(name: str):
        return lambda b: f"{name}[cum={Panel.last_val(b)}]"

    # -- value helpers for FieldPanel --
    @staticmethod
    def field(name: str):
        return lambda f: Panel.to_float(f.get(name))

    @staticmethod
    def norm_usage(v: float) -> float:
        """usage_perc is 0-1 (vLLM docs); tolerate a stray 0-100 from older scrapes."""
        if v == v and v > 1.5:
            return v / 100.0
        return v

    @staticmethod
    def ext_rate(f: dict) -> float:
        denom = int(f["pre"]) + int(f["cac"])
        return (int(f["ext"]) / denom) if denom > 0 else 0.0


class FieldPanel(Panel):
    """Panel whose per-line value comes from one evidence field (or a callable over fields)."""

    def __init__(self, key, ylabel, value, **kw):
        super().__init__(key, ylabel, **kw)
        self._value = value  # field name (str) or callable(fields) -> value | None

    def extract(self, fields: dict):
        if callable(self._value):
            return self._value(fields)
        v = fields.get(self._value)
        return Panel.to_float(v) if v is not None else None


class SlidingPanel(Panel):
    """Sliding-window statistics panel.

    Each output point is the value at ``t_j`` divided by its own elapsed interval
    ``(t_j - t_{j-1})``, averaged over a ``~window_s`` sliding window, then scaled
    to ``[0, 1]`` by ``peak`` (passed in as ``peak_flops`` from the prepare step).
    Subclasses fix the per-snapshot value via :meth:`extract`; :class:`MFUPanel` is
    the realtime-throughput instance (per-snapshot FLOPs → MFU).
    """

    def __init__(self, key: str, ylabel: str, window_s: float, **kw):
        super().__init__(key, ylabel, **kw)
        self.window_s = window_s

    def transform(self, points: list, peak_flops: float) -> list:
        return self._windowed_rate(points, self.window_s, peak_flops)

    @staticmethod
    def _windowed_rate(pts: list, window_s: float, peak: float) -> list:
        """Windowed average of ``value_j / (t_j - t_{j-1})`` over ``~window_s``, scaled by ``peak``.

        Each consecutive pair becomes throughput over its own interval; a sliding
        window then averages that throughput over ``~window_s`` — not a raw sum
        over a wall-clock band, which overcounts when >1 point lands in-band.
        """
        if len(pts) < 2 or peak <= 0:
            return []
        pts = sorted(pts, key=lambda p: p[0])
        win_q: deque = deque()
        dur_sum = val_sum = 0.0
        out = []
        for j in range(1, len(pts)):
            t1, v1 = pts[j]
            dur = (t1 - pts[j - 1][0]).total_seconds()
            if dur <= 0:
                continue
            dv = v1 if v1 == v1 else 0.0
            win_q.append((dur, dv))
            dur_sum += dur
            val_sum += dv
            # Keep ~window_s of coverage: drop oldest while the remainder still covers >= window_s.
            while len(win_q) > 1 and dur_sum - win_q[0][0] >= window_s:
                d_old, v_old = win_q.popleft()
                dur_sum -= d_old
                val_sum -= v_old
            if dur_sum > 0:
                out.append((t1, (val_sum / dur_sum) / peak))
        return out


class MFUPanel(SlidingPanel):
    """MFU: realtime FLOPs throughput as a fraction of peak, via a fixed sliding window."""

    def __init__(self, ylabel: str, **kw):
        super().__init__("mfu", ylabel, _MFU_WINDOW_S, **kw)

    def extract(self, fields: dict):
        return float(fields["flops"]) if fields.get("flops") is not None else None


class CumulativePanel(Panel):
    """Cumulative panel derived from the retained-blocks tally history.

    Walks each replica's history and folds every step into a monotonically
    non-decreasing total via :meth:`accumulate`; :meth:`derive` itself is generic.
    :class:`EvictPanel` counts the drops (block evictions); override
    :meth:`accumulate` for other cumulative semantics.
    """

    def derive(self, retained: dict):
        out = {}
        for rep, hist in retained.items():
            if not hist:
                continue
            cum = 0
            prev = hist[0][1]
            pts = []
            for t, n in hist:
                cum += self.accumulate(prev, n)
                pts.append((t, cum))
                prev = n
            out[rep] = pts
        return out

    @staticmethod
    def accumulate(prev, n):
        """Amount to add to the running total this step. Default: the drop when ``n < prev``."""
        return prev - n if n < prev else 0


class EvictPanel(CumulativePanel):
    """Cumulative GPU block evictions, derived from retained_blocks tally history."""

    def __init__(self, **kw):
        super().__init__("evict", "total gpu block evict", **kw)


# ---- request-dispatch (per-replica) panels ---------------------------------
def _trailing_deltas(dispatch: list, window_s: float):
    """Trailing-window deltas of the per-replica cumulative dispatch counters.

    ``dispatch`` is a list of ``(ts, dispatched, completed, turn_sum, prompt_len_sum)``
    sorted by ``ts``. For each point we subtract the value observed at the latest point
    whose timestamp is ``<= ts - window_s`` (0 before the run starts) — i.e. the
    count of events that landed inside the trailing ``window_s`` window. Returns
    a list of ``(ts, dispatched_delta, completed_delta, turn_sum_delta, prompt_len_sum_delta)``.

    O(n²) over the number of dispatch lines, which is tiny for an offline plot
    pass (the line is time-throttled to ~every 5 s on the producer side).
    """
    out = []
    for i, (ts, d, c, s, p) in enumerate(dispatch):
        cutoff = ts - timedelta(seconds=window_s)
        bd = bc = bs = bp = 0
        for j in range(i, -1, -1):
            if dispatch[j][0] <= cutoff:
                _, bd, bc, bs, bp = dispatch[j]
                break
        out.append((ts, d - bd, c - bc, s - bs, p - bp))
    return out


class TrailingDeltaPanel(Panel):
    """Base for the per-replica request-dispatch panels.

    Subclasses pick one column out of :func:`_trailing_deltas` via
    :meth:`select`. These panels do NOT read per-line evidence fields — they
    derive entirely from the per-replica ``router-dispatch`` cumulative
    buffers — so they are skipped by the evidence extraction loop (like
    ``CumulativePanel``). Produces one series per replica.
    """

    def derive_dispatch(self, dispatch: dict):
        out = {}
        for rep, hist in dispatch.items():
            if not hist:
                continue
            deltas = _trailing_deltas(hist, _DISPATCH_WINDOW_S)
            pts = [(ts, self.select(dd, dc, ds, dp)) for ts, dd, dc, ds, dp in deltas]
            out[rep] = pts
        return out

    @staticmethod
    def select(
        dispatched_delta: float,
        completed_delta: float,
        turn_sum_delta: float,
        prompt_len_sum_delta: float,
    ) -> float:
        raise NotImplementedError


class DispatchedPanel(TrailingDeltaPanel):
    """Requests dispatched in the last 5 min (trailing-window delta)."""

    def __init__(self, **kw):
        super().__init__(
            "dispatched",
            f"dispatched samples\n(last {int(_DISPATCH_WINDOW_S / 60)} min)",
            summary=Panel.last("dispatched", ".0f"),
            **kw,
        )

    @staticmethod
    def select(dispatched_delta, completed_delta, turn_sum_delta, prompt_len_sum_delta):
        return float(dispatched_delta)


class CompletedPanel(TrailingDeltaPanel):
    """Requests completed in the last 5 min (trailing-window delta)."""

    def __init__(self, **kw):
        super().__init__(
            "completed",
            f"completed samples\n(last {int(_DISPATCH_WINDOW_S / 60)} min)",
            summary=Panel.last("completed", ".0f"),
            **kw,
        )

    @staticmethod
    def select(dispatched_delta, completed_delta, turn_sum_delta, prompt_len_sum_delta):
        return float(completed_delta)


class CumulativeCompletedPanel(Panel):
    """Cumulative completed requests per replica — the raw lifetime total.

    Unlike :class:`CompletedPanel` (a trailing-5-min window delta), this plots
    the cumulative ``completed`` counter verbatim at each snapshot: how many
    requests that replica has completed so far, lifetime. One monotonically
    rising line per replica — the realized-throughput share over the whole run.
    """

    def __init__(self, **kw):
        super().__init__(
            "completed_total",
            "cumul. completed\n(lifetime total)",
            summary=Panel.cum("completed_total"),
            **kw,
        )

    def derive_dispatch(self, dispatch: dict):
        """Fold each replica's history into ``(ts, completed_cumulative)``.

        The dispatch buffer already carries the cumulative ``completed``
        counter as its third element, so this panel just reads it directly —
        no windowed delta (that is :class:`CompletedPanel`'s job).
        """
        out = {}
        for rep, hist in dispatch.items():
            if not hist:
                continue
            pts = [(ts, float(c)) for ts, _d, c, _s, _p in hist]
            out[rep] = pts
        return out


class AvgTurnPanel(TrailingDeltaPanel):
    """Average turn of dispatched samples in the last 5 min (turn_sum/dispatched)."""

    def __init__(self, **kw):
        super().__init__(
            "avg_turn",
            f"avg turn\n(dispatched, last {int(_DISPATCH_WINDOW_S / 60)} min)",
            summary=Panel.last("avg_turn", ".3f"),
            **kw,
        )

    @staticmethod
    def select(dispatched_delta, completed_delta, turn_sum_delta, prompt_len_sum_delta):
        # avg turn over the window = (turns accumulated) / (dispatches accumulated).
        # No dispatches in the window → NaN (matplotlib skips the gap).
        return float(turn_sum_delta) / dispatched_delta if dispatched_delta > 0 else float("nan")


class RPMPanel(TrailingDeltaPanel):
    """Requests-per-minute over the last 5 min (completed / window minutes)."""

    def __init__(self, **kw):
        super().__init__(
            "rpm",
            f"RPM\n(completed, last {int(_DISPATCH_WINDOW_S / 60)} min)",
            summary=Panel.last("rpm", ".1f"),
            **kw,
        )

    @staticmethod
    def select(dispatched_delta, completed_delta, turn_sum_delta, prompt_len_sum_delta):
        window_min = _DISPATCH_WINDOW_S / 60.0
        return float(completed_delta) / window_min


class AvgPromptLenPanel(TrailingDeltaPanel):
    """Average dispatched prompt length in the last 5 min (prompt_len_sum/dispatched).

    The request-size signal the router sees at dispatch time: cumulative
    ``len(prompt_ids)`` over the window's dispatched samples divided by the
    window's dispatched count. Mirrors :class:`AvgTurnPanel`'s
    sum-over-counters shape — prompt_len_sum is to request-size what turn_sum
    is to re-dispatch churn.
    """

    def __init__(self, **kw):
        super().__init__(
            "avg_prompt_len",
            f"avg prompt len\n(dispatched, last {int(_DISPATCH_WINDOW_S / 60)} min)",
            summary=Panel.last("avg_prompt_len", ".0f"),
            **kw,
        )

    @staticmethod
    def select(dispatched_delta, completed_delta, turn_sum_delta, prompt_len_sum_delta):
        # avg prompt len over the window = (prompt lengths accumulated) / (dispatches).
        # No dispatches in the window → NaN (matplotlib skips the gap).
        return float(prompt_len_sum_delta) / dispatched_delta if dispatched_delta > 0 else float("nan")


class RouteLatencyPanel(Panel):
    """Global per-dispatch route() latency — one line, one point per dispatch.

    Unlike every other panel, this is a single global series: route() is a
    balancer-wide scoring call, not per-replica. It stores under the pseudo-
    replica key ``_GLOBAL`` so the existing ``{replica: [(ts, val)]}`` series
    structure is reused unchanged, and ``draw()`` plots one line instead of
    iterating the replica order. Instantaneous values (one point per dispatch);
    no windowed aggregation.
    """

    _GLOBAL = "__global__"

    def __init__(self, **kw):
        super().__init__(
            "route_lat", "route latency\n(ms per dispatch)", summary=Panel.last("route", ".2f", "ms"), **kw
        )

    def derive_route(self, route_lat):
        """Fold the flat global [(ts, ms)] buffer into a single-replica series."""
        return {self._GLOBAL: list(route_lat)} if route_lat else None

    def draw(self, ax, data, colors, order) -> None:
        pts = data.get(self.key, {}).get(self._GLOBAL, [])
        if pts:
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color="tab:red",
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                label="route latency",
            )
        ax.set_ylabel(self.ylabel)
        ax.grid(True, alpha=0.3)

    def summarize(self, data, rep):
        # Global series lives under _GLOBAL, not under a replica — ignore `rep`.
        pts = data.get(self.key, {}).get(self._GLOBAL, [])
        return self._summary(pts) if self._summary else ""


class LoadPanel(Panel):
    """Base for per-replica load panels fed from route-load / is-overload buffers.

    Subclasses set ``source`` to ``"route_load"`` or ``"is_overload"``; the
    common :meth:`derive_loads` hook picks the right buffer.  Both draw one
    line per replica (instantaneous load ∈ [0,1]) and a horizontal line at the
    overload threshold — each subclass's ``__init__`` sets ``hline``.
    """

    source: str  # set by subclass: "route_load" or "is_overload"

    def derive_loads(self, route_load, is_overload):
        buf = route_load if self.source == "route_load" else is_overload
        return {rep: list(pts) for rep, pts in buf.items()} or None


class RouteLoadPanel(LoadPanel):
    """Per-replica load that drove the combined-score dispatch (all replicas)."""

    def __init__(self, *, load_threshold: float, **kw):
        super().__init__(
            "route_load_scored",
            f"route load (scored)\n(threshold={load_threshold:.2f})",
            ylim_top=1.0,
            hline=load_threshold,
            summary=Panel.last("rload", ".3f"),
            **kw,
        )
        self.source = "route_load"


class StickyOverloadPanel(LoadPanel):
    """Load of the sticky-bound replica evaluated in is_overloaded (one replica per dispatch)."""

    def __init__(self, *, load_threshold: float, **kw):
        super().__init__(
            "sticky_overload_load",
            f"load (sticky overload chk)\n(threshold={load_threshold:.2f})",
            ylim_top=1.0,
            hline=load_threshold,
            summary=Panel.last("sload", ".3f"),
            **kw,
        )
        self.source = "is_overload"


def build_panels(peak_tflops: float, load_threshold: float = 0.9) -> list[Panel]:
    """The ordered panel list (plot order = axis order = summary order)."""
    return [
        FieldPanel(
            "load",
            "KV Load\n(retained_blocks / num_gpu_blocks)",
            Panel.field("kv"),
            height=2.0,
            ylim_top=_YMAX,
            hline=1.0,
            summary=Panel.rng("load", ".3f"),
        ),
        FieldPanel(
            "usage",
            "vLLM usage_perc\n(running blocks / num_gpu_blocks)",
            lambda f: Panel.norm_usage(Panel.to_float(f.get("usage"))),
            height=2.0,
            ylim_top=_YMAX,
            hline=1.0,
            summary=Panel.rng("usage", ".3f"),
        ),
        MFUPanel(
            f"MFU\n(realtime {int(_MFU_WINDOW_S)}s window)\npeak={peak_tflops:.0f} TFLOPS/NPU",
            height=2.4,
            summary=Panel.last("mfu", ".3f"),
        ),
        FieldPanel(
            "run", "running requests\n(num_requests_running)", Panel.field("run"), summary=Panel.last("run", ".0f")
        ),
        FieldPanel(
            "wait", "waiting requests\n(num_requests_waiting)", Panel.field("wait"), summary=Panel.last("wait", ".0f")
        ),
        EvictPanel(height=2.4, summary=Panel.cum("evict")),
        FieldPanel(
            "hit",
            "gpu prefix hit %",
            lambda f: Panel.to_float(str(f["hit"]).rstrip("%")),
            ylim_top=100.0,
            summary=Panel.last("hit", ".1f", "%"),
        ),
        FieldPanel(
            "prefill",
            "prefill recompute\n(cache-miss tokens / window)",
            lambda f: int(f["pre"]),
            height=2.4,
            summary=Panel.last("prefill", ""),
        ),
        FieldPanel("ext", "external-hit", Panel.ext_rate, ylim_top=1.05, summary=Panel.last("ext", ".4f")),
        # ── Per-replica request-dispatch panels (one line per replica) ──
        DispatchedPanel(height=2.0),
        CompletedPanel(height=2.0),
        CumulativeCompletedPanel(height=2.0),
        AvgTurnPanel(height=2.0),
        RPMPanel(height=2.0),
        AvgPromptLenPanel(height=2.0),
        RouteLatencyPanel(height=2.0),
        # ── Per-replica per-dispatch load panels (one line per replica) ──
        RouteLoadPanel(height=2.0, load_threshold=load_threshold),
        StickyOverloadPanel(height=2.0, load_threshold=load_threshold),
    ]


# ---- pipeline ---------------------------------------------------------------
class Bundle:
    def __init__(
        self,
        series,
        retained,
        dispatch,
        route_lat,
        route_load,
        is_overload,
        replicas,
        t_min,
        t_max,
        log_t_min,
        log_t_max,
        n_ev,
        n_tally,
        n_dispatch,
        n_route,
        n_route_load,
        n_is_overload,
        n_no_ts,
    ):
        self.series = series  # {panel_key: {replica: [(ts, val), ...]}}
        self.retained = retained  # {replica: [(ts, n), ...]}
        self.dispatch = dispatch  # {replica: [(ts, dispatched, completed, turn_sum, prompt_len_sum), ...]}
        self.route_lat = route_lat  # [(ts, ms), ...] global flat — route() is balancer-wide
        self.route_load = route_load  # {replica: [(ts, load), ...]} — score() combined-path per-replica load
        self.is_overload = is_overload  # {replica: [(ts, load), ...]} — is_overloaded() bound-replica load
        self.replicas = replicas
        self.t_min = t_min
        self.t_max = t_max
        # First/last timestamp across ALL log lines (not just signal lines) —
        # the true run walltime (t_min/t_max only covers signal-bearing lines,
        # which starts after vLLM warmup). Drives the figure subtitle.
        self.log_t_min = log_t_min
        self.log_t_max = log_t_max
        self.n_ev = n_ev
        self.n_tally = n_tally
        self.n_dispatch = n_dispatch
        self.n_route = n_route
        self.n_route_load = n_route_load
        self.n_is_overload = n_is_overload
        self.n_no_ts = n_no_ts


def collect(paths, panels: list[Panel]) -> Bundle:
    """Parse logs into per-panel series + retained/dispatch buffers.

    Tracks the global [t_min, t_max] during the single parse pass so later
    truncation needs no second scan.
    """
    # Derived panels (cumulative from tally, dispatch/cumulative-completed from
    # the dispatch buffer) are skipped by the evidence extraction loop — they get
    # their series via derive() / derive_dispatch() in main().
    extract_panels = [
        p
        for p in panels
        if not isinstance(
            p, CumulativePanel | TrailingDeltaPanel | RouteLatencyPanel | CumulativeCompletedPanel | LoadPanel
        )
    ]
    series = {p.key: defaultdict(list) for p in extract_panels}
    retained: dict = defaultdict(list)
    dispatch: dict = defaultdict(list)
    route_lat: list = []
    route_load: dict = defaultdict(list)
    is_overload: dict = defaultdict(list)
    replicas: set[str] = set()
    t_min = t_max = None
    # Walltime window — first/last timestamp across ALL lines (true run span).
    log_t_min = log_t_max = None
    n_ev = n_tally = n_dispatch = n_route = n_route_load = n_is_overload = n_no_ts = 0

    for path in paths:
        try:
            f = open(path, errors="replace")
        except OSError as e:
            print(f"WARN: cannot open {path}: {e}", file=sys.stderr)
            continue
        with f:
            for line in f:
                # Track the true run window across ALL lines (not just signals)
                # for the walltime subtitle. Cheaper than a second scan: LogParser._ts
                # runs two regexes per line, but only on lines that reach here.
                line_ts = LogParser._ts(line)
                if line_ts is not None:
                    if log_t_min is None:
                        log_t_min = log_t_max = line_ts
                    elif line_ts < log_t_min:
                        log_t_min = line_ts
                    elif line_ts > log_t_max:
                        log_t_max = line_ts
                if not LogParser.is_signal(line):
                    continue
                parsed = LogParser.parse(line)
                if parsed is None:
                    continue
                ts, kind, g = parsed
                if ts is None:
                    n_no_ts += 1
                    continue
                if t_min is None:
                    t_min = t_max = ts
                elif ts < t_min:
                    t_min = ts
                elif ts > t_max:
                    t_max = ts
                if kind == "evidence":
                    rep = g["rep"]
                    replicas.add(rep)
                    for p in extract_panels:
                        v = p.extract(g)
                        if v is not None:
                            series[p.key][rep].append((ts, v))
                    n_ev += 1
                elif kind == "tally":
                    try:
                        d = ast.literal_eval(g["d"])
                    except Exception:
                        continue
                    for rep, n in d.items():
                        replicas.add(rep)
                        retained[rep].append((ts, int(n)))
                    n_tally += 1
                elif kind == "dispatch":  # per-replica cumulative snapshot
                    rep = g["rep"]
                    replicas.add(rep)
                    # prompt_len_sum defaults to 0 for older logs that predate the
                    # request-length tracking (regex made it optional).
                    pls = int(g.get("prompt_len_sum") or 0)
                    dispatch[rep].append((ts, int(g["dispatched"]), int(g["completed"]), int(g["turn_sum"]), pls))
                    n_dispatch += 1
                elif kind == "route":  # global per-dispatch latency (not per-replica)
                    route_lat.append((ts, float(g["route"])))
                    n_route += 1
                elif kind == "route_load":  # score() combined-path per-replica load snapshot
                    try:
                        d = ast.literal_eval(g["d"])
                    except Exception:
                        continue
                    for rep, load in d.items():
                        replicas.add(rep)
                        route_load[rep].append((ts, float(load)))
                    n_route_load += 1
                elif kind == "is_overload":  # is_overloaded() bound-replica load check
                    rep = g["rep"]
                    replicas.add(rep)
                    is_overload[rep].append((ts, float(g["load"])))
                    n_is_overload += 1

    return Bundle(
        series,
        retained,
        dispatch,
        route_lat,
        route_load,
        is_overload,
        replicas,
        t_min,
        t_max,
        log_t_min,
        log_t_max,
        n_ev,
        n_tally,
        n_dispatch,
        n_route,
        n_route_load,
        n_is_overload,
        n_no_ts,
    )


def prepare(panels: list[Panel], series: dict, t_cut, max_points: int, peak_flops: float) -> dict:
    """Truncate → per-panel transform → downsample, uniformly for every panel."""
    data = {}
    for p in panels:
        buf = series.get(p.key, {})
        prep = {}
        for rep, pts in buf.items():
            if t_cut is not None:
                pts = [x for x in pts if x[0] <= t_cut]
            pts = p.transform(sorted(pts, key=lambda x: x[0]), peak_flops)
            if not pts:
                continue
            prep[rep] = _downsample(pts, max_points)
        data[p.key] = prep
    return data


def compute_order(load_pts: dict, replicas: set) -> list:
    """Replicas ordered by first load timestamp (stable on name)."""
    first_ts = {r: pts[0][0] for r, pts in load_pts.items() if pts}
    return sorted(replicas, key=lambda r: (first_ts.get(r, datetime.min), r))


def plot(plt, panels: list[Panel], data: dict, order: list, colors: dict, out: str, walltime: str = "-") -> None:
    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=(14, max(31, sum(p.height for p in panels) * 1.55 + 2)),
        gridspec_kw={"height_ratios": [p.height for p in panels]},
    )
    for p, ax in zip(panels, axes, strict=False):
        p.draw(ax, data, colors, order)

    # Legend on the top panel only (colours repeat down the figure).
    ax_top = axes[0]
    for rep in order:
        if data.get(panels[0].key, {}).get(rep):
            ax_top.plot([], [], color=colors[rep], linestyle="-", linewidth=1.5, label=rep)
    ax_top.legend(loc="best", fontsize=8, ncol=max(1, (len(order) + 7) // 8))

    axes[-1].set_xlabel("time")
    fig.suptitle(_TITLE, y=0.995)
    # Subtitle: run walltime (first→last log timestamp). Placed just under the
    # suptitle, centered. tight_layout rect leaves head room so it isn't clipped.
    if walltime and walltime != "-":
        fig.text(0.5, 0.985, f"walltime: {walltime}", ha="center", va="top", fontsize=10, color="#444444")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=140)


def print_summary(
    panels: list[Panel], data: dict, bundle: Bundle, order: list, max_evict, peak_tflops: float, out: str
) -> None:
    print(
        f"OK: {bundle.n_ev} evidence lines, {bundle.n_tally} tally lines, "
        f"{bundle.n_dispatch} dispatch lines, {bundle.n_route_load} route-load lines, "
        f"{bundle.n_is_overload} is-overload lines, {len(bundle.replicas)} replicas -> {out}"
    )
    print(f"   peak={peak_tflops:.0f} TFLOPS/NPU, mfu_window={_MFU_WINDOW_S:.0f}s, max cum evictions={max_evict}")
    print("   per-replica summary:")
    for rep in order:
        parts = [p.summarize(data, rep) for p in panels]
        print(f"     {rep:>16s}: " + " ".join(parts))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-replica KV + MFU + global dispatch + load + route-latency signals (18 time-aligned panels)"
    )
    ap.add_argument("logs", nargs="+", help="log file(s)")
    ap.add_argument("--frac", type=float, default=1.0, help="plot first FRAC of the time window; (0,1]")
    ap.add_argument("--max-points", type=int, default=2000, help="downsample each curve; 0 disables")
    ap.add_argument(
        "--peak-tflops",
        type=float,
        default=560.0,
        help="per-NPU peak FLOPs/s in TFLOPS (MFU denominator). 560=Atlas 800I A3 FP16/NPU (default), 750=800T A3.",
    )
    ap.add_argument(
        "--load-threshold",
        type=float,
        default=0.9,
        help="overload threshold for load panel hlines (default 0.9).",
    )
    args = ap.parse_args(argv)
    if not (0.0 < args.frac <= 1.0):
        ap.error(f"--frac must be in (0, 1], got {args.frac}")
    args.out = str(Path(args.logs[0]).with_suffix(".png"))  # <log-name>.png, next to the first log
    return args


def _import_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("ERROR: matplotlib not installed.  pip install matplotlib", file=sys.stderr)
        return None


def main(argv=None) -> int:
    args = parse_args(argv)
    plt = _import_mpl()
    if plt is None:
        return 2

    panels = build_panels(args.peak_tflops, args.load_threshold)
    bundle = collect(args.logs, panels)
    if bundle.n_no_ts:
        print(f"WARN: {bundle.n_no_ts} signal lines had no parseable timestamp — skipped", file=sys.stderr)
    if not bundle.replicas:
        print(f"ERROR: no vllm-evidence / kv-events tally lines found in {args.logs}", file=sys.stderr)
        return 1

    for p in panels:  # populate derived panels (evictions from tally, dispatch/load from global buffers)
        derived = p.derive(bundle.retained)
        if derived is not None:
            bundle.series[p.key] = derived
        derived_lc = p.derive_dispatch(bundle.dispatch)
        if derived_lc is not None:
            bundle.series[p.key] = derived_lc
        derived_rt = p.derive_route(bundle.route_lat)
        if derived_rt is not None:
            bundle.series[p.key] = derived_rt
        derived_loads = p.derive_loads(bundle.route_load, bundle.is_overload)
        if derived_loads is not None:
            bundle.series[p.key] = derived_loads

    t_cut = None
    if args.frac < 1.0 and bundle.t_min is not None:
        t_cut = bundle.t_min + (bundle.t_max - bundle.t_min) * args.frac
    data = prepare(panels, bundle.series, t_cut, args.max_points, args.peak_tflops * 1e12)

    # Drop panels that got no data (e.g. load panels when the log has zero
    # route-load / is-overload lines — sticky-only or least-inflight-only runs).
    active = [p for p in panels if data.get(p.key)]
    if not active:
        print("ERROR: all panels empty after prepare", file=sys.stderr)
        return 1

    max_evict = max((pts[-1][1] for pts in data.get("evict", {}).values() if pts), default=0)
    order = compute_order(data["load"], bundle.replicas)
    cmap = plt.get_cmap("tab10" if len(order) <= 10 else "tab20")
    colors = {rep: cmap(i % cmap.N) for i, rep in enumerate(order)}

    plot(plt, active, data, order, colors, args.out, _fmt_walltime(bundle.log_t_min, bundle.log_t_max))
    print_summary(active, data, bundle, order, max_evict, args.peak_tflops, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
