"""blindtasks.subway

Two variants:
- easy: 4 stations on the four edges, 1-2 colored paths total
- hard: 4 stations, 1-3 colored paths, larger image (mirroring vlmsareblind)

Both variants involve visually tracing colored routes on a 2D map with multiple paths.
Paths use Manhattan-style routing (right-angle bends through 1-2 random interior points)
to simplify the DFS routing used in vlmsareblind
"""

import numpy as np
from PIL import Image, ImageDraw

from .base import AnswerFormatter, RenderConfig, TaskInstance, count_prompt
from .utils import (
    draw_text_centered,
    load_font,
    sample_render_config,
    supersample_render,
    verify_count,
)

_QUESTIONS = (
    (
        "How many single-colored paths go from station {a} to station {b}?"
    ),
    (
        "Count the coloured routes that directly connect station {a} and station {b}."
    ),
)

_PALETTE = (
    ("red", (214, 39, 40)),
    ("blue", (31, 119, 180)),
    ("green", (44, 160, 44)),
    ("orange", (255, 127, 14)),
    ("purple", (148, 103, 189)),
)
_STATIONS = ("A", "B", "C", "D")
_LINE_WIDTHS = (4, 6, 8)
_REPEAT_PAIR_PROB = 0.5


def _station_positions(s: int) -> dict[str, tuple[int, int]]:
    margin = max(20, s // 12)
    return {
        "A": (s // 2, margin),      # top
        "B": (s - margin, s // 2),  # right
        "C": (s // 2, s - margin),  # bottom
        "D": (margin, s // 2),      # left
    }


def _manhattan_path(
    a: tuple[int, int], b: tuple[int, int], rng: np.random.Generator, s: int
) -> list[tuple[int, int]]:
    inner_margin = max(40, s // 8)
    n_waypoints = int(rng.integers(1, 3)) 
    waypoints: list[tuple[int, int]] = []
    
    for _ in range(n_waypoints):
        wx = int(rng.integers(inner_margin, s - inner_margin + 1))
        wy = int(rng.integers(inner_margin, s - inner_margin + 1))
        waypoints.append((wx, wy))

    pts = [a, *waypoints, b]
    poly: list[tuple[int, int]] = [pts[0]]
    
    for p, q in zip(pts[:-1], pts[1:]):
        corner = (q[0], p[1]) if bool(rng.integers(0, 2)) else (p[0], q[1])
        poly.append(corner)
        poly.append(q)
    
    return poly


def _render_subway(
    cfg: RenderConfig,
    paths: list[dict],
    rng: np.random.Generator,
    line_width: int,
) -> Image.Image:
    s = cfg.image_size
    
    image = Image.new("RGB", (s, s), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    stations = _station_positions(s)

    for p in paths:
        poly = _manhattan_path(stations[p["start"]], stations[p["end"]], rng, s)
        draw.line(poly, fill=p["color_rgb"], width=line_width)

    font = load_font(max(20, s // 12))
    node_r = max(18, s // 14)
    
    for label, (x, y) in stations.items():
        draw.ellipse(
            [x - node_r, y - node_r, x + node_r, y + node_r],
            outline=(0, 0, 0),
            width=3,
            fill=(255, 255, 255),
        )
        draw_text_centered(draw, x, y, label, font, (0, 0, 0))
    return image


def _sample_subway_task(
    seed: int,
    name: str,
    *,
    n_paths_range: tuple[int, int],
    image_sizes: tuple[int, ...],
) -> TaskInstance:
    rng = np.random.default_rng(seed)
    cfg = sample_render_config(rng, image_sizes=image_sizes, allow_invert=False)
    
    line_width = int(rng.choice(_LINE_WIDTHS))
    n_paths = int(rng.integers(n_paths_range[0], n_paths_range[1] + 1))
    
    paths: list[dict] = []
    used_colors: set[str] = set()
    prior_pairs: list[tuple[int, int]] = []

    for _ in range(n_paths):
        if prior_pairs and rng.uniform() < _REPEAT_PAIR_PROB:
            pi, pj = prior_pairs[int(rng.integers(0, len(prior_pairs)))]
            if bool(rng.integers(0, 2)):
                i, j = pj, pi
            else:
                i, j = pi, pj
        else:
            i = int(rng.integers(0, len(_STATIONS)))
            j = int(rng.integers(0, len(_STATIONS)))
            while i == j:
                j = int(rng.integers(0, len(_STATIONS)))
        prior_pairs.append((i, j))

        choices = [c for c in _PALETTE if c[0] not in used_colors] or list(_PALETTE)
        color_name, color_rgb = choices[int(rng.integers(0, len(choices)))]
        used_colors.add(color_name)

        paths.append({
            "start": _STATIONS[i],
            "end": _STATIONS[j],
            "color": color_name,
            "color_rgb": color_rgb,
        })

    pair_counts: dict[frozenset, int] = {}
    for a_idx in range(len(_STATIONS)):
        for b_idx in range(a_idx + 1, len(_STATIONS)):
            key = frozenset((_STATIONS[a_idx], _STATIONS[b_idx]))
            pair_counts[key] = sum(
                1 for p in paths if {p["start"], p["end"]} == set(key)
            )

    achievable = sorted(set(pair_counts.values()))
    target = int(rng.integers(0, n_paths + 1))
    if target not in achievable:
        target = int(rng.choice(achievable))
    
    candidates = [k for k, c in pair_counts.items() if c == target]
    chosen_pair = list(candidates[int(rng.integers(0, len(candidates)))])
    if bool(rng.integers(0, 2)):
        x, y = chosen_pair[0], chosen_pair[1]
    else:
        x, y = chosen_pair[1], chosen_pair[0]

    answer = target

    lo, hi = n_paths_range
    difficulty = (n_paths - lo) / max(1, hi - lo)

    image = supersample_render(
        lambda cfg_hi: _render_subway(
            cfg_hi,
            paths,
            rng,
            line_width * (cfg_hi.image_size // cfg.image_size),
        ),
        cfg,
    )
    paraphrase_idx = int(rng.integers(0, len(_QUESTIONS)))

    return TaskInstance(
        name=name,
        seed=seed,
        question=_QUESTIONS[paraphrase_idx].format(a=x, b=y),
        ground_truth={
            "answer": answer,
            "query_start": x,
            "query_end": y,
            "n_paths": n_paths,
            "paths": [
                {"start": p["start"], "end": p["end"], "color": p["color"]}
                for p in paths
            ],
            "difficulty": difficulty,
            "paraphrase_idx": paraphrase_idx,
            "image_size": cfg.image_size,
            "line_width": line_width,
        },
        image=image,
    )


class SubwayEasyTask:
    name = "blindtasks.subway"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_subway_task(
            seed,
            self.name,
            n_paths_range=(1, 2),
            image_sizes=(256, 336),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


class SubwayHardTask:
    name = "blindtasks.subway.hard"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_subway_task(
            seed,
            self.name,
            n_paths_range=(1, 3),
            image_sizes=(336, 448),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


TASK_EASY = SubwayEasyTask()
TASK_HARD = SubwayHardTask()
