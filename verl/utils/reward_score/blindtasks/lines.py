"""blindtasks.lines

Two variants:
- easy: count k ∈ {2..15} parallel lines. Per-instance variation:
    random global angle {0, 30, 45, 60, 90, 120, 135, 150}°, optional
    jittered spacing, optional varied length, optional perpendicular
    distractor line. The model must still count parallel lines
- hard: two polylines (blue, red) drawn through 3 control points each;
    target is intersection count ∈ {0, 1, 2} (mirroring vlmsareblind)
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from .base import AnswerFormatter, RenderConfig, TaskInstance, count_prompt
from .utils import (
    count_polyline_intersections,
    random_color_pair,
    render_config_meta,
    sample_render_config,
    supersample_render,
    verify_count,
)

_MARGIN_FRAC = 0.10
_PARALLEL_QUESTIONS = (
    (
        "How many parallel lines are in this image? "
        "(Ignore any non-parallel distractor lines.)"
    ),
    (
        "Count the parallel lines in this image. "
        "Ignore any line that is not parallel to the others."
    ),
)
_INTERSECT_QUESTIONS = (
    (
        "Count the number of times the two coloured lines intersect in this image."
    ),
    (
        "How many times do the two coloured curves cross each other?"
    ),
)

_PARALLEL_ANGLES_DEG = (0, 30, 45, 60, 90, 120, 135, 150)
_INTERSECT_TARGETS = (0, 1, 2)
_INTERSECT_LINE_WIDTHS = (2, 3, 4)


def _difficulty_easy(k: int, has_distractor: bool) -> float:
    base = (k - 2) / (15 - 2)
    return min(1.0, base + (0.2 if has_distractor else 0.0))


def _difficulty_hard(target: int) -> float:
    return target / 2


def _line_endpoints_at_angle(
    angle_deg: float,
    offset_frac: float,
    s: int,
    margin: int,
    length_frac: float = 1.0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    cx = cy = s / 2.0
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    
    nx, ny = -dy, dx
    usable = s - 2 * margin
    half_len = usable * length_frac / 2.0
    off = offset_frac * usable / 2.0
    
    px, py = cx + nx * off, cy + ny * off
    a = (int(round(px - dx * half_len)), int(round(py - dy * half_len)))
    b = (int(round(px + dx * half_len)), int(round(py + dy * half_len)))
    
    return a, b


def _render_parallel(
    k: int,
    cfg: RenderConfig,
    angle_deg: float,
    *,
    rng: np.random.Generator | None = None,
    jitter_spacing: bool = False,
    vary_length: bool = False,
    add_distractor: bool = False,
) -> Image.Image:
    s = cfg.image_size
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    margin = max(6, int(s * _MARGIN_FRAC))

    if k == 1:
        offsets = [0.0]
    else:
        offsets = [-1.0 + 2.0 * i / (k - 1) for i in range(k)]
    
    if jitter_spacing and rng is not None and k > 2:
        max_jitter = 0.4 * (2.0 / (k - 1))
        offsets = [o + float(rng.uniform(-max_jitter, max_jitter)) for o in offsets]
        offsets[0] = max(offsets[0], -1.0)
        offsets[-1] = min(offsets[-1], 1.0)

    for off in offsets:
        length_frac = (
            float(rng.uniform(0.55, 1.0)) if (vary_length and rng is not None) else 1.0
        )
        a, b = _line_endpoints_at_angle(angle_deg, off, s, margin, length_frac)
        draw.line([a, b], fill=cfg.fg, width=cfg.line_width)

    if add_distractor and rng is not None:
        a, b = _line_endpoints_at_angle(angle_deg + 90.0, 0.0, s, margin, 0.9)
        draw.line([a, b], fill=cfg.fg, width=cfg.line_width)

    return image


def _scale_poly(
    poly: list[tuple[float, float]], factor: int
) -> list[tuple[float, float]]:
    return [(x * factor, y * factor) for x, y in poly]


def _sample_curve(rng: np.random.Generator, s: int, margin: int) -> list[tuple[float, float]]:
    xs = [margin, s // 2, s - margin]
    ys = [int(rng.integers(margin, s - margin + 1)) for _ in range(3)]
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _sample_intersecting_pair(
    rng: np.random.Generator, target: int, s: int, margin: int, max_attempts: int = 600
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    for _ in range(max_attempts):
        a = _sample_curve(rng, s, margin)
        b = _sample_curve(rng, s, margin)
        
        if count_polyline_intersections(a, b) == target:
            return a, b
    
    # Deterministic fallback
    if target == 0:
        a = [(margin, margin + 4), (s // 2, margin + 8), (s - margin, margin + 4)]
        b = [(margin, s - margin - 4), (s // 2, s - margin - 8), (s - margin, s - margin - 4)]
    elif target == 1:
        a = [(margin, margin), (s // 2, s // 2), (s - margin, s - margin)]
        b = [(margin, s - margin), (s // 2, s // 2), (s - margin, margin)]
    else:
        a = [(margin, margin), (s // 2, s - margin), (s - margin, margin)]
        b = [(margin, s // 2), (s // 2, margin), (s - margin, s // 2)]
    
    return [(float(x), float(y)) for x, y in a], [(float(x), float(y)) for x, y in b]


def _render_intersecting(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
    cfg: RenderConfig,
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
) -> Image.Image:
    s = cfg.image_size
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    draw.line([(int(x), int(y)) for x, y in poly_a], fill=color_a, width=cfg.line_width)
    draw.line([(int(x), int(y)) for x, y in poly_b], fill=color_b, width=cfg.line_width)
    return image


class LinesEasyTask:
    name = "blindtasks.lines"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng)

        k = int(rng.integers(2, 16))
        angle_deg = float(rng.choice(_PARALLEL_ANGLES_DEG))

        jitter_spacing = bool(rng.integers(0, 2))
        vary_length = bool(rng.integers(0, 2))
        add_distractor = bool(rng.integers(0, 4) == 0)

        image = supersample_render(
            lambda cfg_hi: _render_parallel(
                k, cfg_hi, angle_deg,
                rng=rng,
                jitter_spacing=jitter_spacing,
                vary_length=vary_length,
                add_distractor=add_distractor,
            ),
            cfg,
        )
        paraphrase_idx = int(rng.integers(0, len(_PARALLEL_QUESTIONS)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_PARALLEL_QUESTIONS[paraphrase_idx],
            ground_truth={
                "answer": k,
                "angle_deg": angle_deg,
                "jitter_spacing": jitter_spacing,
                "vary_length": vary_length,
                "has_distractor": add_distractor,
                "difficulty": _difficulty_easy(k, add_distractor),
                "paraphrase_idx": paraphrase_idx,
                **render_config_meta(cfg),
            },
            image=image,
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


class LinesHardTask:
    name = "blindtasks.lines.hard"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng, line_widths=_INTERSECT_LINE_WIDTHS, allow_invert=False)

        target = int(rng.choice(_INTERSECT_TARGETS))
        margin = max(8, int(cfg.image_size * _MARGIN_FRAC))
        poly_a, poly_b = _sample_intersecting_pair(rng, target, cfg.image_size, margin)
        color_a, color_b = random_color_pair(rng)

        image = supersample_render(
            lambda cfg_hi: _render_intersecting(
                _scale_poly(poly_a, 2), _scale_poly(poly_b, 2), cfg_hi, color_a, color_b
            ),
            cfg,
        )
        paraphrase_idx = int(rng.integers(0, len(_INTERSECT_QUESTIONS)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_INTERSECT_QUESTIONS[paraphrase_idx],
            ground_truth={
                "answer": target,
                "color_a": list(color_a),
                "color_b": list(color_b),
                "difficulty": _difficulty_hard(target),
                "paraphrase_idx": paraphrase_idx,
                **render_config_meta(cfg),
            },
            image=image,
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


TASK_EASY = LinesEasyTask()
TASK_HARD = LinesHardTask()
