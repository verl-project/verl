"""blindtasks.olympic

Three variants:
- easy: k ∈ {3..6} non-overlapping circles, random placement.
- hard: k ∈ {5..9} smaller non-overlapping circles, harder counting.
- interlocking: 5..9 circles arranged in Olympic-rings-style
"""

import numpy as np
from PIL import Image, ImageDraw

from .base import AnswerFormatter, RenderConfig, TaskInstance, count_prompt
from .utils import (
    place_non_overlapping_circles,
    render_config_meta,
    sample_render_config,
    supersample_render,
    verify_count,
)

_QUESTIONS = (
    "How many circles are in this image?",
    "Count the circles drawn in this image.",
)

_TAB10 = (
    (31, 119, 180), (255, 127, 14), (44, 160, 44),
    (214, 39, 40), (148, 103, 189), (140, 86, 75),
    (227, 119, 194), (127, 127, 127), (188, 189, 34),
    (23, 190, 207),
)

_INTERLOCK_K_VALUES = (5, 6, 7, 8, 9)
_INTERLOCK_OVERLAP = 0.20
_INTERLOCK_RADIUS_FRAC = 0.10

_TAB10_INTERLOCK = (
    (228, 26, 28), (55, 126, 184), (77, 175, 74),
    (152, 78, 163), (255, 127, 0), (255, 255, 51),
    (166, 86, 40), (247, 129, 191), (153, 153, 153),
)


def _render_count(
    cfg: RenderConfig,
    circles: list[tuple[int, int, int]],
    rng: np.random.Generator,
    *,
    monochrome: bool,
    filled: bool,
) -> Image.Image:
    s = cfg.image_size
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    
    for cx, cy, r in circles:
        color = cfg.fg if monochrome else _TAB10[int(rng.integers(0, len(_TAB10)))]
        if filled:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        else:
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                outline=color,
                width=cfg.line_width,
            )
    
    return image


def _difficulty(actual_k: int, k_range: tuple[int, int]) -> float:
    lo, hi = k_range
    raw = (actual_k - lo) / max(1, hi - lo)
    return max(0.0, min(1.0, raw))


def _sample_count_task(
    seed: int,
    name: str,
    *,
    k_range: tuple[int, int],
    radius_frac_range: tuple[float, float],
) -> TaskInstance:
    rng = np.random.default_rng(seed)
    cfg = sample_render_config(rng, line_widths=(1, 2, 3))

    s = cfg.image_size
    k = int(rng.integers(k_range[0], k_range[1] + 1))

    margin = max(8, int(s * 0.05))
    rmin = max(8, int(s * radius_frac_range[0]))
    rmax = max(rmin + 1, int(s * radius_frac_range[1]))

    placed = place_non_overlapping_circles(
        rng, k, s, radius_range=(rmin, rmax), margin=margin
    )
    actual_k = len(placed)

    monochrome = bool(rng.integers(0, 2))
    filled = bool(rng.integers(0, 2))

    image = supersample_render(
        lambda cfg_hi: _render_count(
            cfg_hi,
            [
                (
                    cx * cfg_hi.image_size // cfg.image_size,
                    cy * cfg_hi.image_size // cfg.image_size,
                    r * cfg_hi.image_size // cfg.image_size,
                )
                for cx, cy, r in placed
            ],
            rng,
            monochrome=monochrome,
            filled=filled,
        ),
        cfg,
    )
    paraphrase_idx = int(rng.integers(0, len(_QUESTIONS)))

    return TaskInstance(
        name=name,
        seed=seed,
        question=_QUESTIONS[paraphrase_idx],
        ground_truth={
            "answer": actual_k,
            "monochrome": monochrome,
            "filled": filled,
            "difficulty": _difficulty(actual_k, k_range),
            "paraphrase_idx": paraphrase_idx,
            **render_config_meta(cfg),
        },
        image=image,
    )


class OlympicEasyTask:
    name = "blindtasks.olympic"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_count_task(
            seed,
            self.name,
            k_range=(3, 6),
            radius_frac_range=(0.09, 0.16),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


class OlympicHardTask:
    name = "blindtasks.olympic.hard"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_count_task(
            seed,
            self.name,
            k_range=(5, 9),
            radius_frac_range=(0.06, 0.12),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


def _interlock_centers(k: int, s: int) -> list[tuple[float, float]]:
    r = _INTERLOCK_RADIUS_FRAC * s
    stride = 2 * r * (1 - _INTERLOCK_OVERLAP)
    row1_n = (k + 1) // 2
    row2_n = k - row1_n

    width1 = (row1_n - 1) * stride
    width2 = (row2_n - 1) * stride if row2_n > 0 else 0
    width = max(width1, width2 + stride)
    x0 = (s - width) / 2.0

    y1 = s / 2.0 - r / 2.0
    y2 = y1 + r

    centers: list[tuple[float, float]] = []
    for i in range(row1_n):
        centers.append((x0 + i * stride, y1))
    for j in range(row2_n):
        centers.append((x0 + stride / 2.0 + j * stride, y2))
    return centers


def _render_interlock(
    cfg: RenderConfig,
    centers: list[tuple[float, float]],
    rng: np.random.Generator,
    *,
    monochrome: bool,
) -> Image.Image:
    s = cfg.image_size
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    r = _INTERLOCK_RADIUS_FRAC * s
    for cx, cy in centers:
        color = (
            cfg.fg if monochrome
            else _TAB10_INTERLOCK[int(rng.integers(0, len(_TAB10_INTERLOCK)))]
        )
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=color,
            width=max(2, cfg.line_width),
        )
    return image


def _sample_interlocking(seed: int, name: str) -> TaskInstance:
    rng = np.random.default_rng(seed)
    cfg = sample_render_config(rng, line_widths=(2, 3, 4), allow_invert=False)
    k = int(rng.choice(_INTERLOCK_K_VALUES))
    monochrome = bool(rng.integers(0, 2))

    image = supersample_render(
        lambda cfg_hi: _render_interlock(
            cfg_hi,
            _interlock_centers(k, cfg_hi.image_size),
            rng,
            monochrome=monochrome,
        ),
        cfg,
    )
    paraphrase_idx = int(rng.integers(0, len(_QUESTIONS)))

    return TaskInstance(
        name=name,
        seed=seed,
        question=_QUESTIONS[paraphrase_idx],
        ground_truth={
            "answer": k,
            "monochrome": monochrome,
            "layout": "interlocking",
            "difficulty": _difficulty(
                k, (_INTERLOCK_K_VALUES[0], _INTERLOCK_K_VALUES[-1])
            ),
            "paraphrase_idx": paraphrase_idx,
            **render_config_meta(cfg),
        },
        image=image,
    )


class OlympicInterlockingTask:
    name = "blindtasks.olympic.interlocking"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_interlocking(seed, self.name)

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


TASK_EASY = OlympicEasyTask()
TASK_HARD = OlympicHardTask()
TASK_INTERLOCKING = OlympicInterlockingTask()
