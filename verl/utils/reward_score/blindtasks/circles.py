"""blindtasks.circles

Two variants:
- easy: count k ∈ {2..8} concentric circles
- hard: two filled colored circles, do they overlap? (mirroring vlmsareblind)
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from .base import (
    AnswerFormatter,
    RenderConfig,
    TaskInstance,
    answer_prompt,
    count_prompt,
)
from .utils import (
    circle_in_viewport,
    random_color_pair,
    render_config_meta,
    sample_render_config,
    supersample_render,
    verify_count,
    verify_yes_no,
)

_MARGIN_FRAC = 0.06
_QUESTIONS_COUNT = (
    "How many concentric circles are in this image?",
    "Count the concentric circles drawn in this image.",
)
_QUESTIONS_TOUCH = (
    "Are the two circles touching or overlapping in this image?",
    "Do the two coloured circles touch or overlap?",
)

_TOUCH_RADIUS_FRACS = (0.10, 0.125, 0.15, 0.175)
_TOUCH_DISTANCE_FRACS = (-0.30, -0.15, -0.05, 0.05, 0.15, 0.30)
_TOUCH_MAX_ABS_DIST = 0.30
_TOUCH_CONFIGS = ("horizontal", "vertical", "diagonal_1", "diagonal_2")
_TOUCH_VIEWPORT_RETRIES = 50


def _render_concentric(k: int, cfg: RenderConfig) -> Image.Image:
    s = cfg.image_size
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    cx = cy = s // 2
    margin = max(4, int(s * _MARGIN_FRAC))
    step = (cx - margin) // k
    for i in range(k):
        r = margin + (i + 1) * step
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=cfg.fg, width=cfg.line_width)
    return image


def _touching_centers(
    config: str, dist_frac: float, radius_frac: float, s: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    g = dist_frac * 2 * radius_frac
    half = s / 2.0
    if config == "horizontal":
        x1 = half - (g / 2 + radius_frac) * s
        x2 = half + (g / 2 + radius_frac) * s
        return (x1, half), (x2, half)
    if config == "vertical":
        y1 = half - (g / 2 + radius_frac) * s
        y2 = half + (g / 2 + radius_frac) * s
        return (half, y1), (half, y2)
    a = math.sqrt(2) * (g + 2 * radius_frac) * s / 2
    if config == "diagonal_1":
        return (half - a / 2, half - a / 2), (half + a / 2, half + a / 2)
    return (half - a / 2, half + a / 2), (half + a / 2, half - a / 2)


def _render_touching(
    cfg: RenderConfig,
    config: str,
    dist_frac: float,
    radius_frac: float,
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
) -> Image.Image:
    s = cfg.image_size
    image = Image.new("RGB", (s, s), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    (x1, y1), (x2, y2) = _touching_centers(config, dist_frac, radius_frac, s)
    r = radius_frac * s
    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=color_a)
    draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=color_b)
    return image


def _difficulty_easy(k: int) -> float:
    return (k - 2) / (8 - 2)


def _difficulty_hard(dist_frac: float) -> float:
    return 1.0 - abs(dist_frac) / _TOUCH_MAX_ABS_DIST


class CirclesEasyTask:
    name = "blindtasks.circles"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng)
        k = int(rng.integers(2, 9))
        paraphrase_idx = int(rng.integers(0, len(_QUESTIONS_COUNT)))
        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_QUESTIONS_COUNT[paraphrase_idx],
            ground_truth={
                "answer": k,
                "difficulty": _difficulty_easy(k),
                "paraphrase_idx": paraphrase_idx,
                **render_config_meta(cfg),
            },
            image=supersample_render(
                lambda cfg_hi: _render_concentric(k, cfg_hi), cfg
            ),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


class CirclesHardTask:
    name = "blindtasks.circles.hard"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng, allow_invert=False)
        s = cfg.image_size

        config = ""
        dist_frac = 0.0
        radius_frac = 0.0
        for _ in range(_TOUCH_VIEWPORT_RETRIES):
            config = str(rng.choice(_TOUCH_CONFIGS))
            dist_frac = float(rng.choice(_TOUCH_DISTANCE_FRACS))
            radius_frac = float(rng.choice(_TOUCH_RADIUS_FRACS))
            centers = _touching_centers(config, dist_frac, radius_frac, s)
            r_px = radius_frac * s
            if all(circle_in_viewport(c, r_px, s) for c in centers):
                break

        color_a, color_b = random_color_pair(rng)
        answer = "yes" if dist_frac <= 0.0 else "no"
        paraphrase_idx = int(rng.integers(0, len(_QUESTIONS_TOUCH)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_QUESTIONS_TOUCH[paraphrase_idx],
            ground_truth={
                "answer": answer,
                "configuration": config,
                "distance_frac": dist_frac,
                "radius_frac": radius_frac,
                "color_a": list(color_a),
                "color_b": list(color_b),
                "difficulty": _difficulty_hard(dist_frac),
                "paraphrase_idx": paraphrase_idx,
                "image_size": cfg.image_size,
            },
            image=supersample_render(
                lambda cfg_hi: _render_touching(
                    cfg_hi, config, dist_frac, radius_frac, color_a, color_b
                ),
                cfg,
            ),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_yes_no(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(instance.question, "either yes or no", "yes", fmt)


TASK_EASY = CirclesEasyTask()
TASK_HARD = CirclesHardTask()
