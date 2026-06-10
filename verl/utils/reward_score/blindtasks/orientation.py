"""blindtasks.orientation

Two variants:
- easy: arrow at one of 4 cardinal directions (N/E/S/W) with +-10° jitter
- hard: arrow at one of 8 directions (N/NE/E/SE/S/SW/W/NW) with +-5° jitter
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from .base import (
    AnswerFormatter,
    RenderConfig,
    TaskInstance,
    answer_prompt,
)
from .utils import (
    _DIRECTION_LABELS_4,
    _DIRECTION_LABELS_8,
    random_color,
    render_config_meta,
    sample_render_config,
    supersample_render,
    verify_direction,
)

_LABEL_TO_ANGLE: dict[str, float] = {
    "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
    "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
}
_ARROW_STYLES: tuple[str, ...] = ("filled", "outlined", "chevron")

_QUESTIONS_EASY: tuple[str, ...] = (
    "Which direction does the arrow point?",
    "The arrow points in one of the cardinal directions. Which direction is it?",
)
_QUESTIONS_HARD: tuple[str, ...] = (
    "Which direction does the arrow point?",
    "Identify the direction the arrow points.",
)


def _arrow_polygon(size: int, style: str) -> list[tuple[float, float]]:
    half = size / 2.0
    if style == "chevron":
        return [(0.0, -half), (-half * 0.6, 0.0), (half * 0.6, 0.0)]
    return [
        (0.0, -half),
        (half * 0.8, half * 0.2),
        (half * 0.3, half * 0.2),
        (half * 0.3, half),
        (-half * 0.3, half),
        (-half * 0.3, half * 0.2),
        (-half * 0.8, half * 0.2),
    ]


def _render_arrow(cfg: RenderConfig, angle_deg: float, style: str, color: tuple[int, int, int], cx_offset: int = 0, cy_offset: int = 0) -> Image.Image:
    s = cfg.image_size
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)

    arrow_size = int(s * 0.30)
    cx = s // 2 + cx_offset
    cy = s // 2 + cy_offset

    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    points = _arrow_polygon(arrow_size, style)
    rotated = [
        (cx + x * cos_t - y * sin_t, cy + x * sin_t + y * cos_t)
        for x, y in points
    ]

    if style == "chevron":
        tip, left_arm, right_arm = rotated
        stroke = max(4, cfg.line_width * 2)
        draw.line([left_arm, tip], fill=color, width=stroke)
        draw.line([tip, right_arm], fill=color, width=stroke)
    elif style == "filled":
        draw.polygon(rotated, fill=color)
    else:  # outlined
        draw.polygon(rotated, outline=color, width=max(2, cfg.line_width))
    return image


def _difficulty(n_directions: int) -> float:
    return 0.0 if n_directions == 4 else 1.0


def _sample_orientation_task(seed: int, name: str, *, labels: tuple[str, ...], jitter_deg: float) -> TaskInstance:
    rng = np.random.default_rng(seed)
    cfg = sample_render_config(rng, line_widths=(2, 3))

    label_idx = int(rng.integers(0, len(labels)))
    label = labels[label_idx]
    canonical = _LABEL_TO_ANGLE[label]
    actual_angle = canonical + float(rng.uniform(-jitter_deg, jitter_deg))

    style = _ARROW_STYLES[int(rng.integers(0, len(_ARROW_STYLES)))]
    color = random_color(rng)

    max_offset = int(0.10 * cfg.image_size)
    cx_off = int(rng.integers(-max_offset, max_offset + 1))
    cy_off = int(rng.integers(-max_offset, max_offset + 1))

    image = supersample_render(
        lambda cfg_hi: _render_arrow(
            cfg_hi,
            actual_angle,
            style,
            color,
            cx_offset=cx_off * (cfg_hi.image_size // cfg.image_size),
            cy_offset=cy_off * (cfg_hi.image_size // cfg.image_size),
        ),
        cfg,
    )

    questions = _QUESTIONS_EASY if len(labels) == 4 else _QUESTIONS_HARD
    paraphrase_idx = int(rng.integers(0, len(questions)))

    return TaskInstance(
        name=name,
        seed=seed,
        question=questions[paraphrase_idx],
        ground_truth={
            "answer": label,
            "n_directions": len(labels),
            "angle_deg": actual_angle,
            "arrow_style": style,
            "arrow_color": list(color),
            "position_offset": [cx_off, cy_off],
            "difficulty": _difficulty(len(labels)),
            "paraphrase_idx": paraphrase_idx,
            **render_config_meta(cfg),
        },
        image=image,
    )


class OrientationEasyTask:
    name = "blindtasks.orientation"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_orientation_task(seed, self.name, labels=_DIRECTION_LABELS_4, jitter_deg=10.0)

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_direction(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(
            instance.question,
            "one of these compass labels: N, S, E, W",
            "N",
            fmt,
        )


class OrientationHardTask:
    name = "blindtasks.orientation.hard"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_orientation_task(seed, self.name, labels=_DIRECTION_LABELS_8, jitter_deg=5.0)

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_direction(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(
            instance.question,
            "one of these compass labels: N, NE, E, SE, S, SW, W, NW",
            "NE",
            fmt,
        )


TASK_EASY = OrientationEasyTask()
TASK_HARD = OrientationHardTask()
