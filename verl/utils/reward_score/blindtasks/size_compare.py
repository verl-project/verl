"""blindtasks.size_compare

Two variants:
- easy: 2 circles labelled A/B. Answer is the larger label (ratio ∈ {1.5, 2.0, 3.0})
- hard: 3 circles labelled A/B/C. Answer is the largest label (ratio ∈ {1.3, 1.5, 2.0})
"""

import numpy as np
from PIL import Image, ImageDraw

from .base import (
    AnswerFormatter,
    RenderConfig,
    TaskInstance,
    answer_prompt,
)
from .circled_letter import _verify_label
from .utils import (
    draw_text_centered,
    load_font,
    random_color,
    render_config_meta,
    sample_render_config,
    supersample_render,
)

_LABEL_FILL = (255, 255, 255)
_EASY_RATIOS: tuple[float, ...] = (1.5, 2.0, 3.0)
_HARD_RATIOS: tuple[float, ...] = (1.3, 1.5, 2.0)
_R_LARGE_FRAC: tuple[float, ...] = (0.13, 0.16, 0.18)

_QUESTIONS_EASY: tuple[str, ...] = (
    "Which circle is larger, A or B?",
    "Of the two labelled circles, which has a bigger radius?",
)
_QUESTIONS_HARD: tuple[str, ...] = (
    "Which circle is the largest, A, B, or C?",
    "Of the three labelled circles, which has the biggest radius?",
)


def _render_size_compare(
    cfg: RenderConfig,
    radii: list[int],
    positions: list[tuple[int, int]],
    colors: list[tuple[int, int, int]],
    labels: list[str],
) -> Image.Image:
    s = cfg.image_size
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)

    for r, (cx, cy), col, lab in zip(radii, positions, colors, labels):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        font_size = max(10, int(r * 0.9))
        font = load_font(font_size)
        draw_text_centered(draw, cx, cy, lab, font, _LABEL_FILL)
    return image


def _difficulty_easy(ratio: float) -> float:
    return min(1.0, max(0.0, 1.0 - (ratio - 1.5) / (3.0 - 1.5)))


def _difficulty_hard(gap: float) -> float:
    return min(1.0, max(0.0, 1.0 - (gap - 1.3) / (2.0 - 1.3)))


def _sample_easy(seed: int, name: str) -> TaskInstance:
    rng = np.random.default_rng(seed)
    cfg = sample_render_config(rng, line_widths=(2, 3), allow_invert=False)
    s = cfg.image_size

    r_large_frac = float(rng.choice(_R_LARGE_FRAC))
    ratio = float(rng.choice(_EASY_RATIOS))
    r_large = max(12, int(r_large_frac * s))
    r_small = max(4, int(r_large / ratio))

    larger_label = "A" if bool(rng.integers(0, 2)) else "B"
    if larger_label == "A":
        radii = [r_large, r_small]
    else:
        radii = [r_small, r_large]
    labels = ["A", "B"]

    layout = "horizontal" if bool(rng.integers(0, 2)) else "vertical"
    half = s // 2
    if layout == "horizontal":
        positions = [(s // 4, half), (3 * s // 4, half)]
    else:
        positions = [(half, s // 4), (half, 3 * s // 4)]

    color_a = random_color(rng)
    color_b = random_color(rng)
    colors = [color_a, color_b]

    paraphrase_idx = int(rng.integers(0, len(_QUESTIONS_EASY)))

    image = supersample_render(
        lambda cfg_hi: _render_size_compare(
            cfg_hi,
            [r * (cfg_hi.image_size // s) for r in radii],
            [(x * (cfg_hi.image_size // s), y * (cfg_hi.image_size // s))
             for x, y in positions],
            colors,
            labels,
        ),
        cfg,
    )

    return TaskInstance(
        name=name,
        seed=seed,
        question=_QUESTIONS_EASY[paraphrase_idx],
        ground_truth={
            "answer": larger_label,
            "radii_px": radii,
            "radii_frac": [r / s for r in radii],
            "positions": [list(p) for p in positions],
            "colors_rgb": [list(c) for c in colors],
            "ratio": ratio,
            "layout": layout,
            "difficulty": _difficulty_easy(ratio),
            "paraphrase_idx": paraphrase_idx,
            **render_config_meta(cfg),
        },
        image=image,
    )


def _sample_hard(seed: int, name: str) -> TaskInstance:
    rng = np.random.default_rng(seed)
    cfg = sample_render_config(rng, line_widths=(2, 3), allow_invert=False)
    s = cfg.image_size

    r_large_frac = float(rng.choice(_R_LARGE_FRAC))
    ratio = float(rng.choice(_HARD_RATIOS))
    r_largest = max(12, int(r_large_frac * s))
    r_smallest = max(4, int(r_largest / ratio))

    mid_min = r_smallest
    mid_max = max(mid_min + 1, int(0.8 * r_largest))
    r_middle = int(rng.integers(mid_min, mid_max + 1))

    gap = r_largest / max(r_middle, 1)

    sizes = [r_largest, r_middle, r_smallest]
    perm = np.arange(3)
    rng.shuffle(perm)
    radii_by_label: list[int] = [0, 0, 0]
    labels = ["A", "B", "C"]
    answer_label = ""
    for label_idx, size_rank in enumerate(perm):
        radii_by_label[label_idx] = sizes[int(size_rank)]
        if int(size_rank) == 0:
            answer_label = labels[label_idx]

    margin = max(int(0.10 * s), r_largest + 14)
    positions = [
        (s // 2, margin),
        (margin, s - margin),
        (s - margin, s - margin),
    ]

    colors = [random_color(rng), random_color(rng), random_color(rng)]
    paraphrase_idx = int(rng.integers(0, len(_QUESTIONS_HARD)))

    image = supersample_render(
        lambda cfg_hi: _render_size_compare(
            cfg_hi,
            [r * (cfg_hi.image_size // s) for r in radii_by_label],
            [(x * (cfg_hi.image_size // s), y * (cfg_hi.image_size // s))
             for x, y in positions],
            colors,
            labels,
        ),
        cfg,
    )

    return TaskInstance(
        name=name,
        seed=seed,
        question=_QUESTIONS_HARD[paraphrase_idx],
        ground_truth={
            "answer": answer_label,
            "radii_px": radii_by_label,
            "radii_frac": [r / s for r in radii_by_label],
            "positions": [list(p) for p in positions],
            "colors_rgb": [list(c) for c in colors],
            "ratio": ratio,
            "gap": gap,
            "layout": "triangle",
            "difficulty": _difficulty_hard(gap),
            "paraphrase_idx": paraphrase_idx,
            **render_config_meta(cfg),
        },
        image=image,
    )


class SizeCompareEasyTask:
    name = "blindtasks.size_compare"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_easy(seed, self.name)

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return _verify_label(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(instance.question, "either A or B", "A", fmt)


class SizeCompareHardTask:
    name = "blindtasks.size_compare.hard"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_hard(seed, self.name)

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return _verify_label(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(instance.question, "one of A, B, or C", "A", fmt)


TASK_EASY = SizeCompareEasyTask()
TASK_HARD = SizeCompareHardTask()
