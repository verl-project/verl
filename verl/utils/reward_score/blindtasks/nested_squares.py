"""blindtasks.nested_squares

Two variants:
- easy: perfectly concentric squares, k ∈ {2..6}
- hard: each inner square is randomly offset withing the bounds allowed by its parent,
    k ∈ {2..5} (mirroring vlmsareblind)
"""

import numpy as np
from PIL import Image, ImageDraw

from .base import AnswerFormatter, RenderConfig, TaskInstance, count_prompt
from .utils import (
    render_config_meta,
    sample_render_config,
    supersample_render,
    verify_count,
)

_MARGIN_FRAC = 0.06
_QUESTIONS = (
    "How many nested squares are in this image?",
    "Count the nested squares in this image.",
)

# vlmsareblind hyperparams
_HARD_REDUCTION = 0.75
_HARD_PADDING_FRAC = 0.045


def _difficulty_easy(k: int) -> float:
    return (k - 2) / (6 - 2)


def _difficulty_hard(k: int) -> float:
    return (k - 2) / (5 - 2)


def _render_concentric(k: int, cfg: RenderConfig) -> Image.Image:
    s = cfg.image_size
    
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    
    cx = cy = s // 2
    margin = max(4, int(s * _MARGIN_FRAC))
    step = (cx - margin) // k
    
    for i in range(k):
        half = margin + (i + 1) * step
        draw.rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            outline=cfg.fg,
            width=cfg.line_width,
        )
    
    return image


def _render_offset(k: int, cfg: RenderConfig, rng: np.random.Generator) -> Image.Image:
    s = cfg.image_size
    
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    
    margin = max(6, int(s * _MARGIN_FRAC))
    padding = max(3, int(s * _HARD_PADDING_FRAC))

    cx = cy = s / 2
    size = s - 2 * margin
    
    for _ in range(k):
        half = size / 2
        draw.rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            outline=cfg.fg,
            width=cfg.line_width,
        )
        
        new_size = size * _HARD_REDUCTION - padding
        if new_size <= 4:
            break
        
        max_offset = max(0.0, (size - new_size - padding) / 2.0)
        cx += float(rng.uniform(-max_offset, max_offset))
        cy += float(rng.uniform(-max_offset, max_offset))
        size = new_size
    
    return image


class NestedSquaresEasyTask:
    name = "blindtasks.nested_squares"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng)
        k = int(rng.integers(2, 7))
        paraphrase_idx = int(rng.integers(0, len(_QUESTIONS)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_QUESTIONS[paraphrase_idx],
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


class NestedSquaresHardTask:
    name = "blindtasks.nested_squares.hard"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng, line_widths=(2, 3, 4))
        k = int(rng.integers(2, 6))

        image = supersample_render(
            lambda cfg_hi: _render_offset(k, cfg_hi, rng), cfg
        )
        paraphrase_idx = int(rng.integers(0, len(_QUESTIONS)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_QUESTIONS[paraphrase_idx],
            ground_truth={
                "answer": k,
                "difficulty": _difficulty_hard(k),
                "paraphrase_idx": paraphrase_idx,
                **render_config_meta(cfg),
            },
            image=image,
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


TASK_EASY = NestedSquaresEasyTask()
TASK_HARD = NestedSquaresHardTask()
