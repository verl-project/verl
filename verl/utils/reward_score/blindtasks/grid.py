"""blindtasks.grid

Two variants:
- easy: ask amount of rows OR columns separately (R, C ∈ {2..12})
- hard: ask amount of both rows AND columns simultaneously (mirroring vlsmareblind)
"""

import re
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
    draw_text_centered,
    load_font,
    render_config_meta,
    sample_render_config,
    supersample_render,
    verify_count,
)

_MARGIN_FRAC = 0.09

_QUESTIONS_ROWS = (
    "How many rows does this grid have?",
    "Count the horizontal rows in this grid.",
)
_QUESTIONS_COLS = (
    "How many columns does this grid have?",
    "Count the vertical columns in this grid.",
)
_QUESTIONS_JOINT = (
    "How many rows and columns are in the grid?",
    "Report the row count and the column count of this grid.",
)

_WORD_GRID_LINE_WIDTHS = (2, 3, 4)
_WORD_GRID_WORDS: tuple[str, ...] = (
    "apple", "book", "car", "door", "earth", "fruit", "game", "house",
    "ice", "joke", "kite", "lamp", "moon", "night", "pen", "queen",
    "rain", "ship", "tree", "voice", "water", "yarn", "army", "baby",
    "cake", "dance", "echo", "flag", "happy", "idea", "jelly", "king",
    "lion", "mouse", "north", "opera", "plant", "quiet", "robot", "stone",
    "tiger", "union", "vivid", "world", "youth", "brave", "climb", "dream",
    "alert", "bread", "clean", "drama", "elbow", "flame", "grape", "heart",
    "index", "juice", "lemon", "march", "noble", "ocean", "peace", "reign",
    "shine", "track", "unity", "vapor", "weave", "zone", "angle", "bloom",
    "crisp", "diver", "equip", "fable", "giant", "hover", "ideal", "jolly",
    "lunar", "neigh", "orbit", "plush", "quote", "ridge", "skate", "truth",
    "urban", "valve", "whale", "yield", "azure", "blaze", "charm", "donor",
    "eagle", "flock", "grain", "hinge",
)


def _render_grid(rows: int, cols: int, cfg: RenderConfig) -> Image.Image:
    s = cfg.image_size
    
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    
    margin = max(6, int(s * _MARGIN_FRAC))
    usable = s - 2 * margin
    
    for i in range(rows + 1):
        y = margin + round(i * usable / rows)
        draw.line([(margin, y), (s - margin, y)], fill=cfg.fg, width=cfg.line_width)
    
    for j in range(cols + 1):
        x = margin + round(j * usable / cols)
        draw.line([(x, margin), (x, s - margin)], fill=cfg.fg, width=cfg.line_width)
    
    return image


def _parse_joint(prediction: str) -> tuple[int, int] | None:
    rows_match = re.search(r"rows?\s*[=:]\s*(\d+)", prediction, re.IGNORECASE)
    cols_match = re.search(r"col(?:umn)?s?\s*[=:]\s*(\d+)", prediction, re.IGNORECASE)
    if rows_match and cols_match:
        return int(rows_match.group(1)), int(cols_match.group(1))
    ints = re.findall(r"\d+", prediction)
    if len(ints) >= 2:
        return int(ints[0]), int(ints[1])
    return None


def _verify_joint(prediction: str, rows: int, cols: int) -> float:
    parsed = _parse_joint(prediction)
    if parsed is None:
        return 0.0
    return 1.0 if parsed == (rows, cols) else 0.0


def _verify_joint_shaped(prediction: str, rows: int, cols: int) -> float:
    parsed = _parse_joint(prediction)
    if parsed is None:
        return 0.0
    pr, pc = parsed
    return (0.5 if pr == rows else 0.0) + (0.5 if pc == cols else 0.0)


def _difficulty_easy(rows: int, cols: int) -> float:
    return (max(rows, cols) - 2) / (12 - 2)


def _difficulty_hard(rows: int, cols: int) -> float:
    return (rows + cols - 4) / (12 + 12 - 4)


class GridEasyTask:
    name = "blindtasks.grid"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng, line_widths=(1, 2, 3, 4))

        rows = int(rng.integers(2, 13))
        cols = int(rng.integers(2, 13))

        ask_rows = bool(rng.integers(0, 2))
        pool = _QUESTIONS_ROWS if ask_rows else _QUESTIONS_COLS
        paraphrase_idx = int(rng.integers(0, len(pool)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=pool[paraphrase_idx],
            ground_truth={
                "answer": rows if ask_rows else cols,
                "rows": rows,
                "cols": cols,
                "ask_rows": ask_rows,
                "difficulty": _difficulty_easy(rows, cols),
                "paraphrase_idx": paraphrase_idx,
                **render_config_meta(cfg),
            },
            image=supersample_render(
                lambda cfg_hi: _render_grid(rows, cols, cfg_hi), cfg
            ),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_count(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return count_prompt(instance.question, fmt)


class GridHardTask:
    name = "blindtasks.grid.hard"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng, line_widths=(1, 2, 3, 4))

        rows = int(rng.integers(2, 13))
        cols = int(rng.integers(2, 13))
        paraphrase_idx = int(rng.integers(0, len(_QUESTIONS_JOINT)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_QUESTIONS_JOINT[paraphrase_idx],
            ground_truth={
                "answer": f"rows={rows} columns={cols}",
                "rows": rows,
                "cols": cols,
                "difficulty": _difficulty_hard(rows, cols),
                "paraphrase_idx": paraphrase_idx,
                **render_config_meta(cfg),
            },
            image=supersample_render(
                lambda cfg_hi: _render_grid(rows, cols, cfg_hi), cfg
            ),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return _verify_joint(
            prediction,
            instance.ground_truth["rows"],
            instance.ground_truth["cols"],
        )

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(
            instance.question,
            "formatted exactly as rows=x columns=y",
            "rows=3 columns=4",
            fmt,
        )


def _render_word_grid(
    rows: int, cols: int, cfg: RenderConfig, words: list[str]
) -> Image.Image:
    s = cfg.image_size

    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)

    margin = max(6, int(s * _MARGIN_FRAC))
    usable = s - 2 * margin
    cell_h = usable / rows
    cell_w = usable / cols

    for i in range(rows + 1):
        y = margin + round(i * usable / rows)
        draw.line([(margin, y), (s - margin, y)], fill=cfg.fg, width=cfg.line_width)
    for j in range(cols + 1):
        x = margin + round(j * usable / cols)
        draw.line([(x, margin), (x, s - margin)], fill=cfg.fg, width=cfg.line_width)

    font_size = max(8, int(min(cell_h, cell_w) * 0.30))
    font = load_font(font_size)

    for r in range(rows):
        for c in range(cols):
            word = words[r * cols + c]
            cx = int(margin + (c + 0.5) * cell_w)
            cy = int(margin + (r + 0.5) * cell_h)
            draw_text_centered(draw, cx, cy, word, font, cfg.fg)
    return image


class GridWordTask:
    name = "blindtasks.grid.word"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng, line_widths=_WORD_GRID_LINE_WIDTHS)

        rows = int(rng.integers(2, 9))
        cols = int(rng.integers(2, 9))

        n_cells = rows * cols
        offset = int(rng.integers(0, len(_WORD_GRID_WORDS)))
        words = [
            _WORD_GRID_WORDS[(offset + i) % len(_WORD_GRID_WORDS)]
            for i in range(n_cells)
        ]

        image = supersample_render(
            lambda cfg_hi: _render_word_grid(rows, cols, cfg_hi, words), cfg
        )
        paraphrase_idx = int(rng.integers(0, len(_QUESTIONS_JOINT)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_QUESTIONS_JOINT[paraphrase_idx],
            ground_truth={
                "answer": f"rows={rows} columns={cols}",
                "rows": rows,
                "cols": cols,
                "cell_words": words,
                "difficulty": _difficulty_hard(rows, cols),
                "paraphrase_idx": paraphrase_idx,
                **render_config_meta(cfg),
            },
            image=image,
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return _verify_joint(
            prediction,
            instance.ground_truth["rows"],
            instance.ground_truth["cols"],
        )

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(
            instance.question,
            "formatted exactly as rows=x columns=y",
            "rows=3 columns=4",
            fmt,
        )


TASK_EASY = GridEasyTask()
TASK_HARD = GridHardTask()
TASK_WORD = GridWordTask()
