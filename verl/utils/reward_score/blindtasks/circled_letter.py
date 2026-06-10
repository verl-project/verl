"""blindtasks.circled_letter

Two variants:
- easy: a letter inside one of four labelled circles (A/B/C/D);
    answer is the label
- hard: a long word is rendered in a line of text, one character has a red ellipse around it;
    answer is the circled character (mirroring vlmsareblind)
"""

import re
import string
import numpy as np
from PIL import Image, ImageDraw

from .base import (
    AnswerFormatter,
    RenderConfig,
    TaskInstance,
    answer_prompt,
)
from .utils import (
    draw_text_centered,
    load_font,
    random_font_path,
    render_config_meta,
    sample_render_config,
    supersample_render,
)

_LABELS = ("A", "B", "C", "D")
_N_CIRCLES = 4

_ACCENT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (220, 30, 30),
    (30, 70, 200),
    (40, 140, 40),
    (200, 110, 0),
)

_QUESTIONS_EASY = (
    (
        "A letter is circled. Which circle label (A, B, C, or D) contains the letter?"
    ),
    (
        "Which labelled circle (A, B, C, or D) contains the letter?"
    ),
)
_QUESTIONS_HARD = (
    (
        "A single character in the word shown in the image is circled by a coloured ellipse. "
        "Which character is it?"
    ),
    (
        "One character of the word in the image is highlighted by a coloured ellipse. "
        "Which character is highlighted?"
    ),
)

# vlmsareblind words
_WORDS_HARD = (
    "Acknowledgement",
    "Subdermatoglyphic",
    "Uncopyrightable",
    "tHyUiKaRbNqWeOpXcZvM",
    "Pseudopseudohypoparathyroidism",
    "Floccinaucinihilipilification",
)
_HARD_SCALE = 1.4
_HARD_SUPERSAMPLE = 2
_HARD_THICKNESSES = (3, 4, 5)
_HARD_MIN_WORD_LEN = min(len(w) for w in _WORDS_HARD)
_HARD_MAX_WORD_LEN = max(len(w) for w in _WORDS_HARD)


def _render_labels(circle_id: int, letter: str, cfg: RenderConfig) -> Image.Image:
    s = cfg.image_size
    
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    
    font_label = load_font(max(10, s // 18))
    font_letter = load_font(max(18, s // 9))
    
    cy = s // 2
    spacing = s // (_N_CIRCLES + 1)
    radius = max(10, s//12)
    
    for i, label in enumerate(_LABELS):
        cx = spacing * (i + 1)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=cfg.fg,
            width=cfg.line_width,
        )
        
        draw_text_centered(draw, cx, cy + radius + 10, label, font_label, cfg.fg)
        if i == circle_id:
            draw_text_centered(draw, cx, cy, letter, font_letter, cfg.fg)
    
    return image


def _render_word(
    word: str,
    char_idx: int,
    cfg: RenderConfig,
    thickness: int,
    accent: tuple[int, int, int] | None = None,
    font_path: str | None = None,
) -> Image.Image:
    final_s = cfg.image_size
    s = final_s * _HARD_SUPERSAMPLE
    
    image = Image.new("RGB", (s, s), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    font_size = max(14 * _HARD_SUPERSAMPLE, int(s * 0.85 / max(8, len(word))))
    font = load_font(font_size, path=font_path)

    advances: list[tuple[int, int, int, int]] = []
    for ch in word:
        try:
            l, t, r, b = draw.textbbox((0, 0), ch, font=font)
            advances.append((r - l, b - t, l, t))
        except AttributeError:
            advances.append((font_size, font_size, 0, 0))

    total_w = sum(a[0] for a in advances)
    max_h = max(a[1] for a in advances) if advances else font_size

    x = (s - total_w) // 2
    y = (s - max_h) // 2

    centers: list[tuple[int, int, int, int]] = []
    for (w, h, lo, to), ch in zip(advances, word):
        draw.text((x - lo, y - to), ch, fill=(0, 0, 0), font=font)
        centers.append((x + w // 2, y + h // 2, w, h))
        x += w

    cx, cy, w, h = centers[char_idx]
    rx = int(w * _HARD_SCALE / 2)
    ry = int(h * _HARD_SCALE / 2)

    accent_color = accent if accent is not None else cfg.accent
    scaled_thickness = thickness * _HARD_SUPERSAMPLE

    for off in range(scaled_thickness):
        draw.ellipse(
            [cx - rx - off, cy - ry - off, cx + rx + off, cy + ry + off],
            outline=accent_color,
            width=1,
        )
    return image.resize((final_s, final_s), Image.LANCZOS)


def _verify_label(prediction: str, target: str) -> float:
    match = re.search(r"\b([A-Da-d])\b", prediction)
    if not match:
        match = re.search(r"[A-Da-d]", prediction)
    if not match:
        return 0.0
    return 1.0 if match.group().upper() == target.upper() else 0.0


def _verify_char(prediction: str, target: str) -> float:
    pred = prediction.strip().strip("'\".,!?:; \t\n")
    if not pred:
        return 0.0
    if pred[0] == target:
        return 1.0
    return 1.0 if pred[0].lower() == target.lower() else 0.0


def _difficulty_easy() -> float:
    return 0.0


def _difficulty_hard(word: str) -> float:
    return (len(word) - _HARD_MIN_WORD_LEN) / max(1, _HARD_MAX_WORD_LEN - _HARD_MIN_WORD_LEN)


class CircledLetterEasyTask:
    name = "blindtasks.circled_letter"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng, line_widths=(2, 3))

        circle_id = int(rng.integers(0, _N_CIRCLES))
        letter = string.ascii_uppercase[int(rng.integers(0, 26))]
        paraphrase_idx = int(rng.integers(0, len(_QUESTIONS_EASY)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_QUESTIONS_EASY[paraphrase_idx],
            ground_truth={
                "answer": _LABELS[circle_id],
                "letter": letter,
                "difficulty": _difficulty_easy(),
                "paraphrase_idx": paraphrase_idx,
                **render_config_meta(cfg),
            },
            image=supersample_render(
                lambda cfg_hi: _render_labels(circle_id, letter, cfg_hi), cfg
            ),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return _verify_label(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(instance.question, "one of A, B, C, or D", "A", fmt)


class CircledLetterHardTask:
    name = "blindtasks.circled_letter.hard"

    def sample(self, seed: int) -> TaskInstance:
        rng = np.random.default_rng(seed)
        cfg = sample_render_config(rng, image_sizes=(336, 448), allow_invert=False)

        word = str(rng.choice(_WORDS_HARD))
        char_idx = int(rng.integers(0, len(word)))
        thickness = int(rng.choice(_HARD_THICKNESSES))
        
        accent = _ACCENT_PALETTE[int(rng.integers(0, len(_ACCENT_PALETTE)))]
        font_path = random_font_path(rng)
        paraphrase_idx = int(rng.integers(0, len(_QUESTIONS_HARD)))

        return TaskInstance(
            name=self.name,
            seed=seed,
            question=_QUESTIONS_HARD[paraphrase_idx],
            ground_truth={
                "answer": word[char_idx],
                "word": word,
                "char_index": char_idx,
                "image_size": cfg.image_size,
                "thickness": thickness,
                "accent_color": list(accent),
                "font_path": font_path,
                "paraphrase_idx": paraphrase_idx,
                "difficulty": _difficulty_hard(word),
            },
            image=_render_word(
                word, char_idx, cfg, thickness, accent=accent, font_path=font_path
            ),
        )

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return _verify_char(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(instance.question, "exactly the circled character", "a", fmt)


TASK_EASY = CircledLetterEasyTask()
TASK_HARD = CircledLetterHardTask()
