"""blindtasks.spatial_relation

Two variants:
- easy: 2 distinguishable shapes. Ask 4-way relative position
- hard: 3 shapes (target + reference + distractor). Same 4-way answer
"""

import numpy as np
from PIL import Image, ImageDraw

from .base import (
    AnswerFormatter,
    RenderConfig,
    TaskInstance,
    answer_prompt,
)
from .utils import (
    render_config_meta,
    sample_render_config,
    supersample_render,
    verify_relation,
)

_SHAPES: tuple[str, ...] = ("circle", "square", "triangle")
_RELATIONS: tuple[str, ...] = ("above", "below", "left", "right")

_COLORS_NAMED: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red",     (220, 30, 30)),
    ("blue",    (30, 70, 200)),
    ("green",   (40, 140, 40)),
    ("orange",  (200, 110, 0)),
    ("purple",  (130, 70, 180)),
    ("magenta", (190, 30, 160)),
    ("teal",    (20, 150, 170)),
    ("brown",   (110, 60, 30)),
)

_QUESTION_TEMPLATES: tuple[str, ...] = (
    "Where is the {ca} {sa} relative to the {cb} {sb}?",
    "Is the {ca} {sa} above, below, left of, or right of the {cb} {sb}?",
)

_SHAPE_HALF_FRAC = 0.10
_MIN_SEP_FRAC = 0.25
_DOMINANCE_RATIO = 3.0


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    cx: int,
    cy: int,
    half: int,
    color: tuple[int, int, int],
) -> None:
    if shape == "circle":
        draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=color)
    elif shape == "square":
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color)
    elif shape == "triangle":
        draw.polygon([
            (cx, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ], fill=color)


def _render_relation(cfg: RenderConfig, shapes_specs: list[tuple[str, int, int, tuple[int, int, int]]]) -> Image.Image:
    s = cfg.image_size
    image = Image.new("RGB", (s, s), cfg.bg)
    draw = ImageDraw.Draw(image)
    half = max(6, int(_SHAPE_HALF_FRAC * s))
    for shape_name, cx, cy, col in shapes_specs:
        _draw_shape(draw, shape_name, cx, cy, half, col)
    return image


def _sample_distinguishable_pair(rng: np.random.Generator) -> tuple[int, int, int, int]:
    sa = int(rng.integers(0, len(_SHAPES)))
    sb = int(rng.integers(0, len(_SHAPES)))
    ca = int(rng.integers(0, len(_COLORS_NAMED)))
    cb = int(rng.integers(0, len(_COLORS_NAMED)))
    while sa == sb and ca == cb:
        cb = int(rng.integers(0, len(_COLORS_NAMED)))
    return sa, ca, sb, cb


def _place_pair(rng: np.random.Generator, s: int, answer: str) -> tuple[tuple[int, int], tuple[int, int]]:
    half = max(6, int(_SHAPE_HALF_FRAC * s))
    margin = half + max(4, int(0.05 * s))
    min_sep = max(margin + 1, int(_MIN_SEP_FRAC * s))
    asked_vertical = answer in ("above", "below")

    if asked_vertical:
        if answer == "above":
            ref_y = int(rng.integers(min_sep + margin, s - margin))
            tgt_y = int(rng.integers(margin, ref_y - min_sep + 1))
        else:
            ref_y = int(rng.integers(margin, s - min_sep - margin))
            tgt_y = int(rng.integers(ref_y + min_sep, s - margin + 1))
        y_sep = abs(tgt_y - ref_y)
        max_x_diff = y_sep // int(_DOMINANCE_RATIO)
        ref_x = int(rng.integers(margin, s - margin))
        tgt_x_lo = max(margin, ref_x - max_x_diff)
        tgt_x_hi = min(s - margin, ref_x + max_x_diff)
        tgt_x = int(rng.integers(tgt_x_lo, tgt_x_hi + 1))
    else:
        if answer == "left":
            ref_x = int(rng.integers(min_sep + margin, s - margin))
            tgt_x = int(rng.integers(margin, ref_x - min_sep + 1))
        else:
            ref_x = int(rng.integers(margin, s - min_sep - margin))
            tgt_x = int(rng.integers(ref_x + min_sep, s - margin + 1))
        x_sep = abs(tgt_x - ref_x)
        max_y_diff = x_sep // int(_DOMINANCE_RATIO)
        ref_y = int(rng.integers(margin, s - margin))
        tgt_y_lo = max(margin, ref_y - max_y_diff)
        tgt_y_hi = min(s - margin, ref_y + max_y_diff)
        tgt_y = int(rng.integers(tgt_y_lo, tgt_y_hi + 1))

    return (tgt_x, tgt_y), (ref_x, ref_y)


def _place_distractor(
    rng: np.random.Generator,
    s: int,
    target_xy: tuple[int, int],
    reference_xy: tuple[int, int],
    max_attempts: int = 100,
) -> tuple[int, int]:
    half = max(6, int(_SHAPE_HALF_FRAC * s))
    margin = half + max(4, int(0.05 * s))
    min_sep = 2 * half + 8
    for _ in range(max_attempts):
        dx = int(rng.integers(margin, s - margin))
        dy = int(rng.integers(margin, s - margin))
        sep_t = max(abs(dx - target_xy[0]), abs(dy - target_xy[1]))
        sep_r = max(abs(dx - reference_xy[0]), abs(dy - reference_xy[1]))
        if sep_t >= min_sep and sep_r >= min_sep:
            return dx, dy
    return (
        int(np.clip(2 * reference_xy[0] - target_xy[0], margin, s - margin)),
        int(np.clip(2 * reference_xy[1] - target_xy[1], margin, s - margin)),
    )


def _difficulty(
    target_xy: tuple[int, int],
    reference_xy: tuple[int, int],
    asked_axis: str,
    has_distractor: bool,
) -> float:
    if asked_axis == "vertical":
        asked = abs(target_xy[1] - reference_xy[1])
        orth = abs(target_xy[0] - reference_xy[0])
    else:
        asked = abs(target_xy[0] - reference_xy[0])
        orth = abs(target_xy[1] - reference_xy[1])
    base = orth / max(asked, 1)
    d = min(1.0, base * _DOMINANCE_RATIO)
    if has_distractor:
        d = min(1.0, d + 0.3)
    return d


def _sample_relation_task(
    seed: int,
    name: str,
    *,
    is_hard: bool,
) -> TaskInstance:
    rng = np.random.default_rng(seed)
    cfg = sample_render_config(rng, allow_invert=False)
    s = cfg.image_size

    answer = _RELATIONS[int(rng.integers(0, len(_RELATIONS)))]
    sa_idx, ca_idx, sb_idx, cb_idx = _sample_distinguishable_pair(rng)
    shape_a = _SHAPES[sa_idx]
    name_a, rgb_a = _COLORS_NAMED[ca_idx]
    shape_b = _SHAPES[sb_idx]
    name_b, rgb_b = _COLORS_NAMED[cb_idx]

    target_xy, reference_xy = _place_pair(rng, s, answer)

    distractor_info: dict | None = None
    if is_hard:
        avail_shapes = [i for i in range(len(_SHAPES)) if i not in (sa_idx, sb_idx)]
        avail_colors = [i for i in range(len(_COLORS_NAMED))
                        if i not in (ca_idx, cb_idx)]
        if not avail_shapes:
            avail_shapes = list(range(len(_SHAPES)))
        if not avail_colors:
            avail_colors = list(range(len(_COLORS_NAMED)))
        d_shape = _SHAPES[avail_shapes[int(rng.integers(0, len(avail_shapes)))]]
        d_name, d_rgb = _COLORS_NAMED[
            avail_colors[int(rng.integers(0, len(avail_colors)))]
        ]
        d_xy = _place_distractor(rng, s, target_xy, reference_xy)
        distractor_info = {
            "shape": d_shape,
            "color_name": d_name,
            "color_rgb": d_rgb,
            "position": list(d_xy),
        }

    asked_axis = "vertical" if answer in ("above", "below") else "horizontal"
    difficulty = _difficulty(target_xy, reference_xy, asked_axis, is_hard)

    specs_base: list[tuple[str, int, int, tuple[int, int, int]]] = [
        (shape_a, target_xy[0], target_xy[1], rgb_a),
        (shape_b, reference_xy[0], reference_xy[1], rgb_b),
    ]
    if distractor_info is not None:
        specs_base.append((
            distractor_info["shape"],
            distractor_info["position"][0],
            distractor_info["position"][1],
            distractor_info["color_rgb"],
        ))

    def _scale_specs(cfg_hi):
        scale = cfg_hi.image_size // s
        return [(sn, x * scale, y * scale, col)
                for sn, x, y, col in specs_base]

    image = supersample_render(
        lambda cfg_hi: _render_relation(cfg_hi, _scale_specs(cfg_hi)),
        cfg,
    )

    paraphrase_idx = int(rng.integers(0, len(_QUESTION_TEMPLATES)))
    question = _QUESTION_TEMPLATES[paraphrase_idx].format(
        ca=name_a, sa=shape_a, cb=name_b, sb=shape_b,
    )

    gt = {
        "answer": answer,
        "target_shape": shape_a,
        "target_color_name": name_a,
        "target_color_rgb": list(rgb_a),
        "target_position": list(target_xy),
        "reference_shape": shape_b,
        "reference_color_name": name_b,
        "reference_color_rgb": list(rgb_b),
        "reference_position": list(reference_xy),
        "distractor_shape": distractor_info["shape"] if distractor_info else None,
        "distractor_color_name": (
            distractor_info["color_name"] if distractor_info else None
        ),
        "distractor_position": (
            distractor_info["position"] if distractor_info else None
        ),
        "asked_axis": asked_axis,
        "difficulty": difficulty,
        "paraphrase_idx": paraphrase_idx,
        **render_config_meta(cfg),
    }

    return TaskInstance(
        name=name,
        seed=seed,
        question=question,
        ground_truth=gt,
        image=image,
    )


class SpatialRelationEasyTask:
    name = "blindtasks.spatial_relation"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_relation_task(seed, self.name, is_hard=False)

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_relation(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(
            instance.question,
            "one of above, below, left, or right",
            "above",
            fmt,
        )


class SpatialRelationHardTask:
    name = "blindtasks.spatial_relation.hard"

    def sample(self, seed: int) -> TaskInstance:
        return _sample_relation_task(seed, self.name, is_hard=True)

    def verify(self, instance: TaskInstance, prediction: str) -> float:
        return verify_relation(prediction, instance.ground_truth["answer"])

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        return answer_prompt(
            instance.question,
            "one of above, below, left, or right",
            "above",
            fmt,
        )


TASK_EASY = SpatialRelationEasyTask()
TASK_HARD = SpatialRelationHardTask()
