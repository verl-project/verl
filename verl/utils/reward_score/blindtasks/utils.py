import re
import math
from dataclasses import replace
from typing import Callable

import numpy as np
from PIL import Image, ImageFont

from .base import RenderConfig

_DEFAULT_SUPERSAMPLE = 2

_FONT_PATHS: tuple[str, ...] = (
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    # Linux (Liberation family)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    # Linux (DejaVu family)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    # Linux (FreeFont family)
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
)


def _discover_available_fonts() -> tuple[str, ...]:
    available: list[str] = []
    for path in _FONT_PATHS:
        try:
            ImageFont.truetype(path, 12)
            available.append(path)
        except (OSError, IOError):
            continue
    return tuple(available)


_AVAILABLE_FONTS: tuple[str, ...] = _discover_available_fonts()
_DEFAULT_IMAGE_SIZES = (224, 256, 336)
_DEFAULT_LINE_WIDTHS = (1, 2, 3)
_PALETTE_NORMAL = ((255, 255, 255), (0, 0, 0))
_PALETTE_INVERTED = ((0, 0, 0), (255, 255, 255))

_DISTINCT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (220, 30, 30),
    (30, 70, 200),
    (40, 140, 40),
    (200, 110, 0),
    (130, 70, 180),
    (190, 30, 160),
    (20, 150, 170),
    (110, 60, 30),
)

_WORD_TO_INT: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

_DIRECTION_LABELS_4: tuple[str, ...] = ("N", "E", "S", "W")
_DIRECTION_LABELS_8: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_DIRECTION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "N":  ("north", "up", "upward", "upwards"),
    "S":  ("south", "down", "downward", "downwards"),
    "E":  ("east", "right", "rightward"),
    "W":  ("west", "left", "leftward"),
    "NE": ("northeast", "north-east", "upper-right", "upper right", "top-right", "top right"),
    "SE": ("southeast", "south-east", "lower-right", "lower right", "bottom-right", "bottom right"),
    "SW": ("southwest", "south-west", "lower-left", "lower left", "bottom-left", "bottom left"),
    "NW": ("northwest", "north-west", "upper-left", "upper left", "top-left", "top left"),
}
_SYNONYM_TO_DIRECTION: dict[str, str] = {
    syn: canon for canon, syns in _DIRECTION_SYNONYMS.items() for syn in syns
}

_RELATION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "above": ("above", "over", "on top of", "higher than"),
    "below": ("below", "under", "underneath", "beneath", "lower than"),
    "left":  ("left", "to the left of", "on the left of"),
    "right": ("right", "to the right of", "on the right of"),
}

def load_font(size: int, path: str | None = None) -> ImageFont.ImageFont:
    candidates = (path,) if path else _FONT_PATHS
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def random_font_path(rng: np.random.Generator) -> str | None:
    if not _AVAILABLE_FONTS:
        return None
    return _AVAILABLE_FONTS[int(rng.integers(0, len(_AVAILABLE_FONTS)))]


def draw_text_centered(draw, cx: int, cy: int, text: str, font, fill) -> None:
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top
        draw.text((cx - w // 2 - left, cy - h // 2 - top), text, fill=fill, font=font)
    except AttributeError:
        draw.text((cx, cy), text, fill=fill, font=font)


def sample_render_config(
    rng: np.random.Generator,
    *,
    image_sizes: tuple[int, ...] = _DEFAULT_IMAGE_SIZES,
    line_widths: tuple[int, ...] = _DEFAULT_LINE_WIDTHS,
    allow_invert: bool = True,
    accent: tuple[int, int, int] = (220, 0, 0),
) -> RenderConfig:
    """Sample a RenderConfig with random image size, line width, and colour scheme"""
    image_size = int(rng.choice(image_sizes))
    line_width = int(rng.choice(line_widths))
    
    if allow_invert and bool(rng.integers(0, 2)):
        bg, fg = _PALETTE_INVERTED
    else:
        bg, fg = _PALETTE_NORMAL
    
    return RenderConfig(
        image_size=image_size,
        line_width=line_width,
        bg=bg,
        fg=fg,
        accent=accent,
    )


def _orientation(p, q, r) -> int:
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def _on_segment(p, q, r) -> bool:
    return (
        min(p[0], q[0]) - 1e-9 <= r[0] <= max(p[0], q[0]) + 1e-9
        and min(p[1], q[1]) - 1e-9 <= r[1] <= max(p[1], q[1]) + 1e-9
    )


def segments_intersect(p1, p2, p3, p4) -> bool:
    o1 = _orientation(p1, p2, p3)
    o2 = _orientation(p1, p2, p4)
    o3 = _orientation(p3, p4, p1)
    o4 = _orientation(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, p3):
        return True
    if o2 == 0 and _on_segment(p1, p2, p4):
        return True
    if o3 == 0 and _on_segment(p3, p4, p1):
        return True
    if o4 == 0 and _on_segment(p3, p4, p2):
        return True
    
    return False


def count_polyline_intersections(poly_a, poly_b) -> int:
    n = 0
    for i in range(len(poly_a) - 1):
        for j in range(len(poly_b) - 1):
            if segments_intersect(poly_a[i], poly_a[i + 1], poly_b[j], poly_b[j + 1]):
                n += 1
    return n


def circles_overlap(c1, r1, c2, r2) -> bool:
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return math.sqrt(dx * dx + dy * dy) <= (r1 + r2) + 1e-9


def _extract_count(prediction: str) -> int | None:
    match = re.search(r"-?\d+", prediction)
    if match:
        return int(match.group())
    lower = prediction.lower()
    best_pos = len(lower) + 1
    best_val: int | None = None
    for word, n in _WORD_TO_INT.items():
        m = re.search(rf"\b{word}\b", lower)
        if m and m.start() < best_pos:
            best_pos = m.start()
            best_val = n
    return best_val


def verify_count(prediction: str, target: int) -> float:
    pred = _extract_count(prediction)
    if pred is None:
        return 0.0
    return 1.0 if pred == target else 0.0


def verify_count_shaped(prediction: str, target: int) -> float:
    pred = _extract_count(prediction)
    if pred is None:
        return 0.0
    denom = max(target, 1)
    return max(0.0, 1.0 - abs(pred - target) / denom)


def random_color(
    rng: np.random.Generator,
    palette: tuple[tuple[int, int, int], ...] = _DISTINCT_PALETTE,
) -> tuple[int, int, int]:
    return palette[int(rng.integers(0, len(palette)))]


def random_color_pair(
    rng: np.random.Generator,
    palette: tuple[tuple[int, int, int], ...] = _DISTINCT_PALETTE,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if len(palette) < 2:
        raise ValueError(f"random_color_pair needs >=2 palette entries; got {len(palette)}")
    a = random_color(rng, palette)
    b = random_color(rng, palette)
    while b == a:
        b = random_color(rng, palette)
    return a, b


def circle_in_viewport(
    center: tuple[float, float], radius: float, canvas: int, margin: int = 0
) -> bool:
    cx, cy = center
    return (
        cx - radius >= margin
        and cx + radius <= canvas - margin
        and cy - radius >= margin
        and cy + radius <= canvas - margin
    )


def verify_yes_no(prediction: str, target: str) -> float:
    p = prediction.lower()
    if re.search(r"\byes\b", p):
        pred = "yes"
    elif re.search(r"\bno\b", p):
        pred = "no"
    else:
        return 0.0
    return 1.0 if pred == target else 0.0


def _extract_direction(prediction: str) -> str | None:
    pred = prediction.lower()
    matches: list[tuple[int, int, str]] = []
    for label in ("ne", "se", "sw", "nw"):
        for m in re.finditer(rf"\b{label}\b", pred):
            matches.append((m.start(), 2, label.upper()))
    for label in ("n", "s", "e", "w"):
        for m in re.finditer(rf"\b{label}\b", pred):
            matches.append((m.start(), 1, label.upper()))
    for syn, canon in _SYNONYM_TO_DIRECTION.items():
        pos = pred.find(syn)
        if pos >= 0:
            matches.append((pos, len(syn), canon))
    if not matches:
        return None
    matches.sort(key=lambda x: (x[0], -x[1]))
    return matches[0][2]


def verify_direction(prediction: str, target: str) -> float:
    pred = _extract_direction(prediction)
    if pred is None:
        return 0.0
    return 1.0 if pred == target else 0.0


def verify_direction_shaped(prediction: str, target: str, n_directions: int) -> float:
    pred = _extract_direction(prediction)
    if pred is None:
        return 0.0
    if n_directions == 4:
        labels = _DIRECTION_LABELS_4
    elif n_directions == 8:
        labels = _DIRECTION_LABELS_8
    else:
        raise ValueError(f"n_directions must be 4 or 8, got {n_directions}")
    if pred not in labels:
        return 0.0
    pi = labels.index(pred)
    ti = labels.index(target)
    diff = abs(pi - ti)
    steps_off = min(diff, n_directions - diff)
    return max(0.0, 1.0 - 2.0 * steps_off / n_directions)


def _extract_relation(prediction: str) -> str | None:
    pred = prediction.lower()
    matches: list[tuple[int, int, str]] = []
    for label in ("above", "below", "left", "right"):
        for m in re.finditer(rf"\b{label}\b", pred):
            matches.append((m.start(), len(label), label))
    for canon, syns in _RELATION_SYNONYMS.items():
        for syn in syns:
            pos = pred.find(syn)
            if pos >= 0:
                matches.append((pos, len(syn), canon))
    if not matches:
        return None
    matches.sort(key=lambda x: (x[0], -x[1]))
    return matches[0][2]


def verify_relation(prediction: str, target: str) -> float:
    pred = _extract_relation(prediction)
    if pred is None:
        return 0.0
    return 1.0 if pred == target else 0.0


def supersample_render(
    render_fn: Callable[..., Image.Image],
    cfg: RenderConfig,
    *args,
    factor: int = _DEFAULT_SUPERSAMPLE,
    **kwargs,
) -> Image.Image:
    cfg_hi = replace(cfg, image_size=cfg.image_size * factor, line_width=max(1, cfg.line_width * factor))
    image_hi = render_fn(cfg_hi, *args, **kwargs)
    return image_hi.resize((cfg.image_size, cfg.image_size), Image.LANCZOS)


def render_config_meta(cfg: RenderConfig) -> dict:
    return {
        "image_size": cfg.image_size,
        "line_width": cfg.line_width,
        "inverted": cfg.bg != (255, 255, 255),
    }


def sample_balanced(
    task,
    n_per_class: int,
    *,
    class_fn=None,
    expected_classes: tuple | None = None,
    start_seed: int = 0,
    max_total_attempts: int | None = None,
) -> list:
    if class_fn is None:
        def class_fn(inst):
            return inst.ground_truth["answer"]

    expected = set(expected_classes) if expected_classes is not None else None
    cap = max_total_attempts or n_per_class * 50 * max(len(expected) if expected else 8, 8)

    counts: dict = {}
    accepted: list = []
    seed = start_seed
    attempts = 0

    while attempts < cap:
        inst = task.sample(seed)
        seed += 1
        attempts += 1
        cls = class_fn(inst)
        if expected is not None and cls not in expected:
            continue
        current = counts.get(cls, 0)
        if current >= n_per_class:
            continue
        accepted.append(inst)
        counts[cls] = current + 1

        if expected is not None and all(counts.get(c, 0) >= n_per_class for c in expected):
            break
        if expected is None and counts and min(counts.values()) >= n_per_class:
            break

    return accepted


def place_non_overlapping_circles(
    rng: np.random.Generator,
    n: int,
    canvas: int,
    *,
    radius_range: tuple[int, int],
    margin: int,
    min_gap: int = 4,
    max_attempts: int = 4000,
) -> list[tuple[int, int, int]]:
    """Sample up to n non-overlapping circles (cx, cy, r) inside a square canvas"""
    attempts = 0
    placed: list[tuple[int, int, int]] = []
    
    while len(placed) < n and attempts < max_attempts:
        r = int(rng.integers(radius_range[0], radius_range[1] + 1))
        cx = int(rng.integers(margin + r, canvas - margin - r + 1))
        cy = int(rng.integers(margin + r, canvas - margin - r + 1))
        
        ok = True
        for px, py, pr in placed:
            if circles_overlap((cx, cy), r + min_gap, (px, py), pr):
                ok = False
                break
        
        if ok:
            placed.append((cx, cy, r))
        attempts += 1
    
    return placed
