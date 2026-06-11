from __future__ import annotations

import re

try:
    from reasoning_gym.utils import LOCALIZED_FORMAT_INSTRUCTIONS as _RGYM_FORMAT_INSTRUCTIONS
    print("Loaded RGYM format instructions.")
except Exception:
    _RGYM_FORMAT_INSTRUCTIONS = {}
    print("Failed to load RGYM format instructions, falling back to regex-based stripping.")

RGYM_FORMAT_INSTRUCTIONS_RE = re.compile(
    r"\n{2,}"
    r"[^\n]*\n"
    r"-[^\n]*<answer>\.\.\.</answer>[^\n]*\n"
    r"-[^\n]*<answer>[^\n]*\n"
    r"-[^\n]*<answer>[^\n]*\n"
    r"-[^\n]*$",
    re.DOTALL,
)


def strip_rgym_format_instructions_by_regex(text: str) -> tuple[str, bool]:
    stripped = text.rstrip()
    match = RGYM_FORMAT_INSTRUCTIONS_RE.search(stripped)
    if not match:
        return text, False
    return stripped[: match.start()].rstrip(), True


def strip_rgym_format_instructions(
    text: str, language: str | None = None
) -> tuple[str, bool]:
    """Remove rgym answer-format instructions from a prompt suffix."""
    stripped = text.rstrip()
    instruction = _RGYM_FORMAT_INSTRUCTIONS.get(language or "")
    if instruction:
        instruction = instruction.rstrip()
        if stripped.endswith(instruction):
            return stripped[: -len(instruction)].rstrip(), True
        return text, False

    return strip_rgym_format_instructions_by_regex(text)
