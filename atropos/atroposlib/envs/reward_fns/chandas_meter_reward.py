import logging
import re
from typing import Any, List

try:
    from pkg_resources import resource_filename
except Exception:  # pragma: no cover - optional dependency
    resource_filename = None

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)

# Basic mapping from IAST characters/digraphs to SLP1
_IAST_TO_SLP1 = [
    ("kh", "K"),
    ("gh", "G"),
    ("ch", "C"),
    ("jh", "J"),
    ("ṭh", "W"),
    ("ḍh", "Q"),
    ("th", "T"),
    ("dh", "D"),
    ("ph", "P"),
    ("bh", "B"),
    ("ai", "E"),
    ("au", "O"),
    ("ā", "A"),
    ("ī", "I"),
    ("ū", "U"),
    ("ṛ", "f"),
    ("ṝ", "F"),
    ("ḷ", "x"),
    ("ḹ", "X"),
    ("ṅ", "N"),
    ("ñ", "Y"),
    ("ṭ", "w"),
    ("ḍ", "q"),
    ("ṇ", "R"),
    ("ś", "S"),
    ("ṣ", "z"),
    ("ṃ", "M"),
    ("ṁ", "M"),
    ("ḥ", "H"),
]

_SINGLE_CHAR_MAP = {
    "a": "a",
    "i": "i",
    "u": "u",
    "e": "e",
    "o": "o",
    "k": "k",
    "g": "g",
    "c": "c",
    "j": "j",
    "t": "t",
    "d": "d",
    "n": "n",
    "p": "p",
    "b": "b",
    "m": "m",
    "y": "y",
    "r": "r",
    "l": "l",
    "v": "v",
    "s": "s",
    "h": "h",
}

_DIGRAPH_RE = re.compile("|".join(re.escape(d[0]) for d in _IAST_TO_SLP1), re.UNICODE)


def iast_to_slp1(text: str) -> str:
    """Convert a string from IAST to SLP1."""

    def _replace(match: re.Match) -> str:
        for iast, slp in _IAST_TO_SLP1:
            if match.group(0) == iast:
                return slp
        return match.group(0)

    text = _DIGRAPH_RE.sub(_replace, text)
    return "".join(_SINGLE_CHAR_MAP.get(ch, ch) for ch in text)


@registry.register
class ChandasMeterReward(RewardFunction):
    """Reward based on how closely a poem matches a target Sanskrit meter."""

    def __init__(self, meter: str = "tristubh", weight: float = 1.0, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.meter = meter
        try:
            from chandas import Classifier  # type: ignore

            if resource_filename is not None:
                data_path = resource_filename("chandas", "data/data.json")
                self.classifier = Classifier.from_json_file(data_path)
            else:
                self.classifier = Classifier.from_default_location()
        except Exception as e:  # pragma: no cover - optional dependency
            logger.error("Failed to load chandas Classifier: %s", e)
            self.classifier = None

    def _score_text(self, text: str) -> float:
        if not self.classifier:
            return 0.0
        try:
            slp_text = iast_to_slp1(text)
            result = self.classifier.classify(slp_text)
            if not result:
                return 0.0
            predicted = getattr(result, "name", str(result)).lower()
            raw_score = getattr(
                result, "score", 1.0 if predicted == self.meter.lower() else 0.0
            )
            if predicted != self.meter.lower():
                raw_score = 1.0 - raw_score if raw_score <= 1.0 else 0.0
            return max(0.0, min(1.0, float(raw_score)))
        except Exception as e:  # pragma: no cover - runtime safeguard
            logger.error("Error scoring text with chandas: %s", e)
            return 0.0

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        rewards: List[float] = []
        for completion in completions:
            text = self.get_content(completion)
            rewards.append(self._score_text(text))
        return rewards
