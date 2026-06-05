from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeVerdict:
    verdict: str
    reason: str
    malformed: bool = False


def parse_judge_output(text: str) -> JudgeVerdict:
    verdict: str | None = None
    reason: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().lower() or None
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip() or None
    if verdict not in {"correct", "incorrect", "unresolved"} or reason is None:
        return JudgeVerdict(verdict="unresolved", reason="malformed_judge_output", malformed=True)
    return JudgeVerdict(verdict=verdict, reason=reason, malformed=False)
