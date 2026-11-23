"""SD-Search Appendix J: assistant-only query skeletons and exact causal targets."""

import re
from dataclasses import dataclass


@dataclass
class Query:
    insertion: int
    positions: list
    text: str


def queries(codec, ids, generated):
    """Parse each generated assistant run; tool text can never become hindsight."""
    result = []
    start = 0
    while start < len(ids):
        if not generated[start] or ids[start] in codec.special_ids:
            start += 1
            continue
        end = start
        while end < len(ids) and generated[end] and ids[end] not in codec.special_ids:
            end += 1
        pieces = [codec.token_bytes(t) for t in ids[start:end]]
        raw = b"".join(pieces)
        boundaries = [0]
        for x in pieces:
            boundaries.append(boundaries[-1] + len(x))
        # Ignore tags in reasoning; require a complete unambiguous action.
        from .protocol import parse_action

        if parse_action(raw.decode("utf-8", errors="replace"))[0] == "search":
            action = list(re.finditer(rb"<search>(.*?)</search>", raw, re.S))[-1]
            positions = [
                start + i
                for i in range(len(pieces))
                if action.start(1) <= boundaries[i] < boundaries[i + 1] <= action.end(1)
            ]
            insertion = next(
                start + i for i in range(len(pieces)) if boundaries[i] <= action.start() < boundaries[i + 1]
            )
            if positions:
                result.append(Query(insertion, positions, action[1].decode("utf-8", errors="replace")))
        start = end
    return result


def hindsight(group, outcomes, focal, step):
    label = lambda x: "Correct" if x > 0 else "Incorrect"
    lines = ["\n[Trajectory Hindsight]:\n"]
    seen = set()
    for j, spans in enumerate(group):
        if j == focal:
            continue
        text = " -> ".join(f"<search>{q.text}</search>" for q in spans)
        key = (text, label(outcomes[j]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[Sibling Rollout]: {text} [Outcome]: {key[1]}\n")
    lines.append(" -> ".join(f"<search>{q.text}</search>" for q in group[focal][step:]) + "\n")
    lines.append(f"[Outcome]: {label(outcomes[focal])}\n")
    return "".join(lines)


def build_steps(codec, prompt, response, group, outcomes, focal, hindsight_budget=1024, max_context=8192):
    rows = []
    for step, q in enumerate(group[focal]):
        h = codec.tokenizer.encode(hindsight(group, outcomes, focal, step), add_special_tokens=False)
        truncated = max(0, len(h) - hindsight_budget)
        h = h[-hindsight_budget:]
        stop = q.positions[-1] + 1
        pivot = len(prompt) + q.insertion
        student = prompt + response[:stop]
        teacher = student[:pivot] + h + student[pivot:]
        if len(teacher) > max_context:
            raise ValueError("SD teacher context exceeds limit; refusing target truncation")
        rows.append(
            {
                "student_ids": student,
                "teacher_ids": teacher,
                "student_positions": [len(prompt) + p for p in q.positions],
                "teacher_positions": [len(prompt) + p + len(h) for p in q.positions],
                "hindsight_truncated_tokens": truncated,
            }
        )
    return rows
