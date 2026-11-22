"""Text protocol and outcome scoring, independent of the training runtime."""

import re
import string

SYSTEM_PROMPT = (
    "Answer the user's question. Reason inside <think> and </think>. "
    "When information is missing, emit <search>your query</search>. "
    "The search result will arrive as a separate tool message inside <information> tags. "
    "Use the conversation history to continue reasoning and searching as necessary. "
    "When ready, emit a concise final answer inside <answer> and </answer>. "
    "Emit at most one search or final answer per assistant turn."
)


def parse_action(text):
    """Only accept one complete, nonempty action outside reasoning blocks."""
    outside_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # An unfinished reasoning block cannot contain an executable action.
    if "<think>" in outside_think or "</think>" in outside_think:
        return "invalid", ""
    matches = list(re.finditer(r"<(search|answer)>(.*?)</\1>", outside_think, re.DOTALL))
    if len(matches) != 1:
        return "invalid", ""
    match = matches[0]
    value = match.group(2).strip()
    remainder = outside_think[: match.start()] + outside_think[match.end() :]
    if not value or re.search(r"</?(?:search|answer)>", remainder + value):
        return "invalid", ""
    return match.group(1), value


def normalize_answer(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_final_turn(final_assistant, ground_truth, termination_reason):
    """Never inspect retrieved text or earlier assistant turns for the answer."""
    kind, answer = parse_action(final_assistant)
    if termination_reason != "answer" or kind != "answer":
        return {"score": 0.0, "acc": 0.0, "answer_valid": 0.0}
    targets = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    if isinstance(targets, str):
        targets = [targets]
    acc = float(any(normalize_answer(answer) == normalize_answer(target) for target in targets))
    return {"score": acc, "acc": acc, "answer_valid": 1.0}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Reward-manager fallback; require provenance supplied by our AgentLoop."""
    del data_source, solution_str, kwargs
    info = extra_info or {}
    if "search_r1_final_assistant" not in info:
        raise ValueError("Search-R1 scoring requires final-assistant provenance from SearchR1AgentLoop")
    return score_final_turn(info["search_r1_final_assistant"], ground_truth, info.get("search_r1_termination", ""))
