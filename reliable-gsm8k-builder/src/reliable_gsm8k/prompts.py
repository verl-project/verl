from __future__ import annotations


def build_correct_solution_prompt(question: str, *, compact: bool = False) -> str:
    if compact:
        return (
            "Give the correct final numeric answer.\n"
            "Do not explain.\n"
            "Return exactly one line:\n"
            "FINAL_ANSWER: <numeric answer>\n\n"
            f"Question:\n{question.strip()}"
        )
    return (
        "Solve the GSM8K math problem and output only the final numeric answer.\n"
        "Do not explain.\n"
        "Return exactly this format:\n"
        "FINAL_ANSWER: <final numeric answer>\n"
        "Use Arabic numerals in FINAL_ANSWER.\n"
        "Do not output anything after the FINAL_ANSWER line.\n\n"
        f"Question:\n{question.strip()}"
    )


def build_incorrect_solution_prompt(
    question: str,
    gold_answer: str,
    *,
    banned_final_answers: list[str],
    mistake_hint: str | None = None,
    compact: bool = False,
) -> str:
    banned_block = ""
    if banned_final_answers:
        banned_block = (
            "Do not use any of these wrong final answers again:\n"
            + "\n".join(f"- {answer}" for answer in banned_final_answers)
            + "\n"
        )
    if compact:
        banned_hint = ""
        if banned_final_answers:
            banned_hint = "Avoid these wrong final answers: " + ", ".join(banned_final_answers) + "\n"
        mistake_line = f"Use this kind of mistake: {mistake_hint}\n" if mistake_hint else ""
        return (
            "Give one believable but wrong final numeric answer.\n"
            "The value should look like the result of a realistic math mistake.\n"
            f"Correct final answer: {gold_answer.strip()}\n"
            f"{mistake_line}"
            f"{banned_hint}"
            "Do not explain.\n"
            "Return exactly one line:\n"
            "FINAL_ANSWER: <wrong numeric answer>\n\n"
            f"Question:\n{question.strip()}"
        )
    mistake_block = f"Target this kind of mistake: {mistake_hint}\n" if mistake_hint else ""
    return (
        "Write one plausible but mathematically incorrect final numeric answer for the GSM8K problem.\n"
        "The answer should look like the result of a real student mistake.\n"
        "Do not explain.\n"
        "Return exactly this format:\n"
        "FINAL_ANSWER: <final numeric answer>\n"
        "Use Arabic numerals in FINAL_ANSWER.\n"
        "The FINAL_ANSWER must not equal the correct answer.\n"
        "Do not output anything after the FINAL_ANSWER line.\n"
        f"{mistake_block}"
        f"{banned_block}"
        f"Correct final answer for reference only:\n{gold_answer.strip()}\n\n"
        f"Question:\n{question.strip()}"
    )


def build_judge_prompt(question: str, gold_answer: str, candidate_completion: str) -> str:
    return (
        "You are checking whether a candidate GSM8K solution is mathematically correct.\n"
        "Return exactly:\n"
        "VERDICT: correct|incorrect|unresolved\n"
        "REASON: <short explanation>\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Correct final answer:\n{gold_answer.strip()}\n\n"
        f"Candidate solution:\n{candidate_completion.strip()}"
    )


def build_oe_prompt(question: str) -> str:
    return f"Question:\n{question.strip()}\nSolution:"


def _format_option_block(options: dict[str, str]) -> str:
    blocks: list[str] = []
    for label in sorted(options):
        blocks.append(f"{label}. {options[label]}")
    return "\n".join(blocks)


def build_mc_onecorrect_prompt(question: str, options: dict[str, str]) -> str:
    return (
        f"Question:\n{question.strip()}\n\n"
        "Options:\n"
        f"{_format_option_block(options)}\n\n"
        'Return exactly one line: #### <letter>\n'
        "Choose the letter of the only correct numeric answer."
    )


def build_mc_allwrong_prompt(question: str, options: dict[str, str]) -> str:
    return (
        f"Question:\n{question.strip()}\n\n"
        "Options:\n"
        f"{_format_option_block(options)}\n\n"
        'Return exactly one line: #### <letter-or-NONE>\n'
        "Output NONE if every numeric option is incorrect."
    )
