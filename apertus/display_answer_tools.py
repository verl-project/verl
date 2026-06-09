from __future__ import annotations

from verl.tools.function_tool import function_tool


DISPLAY_ANSWERS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "display_answers",
        "description": "Display the answers to the user",
        "parameters": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The answers to the user",
                },
            },
            "required": ["answers"],
        },
    },
}


@function_tool(schema=DISPLAY_ANSWERS_SCHEMA)
def display_answers(answers: list[str]) -> str:
    """Display the answers to the user.

    Args:
        answers: The answers to the user.
    """
    return "Answers displayed"
