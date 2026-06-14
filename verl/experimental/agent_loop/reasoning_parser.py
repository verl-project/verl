from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningParseResult:
    response_text: str
    reasoning_text: str


class ReasoningParser(ABC):
    _registry: dict[str, type[ReasoningParser]] = {}

    @abstractmethod
    def parse(self, text: str) -> ReasoningParseResult:
        raise NotImplementedError

    @classmethod
    def get_reasoning_parser(cls, name: str) -> ReasoningParser:
        if name not in cls._registry:
            raise ValueError(f"Unknown reasoning parser: {name}")
        return cls._registry[name]()

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[ReasoningParser]) -> type[ReasoningParser]:
            cls._registry[name] = subclass
            return subclass

        return decorator


@ReasoningParser.register("apertus2509")
class Apertus2509ReasoningParser(ReasoningParser):
    start_token = "<|inner_prefix|>"
    end_token = "<|inner_suffix|>"

    def parse(self, text: str) -> ReasoningParseResult:
        response_parts: list[str] = []
        reasoning_parts: list[str] = []
        cursor = 0

        while True:
            start = text.find(self.start_token, cursor)
            if start == -1:
                response_parts.append(text[cursor:])
                break

            response_parts.append(text[cursor:start])
            reasoning_start = start + len(self.start_token)
            end = text.find(self.end_token, reasoning_start)
            if end == -1:
                reasoning_parts.append(text[reasoning_start:])
                break

            reasoning_parts.append(text[reasoning_start:end])
            cursor = end + len(self.end_token)

        return ReasoningParseResult(
            response_text="".join(response_parts),
            reasoning_text="".join(reasoning_parts),
        )


__all__ = [
    "Apertus2509ReasoningParser",
    "ReasoningParseResult",
    "ReasoningParser",
]
