from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


ReasoningBlockKind = Literal["text", "reasoning"]
ReasoningBlock = tuple[ReasoningBlockKind, str]


@dataclass(frozen=True)
class ReasoningParseResult:
    blocks: tuple[ReasoningBlock, ...]
    response_text: str


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
        blocks: list[ReasoningBlock] = []
        cursor = 0

        while True:
            start = text.find(self.start_token, cursor)
            if start == -1:
                self._append_block(blocks, "text", text[cursor:])
                break

            self._append_block(blocks, "text", text[cursor:start])
            reasoning_start = start + len(self.start_token)
            end = text.find(self.end_token, reasoning_start)
            if end == -1:
                self._append_block(blocks, "reasoning", text[reasoning_start:])
                break

            self._append_block(blocks, "reasoning", text[reasoning_start:end])
            cursor = end + len(self.end_token)

        response_text = next(
            (block_text for kind, block_text in reversed(blocks) if kind == "text"),
            "",
        )
        return ReasoningParseResult(
            blocks=tuple(blocks),
            response_text=response_text,
        )

    @staticmethod
    def _append_block(
        blocks: list[ReasoningBlock], kind: ReasoningBlockKind, text: str
    ) -> None:
        if text:
            blocks.append((kind, text))


__all__ = [
    "Apertus2509ReasoningParser",
    "ReasoningBlock",
    "ReasoningBlockKind",
    "ReasoningParseResult",
    "ReasoningParser",
]
