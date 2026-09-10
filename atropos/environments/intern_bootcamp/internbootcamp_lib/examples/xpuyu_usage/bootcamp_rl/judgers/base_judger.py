# Copyright (c) InternLM. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

T = TypeVar("T")
MessageItem = TypedDict("MessageItem", {"role": str, "content": str})
Reward = Union[float, List[float], None]
MetaData = TypedDict("MetaData", {"data_source": str})


@dataclass
class JudgeStatus(Generic[T]):
    ok: bool = True
    reason: Optional[str] = None
    handle: Optional[T] = None


class BaseJudger(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def on_data_received(
        self,
        prompt_messages: List[MessageItem],
        completion_messages: List[MessageItem],
        metadata: dict,
    ) -> JudgeStatus:
        raise NotImplementedError()

    @abstractmethod
    def on_reward_required(
        self,
        status: JudgeStatus,
        timeout: Optional[float] = None,
    ) -> Reward:
        raise NotImplementedError()


registered_judgers: Dict[str, Type[BaseJudger]] = {}


def register_judger(name: str):
    global registered_judgers

    def wrapper(cls):
        assert name not in registered_judgers, f"{name} already in {registered_judgers}"
        registered_judgers[name] = cls
        return cls

    return wrapper
