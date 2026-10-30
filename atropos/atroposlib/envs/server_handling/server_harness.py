import asyncio
from typing import Dict, List, Literal, Union

from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import Completion, CompletionChoice


def create_chat_completion(
    resp: Union[str, List[str]],
    n: int = 1,
    finish_reason: Union[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"],
        List[
            Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        ],
    ] = "stop",
) -> ChatCompletion:
    """
    Simple helper for creating a ChatCompletion object, if you need it
    :param resp:
    :param n:
    :param finish_reason:
    :return:
    """
    choices = [
        Choice(
            finish_reason=(
                finish_reason if isinstance(finish_reason, str) else finish_reason[i]
            ),
            index=i,
            message=ChatCompletionMessage(
                content=resp if isinstance(resp, str) else resp[i],
                role="assistant",
            ),
        )
        for i in range(n)
    ]
    return ChatCompletion(
        id="test_id",
        created=0,
        model="test_model",
        object="chat.completion",
        choices=choices,
    )


def create_completion(
    resp: Union[str, List[str]],
    n: int = 1,
    finish_reason: Union[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"],
        List[
            Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        ],
    ] = "stop",
) -> Completion:
    """
    Simple helper for creating a Completion object, if you need it
    :param resp:
    :param n:
    :param finish_reason:
    :return:
    """
    choices = [
        CompletionChoice(
            finish_reason=(
                finish_reason if isinstance(finish_reason, str) else finish_reason[i]
            ),
            index=i,
            text=resp if isinstance(resp, str) else resp[i],
        )
        for i in range(n)
    ]
    return Completion(
        id="test_id",
        created=0,
        model="test_model",
        object="text_completion",
        choices=choices,
    )


class ServerHarness:
    def __init__(self):
        self.response_map = dict()
        self.sem = asyncio.Semaphore(1)
        self.eval_sem = asyncio.Semaphore(1)
        pass

    def conv_to_dictkey(self, input_message: List[Dict[str, str]]) -> str:
        dictkey = list()
        for item in input_message:
            dictkey.append(f"role:{item['role']}")
            dictkey.append(f"content:{item['content']}")
        return "\n".join(dictkey)

    async def update_weight(self, weight):
        pass

    def set_desired_response(
        self, input_message: List[Dict[str, str]], desired_response: ChatCompletion
    ):
        dictkey = self.conv_to_dictkey(input_message)
        self.response_map[dictkey] = desired_response

    def set_desired_completion(self, input_message: str, completion: Completion):
        self.response_map[input_message] = completion

    async def chat_completion(self, *args, **kwargs) -> ChatCompletion:
        messages = kwargs.get("messages")
        dictkey = self.conv_to_dictkey(messages)
        try:
            return self.response_map.get(dictkey)
        except KeyError as e:
            raise KeyError(f"KeyError: {e} for key:\n{dictkey}")

    async def completion(self, *args, **kwargs) -> Completion:
        prompt = kwargs.get("prompt")
        try:
            return self.response_map.get(prompt)
        except KeyError as e:
            raise KeyError(f"KeyError: {e} for key:\n{prompt}")


if __name__ == "__main__":

    async def main():
        test_compl = create_chat_completion("hello")
        harness = ServerHarness()
        harness.set_desired_response([{"role": "user", "content": "hi"}], test_compl)
        print(harness.response_map)
        print(harness.conv_to_dictkey([{"role": "user", "content": "hi"}]))
        print(
            await harness.chat_completion(messages=[{"role": "user", "content": "hi"}])
        )
        # now, let's test the completion
        test_completion = create_completion("\nhello")
        harness.set_desired_completion("hi", test_completion)
        print(harness.response_map)
        print(await harness.completion(prompt="hi"))

    asyncio.run(main())
