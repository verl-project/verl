import asyncio
import time
from types import SimpleNamespace

import pytest

from examples.search_r1.prompt_cache import TokenizerRuntime


class FakeTokenizer:
    chat_template = "template"

    def convert_tokens_to_ids(self, token):
        return 17

    def convert_ids_to_tokens(self, token_id):
        return "<|im_end|>"

    def encode(self, text, **kwargs):
        return [17]


def test_prompt_cache_singleflight_cancellation_isolation_and_keys():
    async def check():
        tokenizer = FakeTokenizer()
        state = TokenizerRuntime(tokenizer, capacity=2)
        calls = []

        def build(messages):
            calls.append(messages)
            time.sleep(.02)
            return [1, 2, 3]

        builder = SimpleNamespace(chat_template_kwargs={}, build_initial_tokens=build)
        messages = [{"role": "user", "content": "question"}]
        first = asyncio.create_task(state.initial_tokens(tokenizer, builder, messages))
        second = asyncio.create_task(state.initial_tokens(tokenizer, builder, messages))
        await asyncio.sleep(.005)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        result = await second
        result.append(99)
        assert await state.initial_tokens(tokenizer, builder, messages) == [1, 2, 3]
        assert len(calls) == 1 and not state.pending
        builder.chat_template_kwargs = {"different": True}
        await state.initial_tokens(tokenizer, builder, messages)
        tokenizer.chat_template = "changed"
        await state.initial_tokens(tokenizer, builder, messages)
        assert len(calls) == 3 and len(state.prompts) == 2

    asyncio.run(check())


def test_prompt_build_error_is_not_cached():
    async def check():
        state, tokenizer = TokenizerRuntime(FakeTokenizer()), FakeTokenizer()
        count = 0

        def build(messages):
            nonlocal count
            count += 1
            if count == 1:
                raise ValueError("build failed")
            return [1]

        builder = SimpleNamespace(chat_template_kwargs={}, build_initial_tokens=build)
        with pytest.raises(ValueError, match="build failed"):
            await state.initial_tokens(tokenizer, builder, [])
        assert await state.initial_tokens(tokenizer, builder, []) == [1]
        assert count == 2 and not state.pending

    asyncio.run(check())
