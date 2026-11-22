"""Bounded, worker-local caches for the shared text tokenizer."""

import asyncio
import json
from collections import OrderedDict

from omegaconf import OmegaConf


class TokenizerRuntime:
    def __init__(self, tokenizer, capacity=512):
        token = "<|im_end|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            not isinstance(token_id, int)
            or token_id < 0
            or tokenizer.convert_ids_to_tokens(token_id) != token
            or tokenizer.encode(token, add_special_tokens=False) != [token_id]
        ):
            raise ValueError("Search-R1 recipe requires a Qwen ChatML tokenizer")
        self.turn_end = (token_id,)
        self.capacity = capacity
        self.prompts = OrderedDict()
        self.pending = {}
        self.builds = 0

    async def initial_tokens(self, tokenizer, builder, messages):
        kwargs = builder.chat_template_kwargs
        if OmegaConf.is_config(kwargs):
            kwargs = OmegaConf.to_container(kwargs, resolve=True)
        key = (type(builder), json.dumps(
            {"messages": messages, "kwargs": kwargs, "template": tokenizer.chat_template},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ))
        if key in self.prompts:
            self.prompts.move_to_end(key)
            return list(self.prompts[key])
        loop = asyncio.get_running_loop()
        pending_key = (loop, key)
        task = self.pending.get(pending_key)
        if task is None:
            async def build():
                self.builds += 1
                ids = await loop.run_in_executor(None, lambda: builder.build_initial_tokens(messages))
                return tuple(ids)

            task = loop.create_task(build())
            self.pending[pending_key] = task

            def finish(done):
                self.pending.pop(pending_key, None)
                if not done.cancelled() and done.exception() is None:
                    self.prompts[key] = done.result()
                    self.prompts.move_to_end(key)
                    while len(self.prompts) > self.capacity:
                        self.prompts.popitem(last=False)

            task.add_done_callback(finish)
        # One cancelled trajectory must not cancel a build shared by its siblings.
        return list(await asyncio.shield(task))


def tokenizer_runtime(tokenizer):
    state = getattr(tokenizer, "_search_r1_runtime_cache", None)
    if state is None:
        state = TokenizerRuntime(tokenizer)
        tokenizer._search_r1_runtime_cache = state
    return state
