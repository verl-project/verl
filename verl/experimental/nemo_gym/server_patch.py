# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

logger = logging.getLogger(__name__)

def _replace_prefix_tokens(model_prefix, template_prefix, template_ids, tok):
    eos = tok.eos_token_id
    if eos is None or not model_prefix:
        return template_ids
    eos_set = set(eos) if isinstance(eos, list) else {eos}
    cut_model = len(model_prefix)
    if model_prefix[-1] in eos_set:
        cut_model -= 1
    if len(template_ids) <= len(template_prefix):
        return template_ids
    cut = -1
    for pos in reversed(range(len(template_prefix))):
        if template_ids[pos] in eos_set:
            cut = pos
            break
    if cut < 0:
        return template_ids
    return model_prefix[:cut_model] + template_ids[cut:]


def patch_serving_chat_for_nemo_gym() -> None:
    _serving_chat_cls = None
    for _mod in (
        "vllm.entrypoints.openai.chat_completion.serving",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.entrypoints.openai.api_server",
        "vllm.entrypoints.openai.serving_chat",
    ):
        try:
            import importlib
            m = importlib.import_module(_mod)
            if hasattr(m, "OpenAIServingChat"):
                _serving_chat_cls = m.OpenAIServingChat
                break
        except ImportError:
            continue

    if _serving_chat_cls is None:
        logger.warning("[nemo-gym] could not find OpenAIServingChat; skipping retokenization patch.")
        return

    OpenAIServingChat = _serving_chat_cls
    _original_preprocess_chat = OpenAIServingChat._preprocess_chat

    async def _patched_preprocess_chat(
        self, request, messages,
        default_template, default_template_content_format, default_template_kwargs,
        tool_dicts=None, tool_parser=None,
    ):
        required_prefix = getattr(request, "required_prefix_token_ids", None)
        if required_prefix is None:
            for msg in reversed(messages):
                if isinstance(msg, dict) and "prompt_token_ids" in msg:
                    required_prefix = list(msg["prompt_token_ids"]) + list(msg["generation_token_ids"])
                    break
                elif not isinstance(msg, dict) and getattr(msg, "prompt_token_ids", None):
                    required_prefix = list(msg.prompt_token_ids) + list(msg.generation_token_ids)
                    break

        res = await _original_preprocess_chat(
            self, request, messages,
            default_template, default_template_content_format, default_template_kwargs,
            tool_dicts=tool_dicts, tool_parser=tool_parser,
        )

        if required_prefix is None:
            return res

        try:
            tok = self.renderer.get_tokenizer() # avoid concurrent tokenizer access - else already borrowed error w/ hermes tool parser
            engine_prompt = res[1][0]
            engine_prompt["prompt_token_ids"] = _replace_prefix_tokens(
                required_prefix, required_prefix,
                engine_prompt["prompt_token_ids"], tok,
            )
        except Exception as e:
            logger.warning(f"[nemo-gym] retokenization patch failed, skipping: {e}")
        return res

    OpenAIServingChat._preprocess_chat = _patched_preprocess_chat
    logger.info("[nemo-gym] applied retokenization patch to OpenAIServingChat.")
