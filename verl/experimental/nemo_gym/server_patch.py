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
    # matches nemo-rl's implementation
    if not model_prefix:
        return template_ids
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer must have eos_token_id"
    cut_model = len(model_prefix)
    if model_prefix[-1] == eos:
        cut_model -= 1
    assert len(template_ids) > len(template_prefix), (
        f"non-monotonically increasing trajectory: "
        f"template_ids={len(template_ids)} template_prefix={len(template_prefix)}"
    )
    cut = -1
    for pos in reversed(range(len(template_prefix))):
        if template_ids[pos] == eos:
            cut = pos
            break
    assert cut >= 0, "no EOS token found in chat-templated messages"
    return model_prefix[:cut_model] + template_ids[cut:]


def _make_patched_preprocess_chat(original):
    async def _patched(
        self,
        request,
        messages,
        default_template,
        default_template_content_format,
        default_template_kwargs,
        tool_dicts=None,
        tool_parser=None,
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

        res = await original(
            self,
            request,
            messages,
            default_template,
            default_template_content_format,
            default_template_kwargs,
            tool_dicts=tool_dicts,
            tool_parser=tool_parser,
        )

        if required_prefix is None:
            return res

        try:
            # call _preprocess_chat on messages up to last assistant turn (no gen prompt)
            # to get template_prefix_ids for _replace_prefix_tokens
            last_asst = next(
                (
                    i
                    for i in reversed(range(len(messages)))
                    if (
                        messages[i].get("role") if isinstance(messages[i], dict) else getattr(messages[i], "role", None)
                    )
                    == "assistant"
                ),
                None,
            )
            prefix_msgs = messages[: last_asst + 1] if last_asst is not None else messages
            prefix_res = await original(
                self,
                request,
                prefix_msgs,
                default_template,
                default_template_content_format,
                {**(default_template_kwargs or {}), "add_generation_prompt": False},
                tool_dicts=tool_dicts,
                tool_parser=tool_parser,
            )
            template_prefix_ids = prefix_res[1][0]["prompt_token_ids"]

            tok = self.renderer.get_tokenizer()
            engine_prompt = res[1][0]
            engine_prompt["prompt_token_ids"] = _replace_prefix_tokens(
                required_prefix,
                template_prefix_ids,
                engine_prompt["prompt_token_ids"],
                tok,
            )
        except Exception as e:
            logger.warning(f"[nemo-gym] retokenization patch failed, skipping: {e}")
        return res

    return _patched


def patch_serving_chat_for_nemo_gym() -> None:
    import importlib

    targets = {
        "OpenAIServingChat": (
            "vllm.entrypoints.openai.chat_completion.serving",
            "vllm.entrypoints.openai.chat_completion",
            "vllm.entrypoints.openai.api_server",
            "vllm.entrypoints.openai.serving_chat",
        ),
        "OpenAIServingTokenization": (
            "vllm.entrypoints.openai.api_server",
            "vllm.entrypoints.serve.tokenize.serving",
        ),
    }

    patched_any = False
    for cls_name, mods in targets.items():
        cls = None
        for mod in mods:
            try:
                m = importlib.import_module(mod)
                if hasattr(m, cls_name):
                    cls = getattr(m, cls_name)
                    break
            except ImportError:
                continue
        if cls is None:
            logger.warning(f"[nemo-gym] could not find {cls_name}; skipping.")
            continue
        cls._preprocess_chat = _make_patched_preprocess_chat(cls._preprocess_chat)
        logger.warning(f"[nemo-gym] applied retokenization patch to {cls_name}.")
        patched_any = True

    if not patched_any:
        logger.warning("[nemo-gym] retokenization patch not applied to any serving class.")
