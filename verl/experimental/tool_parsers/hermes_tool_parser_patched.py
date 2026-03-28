# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import re

try:
    from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
except ImportError:
    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

try:
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
except ImportError:
    from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


@ToolParserManager.register_module("hermes_patched")
class HermesPatchedToolParser(Hermes2ProToolParser):
    def __init__(self, tokenizer):
        ToolParser.__init__(self, tokenizer)

        try:
            from vllm.transformers_utils.tokenizer import MistralTokenizer

            if isinstance(self.model_tokenizer, MistralTokenizer):
                self.model_tokenizer = self.model_tokenizer.tokenizer
        except ImportError:
            pass

        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
        self.scratch_pad_regex = re.compile(r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL)

        vocab = self.model_tokenizer.get_vocab()
        start_id = vocab.get(self.tool_call_start_token)
        end_id = vocab.get(self.tool_call_end_token)

        self.tool_call_start_token_ids = [start_id] if start_id is not None else []
        self.tool_call_end_token_ids = [end_id] if end_id is not None else []
        self.tool_call_start_token_array = [self.tool_call_start_token]
        self.tool_call_end_token_array = [self.tool_call_end_token]
        self.buffered_delta_text = ""
