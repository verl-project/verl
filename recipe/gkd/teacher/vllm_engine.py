# Copyright 2025 Individual Contributor: furunding
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

import argparse
import random
from typing import NamedTuple

import torch
from codetiming import Timer
from transformers import AutoConfig

# patch FIRST (before any vllm import)
try:
    from . import patch_vllm_logprobs
except ImportError:
    import patch_vllm_logprobs

patch_vllm_logprobs.apply_patch()

from vllm import LLM, SamplingParams

# from vllm.v1.outputs import LogprobsTensors
from vllm.v1.engine.logprobs import LogprobsProcessor


def _update_prompt_logprobs(
    self,
    prompt_logprobs_tensors,
) -> None:
    """Update with prompt logprobs from EngineCore.

    Args:
        prompt_logprobs_tensors: tuple containing the prompt logprobs
                                tensors.

    """

    # Prompt logprobs are enabled.
    assert self.num_prompt_logprobs is not None
    assert self.prompt_logprobs is not None

    self.prompt_logprobs.append(prompt_logprobs_tensors)


def _update_sample_logprobs(self, logprobs_lists) -> None:
    """Update with sample logprobs from EngineCore.

    Outer lists are only of len > 1 if EngineCore made
    >1 tokens in prior step (e.g. in spec decoding).

    Args:
        logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

    """

    assert self.num_logprobs is not None
    assert self.logprobs is not None
    assert self.cumulative_logprob is not None

    # token_ids_lst, logprobs_lst, ranks_lst = logprobs_lists

    # for rank, logprobs, token_ids in zip(ranks_lst, logprobs_lst,
    #                                         token_ids_lst):

    #     # Detokenize (non-incrementally).
    #     decoded_tokens = NONES if self.tokenizer is None else (
    #         convert_ids_list_to_tokens(self.tokenizer, token_ids))

    #     # Sampler puts the sampled logprob in first.
    #     sampled_token_logprob = logprobs[0]
    #     self.cumulative_logprob += sampled_token_logprob

    #     # Update with the Logprob dictionary for this pos.
    #     self.logprobs.append(
    #         self._make_logprob_dict(
    #             logprobs,
    #             token_ids,
    #             decoded_tokens,
    #             rank,
    #             self.num_logprobs,
    #         ))
    self.logprobs.append(logprobs_lists)


LogprobsProcessor._update_prompt_logprobs = _update_prompt_logprobs
LogprobsProcessor._update_sample_logprobs = _update_sample_logprobs


class LogprobsTensors(NamedTuple):
    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs]
    selected_token_ranks: torch.Tensor

    def tolists(self):
        return LogprobsTensors(
            logprob_token_ids=self.logprob_token_ids.cpu(),
            logprobs=self.logprobs.cpu(),
            selected_token_ranks=self.selected_token_ranks.cpu(),
        )

    @staticmethod
    def empty_cpu(num_positions: int, num_tokens_per_position: int) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty((num_positions, num_tokens_per_position), dtype=torch.int32, device="cpu")
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(num_positions, dtype=torch.int32, device="cpu")
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )

    def slice(self, start: int, end: int):
        return LogprobsTensors(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.selected_token_ranks[start:end],
        )


# outputs.LogprobsTensors = LogprobsTensors
# def tolists(self):
#     return self


# LogprobsTensors.tolists = tolists
# setattr(LogprobsTensors, "slice", slice)


class VLLMEngine:
    def __init__(self, ckpt_path, n_logprobs=0, tp_size=1, 
                 enable_prefix_caching=False,
                 gpu_memory_utilization=0.85,
                 max_num_batched_tokens=None):
        self.n_logprobs = n_logprobs
        # self.llm = LLM(ckpt_path, tensor_parallel_size=tp_size, trust_remote_code=True,
        #                enable_chunked_prefill=False, distributed_executor_backend="ray",
        #                max_logprobs=n_logprobs, gpu_memory_utilization=0.7)

        llm_kwargs = {
            "tensor_parallel_size": tp_size,
            "trust_remote_code": True,
            "enable_chunked_prefill": False,
            "max_logprobs": n_logprobs,
            "enable_prefix_caching": enable_prefix_caching,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
            llm_kwargs["max_model_len"] = max_num_batched_tokens
        
        self.llm = LLM(ckpt_path, **llm_kwargs)

    def get_topk_logprobs(self, prompt_token_ids, temperature=0.8, max_new_tokens=1, only_response=False):
        def make_sampling_params(i=None):
            return SamplingParams(
                temperature=temperature,
                top_p=0.95,
                detokenize=False,
                logprobs=self.n_logprobs,
                prompt_logprobs=None if only_response else self.n_logprobs,
                max_tokens=max_new_tokens[i] if (i is not None) else max_new_tokens,
            )

        if isinstance(max_new_tokens, list):
            assert len(prompt_token_ids) == len(max_new_tokens)
            sampling_params = [make_sampling_params(i) for i in range(len(max_new_tokens))]
        else:
            sampling_params = make_sampling_params()

        # NOTE: vllm version problem, not supporting prompt_token_ids anymore..
        # outputs = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        # vLLM API compatibility: newer versions don't accept `prompt_token_ids=` kw.
        # Use TokensPrompt-style dicts via `prompts=` instead.
        try:
            outputs = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        except TypeError:
            # Normalize to list[list[int]]
            if isinstance(prompt_token_ids, (list, tuple)) and len(prompt_token_ids) > 0 and isinstance(prompt_token_ids[0], int):
                ptids_list = [list(prompt_token_ids)]
            else:
                ptids_list = [list(x) for x in prompt_token_ids]

            prompts = [{"prompt_token_ids": x} for x in ptids_list]
            try:
                outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
            except TypeError:
                # Some versions use positional `prompts`
                outputs = self.llm.generate(prompts, sampling_params=sampling_params)


        responses, teacher_topk_logprobs, teacher_topk_indices = [], [], []
        for output in outputs:
            responses.append(torch.tensor(output.outputs[0].token_ids, dtype=torch.int32))
            if self.n_logprobs > 0:
                response_topk_logprobs = torch.tensor(
                    [x.logprobs[0] for x in output.outputs[0].logprobs],
                    dtype=torch.float32,
                )[:, 1:]
                response_topk_indices = torch.tensor(
                    [x.logprob_token_ids[0] for x in output.outputs[0].logprobs],
                    dtype=torch.int32,
                )[:, 1:]
                if only_response:
                    teacher_topk_logprobs.append(response_topk_logprobs)
                    teacher_topk_indices.append(response_topk_indices)
                else:
                    prompt_topk_logprobs = output.prompt_logprobs[1].logprobs[:, 1:].to(torch.float32)
                    prompt_topk_indices = output.prompt_logprobs[1].logprob_token_ids[:, 1:].to(torch.int32)
                    teacher_topk_logprobs.append(torch.vstack([prompt_topk_logprobs, response_topk_logprobs]))
                    teacher_topk_indices.append(torch.vstack([prompt_topk_indices, response_topk_indices]))

        return responses, teacher_topk_logprobs, teacher_topk_indices

    # def get_response_and_topk_logprobs(self, prompt_token_ids, max_tokens=64):
    #     sampling_params = SamplingParams(temperature=0.8, top_p=0.95, detokenize=False,
    #                                      logprobs=self.n_logprobs, max_tokens=max_tokens)

    #     outputs = self.llm.generate(prompt_token_ids=prompt_token_ids,
    #                                 sampling_params=sampling_params)

    #     student_topk_logprobs, student_topk_indices = [], []
    #     for output in outputs:
    #         student_topk_logprobs.append([])
    #         student_topk_indices.append([])
    #         for logprob_list in output.outputs[0].logprobs:
    #             student_topk_logprobs[-1].extend(logprob_list.logprobs)
    #             student_topk_indices[-1].extend(logprob_list.logprob_token_ids)

    #     return student_topk_logprobs, student_topk_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vLLM logprob")
    parser.add_argument("model_dir", help="Model directory")
    parser.add_argument("--tp-size", type=int, default=1, help="TP size")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Test batch size")
    parser.add_argument("--seq-len", "-s", type=int, default=3840, help="Test sequence length")
    parser.add_argument("--token-file", "-t", type=str, help="Input token file")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_dir)
    print(f"Reading configs from {args.model_dir}: {config.vocab_size=}")

    prompt_token_ids = []
    if args.token_file:
        # Init input with tokenid file
        from get_batch import get_batch

        prompt_token_ids = get_batch()
    else:
        # Init input randomly
        prompt_lens = args.batch_size * [args.seq_len]
        for pl in prompt_lens:
            prompt_token_ids.append([random.randint(1, config.vocab_size - 1000) for j in range(pl)])

    engine = VLLMEngine(ckpt_path=args.model_dir, n_logprobs=256, tp_size=args.tp_size)

    with Timer(name="get_topk_logprobs", initial_text=True):
        responses, teacher_topk_logprobs, teacher_topk_indices = engine.get_topk_logprobs(
            prompt_token_ids, temperature=0.7, max_new_tokens=1, only_response=True
        )
    # debug
    import ipdb

    ipdb.set_trace()
