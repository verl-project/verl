# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Megatron Reward Model.
"""

import itertools

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from tensordict import TensorDict

from verl import DataProto
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor, pad_sequence_to_length
from verl.workers.reward_model import BasePPORewardModel


class MegatronRewardModel(BasePPORewardModel):
    def __init__(
        self,
        config,
        model_config,
        reward_model_module: torch.nn.ModuleList,
        hf_config,
        tf_config,
        sft_tokenizer=None,
        rm_tokenizer=None,
    ):
        self.config = config
        self.reward_model_module = reward_model_module
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.model_config = model_config
        self.device = "cuda"
        self.sft_tokenizer = sft_tokenizer
        self.rm_tokenizer = rm_tokenizer
        self.use_different_tokenizer = rm_tokenizer is not None

        print(f"MegatronRewardModel.config: {self.config}")

        if self.config.megatron.param_offload:
            self.offload_params_to_cpu()

    def re_encode_by_rm_tokenizer(self, data: DataProto) -> DataProto:
        assert self.use_different_tokenizer, "re-encode need rm tokenizer not be None!"
        # need to use rm tokenizer to re-generate input_ids, attention_mask and position_ids
        # 1. remove pad for each sequence
        # 2. decode by sft_tokenizer, remove sft system prompts
        # 3. encode by rm_tokenizer with rm system prompts, get rm_input_ids
        # 4. generate attention_mask and position_ids
        input_ids = data.batch["input_ids"]  # (bs, seq_len)
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        ori_values = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
        _, ori_seqlen = input_ids.size(0), input_ids.size(1)
        input_ids_for_rm = []
        attention_mask_for_rm = []
        position_ids_for_rm = []
        print_decode = True
        ori_seqlen = ori_seqlen + 128
        for id, mask in zip(input_ids, attention_mask, strict=True):
            # 1. remove pad for each sequence
            non_zero_indices = torch.nonzero(mask).view(-1)
            begin_pos, end_pos = non_zero_indices[0].item(), non_zero_indices[-1].item()
            valid_id = id[begin_pos : end_pos + 1]
            # 2. decode by sft_tokenizer, remove sft system prompts
            decode_result = self.sft_tokenizer.decode(valid_id)
            # workaround
            decode_with_rm_chat = (
                decode_result.replace("<|user|>\n", "[INST] ")
                .replace("</s>\n<|assistant|>\n", " [/INST]")
                .replace("</s> \n<|assistant|>\n", " [/INST]")
                + "</s>"
            )
            if print_decode and torch.distributed.get_rank() == 0:
                # only print first decode result
                print(
                    f"device {get_device_id()}: sft decode result:\n{decode_result}\n \
                        \ndevice {get_device_id()}: sft decode result with \
                        rm chat template:\n{decode_with_rm_chat}\n\n"
                )
                print_decode = False
            # 3. encode by rm_tokenizer
            rm_input_ids = self.rm_tokenizer(decode_with_rm_chat, return_tensors="pt")["input_ids"][0].to(
                input_ids.device
            )
            # 4. generate attention_mask and position_ids
            rm_attention_mask = torch.ones_like(rm_input_ids, device=input_ids.device)
            cur_seqlen = rm_input_ids.shape[-1]
            # NOTE(gh): the later reward compute will process the shape (bs, seqlen_pad_128)
            if cur_seqlen > ori_seqlen:
                print(f"warninig: rm encode seqlen {cur_seqlen} > sft encode seqlen {ori_seqlen}")
                rm_input_ids = rm_input_ids[:ori_seqlen]
                rm_attention_mask = rm_attention_mask[:ori_seqlen]
            else:
                # right padding
                rm_input_ids = pad_sequence_to_length(rm_input_ids, ori_seqlen, self.rm_tokenizer.pad_token_id)
                rm_attention_mask = pad_sequence_to_length(rm_attention_mask, ori_seqlen, 0)
            rm_position_ids = torch.arange(0, ori_seqlen, device=input_ids.device)
            input_ids_for_rm.append(torch.unsqueeze(rm_input_ids, dim=0))
            attention_mask_for_rm.append(torch.unsqueeze(rm_attention_mask, dim=0))
            position_ids_for_rm.append(torch.unsqueeze(rm_position_ids, dim=0))
        input_ids_for_rm = torch.cat(input_ids_for_rm, dim=0)
        attention_mask_for_rm = torch.cat(attention_mask_for_rm, dim=0)
        position_ids_for_rm = torch.cat(position_ids_for_rm, dim=0)

        # (bs, seqlen) will not change, but input_ids, attention_mask and position_ids will change
        # NOTE(gh): need to replace into origin values after compute reward!
        data.batch["input_ids"] = input_ids_for_rm
        data.batch["attention_mask"] = attention_mask_for_rm
        data.batch["position_ids"] = position_ids_for_rm

        return data, ori_values

    @torch.no_grad()
    def compute_reward(self, data: DataProto) -> DataProto:
        if self.config.megatron.param_offload:
            self.load_params_to_cuda()

        if self.use_different_tokenizer:
            data, ori_values = self.re_encode_by_rm_tokenizer(data)

        input_ids = data.batch["input_ids"]  # (bs, seq_len')
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        assert micro_batch_size is not None, "micro batch size is needed for forward compute"
        if use_dynamic_bsz:
            assert max_token_len is not None, "use_dynamic_bsz is True, but max_token_len is None!"
            max_token_len = max_token_len * self.config.megatron.context_parallel_size

        responses = data.batch["responses"]
        batch_size = responses.size(0)
        response_length = responses.size(1)

        with torch.no_grad():
            output = self.forward_batch(
                data, use_dynamic_bsz=use_dynamic_bsz, micro_batch_size=micro_batch_size, max_token_len=max_token_len
            )
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                logits = torch.cat(output["output"], dim=0)
                if use_dynamic_bsz:
                    indices = output["indices"]
                    indices = list(itertools.chain.from_iterable(indices))
                    assert len(indices) == logits.size(0), f"{len(indices)} vs. {logits.size()}"
                    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                    logits = logits[revert_indices]
            else:
                logits = torch.empty(
                    (input_ids.shape[0], input_ids.shape[1]),
                    device=input_ids.device,
                )
            logits = logits.to(torch.float32)

            # broadcast across pp ranks
            torch.distributed.broadcast(
                tensor=logits,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
                async_op=False,
            )

        # (bs, seqlen', hidden_size) -> (bs, seqlen', 1) -> (bs, seqlen')
        token_level_rewards = logits
        # find the last token reward
        ends = attention_mask.cumsum(dim=-1).argmax(dim=-1).view(-1, 1)  # (bs, 1)
        rewards = torch.gather(token_level_rewards, dim=1, index=ends)  # (bs, 1)

        if self.use_different_tokenizer:
            data.batch.update(ori_values)
            input_ids = ori_values["input_ids"]
            attention_mask = ori_values["attention_mask"]
            position_ids = ori_values["position_ids"]

        token_level_rewards = rewards.expand(attention_mask.shape[0], attention_mask.shape[1])  # (bs, ori_seqlen)

        # assign last valid token reward to ori position
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bs,)
        eos_mask = torch.zeros_like(attention_mask)
        eos_mask[torch.arange(batch_size), eos_mask_idx] = 1.0

        token_level_rewards = token_level_rewards * eos_mask
        token_level_rewards = token_level_rewards[:, -response_length:]

        if self.config.megatron.param_offload:
            self.offload_params_to_cpu()
        else:
            # add empty cache after each compute
            get_torch_device().empty_cache()

        batch = TensorDict({"rm_scores": token_level_rewards}, batch_size=input_ids.shape[0])

        return DataProto(batch=batch)

    def forward_batch(self, data: DataProto, use_dynamic_bsz=False, micro_batch_size=None, max_token_len=None):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        mini_batch = data
        mini_batch.batch = mini_batch.batch.contiguous()
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )

        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)

        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)

        indices = None
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
            total_seqlen = max_token_len
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = mini_batch.batch.split(micro_batch_size)
            seq_len = micro_batches[0]["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        n_micro_batch = len(micro_batches)

        # compute input shapes for pp stages
        forward_backward_func = get_forward_backward_func()

        def loss_func(output):
            return torch.tensor(1.0, device=output.device), output

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]
            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"], indices)
            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                multi_modal_inputs,
                value_model=True,
            )

            return output, loss_func

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.reward_model_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.reward_model_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=True,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.reward_model_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=True,
            )

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")
        # loss_reduces contains the stats returned from loss_func
        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced

    def offload_params_to_cpu(self):
        if self.device in ["cuda", "npu"]:
            for reward_model_module in self.reward_model_module:
                for name, param in reward_model_module.named_parameters():
                    param.data = param.data.to("cpu", non_blocking=True)
            self.device = "cpu"
            get_torch_device().empty_cache()

    def load_params_to_cuda(self):
        if self.device == "cpu":
            for reward_model_module in self.reward_model_module:
                for name, param in reward_model_module.named_parameters():
                    param.data = param.data.to(get_device_id(), non_blocking=True)
            self.device = get_device_name()
