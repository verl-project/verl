# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
# Copyright 2025 Individual Contributor: Brilliant Hanabi, funrunding
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

import asyncio
import logging
import os
import time

import numpy as np
import psutil
import torch
from codetiming import Timer
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from .megatron_kl_loss import vocab_parallel_kl_divergence
from omegaconf import DictConfig, OmegaConf
from torch import nn

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.flops_counter import FlopsCounter
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron_utils import get_model_config
from verl.utils.profiler import (
    DistProfiler,
    GPUMemoryLogger,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import gather_timing
from verl.utils.profiler.profile import Profiler
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.megatron_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TensorBuffer:
    def __init__(self, memory_alloc, dtype):
        self.device = get_device_id()
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        self.capacity = memory_alloc // dtype_size
        self.dtype = dtype
        self.tensor = torch.empty(self.capacity, dtype=self.dtype, device=self.device)
        self.keys = []
        self.shapes = []

    @property
    def size(self):
        return sum(shape.numel() for shape in self.shapes)

    def clear(self):
        self.keys.clear()
        self.shapes.clear()
        self.tensor = torch.empty(self.capacity, dtype=self.dtype, device=self.device)

    def append(self, key, shape, weight=None):
        if weight is not None:
            self.tensor[self.size : self.size + shape.numel()] = weight.view(-1)
        self.keys.append(key)
        self.shapes.append(shape)

    def to_tensors(self):
        tensors = []
        start = 0
        for key_, shape_ in zip(self.keys, self.shapes, strict=False):
            tensors.append((key_, self.tensor[start : start + shape_.numel()].view(shape_)))
            start += shape_.numel()
        return tensors


def record_time(func):
    def wrapper(*args, **kwargs):
        tik = time.time()
        func(*args, **kwargs)
        tok = time.time()
        return tok - tik

    return wrapper


class OnPolicyDistillActor:
    """
    Responsible purely for the training step (forward-backward + optimizer).
    """

    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            hf_config (PretrainedConfig): huggingface config
            tf_config (TransformerConfig): mcore transformer config
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this
                pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for
                more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation.
                Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron.
                It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          hf_config=hf_config,
        >>>                          tf_config=tf_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        self.config = config
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.prof = Profiler(self.config.profiler)
        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "pipeline_model_parallel_split_rank": None,
                "reduce_grads_use_alltoall": False,
            }
        )

        config = get_model_config(self.actor_module[0])
        print(config)
        config.finalize_model_grads_func = finalize_model_grads

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    def forward_backward_batch(
        self,
        data: DataProto,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        # broadcast_dict_tensor(
        #     data.batch,
        #     src=mpu.get_pipeline_model_parallel_last_rank(),
        #     group=mpu.get_pipeline_model_parallel_group(),
        # )
        # split into micro-batches
        # NOTE: tensordict lock error
        # data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)
        data.batch.set_("attention_mask", data.batch["attention_mask"].to(torch.bool))

        indices = None
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=data.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {len(micro_batches)} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(batch=data.batch, max_token_len=max_token_len)
            # total_seqlen = max_token_len
            if mpu.is_pipeline_last_stage():
                teacher_topk_logps_tensor = torch.tensor(data.non_tensor_batch["teacher_topk_logps"])
                teacher_topk_indices_tensor = torch.tensor(data.non_tensor_batch["teacher_topk_indices"])
                teacher_topk_logps, teacher_topk_indices = [], []
                for partition in indices:
                    curr_logp_micro_batch, curr_idx_micro_batch = [], []
                    for idx in partition:
                        curr_logp_micro_batch.append(teacher_topk_logps_tensor[idx : idx + 1])
                        curr_idx_micro_batch.append(teacher_topk_indices_tensor[idx : idx + 1])
                    curr_logp_micro_batch = torch.cat(curr_logp_micro_batch)
                    curr_idx_micro_batch = torch.cat(curr_idx_micro_batch)

                    teacher_topk_logps.append(curr_logp_micro_batch)
                    teacher_topk_indices.append(curr_idx_micro_batch)

                for i, mb in enumerate(micro_batches):
                    responses = mb["responses"]
                    response_length = responses.size(1)
                    calc_kl_mask = mb["attention_mask"].clone()
                    calc_kl_mask[:, : (-response_length - 1)] = False
                    mb["calc_kl_mask"] = calc_kl_mask
                    mb["kl_losses"] = torch.zeros_like(calc_kl_mask, dtype=torch.float32)
                    mb["teacher_topk_logps"] = teacher_topk_logps[i].pin_memory()
                    mb["teacher_topk_indices"] = teacher_topk_indices[i].pin_memory()
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = data.batch.split(micro_batch_size)
            # seq_len = micro_batches[0]["input_ids"].shape[1]
            # total_seqlen = micro_batch_size * seq_len
            if mpu.is_pipeline_last_stage():
                teacher_topk_logps = np.array_split(data.non_tensor_batch["teacher_topk_logps"], len(micro_batches))
                teacher_topk_indices = np.array_split(data.non_tensor_batch["teacher_topk_indices"], len(micro_batches))
                
                for i, mb in enumerate(micro_batches):
                    responses = mb["responses"]
                    response_length = responses.size(1)
                    calc_kl_mask = mb["attention_mask"].clone()
                    # NOTE: otherwise incorrect indexing using int64
                    # print(f"[DEBUG-CREATE] attention_mask.shape={mb['attention_mask'].shape}, response_length={response_length}")
                    calc_kl_mask[:, : (-response_length - 1)] = False
                    calc_kl_mask = calc_kl_mask.bool()
                    # print(f"[DEBUG-CREATE] calc_kl_mask.shape={calc_kl_mask.shape}, calc_kl_mask.dtype={calc_kl_mask.dtype}")
                    # NOTE: TensorDict is locked, need to unlock to add new keys
                    mb.unlock_()
                    mb["calc_kl_mask"] = calc_kl_mask
                    mb["kl_losses"] = torch.zeros_like(calc_kl_mask, dtype=torch.float32)
                    mb["teacher_topk_logps"] = torch.tensor(teacher_topk_logps[i]).pin_memory()
                    mb["teacher_topk_indices"] = torch.tensor(teacher_topk_indices[i]).pin_memory()
                    mb.lock_()


        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        def loss_func(output):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            metrics = {}

            ret_entropy = None
            stats = {}
            kl_losses = output["kl_losses"]
            calc_kl_mask = output["calc_kl_mask"]
            # inf_cnt = masked_kl_lossed.isinf().sum().item()
            # nan_cnt = masked_kl_lossed.isnan().sum().item()
            # total_cnt = masked_kl_lossed.nelement()
            # print(f"rank: {rank}, kl_loss inf_cnt/nan_cnt/total_cnt: {inf_cnt} / {nan_cnt} /{total_cnt}")
            masked_kl_lossed = kl_losses[calc_kl_mask]
            mean_kl_loss = masked_kl_lossed.mean()
            stats.update({"actor/kl_loss": mean_kl_loss.detach().item()})

            append_to_dict(metrics, stats)
            return mean_kl_loss, [metrics, ret_entropy]

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            # attention_mask = batch["attention_mask"]
            attention_mask = batch["attention_mask"].to(torch.bool) # to avoid incorrect int64 indexing in downstream
            position_ids = batch["position_ids"]


            def logits_processor(logits, teacher_topk_logps, teacher_topk_indices, calc_kl_mask, kl_losses):
                # print(f"[DEBUG-LOGITS] logits.shape: {logits.shape}")
                # print(f"[DEBUG-LOGITS] calc_kl_mask.shape: {calc_kl_mask.shape}")
                # print(f"[DEBUG-LOGITS] teacher_topk_logps.shape: {teacher_topk_logps.shape}")
                # print(f"[DEBUG-LOGITS] teacher_topk_indices.shape: {teacher_topk_indices.shape}")
                assert logits.shape[:2] == calc_kl_mask.shape[:2]
                assert logits.shape[:2] == teacher_topk_indices.shape[:2]
                assert logits.shape[:2] == teacher_topk_logps.shape[:2]

                masked_logits = logits[calc_kl_mask]
                masked_teacher_topk_logps = teacher_topk_logps[calc_kl_mask]
                masked_teacher_topk_indices = teacher_topk_indices[calc_kl_mask]

                kl_losses[calc_kl_mask] = vocab_parallel_kl_divergence(
                    masked_logits, masked_teacher_topk_logps, masked_teacher_topk_indices
                )
                return {"kl_losses": kl_losses, "calc_kl_mask": calc_kl_mask}

            if mpu.is_pipeline_last_stage():
                device = get_device_id()
                teacher_topk_logps = batch["teacher_topk_logps"].to(device, non_blocking=True)
                teacher_topk_indices = batch["teacher_topk_indices"].to(device, non_blocking=True)
                ''' 
                NOTE: logits_processor needs logits as first input, but it is not in the args here
                because it will be popuated in forward_fn such as verl/verl/models/mcore/model_forward.py -> model_forward()
                
                input_args = dict(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
                    packed_seq_params=packed_seq_params,
                    **model_kwargs,
                )
                output_orig = model(**input_args)
                output_dict = logits_processor(output_orig, **args)
                '''
                logits_processor_args = {
                    "calc_kl_mask": batch["calc_kl_mask"],
                    "kl_losses": batch["kl_losses"],
                    "teacher_topk_logps": teacher_topk_logps,
                    "teacher_topk_indices": teacher_topk_indices,
                }
            else:
                logits_processor_args = None

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"], indices)

            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)

            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
            )

            return output, loss_func

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=-1,  # no use when variable_seq_lengths was set
            micro_batch_size=-1,  # no use when variable_seq_lengths was set
            forward_only=False,
        )

        # loss_reduces contains the stats returned from loss_func

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, data: DataProto) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        # self.prof.start()
        data.to(get_device_id())

        self.actor_optimizer.zero_grad()
        # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        for chunk in self.actor_module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

        micro_batch_size = self.config.micro_batch_size
        max_token_len = None
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.max_token_len * self.config.megatron.context_parallel_size

        metric_micro_batch = self.forward_backward_batch(
            data,
            use_dynamic_bsz=self.config.use_dynamic_bsz,
            micro_batch_size=micro_batch_size,
            max_token_len=max_token_len,
        )

        metric_micro_batch = metric_micro_batch["output"]
        for metric in metric_micro_batch:
            # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
            append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.

        update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
        data = {"actor/grad_norm": grad_norm}
        append_to_dict(metrics, data)

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError
        # self.prof.step()
        # add empty cache after each compute
        # self.prof.stop_and_save()
        # self.prof.stop_trace()
        get_torch_device().empty_cache()
        return metrics


class MegatronOnPolicyDistillActorWorker(ActorRolloutRefWorker):
    """
    Actor-only worker: owns the trainable Megatron model and optimizer, performs update_actor.
    """

    def __init__(self, config: DictConfig, role: str):
        # Ensure we run as actor-only worker
        is_struct = OmegaConf.is_struct(config) or False
        OmegaConf.set_struct(config, False)
        OmegaConf.set_struct(config, is_struct)

        super().__init__(config, role)
        assert self._is_actor and not self._is_rollout, "Actor worker must be actor-only."

    def _get_actor_params_generator(self):
        assert self._is_actor
        if self.bridge is not None:
            generator = self.bridge.export_weights(self.actor.actor_module)
        else:
            from verl.utils.megatron_utils import per_tensor_generator
            # from megatron_utils import per_tensor_generator

            from verl.models.mcore import get_mcore_weight_converter

            layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            generator = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                weight_converter,
                self.tf_config,
                layer_name_mapping,
            )
        return generator

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_transformer_config = OmegaConf.to_container(
            self.config.actor.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True
        )

        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        # we need the model for actor and rollout
        optim_config = self.config.actor.optim
        (
            self.actor_module,
            self.actor_optimizer,
            self.actor_optimizer_scheduler,
            self.actor_model_config,
            self.actor_optim_config,
        ) = self._build_model_optimizer(
            model_path=self.config.model.path,
            optim_config=optim_config,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )

        self.actor = OnPolicyDistillActor(
            config=self.config.actor,
            model_config=self.actor_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.actor_optimizer,
        )
        log_gpu_memory_usage("After OnPolicyDistillActor init", logger=logger)

        self.flops_counter = FlopsCounter(self.actor_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.actor.checkpoint,
            model_config=self.actor_model_config,
            transformer_config=self.tf_config,
            role="actor",
            model=self.actor_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.actor_optimizer,
            optimizer_scheduler=self.actor_optimizer_scheduler,
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
        )
        get_torch_device().empty_cache()
        log_gpu_memory_usage("Actor init_model finished", logger=logger)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red")
    def update_actor(self, data: DataProto):
        assert self._is_actor and not self._is_rollout

        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(data=data)

        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert self._is_actor and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        params_generator = self._get_actor_params_generator()

        from ray.util.collective import collective

        update_weights_bucket_bytes = int(self.config.rollout.update_weights_bucket_megabytes) << 20
        tensor_buffer = TensorBuffer(update_weights_bucket_bytes, self.param_dtype)

        for key, shape, dtype in self._weights_info:
            weight_key, weight = next(params_generator)
            assert key == weight_key
            assert shape == weight.size()
            try:
                assert dtype == weight.dtype
            except AssertionError:
                if not key.endswith("e_score_correction_bias"):
                    raise
                # weight = weight.to(dtype)

            if shape.numel() > tensor_buffer.capacity:
                collective.broadcast(weight, src_rank=0, group_name="actor_rollout")
            else:
                if tensor_buffer.size + shape.numel() > tensor_buffer.capacity:
                    collective.broadcast(tensor_buffer.tensor, src_rank=0, group_name="actor_rollout")
                    tensor_buffer.clear()
                tensor_buffer.append(key, shape, weight)
        if tensor_buffer.size > 0:
            collective.broadcast(tensor_buffer.tensor, src_rank=0, group_name="actor_rollout")
            tensor_buffer.clear()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info

        params_generator = self._get_actor_params_generator()
        ret = []
        for key, tensor in params_generator:
            ret.append((key, tensor.size(), tensor.dtype))

        self._weights_info = ret
        return ret


class MegatronOnPolicyDistillRolloutWorker(ActorRolloutRefWorker):
    """
    Rollout-only worker: owns the inference engine (vLLM/SGlang, or Megatron forward) and generates sequences.
    """

    def __init__(self, config: DictConfig, role: str):
        # Ensure we run as rollout-only worker
        # is_struct = OmegaConf.is_struct(config) or False
        # OmegaConf.set_struct(config, False)
        # # Set a safe minimal rollout micro-batch size if not provided by config
        # if OmegaConf.select(config, "actor.ppo_mini_batch_size") is None:
        #     config.actor.ppo_mini_batch_size = 2
        # if OmegaConf.select(config, "rollout.n") is None:
        #     config.rollout.n = 1
        # OmegaConf.set_struct(config, is_struct)
        import datetime

        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.device import (
            get_nccl_backend,
            get_torch_device,
        )
        from verl.utils.distributed import set_numa_affinity
        from verl.utils.fs import copy_to_local
        from verl.utils.model import get_generation_config
        from verl.utils.profiler import DistProfilerExtension, ProfilerConfig
        from verl.workers.megatron_workers import MegatronWorker

        MegatronWorker.__init__(self)
        self.config = config
        self.local_path = copy_to_local(self.config.model.path)

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

        self.role = role
        assert self.role == "rollout"

        self._is_actor = False
        self._is_rollout = True
        self._is_ref = False

        # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
        # This is for extendability in AsyncRL cases
        omega_profiler_config = config.rollout.get("profiler", {})

        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # self._build_rollout will use this variable
        self.bridge = "none"
        self.generation_config = get_generation_config(self.local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Build the actor module only for inference + rollout engine; no optimizer/updates.
        """
        from verl.utils.torch_dtypes import PrecisionType

        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init rollout model", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        self.rollout_device_mesh = self.rollout.device_mesh
        log_gpu_memory_usage("After rollout init", logger=logger)
        get_torch_device().empty_cache()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        """
        Asynchronous-friendly rollout. When called via Ray with blocking=False,
        returns immediately with a future. The actual method execution generates
        sequences and optionally fetches teacher knowledge, and returns DataProto.
        """
        assert self._is_rollout and not self._is_actor
        prompts.batch = prompts.batch.to(get_device_name())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        # No context switching here; rollout-only worker always in rollout mode.

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = gather_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        # clear kv cache
        get_torch_device().empty_cache()

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
    def async_generate_sequences(self, *args, **kwargs):
        return self.generate_sequences(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        from ray.util.collective import collective

        assert self._is_rollout and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None
        rollout_name = self.config.rollout.name
        if rollout_name == "vllm":
            inference_model = (
                self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            )
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(inference_model)
        elif rollout_name == "sglang":
            from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights

            inference_model = self.rollout._engine

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def update_weights(inference_engine, params):
                await sgl_update_weights(
                    engine=inference_engine,
                    params_batch=params,
                    device_mesh_key="infer_tp",
                    device_mesh=self.rollout_device_mesh,
                )

                if self.rollout_device_mesh["infer_tp"].get_local_rank() == 0:
                    await inference_engine.flush_cache()
        else:
            raise NotImplementedError(f"Unknown rollout name: {rollout_name}")

        update_weights_bucket_bytes = int(self.config.rollout.update_weights_bucket_megabytes) << 20
        tensor_buffer = TensorBuffer(update_weights_bucket_bytes, self.param_dtype)

        def group_tensor_generator():
            for key, shape, dtype in self._weights_info:
                assert dtype == self.param_dtype, key
                if shape.numel() > tensor_buffer.capacity:
                    tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                    collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")
                    yield [(key, tensor)]
                else:
                    if tensor_buffer.size + shape.numel() > tensor_buffer.capacity:
                        collective.broadcast(tensor_buffer.tensor, src_rank=0, group_name="actor_rollout")
                        yield tensor_buffer.to_tensors()
                        tensor_buffer.clear()
                    tensor_buffer.append(key, shape)
            if tensor_buffer.size > 0:
                collective.broadcast(tensor_buffer.tensor, src_rank=0, group_name="actor_rollout")
                yield tensor_buffer.to_tensors()
                tensor_buffer.clear()

        for tensors in group_tensor_generator():
            if rollout_name == "vllm":
                inference_model.load_weights(tensors)
            elif rollout_name == "sglang":
                loop.run_until_complete(update_weights(inference_model, tensors))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info
