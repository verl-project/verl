import torch
from typing import Any

from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.enums import ModelType
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_attr_wrapped_model

from verl.utils.device import get_device_id

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from typing import List
from tests.unit_tests.test_utilities import Utils
from megatron.core.distributed.param_and_grad_buffer import (
    _ParamAndGradBuffer,
)
from verl.utils.model import normalize_model_name
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core import parallel_state as mpu
from megatron.core.transformer import TransformerConfig
from loguru import logger

# from verl.utils.device import get_device_id, get_device_name


def get_model_config(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    use_distributed_optimizer=True,
    transformer_config=None,
    override_ddp_config=None,
):
    """Build the model."""
    # Build model.
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and mpu.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert model_type != ModelType.encoder_and_decoder, (
            "Interleaved schedule not supported for model with both encoder and decoder"
        )
        model = []
        for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
        mpu.set_virtual_pipeline_model_parallel_rank(0)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert mpu.get_pipeline_model_parallel_split_rank() is not None, (
                    "Split rank needs to be specified for model with both encoder and decoder"
                )
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = mpu.get_pipeline_model_parallel_split_rank()
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder,
            )
        else:
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param
            )

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum(
                    [
                        sum([p.nelement() for p in model_module.parameters()])
                        for model_module in model
                    ]
                ),
            ),
            flush=True,
        )

    # GPU allocation.
    if transformer_config is None or (not transformer_config.use_cpu_initialization):
        for model_module in model:
            model_module.to(torch.cuda.current_device())

    # Fp16 conversion.
    config: TransformerConfig = get_model_config(model[0])
    config.fp8 = None
    tfconfig: TransformerConfig = model[0].config
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        ddp_models = []
        ddp_config_dict = {
            "use_distributed_optimizer": use_distributed_optimizer,
            "grad_reduce_in_fp32": True,
            "overlap_grad_reduce": False,
        }
        if override_ddp_config is not None:
            ddp_config_dict.update(override_ddp_config)
        ddp_config = DistributedDataParallelConfig(**ddp_config_dict)
        for model_chunk_idx, model_chunk in enumerate(model):
            ddp_model = DDP(
                config=tfconfig,
                module=model_chunk,
                disable_bucketing=(model_chunk_idx > 0),
                ddp_config=ddp_config,
            )
            ddp_models.append(ddp_model)
        model = ddp_models
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # # if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()
    return model


def get_named_tensor_buckets(
    expert_buffers: List[_ParamAndGradBuffer],
    transformer_config,
    bucket_bytes: int,
    num_experts_per_rank: int,
    ep_rank: int,
    pp_rank: int,
    vpp_rank: int = 0,
):
    """Group tensors into buckets based on a specified size in megabytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]

    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    def get_flattened_params(expert_buffer, param_infos):
        start_index = param_infos[0][2]
        end_index = param_infos[-1][3]
        return expert_buffer.param_data[start_index:end_index]

    def get_param_info(
        expert_buffer,
        param,
    ):
        param_name = expert_buffer.param_to_name[param]
        name_prefix, local_expert_id = param_name.split(".weight")
        local_expert_id = int(local_expert_id)
        global_expert_id = num_experts_per_rank * ep_rank + local_expert_id
        return (
            normalize_model_name(
                f"{name_prefix}.weight{global_expert_id}",
                pp_rank,
                vpp_rank,
                transformer_config,
            ),
            param.shape,
            expert_buffer.param_index_map[param][0],
            expert_buffer.param_index_map[param][1],
        )

    for expert_buffer in expert_buffers:
        param_infos = []
        current_size = 0
        for param in expert_buffer.params[::-1]:
            param_info = get_param_info(expert_buffer, param)
            tensor_size = param.element_size() * param.numel()
            if current_size + tensor_size > bucket_bytes:
                if param_infos:
                    yield param_infos, get_flattened_params(expert_buffer, param_infos)

                param_infos = [param_info]
                current_size = tensor_size

            else:
                param_infos.append(param_info)
                current_size += tensor_size

        if param_infos:
            yield param_infos, get_flattened_params(expert_buffer, param_infos)


def broadcast_from_megatron_pp(tensor: torch.Tensor):
    # tensor is not None only in one of the pp ranks
    if tensor is not None:
        shape = tensor.shape
        dtype = tensor.dtype
        tensor_parallel = getattr(tensor, "tensor_model_parallel", None)
        partition_dim = getattr(tensor, "partition_dim", None)
        tensor_spec = (shape, dtype, tensor_parallel, partition_dim)
    else:
        tensor_spec = None
    tensor_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=tensor_spec_output,
        obj=tensor_spec,
        group=mpu.get_pipeline_model_parallel_group(),
    )
    # find the src rank
    target_tensor_spec = None
    src_rank = None
    for rank, tensor_spec in enumerate(tensor_spec_output):
        if tensor_spec is not None:
            if target_tensor_spec is None:
                target_tensor_spec = tensor_spec
            else:
                raise ValueError("A tensor exists on two pp ranks")
            src_rank = rank
    assert target_tensor_spec is not None
    if tensor is None:
        tensor = torch.empty(
            size=target_tensor_spec[0],
            dtype=target_tensor_spec[1],
            device=get_device_id(),
        )
        if target_tensor_spec[2] is not None:
            tensor.tensor_model_parallel = target_tensor_spec[2]
        if target_tensor_spec[3] is not None:
            tensor.partition_dim = target_tensor_spec[3]

    global_rank = torch.distributed.get_global_rank(
        group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank
    )
    torch.distributed.broadcast(
        tensor=tensor, src=global_rank, group=mpu.get_pipeline_model_parallel_group()
    )
    return tensor


def broadcast_str_from_megatron_pp(obj: Any):
    obj_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=obj_output, obj=obj, group=mpu.get_pipeline_model_parallel_group()
    )

    src_rank = None
    target_obj = None
    for rank, item in enumerate(obj_output):
        if item is not None:
            if target_obj is not None:
                raise ValueError("An object exists on two pp ranks")
            target_obj = item
            src_rank = rank

    assert target_obj is not None, "No valid object found to broadcast."

    global_rank = torch.distributed.get_global_rank(
        group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank
    )

    obj_output = [None] * torch.distributed.get_world_size(
        group=mpu.get_pipeline_model_parallel_group()
    )
    obj_output[0] = target_obj
    torch.distributed.broadcast_object_list(
        object_list=obj_output,
        src=global_rank,
        group=mpu.get_pipeline_model_parallel_group(),
    )

    return obj_output[0]


def get_param(
    param_data: torch.Tensor,
    shape: torch.Size,
    start_index: int,
    flattened_experts_start_index: int = 0,
):
    end_index = start_index + shape.numel() - flattened_experts_start_index
    assert end_index <= param_data.numel(), "Requested tensor is out of buffer range"
    buffer_tensor = param_data[start_index:end_index]
    return buffer_tensor.view(shape)


ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def apply_offset():
    pass


def get_info_name(infos):
    def reset_info_list(info_list):
        start_offset = info_list[0][2]

        def reset(info_tuple):
            return (
                info_tuple[0],
                info_tuple[1],
                info_tuple[2] - start_offset,
                info_tuple[3] - start_offset,
            )

        return list(map(reset, info_list))

    infos_list = list(map(reset_info_list, infos))

    logger.info(infos_list)


def bucketed_all_gather_param_generator(
    expert_parallel_buffers,
    transformer_config,
    num_moe_experts,
    ep_size: int,
    ep_rank: int,
    pp_size: int,
    pp_rank: int,
    vpp_rank: int,
):
    bucket_size = (1 << 30) * 10
    rank = torch.distributed.get_rank()

    def expert_generator(expert_infos, flattened_params):
        for param_info, flattened_param in zip(expert_infos, flattened_params):
            for info in param_info:
                param_name, shape, start_index, _ = info
                param = get_param(flattened_param, shape, start_index)
                logger.debug(f"{param_name=}{param=}")

    for experts_param_infos, flattened_experts in get_named_tensor_buckets(
        expert_parallel_buffers,
        transformer_config=transformer_config,
        bucket_bytes=bucket_size,
        num_experts_per_rank=num_moe_experts // ep_size,
        ep_rank=ep_rank,
        pp_rank=pp_rank,
        vpp_rank=vpp_rank,
    ):
        for scan_pp_rank in range(pp_size):
            if scan_pp_rank == pp_rank:
                all_experts_param_infos = [None for _ in range(ep_size)]
                torch.distributed.all_gather_object(
                    all_experts_param_infos,
                    experts_param_infos,
                    group=mpu.get_expert_model_parallel_group(),
                )
                if rank == 0:
                    logger.debug(all_experts_param_infos)
                    logger.info(f"{flattened_experts.shape=}")

                all_experts_flattened_param = [
                    torch.empty_like(flattened_experts) for _ in range(ep_size)
                ]
                all_experts_flattened_param = torch.zeros(
                    flattened_experts.numel() * ep_size,
                    dtype=flattened_experts.dtype,
                    device=torch.cuda.current_device(),
                )

                torch.distributed.all_gather_into_tensor(
                    all_experts_flattened_param,
                    flattened_experts,
                    group=mpu.get_expert_model_parallel_group(),
                )
            else:
                all_experts_param_infos = None
                all_experts_flattened_param = None

            all_experts_param_infos = broadcast_str_from_megatron_pp(
                all_experts_param_infos
            )
            all_experts_flattened_param = broadcast_from_megatron_pp(
                all_experts_flattened_param
            )
            get_info_name(all_experts_param_infos)
            experts_flattened_param = torch.chunk(
                all_experts_flattened_param, chunks=ep_size
            )
            logger.debug(experts_flattened_param)
            expert_generator(all_experts_param_infos, experts_flattened_param)

    pass


def param_generator(
    param_infos,
    flattened_param_data,
    start_from_zero: bool = True,
):
    for param_info in param_infos:
        param_name, shape, start_index, _ = param_info
        param = get_param(flattened_param_data, shape, start_index)


def dense_param_generator(
    buffers,
    transformer_config,
    pp_size: int,
    pp_rank: int,
    vpp_rank: int,
):
    def get_dense_param_infos(buffer):
        return [
            (
                normalize_model_name(
                    buffer.param_to_name[param],
                    pp_rank,
                    vpp_rank,
                    transformer_config,
                ),
                param.shape,
                buffer.param_index_map[param][0],
                buffer.param_index_map[param][1],
            )
            for param in buffer.params[::-1]
        ]

    for buffer in buffers:
        for scan_pp_rank in range(pp_size):
            if scan_pp_rank == pp_rank:
                dense_param_infos = get_dense_param_infos(buffer)
                flattened_dense_param_data = buffer.param_data
            else:
                dense_param_infos = None
                flattened_dense_param_data = None
            dense_param_infos = broadcast_str_from_megatron_pp(dense_param_infos)
            flattened_dense_param_data = broadcast_from_megatron_pp(
                flattened_dense_param_data
            )
            param_generator(dense_param_infos, flattened_dense_param_data)


def ddp_all_gather_param_generator(
    module,
):
    vpp_size = len(module)
    pass


class TestMoELayerInit:
    def __init__(
        self,
        expert_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=0,
    ) -> None:
        self.expert_model_parallel_size = expert_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        Utils.initialize_model_parallel(
            expert_model_parallel_size=expert_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            # virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        )

    def _get(
        self,
        param_data: torch.Tensor,
        shape: torch.Size,
        start_index: int,
        numel: int,
    ) -> torch.Tensor:
        end_index = start_index + shape.numel()
        assert end_index <= numel, "Requested tensor is out of buffer range"
        buffer_tensor = param_data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def test_te_moe_layer(
        self,
    ):
        self.num_moe_experts = 8
        self.grouped_gemm = True
        self.num_layers = 4
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=self.num_layers,
            hidden_size=4096,
            num_attention_heads=64,
            num_moe_experts=self.num_moe_experts,
            use_cpu_initialization=False,
            moe_shared_expert_intermediate_size=1024,
            # moe_token_dispatcher_type=moe_token_dispatcher_type,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            expert_model_parallel_size=self.expert_model_parallel_size,
            pipeline_dtype=torch.float32,
            # virtual_pipeline_model_parallel_size=self.virtual_pipeline_model_parallel_size,
            moe_router_topk=4,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=self.grouped_gemm,
            moe_ffn_hidden_size=12288,
            add_bias_linear=False,
        )
        self.transformer_config.first_pipeline_num_layers = None
        self.transformer_config.last_pipeline_num_layers = None
        transformer_layer_spec = get_gpt_decoder_block_spec(
            self.transformer_config, use_transformer_engine=True
        )

        def megatron_actor_model_provider(pre_process, post_process):
            vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            gpt_model = GPTModel(
                config=self.transformer_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                max_sequence_length=4,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vpp_rank,
            ).cuda()
            return gpt_model

        rank = torch.distributed.get_rank()
        ep_rank = mpu.get_expert_model_parallel_rank()
        ep_size = mpu.get_expert_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        actor_module = get_model(
            model_provider_func=megatron_actor_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
        )
        vpp_size = len(actor_module)
        for vpp_rank in range(vpp_size):
            vpp_ddp_module = actor_module[vpp_rank]
            # expert_parallel_buffers = vpp_ddp_module.expert_parallel_buffers[0]
            # expert_parallel_buffers = vpp_ddp_module.expert_parallel_buffers
            # buffers = vpp_ddp_module.buffers[0]
            dense_param_generator(
                vpp_ddp_module.buffers,
                self.transformer_config,
                pp_size,
                pp_rank,
                vpp_rank,
            )

            bucketed_all_gather_param_generator(
                expert_parallel_buffers=vpp_ddp_module.expert_parallel_buffers,
                transformer_config=self.transformer_config,
                num_moe_experts=self.num_moe_experts,
                ep_size=ep_size,
                ep_rank=ep_rank,
                pp_rank=pp_rank,
                pp_size=pp_size,
                vpp_rank=vpp_rank,
            )


    def destory(self):
        Utils.destroy_model_parallel()


if __name__ == "__main__":
    test_moe = TestMoELayerInit(
        expert_model_parallel_size=2,
        pipeline_model_parallel_size=2,
    )
    test_moe.test_te_moe_layer()
    test_moe.destory()
