# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import os

os.environ["NCCL_DEBUG"] = "WARN"

from functools import partial

import numpy as np
import pytest
import ray
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Qwen3Config,
    Qwen3MoeConfig,
)

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.torch_functional import logprobs_from_logits_naive
from verl.workers.config import (
    ActorConfig,
    CriticConfig,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    McoreEngineConfig,
    McoreOptimizerConfig,
)
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.utils.losses import ppo_loss, sft_loss, value_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def get_test_language_model(device_count):
    if device_count == 1:
        model = "~/models/HuggingFaceTB/SmolLM2-135M-Instruct"
    else:
        model = "~/models/Qwen/Qwen2.5-0.5B"
    model = os.path.expanduser(model)
    return model


def create_training_config(model_type, strategy, device_count, model):
    if device_count == 1:
        tp = pp = cp = fsdp_size = 1
    else:
        tp = pp = cp = 2
        fsdp_size = 4

    path = os.path.expanduser(model)
    model_config = HFModelConfig(path=path, use_remove_padding=True)

    kwargs = dict(
        param_offload=True,
        optimizer_offload=True,
        grad_offload=True,
        use_dynamic_bsz=True,
        use_remove_padding=True,
        max_token_len_per_gpu=500,
        infer_max_token_len_per_gpu=1000,
    )

    if strategy == "megatron":
        engine_config = McoreEngineConfig(
            forward_only=False,
            use_mbridge=True,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            **kwargs,
        )
        optimizer_config = McoreOptimizerConfig(lr_decay_steps=10)
    elif strategy in ["fsdp", "fsdp2"]:
        engine_config = FSDPEngineConfig(
            forward_only=False, fsdp_size=fsdp_size, strategy=strategy, ulysses_sequence_parallel_size=cp, **kwargs
        )
        optimizer_config = FSDPOptimizerConfig()
    else:
        raise NotImplementedError(f"strategy {strategy} is not supported")

    config = TrainingWorkerConfig(
        model_type=model_type,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=None,
    )
    return config


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "megatron"])
def test_actor_engine(strategy):
    ray.init()
    device_count = torch.cuda.device_count()
    config = create_training_config(
        model_type="language_model",
        strategy=strategy,
        device_count=device_count,
        model=get_test_language_model(device_count),
    )
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=config)
    resource_pool = RayResourcePool(process_on_nodes=[device_count])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    # init model
    wg.reset()

    sft_loss_ = partial(sft_loss, config=config)

    wg.set_loss_fn(sft_loss_)

    batch_size = 8
    seqlen = 32

    response_length = seqlen // 2

    torch.manual_seed(1)
    np.random.seed(1)

    input_ids = torch.randint(0, config.model_config.hf_config.vocab_size, (batch_size, seqlen))
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_valid_token=0.8, max_ratio_of_left_padding=0.2, min_ratio_of_valid_token=0.6
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    global_token_num = torch.sum(attention_mask, dim=-1).tolist()

    print(input_ids.float().mean(), attention_mask.float().mean())

    responses = input_ids[:, response_length:]
    response_mask = attention_mask[:, response_length:]

    assert torch.all(response_mask[:, 0] == 1)

    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :response_length],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
        },
        meta_info={"temperature": 1.0, "global_token_num": global_token_num, "compute_loss": False},
    )

    data_td = data.to_tensordict()
    data_td = left_right_2_no_padding(data_td)

    # eval
    output = wg.infer_batch(data_td)
    output = output.get()
    logprobs_unpad = tu.get(output, "log_probs").cpu()
    logprobs = no_padding_2_padding(logprobs_unpad, data_td)

    output = DataProto.from_single_dict({"old_log_probs": logprobs})

    # load hf model and compare results with hf model
    path = config.model_config.path
    hf_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    hf_output = hf_model(input_ids, attention_mask=attention_mask)
    hf_logprobs = logprobs_from_logits_naive(
        hf_output.logits[:, -response_length - 1 : -1, :].float(), input_ids[:, -response_length:]
    )
    hf_logprobs_mean = torch.mean(hf_logprobs * response_mask)
    mcore_logprobs_mean = torch.mean(output.batch["old_log_probs"] * response_mask)

    torch.testing.assert_close(hf_logprobs_mean, mcore_logprobs_mean, atol=1e-3, rtol=1e-2)

    data = data.union(output)

    # TODO: sft_loss_ is not compatible with ActorWorker until we replace DataProto with torch.jagged TensorDict
    # wg.set_loss_fn(sft_loss_)

    # train for one step
    # metrics = wg.update_actor(data)
    # print(metrics)

    # add ppo data
    data.batch["advantages"] = torch.rand_like(responses, dtype=torch.float32)
    data.batch["ref_log_prob"] = torch.rand_like(responses, dtype=torch.float32)

    # construct actor config
    actor_config = ActorConfig(strategy=strategy, rollout_n=1, ppo_micro_batch_size_per_gpu=-1)

    # set ppo loss
    ppo_loss_ = partial(ppo_loss, config=actor_config)
    wg.set_loss_fn(ppo_loss_)

    # update again
    data_td = data.to_tensordict()
    data_td = left_right_2_no_padding(data_td)

    # auto load/offload
    tu.assign_non_tensor(data_td, global_batch_size=data_td.shape[0])
    ppo_metrics = wg.train_batch(data_td)
    ppo_metrics = ppo_metrics.get()
    ppo_metrics = tu.get(ppo_metrics, "metrics")
    print(ppo_metrics)

    # test manual load/offload
    tu.assign_non_tensor(data_td, disable_auto_offload=True)
    wg.to("device")
    ppo_metrics = wg.train_batch(data_td)
    ppo_metrics = ppo_metrics.get()
    ppo_metrics = tu.get(ppo_metrics, "metrics")
    print(ppo_metrics)
    wg.to("cpu")

    ray.shutdown()


def create_value_model(language_model_path, output_path):
    config = AutoConfig.from_pretrained(language_model_path)
    config.num_labels = 1
    config.classifier_dropout = 0
    config.tie_word_embeddings = False
    model = AutoModelForTokenClassification.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(language_model_path))
    assert model.config.num_labels == 1
    path = os.path.expanduser(output_path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    config.save_pretrained(path)
    return path


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_critic_engine(strategy):
    device_count = torch.cuda.device_count()
    value_model_path = os.path.expanduser("~/models/test_model")
    language_model_path = get_test_language_model(device_count=device_count)
    create_value_model(language_model_path, value_model_path)

    torch.manual_seed(1)
    np.random.seed(1)

    ray.init()

    config = create_training_config(
        model_type="value_model", strategy=strategy, device_count=device_count, model=value_model_path
    )
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=config)
    resource_pool = RayResourcePool(process_on_nodes=[device_count])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    # init model
    wg.reset()

    batch_size = 8
    seqlen = 32

    response_length = seqlen // 2
    input_ids = torch.randint(0, config.model_config.hf_config.vocab_size, (batch_size, seqlen))
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_valid_token=0.8, max_ratio_of_left_padding=0.2, min_ratio_of_valid_token=0.6
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    global_token_num = torch.sum(attention_mask, dim=-1).tolist()

    print(input_ids.float().mean(), attention_mask.float().mean())

    responses = input_ids[:, response_length:]
    response_mask = attention_mask[:, response_length:]

    assert torch.all(response_mask[:, 0] == 1)

    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :response_length],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
        },
        meta_info={"temperature": 1.0, "global_token_num": global_token_num, "compute_loss": False},
    )

    data_td = data.to_tensordict()
    data_td = left_right_2_no_padding(data_td)

    # eval
    output = wg.infer_batch(data_td)
    output = output.get()

    values_unpad = tu.get(output, "values").float().cpu()
    values = no_padding_2_padding(values_unpad, data_td)

    output = DataProto.from_single_dict({"values": values})

    # load hf model and compare results with hf model
    with torch.device("cuda"), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hf_model = AutoModelForTokenClassification.from_pretrained(
            value_model_path, torch_dtype=torch.float32, attn_implementation="flash_attention_2"
        )
        hf_output = hf_model(input_ids.cuda(), attention_mask=attention_mask.cuda())
        hf_values = hf_output.logits[:, -response_length - 1 : -1, :].float().squeeze(-1).cpu()

    hf_values_mean = torch.mean(hf_values * response_mask)
    engine_values = torch.mean(output.batch["values"] * response_mask)

    torch.testing.assert_close(hf_values_mean, engine_values, atol=1e-2, rtol=1e-2)

    data = data.union(output)

    # add ppo data
    data.batch["returns"] = torch.rand_like(responses, dtype=torch.float32)

    # update again
    # create critic config
    critic_config = CriticConfig(
        strategy=strategy, rollout_n=1, ppo_micro_batch_size_per_gpu=-1, model_config=config.model_config
    )
    value_loss_ = partial(value_loss, config=critic_config)
    wg.set_loss_fn(value_loss_)

    # update again
    data_td = data.to_tensordict()
    data_td = left_right_2_no_padding(data_td)

    # auto load/offload
    tu.assign_non_tensor(data_td, global_batch_size=data_td.shape[0])
    ppo_metrics = wg.train_batch(data_td)
    ppo_metrics = ppo_metrics.get()
    ppo_metrics = tu.get(ppo_metrics, "metrics")
    print(ppo_metrics)

    ray.shutdown()


def create_actor_model(tmp_path, config):
    model = AutoModelForCausalLM.from_config(config)
    path = os.path.join(tmp_path, "test_model")
    model.save_pretrained(path)
    config.save_pretrained(path)
    return path


def _worker(rank: int, world_size: int, rendezvous_file: str, strategy: str, model_path: str):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    ref_model_config = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        ref_model = AutoModelForCausalLM.from_config(ref_model_config)

    from verl.workers.engine import BaseEngine, EngineRegistry

    # construct configs
    model_config = HFModelConfig(path=model_path, load_tokenizer=False)

    if strategy == "megatron":
        engine_config = McoreEngineConfig(
            forward_only=False,
            use_mbridge=True,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
        )
        optimizer_config = McoreOptimizerConfig(lr_decay_steps=10)
    elif strategy in ["fsdp", "fsdp2"]:
        engine_config = FSDPEngineConfig(
            forward_only=False, fsdp_size=4, strategy=strategy, ulysses_sequence_parallel_size=2
        )
        optimizer_config = FSDPOptimizerConfig()
    else:
        raise NotImplementedError(f"strategy {strategy} is not supported")

    checkpoint_config = CheckpointConfig()

    # build model engine
    engine: BaseEngine = EngineRegistry.new(
        model_type="language_model",
        backend=engine_config.strategy,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )

    engine.initialize()

    # get per tensor parameter
    per_tensor_params, _ = engine.get_per_tensor_param()

    ref_state_dict = ref_model.state_dict()

    # load ground truth and compare
    for key, value in per_tensor_params:
        assert key in ref_state_dict, f"{key} not in ref_state_dict"
        assert value.shape == ref_state_dict[key].shape, (
            f"{key} shape not equal, {value.shape} != {ref_state_dict[key].shape}"
        )
        if rank == 0:
            print(key, value.shape)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("config", [Qwen3Config(num_hidden_layers=2), Qwen3MoeConfig(num_hidden_layers=2)])
@pytest.mark.parametrize("strategy", ["megatron", "fsdp", "fsdp2"])
def test_per_tensor_generator(world_size, tmp_path, config, strategy):
    rendezvous_file = str(tmp_path / "rdzv_mask")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)
    # create a model
    model_path = create_actor_model(tmp_path, config)
    # spawn workers
    mp.spawn(
        fn=_worker,
        args=(world_size, rendezvous_file, strategy, model_path),
        nprocs=world_size,
        join=True,
    )
