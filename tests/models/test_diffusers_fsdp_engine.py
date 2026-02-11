# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from functools import partial

import numpy as np
import pytest
import ray
import torch

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.workers.config import DiffusersModelConfig, FSDPActorConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.diffusers_patch.schedulers import FlowMatchSDEDiscreteScheduler
from verl.workers.utils.diffusers_patch.utils import set_timesteps
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import embeds_padding_2_no_padding


def create_training_config(model_type, strategy, device_count, model):
    if device_count == 1:
        cp = fsdp_size = 1
    else:
        cp = 1  # TODO (mike): test with cp = 2
        fsdp_size = 4
    path = os.path.expanduser(model)
    tokenizer_path = os.path.join(path, "tokenizer")
    model_config = DiffusersModelConfig(
        path=path,
        tokenizer_path=tokenizer_path,
        use_remove_padding=True,
    )

    if strategy in ["fsdp", "fsdp2"]:
        from hydra import compose, initialize_config_dir

        from verl.utils.config import omega_conf_to_dataclass

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/model")):
            cfg = compose(
                config_name="diffusion_model",
                overrides=[
                    "path=" + path,
                    "tokenizer_path=" + tokenizer_path,
                    "lora_rank=8",
                    "lora_alpha=16",
                ],
            )
        model_config: DiffusersModelConfig = omega_conf_to_dataclass(cfg)

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(
                config_name="dp_actor",
                overrides=[
                    "strategy=" + strategy,
                    "clip_ratio=0.0001",
                    "clip_ratio_high=5.0",
                    "ppo_mini_batch_size=4",
                    "ppo_micro_batch_size_per_gpu=4",
                    "optim.lr=1e-4",
                    "optim.weight_decay=0.0001",
                    "fsdp_config.param_offload=False",
                    "fsdp_config.optimizer_offload=False",
                    "fsdp_config.model_dtype='bfloat16'",
                    "fsdp_config.dtype='bfloat16'",
                    "+fsdp_config.mixed_precision.param_dtype='bfloat16'",
                    "fsdp_config.forward_only=False",
                    "fsdp_config.fsdp_size=" + str(fsdp_size),
                    "fsdp_config.ulysses_sequence_parallel_size=" + str(cp),
                    "policy_loss.loss_mode='flow_grpo'",
                ],
            )
        actor_config: FSDPActorConfig = omega_conf_to_dataclass(cfg)

        engine_config = actor_config.engine
        optimizer_config = actor_config.optim
        checkpoint_config = actor_config.checkpoint
    else:
        raise NotImplementedError(f"strategy {strategy} is not supported")

    training_config = TrainingWorkerConfig(
        model_type=model_type,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )
    return training_config, actor_config


def create_data_samples(num_device: int, model_config: DiffusersModelConfig) -> DataProto:
    from tensordict import TensorDict

    scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_config.local_path, subfolder="scheduler"
    )
    set_timesteps(scheduler, model_config)

    batch_size = 8 * num_device
    seq_len = 64
    img_size = 512
    latent_dim = 64
    encoder_latent_dim = 3584
    inference_steps = 40
    vocab_size = 99
    vae_scale_factor = 8
    height, width = img_size, img_size
    latent_height, latent_width = height // vae_scale_factor // 2, width // vae_scale_factor // 2
    num_diffusion_steps = 10
    timesteps = scheduler.timesteps[None].repeat(batch_size, 1)

    torch.manual_seed(1)
    np.random.seed(1)

    batch = TensorDict(
        {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, inference_steps)),
            "response_mask": torch.ones((batch_size, inference_steps)),
            "old_log_probs": torch.randn((batch_size, num_diffusion_steps)),
            "advantages": torch.randn((batch_size, num_diffusion_steps)),
            "responses": torch.randn((batch_size, 3, height, width)),
            "all_latents": torch.randn((batch_size, inference_steps, latent_height * latent_width, latent_dim)),
            "rollout_log_probs": torch.randn((batch_size, num_diffusion_steps)),
            "all_timesteps": timesteps,
            "prompt_embeds": torch.randn((batch_size, seq_len, encoder_latent_dim)),
            "prompt_embeds_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
            "negative_prompt_embeds": torch.randn((batch_size, seq_len, encoder_latent_dim)),
            "negative_prompt_embeds_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
            "loss_mask": torch.ones((batch_size, inference_steps), dtype=torch.int32),
        },
        batch_size=batch_size,
    )
    data = DataProto(batch=batch)
    data.meta_info["global_token_num"] = torch.sum(data.batch["attention_mask"], dim=-1).tolist()
    data.meta_info["use_dynamic_bsz"] = False
    data.meta_info["micro_batch_size_per_gpu"] = 4
    data.meta_info["height"] = height
    data.meta_info["width"] = width

    return data


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_diffusers_fsdp_engine(strategy):
    # Create configs
    ray.init()
    device_count = torch.cuda.device_count()
    training_config, actor_config = create_training_config(
        model_type="diffusion_model",
        strategy=strategy,
        device_count=device_count,
        model="~/models/Qwen/Qwen-Image",
    )
    # init model
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=training_config)
    resource_pool = RayResourcePool(process_on_nodes=[device_count])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)  # TrainigWorker
    wg.reset()

    # forward only without loss function
    data_td = create_data_samples(device_count, training_config.model_config).to_tensordict()
    data_td = embeds_padding_2_no_padding(data_td)
    tu.assign_non_tensor(
        data_td,
        compute_loss=False,
        image_height=training_config.model_config.get("image_height", 512),
        image_width=training_config.model_config.get("image_width", 512),
        vae_scale_factor=training_config.model_config.get("vae_scale_factor", 8),
    )
    output = wg.infer_batch(data_td)
    output_dict = output.get()

    print("Output:", output_dict)
    for key in ["log_probs", "metrics"]:
        assert key in output_dict

    # forward and backward with loss function
    # set loss function
    loss_fn = partial(ppo_loss, config=actor_config)
    wg.set_loss_fn(loss_fn)

    # train batch
    data_td = create_data_samples(device_count, training_config.model_config).to_tensordict()
    data_td = embeds_padding_2_no_padding(data_td)
    ppo_mini_batch_size = 4
    ppo_epochs = actor_config.ppo_epochs
    seed = 42
    shuffle = actor_config.shuffle
    tu.assign_non_tensor(
        data_td,
        global_batch_size=ppo_mini_batch_size * device_count,
        mini_batch_size=ppo_mini_batch_size * device_count,
        epochs=ppo_epochs,
        seed=seed,
        dataloader_kwargs={"shuffle": shuffle},
        image_height=training_config.model_config.get("image_height", 512),
        image_width=training_config.model_config.get("image_width", 512),
        vae_scale_factor=training_config.model_config.get("vae_scale_factor", 8),
    )
    output = wg.train_mini_batch(data_td)
    output_dict = output.get()

    print("Output:", output_dict)
    assert "metrics" in output_dict.keys()

    ray.shutdown()
