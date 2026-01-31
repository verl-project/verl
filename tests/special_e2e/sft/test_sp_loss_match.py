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

import torch
import torch.distributed
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils.distributed import initialize_global_process_group
from verl.workers.engine.utils import enable_full_determinism


def test_trainer_forward_consistency(trainer: FSDPSFTTrainer, total_steps: int = 4):
    """Test consistency between original forward pass and SP+rmpad forward passes.

    Args:
        trainer: The FSDPSFTTrainer instance to test
        total_steps: Number of steps to test (default: 4)
    """
    if trainer.device_mesh.get_rank() == 0:
        print("\nStarting debug comparison between original and SP+rmpad forward passes...")
        print(f"Sequence parallel size: {trainer.config.ulysses_sequence_parallel_size}")
        print(f"Remove padding: {trainer.use_remove_padding}\n")

    steps_remaining = total_steps
    for epoch in range(1):  # Just one epoch for testing
        trainer.train_sampler.set_epoch(epoch=epoch)
        for data in trainer.train_dataloader:
            data = TensorDict(data, batch_size=trainer.config.data.train_batch_size).cuda()
            trainer.fsdp_model.train()
            micro_batches = data.split(trainer.config.data.micro_batch_size_per_gpu)

            for idx, micro_batch in enumerate(micro_batches):
                if trainer.device_mesh.get_rank() == 0:
                    print(f"\nProcessing micro batch {idx + 1}/{len(micro_batches)}")

                # Compute losses using both methods
                # Disable SP and rmpad
                trainer.use_remove_padding = False
                old_sp = trainer.config.ulysses_sequence_parallel_size
                trainer.config.ulysses_sequence_parallel_size = 1
                loss_ref = trainer._compute_loss_and_backward(micro_batch.copy(), do_backward=False)

                # Do SP and rmpad
                trainer.config.ulysses_sequence_parallel_size = old_sp
                trainer.use_remove_padding = True
                loss_sp = trainer._compute_loss_and_backward(micro_batch.copy(), do_backward=False)

                # Collect losses across all ranks
                loss_ref_all = loss_ref.clone()
                loss_sp_all = loss_sp.clone()
                torch.distributed.all_reduce(loss_ref_all, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(loss_sp_all, op=torch.distributed.ReduceOp.AVG)

                # Calculate relative difference of averaged losses
                rel_diff = torch.abs(loss_ref_all - loss_sp_all) / (torch.abs(loss_ref_all) + 1e-8)
                torch.testing.assert_close(
                    loss_ref_all, loss_sp_all, rtol=0.0, atol=0.0, msg="Losses from the two methods do not match!"
                )

                if trainer.device_mesh.get_rank() == 0:
                    print("\nComparison Results (Averaged across ranks):")
                    print(f"Reference Loss: {loss_ref_all.item():.6f}")
                    print(f"SP+rmpad Loss: {loss_sp_all.item():.6f}")
                    print(f"Relative Difference: {rel_diff.item():.6f}")

                    assert rel_diff.item() < 1e-2, "Significant difference detected between averaged losses!"
                    print("Loss difference is within the acceptable range.")

                steps_remaining -= 1
                if steps_remaining == 0:
                    break
            if steps_remaining == 0:
                break
        break

    if trainer.device_mesh.get_rank() == 0:
        print("\nDebug comparison completed successfully.")


def _check_backward_determinism(trainer: FSDPSFTTrainer, micro_batch: TensorDict, mode_label: str):
    if trainer.device_mesh.get_rank() == 0:
        print(f"\nChecking backward determinism ({mode_label})...")

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()

    trainer.fsdp_model.zero_grad(set_to_none=True)
    trainer._compute_loss_and_backward(micro_batch.copy(), do_backward=True)
    grads_1 = {}
    for name, param in trainer.fsdp_model.named_parameters():
        grads_1[name] = None if param.grad is None else param.grad.detach().clone()

    trainer.fsdp_model.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_states)
    trainer._compute_loss_and_backward(micro_batch.copy(), do_backward=True)
    grads_2 = {}
    for name, param in trainer.fsdp_model.named_parameters():
        grads_2[name] = None if param.grad is None else param.grad.detach().clone()

    for name in grads_1:
        g1 = grads_1[name]
        g2 = grads_2[name]
        assert g1 is not None, f"Grad missing for {name} in first backward"
        assert g2 is not None, f"Grad missing for {name} in second backward"
        torch.testing.assert_close(g1, g2, rtol=0.0, atol=0.0, msg=f"Grad mismatch for {name}")


def test_trainer_backward_determinism(trainer: FSDPSFTTrainer):
    """Test backward determinism for both ref and SP+rmpad modes on one micro-batch."""
    if trainer.device_mesh.get_rank() == 0:
        print("\nStarting backward determinism check...")

    trainer.train_sampler.set_epoch(epoch=0)
    for data in trainer.train_dataloader:
        data = TensorDict(data, batch_size=trainer.config.data.train_batch_size).cuda()
        trainer.fsdp_model.train()
        micro_batch = data.split(trainer.config.data.micro_batch_size_per_gpu)[0]

        old_sp = trainer.config.ulysses_sequence_parallel_size

        # Ref path: no SP, no rmpad
        trainer.use_remove_padding = False
        trainer.config.ulysses_sequence_parallel_size = 1
        _check_backward_determinism(trainer, micro_batch, mode_label="ref (no SP, no rmpad)")

        # SP + rmpad path
        trainer.config.ulysses_sequence_parallel_size = old_sp
        trainer.use_remove_padding = True
        _check_backward_determinism(trainer, micro_batch, mode_label="SP + rmpad")
        break

    if trainer.device_mesh.get_rank() == 0:
        print("\nBackward determinism check completed successfully.")


def create_trainer(config):
    """Create and initialize a trainer instance with the given config.

    Args:
        config: Configuration object with training parameters

    Returns:
        FSDPSFTTrainer: Initialized trainer instance
    """
    local_rank, rank, world_size = initialize_global_process_group()

    seed = 42
    enable_full_determinism(seed=seed)

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp")
    )

    # build tokenizer and datasets first
    from verl.trainer.fsdp_sft_trainer import create_sft_dataset
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(
        config.data.train_files, config.data, tokenizer, max_samples=config.data.get("train_max_samples", -1)
    )
    val_dataset = create_sft_dataset(
        config.data.val_files, config.data, tokenizer, max_samples=config.data.get("val_max_samples", -1)
    )

    return FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


def main(config):
    """Main function to run trainer tests.

    Args:
        config: Configuration object with training parameters
    """
    trainer = create_trainer(config)
    test_trainer_forward_consistency(trainer)
    test_trainer_backward_determinism(trainer)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../../../verl/trainer/config", config_name="sft_trainer")
    def hydra_entry(cfg: DictConfig) -> None:
        main(cfg)

    hydra_entry()
