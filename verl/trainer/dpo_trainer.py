# Copyright 2026 Amazon.com Inc and/or its affiliates
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
Offline DPO trainer (SPMD, one process per GPU), built like ``sft_trainer.py``.

Data flow for one step:
  1. DPODataset packs each (chosen, rejected) pair into one row with a ``boundary_offset``.
  2. The reference model scores every chosen and rejected sequence: the policy with its LoRA adapter
     disabled, a separate forward-only worker for full fine-tuning, or nothing when reference_free.
  3. The policy trains on the packed rows. Micro-batching sees whole pairs, so it cannot split or
     reorder a pair; ``prepare_micro_batches`` then splits each micro-batch into interleaved
     ``[chosen_0, rejected_0, ...]`` sequences (the ``dpo_expand_packed_sequences`` flag), and
     ``dpo_loss`` reads even rows as chosen and odd rows as rejected.
"""

import dataclasses
import logging
import os
from functools import partial

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra
import torch
import torch.distributed
from omegaconf import OmegaConf
from tensordict.tensorclass import NonTensorData
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.trainer.dpo.dpo_losses import DPOLossConfig
from verl.trainer.dpo.utils import (
    DPO_EXPAND_FLAG,
    expand_packed_sequences,
    get_boundary_offsets,
    sum_logps_per_sequence,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.dataset_utils import DPOTensorCollator
from verl.utils.dataset.dpo_dataset import DPODataConfig, DPODataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.losses import dpo_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_DPO_LOGGING_LEVEL", "WARN"))


# ─── Batch helpers (no distributed state, unit-tested on CPU) ──────────────


def prepare_packed_batch(data: dict) -> dict:
    """Return the packed tensors of a collated DPO batch, with position ids for each packed row.

    Position ids here only label the packed rows for micro-batching; the expansion after
    micro-batching gives every chosen and rejected sequence its own ids starting at 0.
    """
    input_ids = data["input_ids"]
    if not input_ids.is_nested:
        raise NotImplementedError("The DPO trainer requires data.pad_mode=no_padding")
    lengths = input_ids.offsets().diff().tolist()
    position_ids = torch.nested.as_nested_tensor(
        [torch.arange(n, dtype=torch.long) for n in lengths], layout=torch.jagged
    )
    return {
        "input_ids": input_ids,
        "loss_mask": data["loss_mask"],
        "position_ids": position_ids,
        "boundary_offset": data["boundary_offset"],
    }


def expanded_seqlens(packed: dict) -> list[int]:
    """Lengths of the 2B chosen and rejected sequences, interleaved, from a packed batch."""
    lengths = packed["input_ids"].offsets().diff().tolist()
    offsets = get_boundary_offsets(packed)
    return [n for total, off in zip(lengths, offsets, strict=True) for n in (off, total - off)]


def reduce_dpo_metrics(metrics: dict) -> dict:
    """Combine the per-micro-batch ``dpo/*`` metrics returned by TrainingWorker.

    ``dpo_loss`` pre-scales each micro-batch value like the loss, so the global value is the sum over
    micro-batches averaged over data-parallel ranks. Other metrics are returned unchanged.
    """
    reduced = {}
    for key, value in metrics.items():
        if key.startswith("dpo/") and isinstance(value, list):
            if value and isinstance(value[0], list):
                value = sum(sum(rank) for rank in value) / len(value)
            else:
                value = sum(value)
        reduced[key] = value
    return reduced


# ─── Trainer ───────────────────────────────────────────────────────────────


class DPOTrainer:
    def __init__(self, config):
        self.config = config

        log_gpu_memory_usage(f"rank {torch.distributed.get_rank()}: Before DPOTrainer init", logger=logger)
        self.rank = torch.distributed.get_rank()

        self._build_config()
        self._build_dataset()
        self._build_engine()
        self._build_dataloader()
        self._init_engine()
        self._build_ckpt_handler()

        self.resume_global_step = self.ckpt_handler.load_checkpoint()
        self.device_name = self.config.trainer.device

        if self.rank == 0:
            print(self.config)
        log_gpu_memory_usage(f"rank {self.rank}: After DPOTrainer init", logger=logger)

    def _build_config(self):
        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)
        self.data_config = omega_conf_to_dataclass(self.config.data, DPODataConfig)
        trainer = self.config.trainer
        self.loss_config = DPOLossConfig(
            loss_type=trainer.get("loss_type", "sigmoid"),
            beta=trainer.get("beta", 0.1),
            label_smoothing=trainer.get("label_smoothing", 0.0),
            reference_free=trainer.get("reference_free", False),
        )
        if self.data_config.pad_mode != "no_padding":
            raise NotImplementedError("The DPO trainer requires data.pad_mode=no_padding")

        self.is_lora = self.model_config.lora_rank > 0 or self.model_config.get("lora_adapter_path") is not None

        self.profiler_interval = trainer.profile_interval
        assert len(self.profiler_interval) == 2
        self.start_profile_step, self.end_profile_step = self.profiler_interval
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_dataset(self):
        tokenizer = self.model_config.tokenizer
        self.train_dataset = create_dpo_dataset(
            self.data_config.train_files, self.data_config, tokenizer, max_samples=self.data_config.train_max_samples
        )
        self.val_dataset = None
        if self.data_config.val_files:
            self.val_dataset = create_dpo_dataset(
                self.data_config.val_files, self.data_config, tokenizer, max_samples=self.data_config.val_max_samples
            )

    def _worker_config(self, engine_config):
        from verl.workers.engine_workers import TrainingWorkerConfig

        return TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

    def _build_engine(self):
        self.loss_fn = partial(dpo_loss, config=self.loss_config)
        self.training_client = TrainingWorker(config=self._worker_config(self.engine_config))
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        # Note that in SPMD world, this abstraction has to break
        self.engine = self.training_client.engine

        # Full fine-tuning needs a frozen copy of the initial model as the reference. With LoRA the
        # policy itself is the reference once its adapter is disabled.
        self.ref_client = None
        if not self.is_lora and not self.loss_config.reference_free:
            ref_engine_config = dataclasses.replace(self.engine_config, forward_only=True)
            self.ref_client = TrainingWorker(config=self._worker_config(ref_engine_config))
            self.ref_client.reset()

    def _build_dataloader(self):
        device_name = get_device_name()
        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.global_batch_size = self.data_config.train_batch_size
        if self.global_batch_size % dp_size != 0:
            raise ValueError(f"data.train_batch_size {self.global_batch_size} must be divisible by dp size {dp_size}")
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = DPOTensorCollator(self.data_config.pad_mode)

        def make_loader(dataset, shuffle):
            sampler = DistributedSampler(dataset, shuffle=shuffle, num_replicas=dp_size, rank=dp_rank, drop_last=True)
            loader = StatefulDataLoader(
                dataset=dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=sampler,
                collate_fn=self.collate_fn,
                num_workers=self.data_config.dataloader_num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
            return sampler, loader

        self.train_sampler, self.train_dataloader = make_loader(self.train_dataset, shuffle=True)
        self.val_dataloader = None
        if self.val_dataset:
            _, self.val_dataloader = make_loader(self.val_dataset, shuffle=False)

    def _init_engine(self):
        self.steps_per_epoch = len(self.train_dataloader)
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch
        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.training_client.reset()

        # global_batch_size counts pairs; dpo_loss uses it to normalise the loss.
        self.meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.data_config.use_dynamic_bsz,
            "max_token_len_per_gpu": self.data_config.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.data_config.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.data_config.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

    def _build_ckpt_handler(self):
        trainer = self.config.trainer
        lora_train_meta = None
        if self.is_lora:
            lora_train_meta = {
                "r": int(self.config.model.get("lora_rank", 0) or 0),
                "lora_alpha": int(self.config.model.get("lora_alpha", 0) or 0),
                "task_type": str(self.config.model.get("task_type", None) or "CAUSAL_LM"),
            }
        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=trainer.default_local_dir,
            max_ckpt_to_keep=trainer.get("max_ckpt_to_keep", None),
            default_hdfs_dir=trainer.get("default_hdfs_dir", None),
            resume_mode=trainer.get("resume_mode", "auto"),
            resume_from_path=trainer.get("resume_from_path", None),
            lora_train_meta=lora_train_meta,
        )

    # ─── DPO steps ────────────────────────────────────────────────────────

    def _gather_seqlens(self, seqlens: list[int]) -> list[int]:
        """All-gather per-sequence lengths over the data-parallel group (used for MFU)."""
        dp_group = self.engine.get_data_parallel_group()
        dp_size = self.engine.get_data_parallel_size()
        if dp_size == 1 or dp_group is None:
            return seqlens
        local = torch.tensor(seqlens, dtype=torch.long, device=self.device_name)
        output = torch.empty(local.shape[0] * dp_size, dtype=local.dtype, device=self.device_name)
        torch.distributed.all_gather_into_tensor(output_tensor=output, input_tensor=local, group=dp_group)
        return output.tolist()

    def _broadcast_from_sp_src(self, value, numel: int) -> torch.Tensor:
        """Only the first rank of a sequence-parallel group receives worker outputs; share them with the group."""
        mesh = getattr(self.engine, "ulysses_device_mesh", None)
        if mesh is None or mesh["sp"].size() == 1:
            return value
        group = mesh["sp"].get_group()
        if value is None:
            buf = torch.empty(numel, dtype=torch.float32, device=self.device_name)
        else:
            buf = value.to(device=self.device_name, dtype=torch.float32).contiguous()
        torch.distributed.broadcast(buf, group=group, group_src=0)
        return buf.cpu()

    def _compute_ref_logps(self, packed: dict):
        """Summed reference log-probs of every chosen and rejected sequence, as a (B, 2) tensor."""
        if self.loss_config.reference_free:
            return None
        expanded = expand_packed_sequences(
            list(packed["input_ids"].unbind()), list(packed["loss_mask"].unbind()), get_boundary_offsets(packed)
        )
        ref_data = tu.get_tensordict(tensor_dict=expanded, non_tensor_dict=self.meta_info)
        tu.assign_non_tensor(ref_data, compute_loss=False)

        if self.ref_client is not None:
            output = self.ref_client.infer_batch(ref_data)
        else:
            tu.assign_non_tensor(ref_data, no_lora_adapter=True)
            output = self.training_client.infer_batch(ref_data)

        per_seq = None
        if output is not None:
            per_seq = sum_logps_per_sequence(output["log_probs"], expanded["loss_mask"]).float()
        num_seqs = expanded["input_ids"].offsets().numel() - 1
        per_seq = self._broadcast_from_sp_src(per_seq, num_seqs)
        return per_seq.detach().reshape(-1, 2)

    def _build_policy_batch(self, data: dict, update_lr_scheduler: bool):
        packed = prepare_packed_batch(data)
        ref_logps = self._compute_ref_logps(packed)
        if ref_logps is not None:
            packed["ref_logps"] = ref_logps
        batch = tu.get_tensordict(tensor_dict=packed, non_tensor_dict=self.meta_info)
        seqlens = self._gather_seqlens(expanded_seqlens(packed))
        # NonTensorData keeps the list as one value instead of a NonTensorStack.
        tu.assign_non_tensor(
            batch,
            update_lr_scheduler=update_lr_scheduler,
            global_token_num=NonTensorData(seqlens),
            **{DPO_EXPAND_FLAG: True},
        )
        return batch, seqlens

    def _validate(self) -> dict:
        totals, count = {}, 0
        for val_data in self.val_dataloader:
            batch, _ = self._build_policy_batch(val_data, update_lr_scheduler=False)
            tu.assign_non_tensor(batch, compute_loss=True)
            output = self.training_client.infer_batch(batch)
            if self.engine.is_mp_src_rank_with_outputs():
                metrics = reduce_dpo_metrics(tu.get(output, "metrics"))
                for key in ("loss", *[k for k in metrics if k.startswith("dpo/")]):
                    totals[key] = totals.get(key, 0.0) + float(metrics[key])
                count += 1
        if not count:
            return {}
        return {f"val/{key}": value / count for key, value in totals.items()}

    def fit(self):
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0
        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step
        last_valid_metric = None
        log_with_rank(f"Total training steps: {self.total_training_steps}", logger=logger, rank=0, log_only_rank_0=True)
        start_epoch = global_step // self.steps_per_epoch
        total_tokens = 0

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            aggressive_empty_cache(force_sync=True)

            for data in tqdm(
                self.train_dataloader,
                initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=not is_logging,
            ):
                global_step += 1
                batch, seqlens = self._build_policy_batch(data, update_lr_scheduler=True)

                if global_step == self.start_profile_step:
                    self.training_client.start_profile()
                output = self.training_client.train_batch(data=batch)
                self.training_client.step_profile()
                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = reduce_dpo_metrics(tu.get(output, "metrics"))
                    for key in [k for k in metrics if k in ("loss", "grad_norm", "lr", "mfu") or k.startswith("dpo/")]:
                        metrics[f"train/{key}"] = metrics.pop(key)
                    metrics["train/global_tokens"] = sum(seqlens)
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9
                    if is_logging:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = self.test_freq > 0 and global_step % self.test_freq == 0
                is_save_step = self.save_freq > 0 and global_step % self.save_freq == 0

                if self.val_dataloader is not None and (is_last_step or is_valid_step):
                    val_metrics = self._validate()
                    if is_logging:
                        tracking.log(data=val_metrics, step=global_step)
                        last_valid_metric = val_metrics
                    torch.distributed.barrier()

                if is_last_step or is_save_step:
                    aggressive_empty_cache(force_sync=True)
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def create_dpo_dataset(data_paths, data_config: DPODataConfig, tokenizer, max_samples=-1):
    """Create the DPO dataset, or the class named by data.custom_cls."""
    dataset_cls = DPODataset
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
    return dataset_cls(data_files=data_paths, tokenizer=tokenizer, config=data_config, max_samples=max_samples)


def run_dpo(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = DPOTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_dpo(config)


if __name__ == "__main__":
    main()
