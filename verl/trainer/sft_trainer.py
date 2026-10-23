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


import os
from functools import partial

from tensordict.tensorclass import NonTensorData

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class SFTTrainer:
    def _configure_gradient_accumulation(self, dp_size: int) -> None:
        """Resolve the gradient-accumulation contract, if the config asks for one.

        The knob is opt-in: with ``data.gradient_accumulation_steps`` unset the
        trainer keeps its historical behaviour untouched (micro-batches follow
        ``use_dynamic_bsz`` / ``max_token_len_per_gpu`` or
        ``micro_batch_size_per_gpu``, and nothing is validated), so a config that
        never asked for accumulation cannot start failing or training a different
        batch because of this code.

        When it *is* set, ``train_batch_size`` stays authoritative -- it decides
        sampling and the step count -- and must agree exactly with
        ``micro_batch_size_per_gpu * gradient_accumulation_steps *
        data_parallel_size``, so the effective batch is never silently different
        from the one requested. Dynamic batch sizing forms micro-batches from
        ``max_token_len_per_gpu`` and never consults ``micro_batch_size_per_gpu``,
        so accumulation is undefined there and is rejected.
        """
        data_config = self.config.data
        raw_gas = getattr(data_config, "gradient_accumulation_steps", None)
        if raw_gas is None:
            # Not opted in: keep the historical behaviour, but a global batch that
            # the data-parallel degree cannot tile still trains on
            # dp * floor(global / dp) samples per step -- fewer than configured.
            # The sampler's drop_last only equalises shard lengths; it cannot make
            # a non-divisible batch whole, so say so instead of leaving it silent.
            if dp_size > 1 and self.global_batch_size % dp_size != 0:
                logger.warning(
                    "[sft] train_batch_size=%s is not divisible by data_parallel_size=%s: each "
                    "step will consume %s samples (%s per rank), not the configured %s. Set "
                    "train_batch_size to a multiple of the data-parallel degree.",
                    self.global_batch_size,
                    dp_size,
                    dp_size * self.train_batch_size_per_dp,
                    self.train_batch_size_per_dp,
                    self.global_batch_size,
                )
            self.gradient_accumulation_steps = None
            return
        if isinstance(raw_gas, bool) or not float(raw_gas).is_integer():
            raise ValueError(
                f"data.gradient_accumulation_steps must be a positive integer, got {raw_gas!r}"
            )
        gas = int(raw_gas)
        if gas < 1:
            raise ValueError(f"data.gradient_accumulation_steps must be >= 1, got {gas}")
        if data_config.use_dynamic_bsz:
            raise ValueError(
                "data.gradient_accumulation_steps requires data.use_dynamic_bsz=False: "
                "dynamic batch sizing forms micro-batches from max_token_len_per_gpu and "
                "ignores micro_batch_size_per_gpu, so the accumulation contract is undefined."
            )
        micro = int(data_config.micro_batch_size_per_gpu or 1)
        expected = micro * gas * dp_size
        if self.global_batch_size != expected:
            raise ValueError(
                f"train_batch_size={self.global_batch_size} contradicts "
                f"micro_batch_size_per_gpu({micro}) * gradient_accumulation_steps({gas}) * "
                f"data_parallel_size({dp_size}) = {expected}. Set "
                "train_batch_size = micro_batch_size_per_gpu * "
                "gradient_accumulation_steps * data_parallel_size."
            )
        self.gradient_accumulation_steps = gas
        if self.rank == 0:
            logger.info(
                "[sft] global_batch=%s dp=%s per_dp=%s micro=%s gradient_accumulation_steps=%s "
                "(per-rank effective batch %s; micro-batches accumulated per optimizer step)",
                self.global_batch_size,
                dp_size,
                self.train_batch_size_per_dp,
                micro,
                gas,
                micro * gas,
            )

    def __init__(
        self,
        config,
    ):
        self.config = config

        log_gpu_memory_usage(f"rank {torch.distributed.get_rank()}: Before SFTTrainer init", logger=logger)

        self.rank = torch.distributed.get_rank()

        self._build_config()
        self._build_dataset()

        self._build_engine()

        self._build_dataloader()

        self._init_engine()

        self._build_ckpt_handler()

        # Initialize resume-related variables
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        if self.rank == 0:
            print(self.config)

        log_gpu_memory_usage(f"rank {self.rank}: After SFTTrainer init", logger=logger)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)
        lora_train_meta = self._get_lora_train_meta()

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
            lora_train_meta=lora_train_meta,
        )

    def _get_lora_train_meta(self):
        lora_adapter_path = self.config.model.get("lora_adapter_path", None)
        lora_rank = int(getattr(self.config.model, "lora_rank", 0) or 0)

        if lora_adapter_path is None and lora_rank <= 0:
            return None

        raw_lora_alpha = self.config.model.get("lora_alpha", None)
        if raw_lora_alpha is None:
            log_with_rank(
                "LoRA is enabled but `model.lora_alpha` is not set; fallback to 0 in checkpoint metadata.",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            lora_alpha = 0
        else:
            lora_alpha = int(raw_lora_alpha)
            if lora_alpha == 0:
                log_with_rank(
                    "LoRA is enabled but `model.lora_alpha` is 0; this may lead to ineffective LoRA scaling.",
                    logger=logger,
                    rank=self.rank,
                    level=logging.WARNING,
                    log_only_rank_0=True,
                )

        task_type = self.config.model.get("task_type", None)
        if task_type is None:
            task_type = "CAUSAL_LM"

        return {
            "r": lora_rank,
            "lora_alpha": int(lora_alpha or 0),
            "task_type": str(task_type),
        }

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)

        # check profile interval
        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()

    def _validate_profiler_interval(self):
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        # Distillation replaces CE when a distillation config is present: the engine
        # then asks this object for the per-position divergence (it owns the logits)
        # and reduces it afterwards, exactly like sft_loss does for CE.
        # Teacher distributions for distillation. They are keyed by the dataset's
        # `sample` column and attached to each batch here, because they are ragged
        # [tokens, topk] payloads that the generic collator cannot nest: routing them
        # through the dataset turned them into empty non-tensor stacks.
        self.teacher_data = None
        if self.config.trainer.get("teacher_data", None) is not None:
            # Registered process-wide: the engine rebuilds the loss object, so path or
            # payload state has to live outside the instance to reach it.
            from verl.workers.utils.distill_sft_loss import set_teacher_data

            set_teacher_data(str(self.config.trainer.teacher_data))
            raw = torch.load(str(self.config.trainer.teacher_data), map_location="cpu")
            # normalize to {key: (ids, logps)}
            self.teacher_data = {
                k: (v["ids"] if isinstance(v, dict) else v[0], v["logps"] if isinstance(v, dict) else v[1])
                for k, v in raw.items()
            }
        self.distillation_config = None
        if self.config.get("distillation", None) is not None:
            from verl.workers.config import DistillationConfig
            from verl.workers.utils.distill_sft_loss import SFTDistillationLoss

            self.distillation_config = (
                self.config.distillation
                if isinstance(self.config.distillation, DistillationConfig)
                else DistillationConfig(**dict(self.config.distillation))
            )
            self.loss_fn = SFTDistillationLoss(
                self.distillation_config,
                teacher_lookup=self.teacher_data,
                teacher_data_path=(
                    str(self.config.trainer.teacher_data)
                    if self.config.trainer.get("teacher_data", None) is not None
                    else None
                ),
            )
        else:
            self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        # Note that in SPMD world, this abstraction has to break
        self.engine = self.training_client.engine

    def _init_engine(self):
        # patch optimizer config
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.training_client.reset()

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self._configure_gradient_accumulation(dp_size)
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            # With gradient accumulation the per-DP training batch can exceed the
            # per-rank validation shard; the loader would then yield zero batches and
            # the trainer would average an empty list (torch.mean(torch.tensor([])) is
            # NaN). micro_batch_size_per_gpu doubles as the val batch size (see
            # sft_trainer_engine.yaml), so bound it there. Without accumulation the
            # historical batch size is kept exactly.
            val_batch_size = self.train_batch_size_per_dp
            if self.gradient_accumulation_steps and self.gradient_accumulation_steps > 1:
                val_batch_size = max(1, min(int(config.data.micro_batch_size_per_gpu or 1),
                                            self.train_batch_size_per_dp))
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

    def _get_batch_seqlens(self, data):
        # mean over dp group
        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        batch_seqlens = batch_seqlens.to(self.device_name)  # (global_bsz // dp)
        if self.engine.get_data_parallel_size() > 1:
            output_tensor = torch.empty(
                (batch_seqlens.shape[0] * self.engine.get_data_parallel_size(),),
                dtype=batch_seqlens.dtype,
                device=self.device_name,
            )  # (global_bsz,)

        dp_group = self.engine.get_data_parallel_group()
        # Take the size from the group itself. Under context parallelism the engine's
        # data-parallel size and the group it hands to the collective can differ (the
        # group may include the CP dimension), and a buffer sized by the wrong one makes
        # all_gather_into_tensor hang instead of failing.
        dp_size = torch.distributed.get_world_size(dp_group) if dp_group is not None else 1


        if dp_size == 1 or dp_group is None:
            return batch_seqlens.tolist()

        # Context parallelism: this gather only feeds the token-count metrics, and the
        # collective over the data-parallel group did not complete on this stack once CP
        # was on (consistent group and sizes on every rank, still a PCCP broadcast
        # timeout). Metrics stay rank-local rather than blocking training.
        try:
            import megatron.core.parallel_state as _mpu

            if _mpu.get_context_parallel_world_size() > 1:
                return batch_seqlens.tolist()
        except Exception:  # noqa: BLE001 - not a megatron run
            pass

        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * dp_size,),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )  # (global_bsz,)

        torch.distributed.all_gather_into_tensor(
            output_tensor=output_tensor,
            input_tensor=batch_seqlens,
            group=dp_group,
        )

        batch_seqlens = output_tensor.tolist()
        return batch_seqlens

    def fit(self):
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

        # TODO: add a unified tracking
        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps or 1,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }
        if self.distillation_config is not None:
            # keys MegatronEngine's forward off its distillation branch
            meta_info["distillation_use_topk"] = True

        train_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"rank {self.rank}: At start of epoch {epoch}", logger=logger)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                # construct tensordict
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data)
                # this is necessary. Otherwise, it is interpreted as NonTensorStack
                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)

                # start profile in SPMD mode
                if global_step == self.start_profile_step:
                    self.training_client.start_profile()
                # train for on batch
                output = self.training_client.train_batch(data=data)
                # SFT has one train_batch per step (no PPO-style mini-batch loop), so advancing
                # the profiler here is the per-step boundary; it also drives a torch.profiler
                # schedule when one is configured (its unit is one such training step).
                self.training_client.step_profile()

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = tu.get(output, "metrics")

                    # TODO: we can actual accumulate metrics for N steps and perform aggregate metrics
                    for k in ["loss", "grad_norm", "lr", "mfu"]:
                        if k in metrics.keys():
                            value = metrics.pop(k)
                            metrics[f"train/{k}"] = value

                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9

                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                # early exit or validation step
                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)

                        if self.engine.is_mp_src_rank_with_outputs():
                            metrics = tu.get(output, "metrics")
                            val_losses.append(metrics["loss"])

                    if self.engine.is_mp_src_rank_with_outputs():
                        val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
                        # average over data parallel group
                        dp_group = self.engine.get_data_parallel_group()
                        if dp_group is not None:
                            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)

                    if is_logging:
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    aggressive_empty_cache(force_sync=True)
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


    def dump_teacher_topk(self) -> None:
        """Forward-only pass that exports the model's top-k next-token distribution.

        On-policy self-distillation reads the teacher (this same model, under a
        privileged context) at the student's rollout positions. Those positions
        usually live in a different dataset, so the distribution is exported once
        and replayed by the training run instead of being recomputed in-step.

        The dump is self-checked against the engine's own ``log_probs``: for every
        position whose true next token falls inside the exported top-k, the
        exported log-prob must equal ``log_probs`` exactly. That is what separates
        "a file was written" from "the distribution is the model's".
        """
        from verl.workers.utils.teacher_dump import DUMP_IDS_KEY, TeacherTopKDump

        cfg = self.config.trainer.teacher_dump
        out_path = str(cfg.path)
        topk = int(cfg.topk)
        max_steps = int(cfg.get("max_steps", 0)) or None

        dump = TeacherTopKDump(topk)
        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "gradient_accumulation_steps": 1,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
            # The engine keys its distillation branch off this non-tensor flag.
            "distillation_use_topk": True,
        }

        save_tensors = bool(cfg.get("save_tensors", False))
        is_src = self.engine.is_mp_src_rank_with_outputs()
        records = []
        payload_tensors = []
        for step, raw in enumerate(self.train_dataloader):
            data = tu.get_tensordict(tensor_dict=raw, non_tensor_dict=meta_info)
            batch_seqlens = self._get_batch_seqlens(data=data)
            tu.assign_non_tensor(data, update_lr_scheduler=False, global_token_num=NonTensorData(batch_seqlens))
            out = self.engine.infer_batch(data, loss_function=dump)
            model_output = ((out or {}).get("model_output") or {}) if is_src else {}

            if save_tensors:
                from verl.workers.utils.teacher_dump import capture_dump_tensors

                # Every data-parallel rank holds a different slice of the batch, so a
                # dump that kept only the source rank's share would silently export
                # 1/dp_size of the corpus. The collective runs on every rank — the ones
                # without outputs contribute an empty capture — and costs one gather per
                # step, which is nothing next to the forward.
                captured = (
                    capture_dump_tensors(model_output, data["input_ids"])
                    if is_src
                    else {"ids": None, "empty_rank": True}
                )
                world = torch.distributed.get_world_size()
                gathered = [None] * world
                torch.distributed.all_gather_object(gathered, captured)
                payload_tensors.extend(g for g in gathered if g is not None and not g.get("empty_rank"))

            if is_src:
                if DUMP_IDS_KEY not in model_output:
                    raise RuntimeError(
                        "the engine produced no top-k: the batch lost its distillation_use_topk flag "
                        "or the logits processor was not called"
                    )
                stats = self._dump_consistency_stats(model_output, data)
                stats["step"] = step
                stats["topk"] = topk
                records.append(stats)
                log_with_rank(
                    f"teacher dump step {step}: coverage={stats['coverage']:.4f} "
                    f"max|dump - log_probs|={stats['max_abs_diff']:.3e}",
                    logger=logger,
                    rank=0,
                    log_only_rank_0=True,
                )
            if max_steps is not None and step + 1 >= max_steps:
                break

        if is_src:
            payload = {
                "records": records,
                "tensors": payload_tensors,
                "topk": topk,
                "model_path": self.config.model.path,
                "pad_mode": str(self.config.data.pad_mode),
                "use_remove_padding": self.config.model.use_remove_padding,
            }
            torch.save(payload, out_path)
            log_with_rank(f"teacher dump written to {out_path}", logger=logger, rank=0, log_only_rank_0=True)

    @staticmethod
    def _dump_consistency_stats(model_output, data) -> dict:
        """Compare the exported top-k against the engine's own log-probs."""
        from verl.workers.utils.teacher_dump import dump_consistency_stats

        return dump_consistency_stats(model_output, data["input_ids"])

def run_sft(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = SFTTrainer(config=config)
    if config.trainer.get("teacher_dump", None) is not None:
        # Forward-only export of the model's top-k distribution (teacher replay).
        trainer.dump_teacher_topk()
    else:
        trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to multi-turn dataset
        dataset_cls = MultiTurnSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples
    )
    return dataset


if __name__ == "__main__":
    main()
