"""Atropos-integrated GRPO trainer for verl.

Subclasses RayPPOTrainer and overrides fit() to pull pre-scored rollouts from
the Atropos trajectory API instead of generating rollouts internally.
"""

import os

import torch
from tqdm import tqdm

from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer

from .atropos_client import AtroposClient
from .atropos_data import scored_data_to_dataproto


class RayAtroposTrainer(RayPPOTrainer):
    """Trainer that pulls pre-scored rollouts from the Atropos trajectory API.

    Extends RayPPOTrainer, overriding:
    - _create_dataloader: bypassed (data comes from Atropos, not a dataset)
    - _save_checkpoint: simplified (no dataloader state to persist)
    - fit: custom training loop polling the Atropos API instead of generating rollouts
    """

    def _load_checkpoint(self):
        """Load checkpoint, skipping dataloader state (no dataloader in Atropos mode)."""
        from verl.trainer.ppo.ray_trainer import find_latest_ckpt_path

        if self.config.trainer.resume_mode == "disable":
            return

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str)
            assert "global_step_" in self.config.trainer.resume_from_path
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        print(f"Resuming from {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

    def _save_checkpoint(self):
        """Save checkpoint without dataloader state."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        local_mkdir_safe(local_global_step_folder)

        # latest checkpointed iteration tracker
        ckpt_cfg = self.config.actor_rollout_ref.actor.checkpoint
        if ckpt_cfg.get("async_save", False):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """Skip dataset creation — data comes from Atropos trajectory API."""
        from omegaconf import OmegaConf, open_dict

        self.train_dataloader = None
        self.val_dataloader = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        total_training_steps = self.config.trainer.total_training_steps
        if total_training_steps is None:
            raise ValueError(
                "trainer.total_training_steps must be set for Atropos training "
                "(there is no train dataloader to infer step count from)"
            )
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config: {e}")

    def fit(self):
        """Training loop that polls Atropos for scored data and runs GRPO updates."""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        atropos_cfg = self.config.get("atropos", {})
        api_url = atropos_cfg.get("api_url", "http://localhost:8000")
        poll_timeout = atropos_cfg.get("poll_timeout", 300)
        proxy_url = atropos_cfg.get("proxy_url", None)
        drain_timeout = atropos_cfg.get("proxy_drain_timeout", 300)
        client = AtroposClient(api_url=api_url)

        # sync client for pause/resume calls to the generate proxy.
        # the trainer must drain in-flight requests before sleeping vLLM,
        # otherwise active CUDA operations crash when GPU memory is freed.
        # httpx timeout = drain_timeout + 10s headroom so the proxy always
        # responds (with 504 on timeout) before the client gives up.
        _proxy_client = None
        if proxy_url:
            import httpx

            _proxy_client = httpx.Client(timeout=httpx.Timeout(drain_timeout + 10, connect=10))

        max_prompt_length = self.config.data.max_prompt_length
        max_response_length = self.config.data.max_response_length
        # +1 because atropos base class uses >= (exclusive upper bound)
        max_token_len = max_prompt_length + max_response_length + 1
        group_size = self.config.actor_rollout_ref.rollout.n

        # batch_size = total sequences per training step (prompts × group_size)
        atropos_batch_size = self.config.data.train_batch_size * group_size
        client.register_trainer(
            batch_size=atropos_batch_size,
            max_token_len=max_token_len,
            num_steps=self.total_training_steps,
            checkpoint_dir=self.config.trainer.default_local_dir,
            save_checkpoint_interval=self.config.trainer.save_freq,
            wandb_project=self.config.trainer.project_name,
        )

        self.global_steps = 0

        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # signal that init is complete by writing the internal vLLM address.
        # the launch script waits for this, then starts the generate proxy
        # pointing at this address.
        ready_file = atropos_cfg.get("ready_file", None)
        if ready_file:
            vllm_addresses = self.async_rollout_manager.server_addresses
            with open(ready_file, "w") as f:
                f.write(",".join(vllm_addresses))

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Atropos Training")

        self.global_steps += 1

        for step in range(self.global_steps, self.total_training_steps + 1):
            self.global_steps = step
            is_last_step = step >= self.total_training_steps
            metrics = {}
            timing_raw = {}

            # finalize any pending async operations from previous step
            if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

            with marked_timer("step", timing_raw):
                # step 1: poll atropos API for scored batch
                with marked_timer("atropos_poll", timing_raw, "yellow"):
                    scored_data_list = client.poll_batch(timeout=poll_timeout)
                    metrics["atropos/batch_groups"] = len(scored_data_list)

                    all_scores = []
                    groups_with_identical_scores = 0
                    total_sequences = 0
                    for sd in scored_data_list:
                        scores = sd.get("scores", [])
                        all_scores.extend(scores)
                        total_sequences += len(scores)
                        if len(set(scores)) <= 1:
                            groups_with_identical_scores += 1
                    metrics["atropos/total_sequences"] = total_sequences
                    if all_scores:
                        metrics["atropos/mean_score"] = sum(all_scores) / len(all_scores)
                    if scored_data_list:
                        metrics["atropos/identical_score_rate"] = groups_with_identical_scores / len(scored_data_list)

                # step 2: convert ScoredData → DataProto
                with marked_timer("data_convert", timing_raw, "cyan"):
                    batch = scored_data_to_dataproto(
                        scored_data_list,
                        max_prompt_length=max_prompt_length,
                        max_response_length=max_response_length,
                        pad_token_id=self.tokenizer.pad_token_id or 0,
                    )

                batch.batch["response_mask"] = compute_response_mask(batch)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                response_lengths = batch.batch["response_mask"].sum(dim=-1).float()
                metrics["atropos/avg_response_length"] = response_lengths.mean().item()
                metrics["atropos/empty_response_count"] = (response_lengths == 0).sum().item()

                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # drain in-flight proxy requests before sleeping vLLM — the env
                # generates asynchronously, so requests from the NEXT batch may be
                # in-flight when we receive the current batch. sleeping vLLM while
                # generation is active frees GPU memory mid-GEMM → CUBLAS crash.
                if _proxy_client:
                    resp = _proxy_client.post(f"{proxy_url}/pause")
                    resp.raise_for_status()
                    print(f"proxy paused: {resp.json()}")

                try:
                    # ensure replicas are in training mode before forward passes
                    self.checkpoint_manager.sleep_replicas()

                    # step 3: compute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        metrics.update(
                            {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                        )
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # step 4: compute ref_log_probs if needed
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # steps 5-6: apply KL penalty and compute advantages
                    with marked_timer("adv", timing_raw, "brown"):
                        batch.batch["token_level_scores"] = batch.batch["token_level_rewards"].clone()

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # step 7: update actor
                    with marked_timer("update_actor", timing_raw, "red"):
                        actor_output = self._update_actor(batch)

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    gradient_norm = metrics.get("actor/grad_norm", None)
                    metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                    # step 8: save checkpoint (for persistence only, not weight sync)
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                    # sync weights to internal vLLM (wakes from sleep, zero-copy on naive backend)
                    with marked_timer("update_weights", timing_raw, "red"):
                        self.checkpoint_manager.update_weights()
                finally:
                    # always resume proxy — if training crashes mid-step, requests
                    # waiting on _resume_event.wait() would hang forever without this
                    if _proxy_client:
                        try:
                            resp = _proxy_client.post(f"{proxy_url}/resume")
                            resp.raise_for_status()
                        except Exception as e:
                            print(f"WARNING: failed to resume proxy: {e}")

            # step 9: collect and log metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            metrics["training/global_step"] = self.global_steps

            logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)

            if is_last_step:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                progress_bar.close()
                if _proxy_client:
                    _proxy_client.close()
                return
