# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from copy import deepcopy
from typing import Any

import torch
from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client
from arctic_platform.rl.ray_server import ArcticRLRayServerState
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.remote_backend.base import RemoteBackend, RemoteBackendRegistry


def _no_padding_2_padding_prompt_response(tensor: torch.Tensor, data, pad_token_id):
    """Convert a jagged tensor into a left-padded prompt + right-padded
    response of shape ``[bsz, max_prompt_len + max_response_len]`` —
    the wire format Arctic expects. Arctic-specific; kept here rather
    than in :mod:`verl.workers.utils.padding` so that generic verl code
    stays untouched.
    """
    import torch.nn.functional as F

    from verl.utils import tensordict_utils as tu

    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]

    max_prompt_len = tu.get_non_tensor_data(data=data, key="max_prompt_len", default=-1)
    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_prompt_len < 0:
            max_prompt_len = prompt_lens.max().item()
        if max_response_len < 0:
            max_response_len = response_lens.max().item()
    else:
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_prompt_len = prompt_ids.shape[1]
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0], f"{sequence_offsets[-1].item()} != {values.shape[0]}"

    rows = []
    for prompt_len, resp_len, seq_offset in zip(prompt_lens, response_lens, sequence_offsets, strict=True):
        prompt_pad = max_prompt_len - prompt_len
        response_pad = max_response_len - resp_len
        prompt = values[seq_offset - prompt_len - resp_len : seq_offset - resp_len]
        response = values[seq_offset - resp_len : seq_offset]
        prompt_padded_left = F.pad(prompt, (prompt_pad, 0), value=pad_token_id)
        response_padded_right = F.pad(response, (0, response_pad), value=pad_token_id)
        rows.append(torch.cat((prompt_padded_left, response_padded_right)))

    return torch.stack(rows, dim=0), max_prompt_len, max_response_len


def _prepare_padded_arctic_batch_dict(data, pad_token_id, *, drop_position_ids: bool = False):
    """Convert verl's nested-jagged ``input_ids`` / ``position_ids`` into the dense padded shape Arctic expects on the
    wire, and report the max prompt / response lengths. With ``drop_position_ids=True`` the server reconstructs
    position_ids from attention_mask (set False for mrope / 3D rope)."""
    input_ids = data["input_ids"]
    input_ids, max_prompt_len, max_response_len = _no_padding_2_padding_prompt_response(
        tensor=input_ids, data=data, pad_token_id=pad_token_id
    )

    batch_dict = dict(
        input_ids=input_ids,
        attention_mask=data["attention_mask"],
        prompts=data["prompts"],
    )
    if not drop_position_ids:
        position_ids, _, _ = _no_padding_2_padding_prompt_response(
            tensor=data["position_ids"], data=data, pad_token_id=0
        )
        batch_dict["position_ids"] = position_ids
    return batch_dict, max_prompt_len, max_response_len


@RemoteBackendRegistry.register("arctic")
class ArcticRLClientWrapper(RemoteBackend):
    """Arctic backend: a thin wrapper around ArcticRL's ``ArcticRLClient``.

    Implements the generic :class:`verl.remote_backend.RemoteBackend`
    interface. Registered under the name ``"arctic"`` so
    :class:`verl.remote_backend.trainer.RemoteBackendTrainer` can resolve
    it via ``config.trainer.remote_backend = "arctic"``.
    """

    def __init__(self, config, reconnect_job_config: dict = None, rl_server_state: ArcticRLRayServerState = None):
        self.config = config
        # Per-backend yaml is loaded flat at `config.remote_backend`; the file
        # name (arctic.yaml) names the backend, so no extra nesting is needed.
        self._backend_config = config.remote_backend
        self._client = None
        self.zorro_train_enable = self._backend_config.zorro_train.enable
        self.zorro_train_max_rollouts = self._backend_config.zorro_train.max_rollouts
        # Set False for non-arange position_ids (mrope / 3D rope).
        self.drop_position_ids = self._backend_config.get("drop_position_ids", True)
        self.logits_optimization = self._backend_config.get("logits_optimization", "none")
        self.logits_optimization_peak_mem_size_in_gib = self._backend_config.get(
            "logits_optimization_peak_mem_size_in_gib", 4
        )
        self.logits_compute_in_fp32 = self._backend_config.get("logits_compute_in_fp32", False)
        self.logits_compute_from_fp32_inputs = self._backend_config.get("logits_compute_from_fp32_inputs", False)
        # No server consumer yet; forwarded for forward-compat with group-balanced routing.
        self.zorro_train_load_balancer = self._backend_config.zorro_train.get("load_balancer", True)
        # CUDA-IPC bypasses NCCL for weight sync; only valid when colocate=True.
        self.cuda_ipc_weight_sync = self._backend_config.get("cuda_ipc_weight_sync", False)
        self.low_memory_weight_sync = self._backend_config.get("low_memory_weight_sync", False)
        self.use_liger = self.config.actor_rollout_ref.model.use_liger
        self._max_token_len_per_gpu = self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.actor_rollout_ref.model.path)
        self._client = self._initialize_client(reconnect_job_config, rl_server_state)

    # ------------------------------------------------------------------ #
    # RemoteBackend interface
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        main_config,
        *,
        handle: dict | None = None,
    ) -> "ArcticRLClientWrapper":
        """Sole public constructor.

        * ``handle=None`` (driver path): build a fresh ``ArcticRLClient``
          and the underlying Ray actors.
        * ``handle=<reconnect_handle()>``: re-attach to the existing
          driver-side instance (used by forwarder workers and
          ``ArcticReplica``).
        """
        if handle is None:
            return cls(config=main_config)
        return cls(
            config=main_config,
            reconnect_job_config=handle["rl_client_reconnect_config"],
            rl_server_state=handle["rl_server_state"],
        )

    def reconnect_handle(self) -> dict:
        """Serializable bundle the forwarder workers / rollout replicas
        pass back to :meth:`from_config` (as ``handle=...``) to re-attach
        to this same backend instance."""
        return {
            "rl_client_reconnect_config": self.reconnect_config(),
            "rl_server_state": self.get_server_state(),
        }

    # Parallelism contract --------------------------------------------- #

    def requires_single_forwarder(self) -> bool:
        # Arctic owns its own training/sampling parallelism; the verl-side
        # worker group must be a single forwarder.
        return True

    # ------------------------------------------------------------------ #
    # Core RL ops — called by `ArcticRLActorRolloutRefWorker`
    # ------------------------------------------------------------------ #
    # Build Arctic's payload shape (dense padded batch, ``meta`` dict,
    # ``processing`` pipeline) and dispatch through the private
    # `_send_*` wire helpers below.

    async def compute_log_prob(
        self,
        data,
        *,
        ref: bool,
        calculate_entropy: bool,
        rollout_n: int,
        temperature,
        pad_token_id: int,
    ) -> dict:
        batch, max_prompt_len, max_response_len = _prepare_padded_arctic_batch_dict(
            data, pad_token_id, drop_position_ids=self.drop_position_ids
        )
        meta = dict(
            zorro_train_enable=self.zorro_train_enable,
            zorro_train_max_rollouts=self.zorro_train_max_rollouts,
            zorro_train_load_balancer=self.zorro_train_load_balancer,
            rollout_n=rollout_n,
            max_prompt_len=max_prompt_len,
            max_response_len=max_response_len,
            max_token_len_per_gpu=self._max_token_len_per_gpu,
            temperature=data["temperature"],
            calculate_entropy=calculate_entropy,
            pad_token_id=pad_token_id,
            drop_position_ids=self.drop_position_ids,
            logits_optimization=self.logits_optimization,
            logits_optimization_peak_mem_size_in_gib=self.logits_optimization_peak_mem_size_in_gib,
            logits_compute_in_fp32=self.logits_compute_in_fp32,
        )
        payload = dict(batch=batch, meta=meta)

        response = await (self._send_compute_ref_log_prob(payload) if ref else self._send_compute_log_prob(payload))

        model_output = {"log_probs": response["batch"]["log_probs"]}
        if calculate_entropy and "entropy" in response["batch"]:
            model_output["entropy"] = response["batch"]["entropy"]

        metrics = dict(response.get("metrics", {}))
        return {"model_output": model_output, "metrics": metrics}

    async def update_actor(
        self,
        data,
        *,
        actor_config,
        pad_token_id: int,
        rollout_n: int,
        temperature,
    ) -> dict:
        import json

        global_token_num = data["input_ids"].offsets().diff().tolist()

        input_ids, max_prompt_len, max_response_len = _no_padding_2_padding_prompt_response(
            tensor=data["input_ids"], data=data, pad_token_id=pad_token_id
        )

        batch = dict(
            input_ids=input_ids,
            attention_mask=data["attention_mask"],
            prompts=data["prompts"],
            responses=data["responses"],
            response_mask=data["response_mask"],
            old_log_probs=data["old_log_probs"],
            advantages=data["advantages"],
        )
        if not self.drop_position_ids:
            position_ids, _, _ = _no_padding_2_padding_prompt_response(
                tensor=data["position_ids"], data=data, pad_token_id=0
            )
            batch["position_ids"] = position_ids
        if actor_config.use_kl_loss:
            batch["ref_log_prob"] = data["ref_log_prob"]

        # Per-chunk loss normalizers (mirror recipe/rl-correctness arctic_workers.train_global_batch).
        per_step_global_bsz = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * rollout_n
        meta = dict(
            zorro_train_enable=self.zorro_train_enable,
            zorro_train_max_rollouts=self.zorro_train_max_rollouts,
            zorro_train_load_balancer=self.zorro_train_load_balancer,
            rollout_n=rollout_n,
            max_prompt_len=max_prompt_len,
            max_response_len=max_response_len,
            max_token_len_per_gpu=self._max_token_len_per_gpu,
            temperature=data["temperature"],
            calculate_entropy=actor_config.calculate_entropy,
            pad_token_id=pad_token_id,
            drop_position_ids=self.drop_position_ids,
            logits_optimization=self.logits_optimization,
            logits_optimization_peak_mem_size_in_gib=self.logits_optimization_peak_mem_size_in_gib,
            logits_compute_in_fp32=self.logits_compute_in_fp32,
            global_batch_size=per_step_global_bsz,
            batch_num_tokens=batch["response_mask"].sum(),
            rollout_is_weights=data.get("rollout_is_weights", None),
        )

        def _safe_serialize(obj):
            return json.loads(json.dumps(obj, default=lambda o: None))

        meta["actor_config"] = _safe_serialize(vars(actor_config))
        # `policy_loss` may be None on configs that never wired up a custom
        # loss (e.g. early test/dev configs); send an empty dict instead of
        # crashing in `vars(None)`.
        policy_loss = getattr(actor_config, "policy_loss", None)
        meta["policy_loss_config"] = _safe_serialize(vars(policy_loss)) if policy_loss is not None else {}

        payload = dict(batch=batch, meta=meta)
        response = await self._send_update_actor(payload)

        metrics = dict(response["metrics"])
        loss = metrics.pop("loss")
        if "last_lr" in metrics:
            metrics["lr"] = metrics.pop("last_lr")

        return {
            "loss": loss,
            "metrics": metrics,
            "global_token_num": global_token_num,
        }

    def _create_ds_config(self, n_gpus: int, training: bool = True) -> dict[str, Any]:
        # Pass the entire `remote_backend.deepspeed` YAML block through to the
        # DS engine (torch_autocast / communication_data_type / data_types /
        # zero_optimization / log_level / ...) and merge in the per-step batch
        # sizing. Matches recipe/rl-correctness arctic_rl_client._create_ds_config.
        actor_cfg = self.config.actor_rollout_ref.actor
        data_cfg = self.config.data
        rollout_n = self.config.actor_rollout_ref.rollout.n
        deepspeed_config = OmegaConf.to_container(self._backend_config.deepspeed, resolve=True)

        micro_batch_size = actor_cfg.ppo_micro_batch_size_per_gpu or 1
        # DS train_batch_size = samples consumed per optimizer step.
        # We do one optimizer step per PPO mini-batch (matching verl baseline's
        # dp_actor.update_policy loop), not per global batch, so the DS engine's
        # gradient-accumulation gating must reflect one mini-batch, not BSZ.
        global_batch_size = data_cfg.train_batch_size * rollout_n
        train_batch_size = actor_cfg.ppo_mini_batch_size * rollout_n
        assert global_batch_size % train_batch_size == 0, (
            f"data.train_batch_size ({data_cfg.train_batch_size}) must be divisible by "
            f"actor.ppo_mini_batch_size ({actor_cfg.ppo_mini_batch_size}); "
            f"got global={global_batch_size}, per-step={train_batch_size}"
        )
        grad_accum_steps = max(1, train_batch_size // (micro_batch_size * n_gpus))
        train_seq_parallel_size = actor_cfg.fsdp_config.get("ulysses_sequence_parallel_size", 1)

        ds_config = deepcopy(deepspeed_config)
        ds_config.update(
            {
                "train_micro_batch_size_per_gpu": micro_batch_size,
                "train_batch_size": train_batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "sequence_parallel_size": train_seq_parallel_size,
            }
        )
        if not training:
            ds_config.pop("data_types", None)

        return ds_config

    def reconnect_config(self):
        return self._client.reconnect_config()

    def get_server_state(self):
        return self._client.get_server_state()

    def _create_ds_worker_config(self):
        attn_implementation = self.config.actor_rollout_ref.model.override_config.get(
            "attn_implementation", "flash_attention_2"
        )
        ds_worker_config = dict(
            use_liger=self.use_liger,
            enable_gradient_checkpointing=self.config.actor_rollout_ref.model.enable_gradient_checkpointing,
            attn_implementation=attn_implementation,
        )

        if self.zorro_train_enable:
            # XXX: can't find where it's configured
            use_unpad = True

            ds_worker_config.update(
                zorro_train_enable=True,
                response_len=self.config.data.max_response_length,
                # Matches recipe/rl-correctness: read per-GPU token budget
                # from actor.ppo_max_token_len_per_gpu (the budget the train
                # engine sees), not rollout.max_num_batched_tokens.
                max_token_len=self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
                rollout_n=self.config.actor_rollout_ref.rollout.n,
                temperature=self.config.actor_rollout_ref.rollout.temperature,
                use_unpad=use_unpad,
                # Server reads these from ds_worker_config (arctic_platform/rl/deepspeed_worker.py), not per-call meta.
                logits_optimization=self.logits_optimization,
                logits_optimization_peak_mem_size_in_gib=self.logits_optimization_peak_mem_size_in_gib,
                logits_compute_from_fp32_inputs=self.logits_compute_from_fp32_inputs,
                logits_compute_in_fp32=self.logits_compute_in_fp32,
            )

        return ds_worker_config

    def _initialize_client(self, reconnect_job_config: dict = None, rl_server_state: ArcticRLRayServerState = None):
        if rl_server_state is not None:
            return create_arctic_rl_client(reconnect_job_config, rl_server_state)

        if reconnect_job_config is not None:
            return create_arctic_rl_client(reconnect_job_config)

        model_name = self.config.actor_rollout_ref.model.path
        n_training_gpus = self._backend_config.get("training_gpus", self.config.trainer.n_gpus_per_node)
        n_sampling_gpus = self._backend_config.get("sampling_gpus", self.config.trainer.n_gpus_per_node)
        n_log_prob_gpus = self._backend_config.get("log_prob_gpus", self.config.trainer.n_gpus_per_node)
        colocate = self._backend_config.get("colocate", True)
        attn_implementation = self.config.actor_rollout_ref.model.override_config.get("attn_implementation", "eager")

        actor_cfg = self.config.actor_rollout_ref.actor
        optim_cfg = actor_cfg.optim
        data_cfg = self.config.data

        max_length = data_cfg.max_prompt_length + data_cfg.max_response_length

        # LR-scheduler horizon = global_steps * ppo_epochs * num_minibatches
        # (matches recipe/rl-correctness arctic_rl_client.py). verl injects
        # actor.optim.total_training_steps after dataloader build; fall back
        # to trainer.total_training_steps for early-construction paths.
        global_training_steps = optim_cfg.get("total_training_steps", None)
        if not global_training_steps or int(global_training_steps) <= 0:
            global_training_steps = self.config.trainer.get("total_training_steps", None)
        assert global_training_steps and int(global_training_steps) > 0, (
            "total_training_steps not populated before Arctic client build; expected "
            "ray_trainer._create_dataloader to inject actor.optim.total_training_steps"
        )
        num_minibatches = data_cfg.train_batch_size // actor_cfg.ppo_mini_batch_size
        opt_steps_per_global_step = actor_cfg.ppo_epochs * num_minibatches
        training_horizon = int(global_training_steps) * opt_steps_per_global_step
        lr_scheduler_type = optim_cfg.get("lr_scheduler_type", "constant")
        print(
            f"[ArcticRLClientWrapper] lr_scheduler: type={lr_scheduler_type} "
            f"training_horizon={training_horizon} (global_steps={int(global_training_steps)} "
            f"x ppo_epochs={actor_cfg.ppo_epochs} x num_minibatches={num_minibatches}); "
            f"warmup_steps={int(optim_cfg.lr_warmup_steps_ratio * training_horizon)}",
            flush=True,
        )

        rollout_cfg = self.config.actor_rollout_ref.rollout
        vllm_config = {
            "tensor_parallel_size": self._backend_config.sampling_tp_size,
            "gpu_memory_utilization": rollout_cfg.gpu_memory_utilization,
            "max_model_len": rollout_cfg.get("max_model_len") or max_length,
            "max_num_seqs": rollout_cfg.max_num_seqs,
            "enforce_eager": rollout_cfg.enforce_eager,
            "enable_chunked_prefill": rollout_cfg.enable_chunked_prefill,
        }
        if rollout_cfg.get("quantization"):
            vllm_config["quantization"] = rollout_cfg.quantization

        # Forward grad_clip to the DeepSpeed engine so it clips global grad-norm
        # to the same threshold verl applies in the FSDP path (otherwise DS
        # defaults to no clipping -> trajectory divergence vs verl baseline).
        optimizer_config = {
            "lr": optim_cfg.lr,
            "weight_decay": optim_cfg.weight_decay,
            "betas": list(optim_cfg.betas),
        }
        grad_clip = optim_cfg.get("clip_grad", None)
        if grad_clip is None:
            grad_clip = actor_cfg.get("grad_clip", None)
        if grad_clip is not None and float(grad_clip) > 0:
            optimizer_config["gradient_clipping"] = float(grad_clip)

        lr_scheduler_config = {
            "type": lr_scheduler_type,
            "warmup_ratio": optim_cfg.lr_warmup_steps_ratio,
            "min_lr_ratio": optim_cfg.get("min_lr_ratio", None),
        }

        rl_config = ArcticRLClientConfig(
            host=None if self._backend_config.comm_protocol == "ray" else "localhost",
            port=None if self._backend_config.comm_protocol == "ray" else 7000,
            comm_protocol=self._backend_config.comm_protocol,
            backend="local",
            training_gpus=n_training_gpus,
            sampling_gpus=n_sampling_gpus,
            log_prob_gpus=n_log_prob_gpus,
            colocate=colocate,
            log_prob_engine="deepspeed",
            model_name=model_name,
            ds_config=self._create_ds_config(n_training_gpus),
            log_prob_ds_config=None if n_log_prob_gpus == 0 else self._create_ds_config(n_log_prob_gpus),
            training_config={
                "training_horizon": training_horizon,
                "optimizer": optimizer_config,
                "lr_scheduler": lr_scheduler_config,
                "max_length": max_length,
                "model_config": None,
                "attn_implementation": attn_implementation,
            },
            ds_worker_config=self._create_ds_worker_config(),
            vllm_config=vllm_config,
            checkpoint_path=self.config.trainer.default_local_dir,
        )

        # ArcticRLClient is constructed as a ray remote actor with num_gpus=0,
        # which causes CUDA_VISIBLE_DEVICES to be empty.
        if colocate:
            num_visible = n_training_gpus + n_sampling_gpus + n_log_prob_gpus
        else:
            num_visible = rl_config.training_gpus + rl_config.sampling_gpus + rl_config.log_prob_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_visible))

        return create_arctic_rl_client(rl_config)

    _default_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 1024,
    }

    async def generate(self, prompt_ids, sampling_params, routing_key=None) -> list:
        # `routing_key` kept for caller-API compat; arctic_platform.rl handles routing internally.
        prompts = [self.tokenizer.decode(prompt_ids)]
        merged_params = {**self._default_sampling_params, **sampling_params}
        return await self._client.generate(
            prompts=prompts,
            sampling_params=merged_params,
        )

    # ------------------------------------------------------------------ #
    # Arctic wire helpers (private; not on `RemoteBackend`)
    # ------------------------------------------------------------------ #
    # Invoked from `compute_log_prob` / `update_actor` above. Future
    # backends are free to pick their own wire format; verl never calls
    # these directly.

    async def _send_compute_ref_log_prob(self, payload: dict):
        payload["processing"] = {"post": ["compute_entropy_and_logprobs"], "loss_fn": None}
        response = await self._client.fwd_no_grad(payload, reference_model=True)
        response["batch"]["log_probs"] = response["batch"].pop("logprobs")
        return response

    async def _send_compute_log_prob(self, payload: dict):
        payload["processing"] = {"post": ["compute_entropy_and_logprobs"], "loss_fn": None}
        response = await self._client.fwd_no_grad(payload, reference_model=False)
        response["batch"]["log_probs"] = response["batch"].pop("logprobs")
        return response

    async def _send_update_actor(self, payload: dict):
        payload["processing"] = {
            "post": ["apply_temperature", "compute_entropy_and_logprobs"],
            "loss_fn": "verl_grpo",
        }

        def _left_pad(t: torch.Tensor, seq_len: int) -> torch.Tensor:
            """Left-pad a response-only tensor to full sequence length with zeros."""
            pad_len = seq_len - t.shape[-1]
            if pad_len <= 0:
                return t
            pad = torch.zeros(*t.shape[:-1], pad_len, dtype=t.dtype, device=t.device)
            return torch.cat([pad, t], dim=-1)

        seq_len = payload["batch"]["input_ids"].shape[-1]
        for name in ["old_log_probs", "advantages", "response_mask", "ref_log_prob"]:
            if name in payload["batch"]:
                payload["batch"][name] = _left_pad(payload["batch"][name], seq_len)
        payload["batch"]["loss_mask"] = payload["batch"]["response_mask"]

        fwd_bwd_response = await self._client.fwd_bwd(payload)
        step_response = await self._client.step()
        step_response["metrics"].update(**fwd_bwd_response["metrics"])
        return step_response

    async def save_checkpoint(self):
        return await self._client.save_checkpoint()

    async def update_weights(self):
        return await self._client.sync_weights(
            cuda_ipc=self.cuda_ipc_weight_sync,
            low_memory=self.low_memory_weight_sync,
        )

    async def destroy(self) -> None:
        if self._client is not None:
            await self._client.shutdown()
