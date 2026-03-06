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

from __future__ import annotations

from typing import Literal

import torch
from onnx_ir import Tensor
from torch import nn
from torch.distributed.fsdp import register_fsdp_forward_method
from torch.distributions import Normal
from transformers import PreTrainedModel
from typing_extensions import override

from verl.protocol import DataProto
from verl.utils.device import get_device_name

from ...sac.base import SupportSACTraining
from ..modules.mlp import MLP
from .configuration_pi0_torch import PI0TorchConfig
from .model.modeling_pi0 import PI0Model, make_att_2d_masks
from .pi0_utils import (
    ImageTransform,
    Normalize,
    PromptTokenizerTransform,
    Unnormalize,
)
from .policy.base import Pi0Output


class PI0ForActionPrediction(PreTrainedModel, SupportSACTraining):
    config_class = PI0TorchConfig
    base_model_prefix = "pi0_torch"

    def __init__(self, config: PI0TorchConfig):
        super().__init__(config)
        self.model: PI0Model = None
        self.state_norm_stats = config.state_norm_stats
        self.action_norm_stats = config.action_norm_stats
        self.pi05_enabled = config.pi05_enabled

        assert self.state_norm_stats, "state_norm_stats must be provided in PI0TorchConfig"
        assert self.action_norm_stats, "action_norm_stats must be provided in PI0TorchConfig"
        assert isinstance(self.pi05_enabled, bool), "pi05_enabled must be provided in PI0TorchConfig"

        # Input transforms
        self.state_normalize_transform = Normalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_normalize_transform = Normalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)
        self.image_transform = ImageTransform(resize_imgs_with_padding=(224, 224), enable_image_aug=False)
        max_length = 200 if self.pi05_enabled else 48
        self.prompt_tokenizer_transform = PromptTokenizerTransform(max_length=max_length, discrete_state_input=False)

        # Output transforms
        self.state_unnormalize_transform = Unnormalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)

        self._to(get_device_name())

        ##### SAC Algorithm Support #####
        if getattr(self.config, "sac_enable", False):
            head_num = 2 if getattr(self.config, "double_q", True) else 1

            self.critic_heads = nn.ModuleList(
                [
                    MLP(
                        input_dim=2150,  # 2048(prefix mean) + 32(state) + 10*7(action flat)
                        hidden_dims=[1024, 512, 256],
                        output_dim=1,
                        activation="relu",
                        init_method="normal",
                    )
                    for _ in range(head_num)
                ]
            )

            self.target_network_heads = nn.ModuleList(
                [
                    MLP(
                        input_dim=2150,
                        hidden_dims=[1024, 512, 256],
                        output_dim=1,
                        activation="relu",
                        init_method="normal",
                    )
                    for _ in range(head_num)
                ]
            )

    def _to(self, device: torch.device | str):
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_normalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        return self

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tensor:
        """Full forward pass for one diffusion denoising step.

        Args:
            images: List of image tensors, each shaped (B, C, H, W) after batching.
            img_masks: List of boolean masks corresponding to images, each (B,).
            lang_tokens: Language token ids (B, L).
            lang_masks: Language attention mask (B, L) with True for valid tokens.
            state: State tensor (B, state_dim) if pi05 is disabled else ignored.
            x_t: Noisy action tokens (B, n_action_steps, action_dim).
            timestep: Diffusion timestep as float tensor (B,).

        Returns:
            Predicted v_t with shape (B, n_action_steps, action_dim).
        """

        if self.model is None:
            raise RuntimeError("PI0ForActionPrediction.model is not initialized. Did from_pretrained() run?")

        return self.model(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            x_t,
            timestep,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        env_obs: DataProto,
        tokenizer,
    ) -> tuple[Pi0Output, dict, dict]:
        """Run one forward pass from raw inputs to final action sequence.

        Args:
            env_obs: The environment observations as DataProto.
            tokenizer: The tokenizer used for prompt tokenization.

        Returns:
            A tuple of (pi0_output, s, a):
                - pi0_output: The Pi0Output containing the predicted actions.
                - s: Dictionary of tensors representing the states, with keys
                    - "images": torch.Tensor of shape (B, n_images, C, H, W)
                    - "image_masks": torch.Tensor of shape (B, n_images)
                    - "lang_tokens": torch.Tensor of shape (B, L)
                    - "lang_masks": torch.Tensor of shape (B, L)
                    - "states": torch.Tensor of shape (B, state_dim)
                - a: Dictionary of tensors representing actions, with key:
                    - "full_action": torch.Tensor of shape (B, action_steps, action_dim)
        """

        from .policy.libero_policy import LiberoPi0Input

        pi0_input = LiberoPi0Input.from_env_obs(env_obs)

        # Input transforms
        state = self.state_normalize_transform(pi0_input.state)
        images, _ = self.image_transform.call_batch(pi0_input.images)
        lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
            {"task": pi0_input.task, "observation.state": state}, tokenizer
        )

        # Inference
        pred_action = self.model.sample_actions(images, pi0_input.img_masks, lang_tokens, lang_masks, state=state)

        # Output transforms
        # state = self.state_unnormalize_transform(state)
        pred_action = self.action_unnormalize_transform(pred_action)

        from .policy.libero_policy import LiberoPi0Output

        pi0_output = LiberoPi0Output.from_model_output({"full_action": pred_action})
        s = {
            "states": state,
            "images": torch.stack(images, dim=1),
            "image_masks": torch.stack(pi0_input.img_masks, dim=1),
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }
        a = {
            "full_action": pred_action,
        }

        return pi0_output, s, a

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            config = PI0TorchConfig.from_pretrained(pretrained_model_name_or_path)

        policy = cls(config)
        policy.model = PI0Model.from_pretrained(pretrained_model_name_or_path)
        return policy

    def freeze_vision_tower(self) -> None:
        """Freeze the vision tower parameters."""

        if self.model is None:
            raise RuntimeError("PI0ForActionPrediction.model is not initialized. Did from_pretrained() run?")
        vision_tower = self.model.paligemma_with_expert.vision_tower
        vision_tower.requires_grad_(False)
        vision_tower.eval()

    # --- SAC Algorithm Support ---

    def _multi_heads_value(
        self, value_heads: nn.ModuleList, input_tensor: torch.Tensor, method: Literal["cat", "min"] = "cat"
    ) -> torch.Tensor:
        q_values = [head(input_tensor) for head in value_heads]
        if method == "cat":
            q_values = torch.cat(q_values, dim=-1)
        elif method == "min":
            q_values = torch.min(torch.cat(q_values, dim=-1), dim=-1).values
        else:
            raise ValueError(f"Unknown method: {method}")

        return q_values

    def _build_kv_cache_from_prefix(
        self,
        prefix_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """Build KV cache for prefix. No grad needed."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_features
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        with torch.no_grad():
            _, past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.model.use_cache,
                fill_kv_cache=True,
                adarms_cond=[None, None],
            )
        return past_key_values

    def _get_logprobs(
        self,
        s: dict[str, torch.Tensor],
        prefix_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        x_t: torch.Tensor | None = None,  # (B, T, A)
        x_next: torch.Tensor | None = None,  # (B, T, A)
        v_t: torch.Tensor | None = None,  # (B, T, A)
        t: torch.Tensor | None = None,  # (B,)
        step_idx: torch.Tensor | None = None,  # (B,)
    ) -> torch.Tensor:
        """
        Compute log-probability of x_{t+1} given (x_t, v_t) under the Flow-SDE formulation.
        See https://arxiv.org/abs/2510.25889
        """

        prefix_embs, prefix_pad_masks, _ = prefix_features
        states = s["states"]
        B = prefix_embs.shape[0]
        device = prefix_embs.device

        past_key_values = self._build_kv_cache_from_prefix(prefix_features)

        if x_t is None or x_next is None or v_t is None or t is None:
            actions_shape = (B, self.model.n_action_steps, self.model.max_action_dim)
            x = self.model.sample_noise(actions_shape, device=device)

            dt = -1.0 / float(self.model.num_steps)
            t_grid = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

            x_prev, v_prev, t_prev = None, None, None
            for tt in t_grid:
                x_prev = x
                t_prev = tt
                v_prev = self.model.denoise_step(
                    states,
                    prefix_pad_masks,
                    past_key_values,
                    x,
                    tt.expand(B),
                )
                x = x + dt * v_prev

            x_t = x_prev
            x_next = x
            v_t = v_prev
            t = t_prev.expand(B)

        # sigma schedule step index
        K = int(self.model.num_steps)
        if step_idx is None:
            step_idx = torch.full((B,), K - 1, device=device, dtype=torch.long)

        # one-step mean/std
        dt_pos = 1.0 / float(K)
        t_b = t[:, None, None]  # (B,1,1)
        dt_b = torch.full_like(t_b, dt_pos)

        x0_pred = x_t - v_t * t_b
        x1_pred = x_t + v_t * (1.0 - t_b)

        # heuristic sigma schedule (ported family)
        noise_level = 0.5
        t_grid_full = torch.arange(1.0, -dt_pos / 2, -dt_pos, dtype=torch.float32, device=device)  # len=K+1
        t_for_sigma = torch.where(t_grid_full == 1.0, t_grid_full[1], t_grid_full)
        sigmas = noise_level * torch.sqrt(t_grid_full / (1.0 - t_for_sigma).clamp_min(1e-6))
        sigmas = sigmas[:-1]  # len=K

        sigma_i = sigmas[step_idx][:, None, None].clamp_min(1e-6)  # (B,1,1)

        x0_weight = torch.ones_like(t_b) - (t_b - dt_b)
        x1_weight = t_b - dt_b - (sigma_i**2) * dt_b / (2.0 * t_b.clamp_min(1e-6))

        x_next_mean = x0_pred * x0_weight + x1_pred * x1_weight
        x_next_std = (dt_b.sqrt() * sigma_i).clamp_min(1e-6)

        dist = Normal(x_next_mean.float(), x_next_std.float())
        log_probs = dist.log_prob(x_next.float()).sum(dim=2).mean(dim=1)  # (B,)
        return log_probs

    def _sample_actions_and_logprobs_from_prefix(
        self,
        states: torch.Tensor,
        prefix_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions amd compute logprob aligned with those sampled actions.

        Args:
            states: (B, state_dim)
            prefix_features: tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks)

        Returns:
            actions: (B, n_action_steps, action_dim)
            log_probs: (B,)
        """

        prefix_embs, prefix_pad_masks, _ = prefix_features
        B = prefix_embs.shape[0]
        device = prefix_embs.device

        past_key_values = self._build_kv_cache_from_prefix(prefix_features)

        actions_shape = (B, self.model.n_action_steps, self.model.max_action_dim)
        x = self.model.sample_noise(actions_shape, device=device)

        dt = -1.0 / float(self.model.num_steps)
        t_grid = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)  # len=K

        x_prev, v_prev, t_prev = None, None, None
        for tt in t_grid:
            x_prev = x
            t_prev = tt
            v_prev = self.model.denoise_step(
                states,
                prefix_pad_masks,
                past_key_values,
                x,
                tt.expand(B),
            )
            x = x + dt * v_prev

        actions = x  # x_K

        # aligned logprob: use last transition (K-1)
        step_idx = torch.full((B,), int(self.model.num_steps) - 1, device=device, dtype=torch.long)
        log_probs = self._get_logprobs(
            {"states": states},
            prefix_features,
            x_t=x_prev,
            x_next=actions,
            v_t=v_prev,
            t=t_prev.expand(B),
            step_idx=step_idx,
        )

        return actions, log_probs

    @override
    def sac_init(self):
        """Initialize SAC-related components."""

        self.freeze_vision_tower()

        register_fsdp_forward_method(self, "sac_forward_critic")
        register_fsdp_forward_method(self, "sac_forward_actor")
        register_fsdp_forward_method(self, "sac_update_target_network")
        register_fsdp_forward_method(self, "sac_forward_state_features")

    @override
    def sac_forward_actor(
        self,
        state_features: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_features, states = state_features
        actions, log_probs = self._sample_actions_and_logprobs_from_prefix(states, prefix_features)
        return actions, log_probs

    @override
    def sac_forward_critic(
        self,
        a: dict[str, torch.Tensor],
        state_features: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ):
        critic_head = self.target_network_heads if use_target_network else self.critic_heads
        for p in critic_head.parameters():
            p.requires_grad_(requires_grad)

        prefix_features, states = state_features
        prefix_embs, _, _ = prefix_features
        mean_prefix_embs = prefix_embs.mean(dim=1, keepdim=False)  # (B, 2048)
        actions = self.action_normalize_transform(a["full_action"])  # (B, 50, 32)
        actions = actions[:, :10, :7]  # (B, 10, 7)
        flattened_actions = actions.reshape(actions.shape[0], -1)  # (B, 70)
        critic_input = torch.cat([mean_prefix_embs, states, flattened_actions], dim=-1)  # (B, 2150)

        q_values = self._multi_heads_value(critic_head, critic_input, method=method)

        return q_values

    @override
    def sac_forward_state_features(
        self, s: dict[str, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            prefix_features = self.model.embed_prefix(
                images=s["images"].unbind(dim=1),
                img_masks=s["image_masks"].unbind(dim=1),
                lang_tokens=s["lang_tokens"],
                lang_masks=s["lang_masks"],
            )
        return (prefix_features, s["states"])

    @override
    @torch.no_grad()
    def sac_update_target_network(self, tau: float):
        for target_head, head in zip(self.target_network_heads, self.critic_heads, strict=False):
            for target_param, param in zip(target_head.parameters(), head.parameters(), strict=False):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
