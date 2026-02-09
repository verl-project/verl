# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Giga Team. and/or its affiliates
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
#
# from https://github.com/open-gigaai/giga-models


import math

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn

from .paligemma_with_expert import PaliGemmaWithExpertModel


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Mps is currently not compatible with float64."""
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device: str | torch.device = "cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar
    positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      pad_masks: bool[B, N] indicating valid (true) vs. padding (false) tokens.
      att_masks: int[B, N] defining attention type. A `1` at a position
                 indicates the start of a new causal block.

    Returns:
        A 2D boolean attention mask of shape (B, N, N).
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


class PI0Model(ModelMixin, ConfigMixin):
    """pi0: A Vision-Language-Action Flow Model for General Robot Control.

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    @register_to_config
    def __init__(
        self,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        proj_width: int = 1024,
        n_action_steps: int = 50,
        num_steps: int = 10,
        use_cache: bool = True,
        pi05_enabled: bool = False,
    ):
        super().__init__()

        # Store the parameters
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.proj_width = proj_width
        self.n_action_steps = n_action_steps
        self.num_steps = num_steps
        self.use_cache = use_cache
        self.pi05_enabled = pi05_enabled

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            pi05_enabled=pi05_enabled,
        )

        # Projections are float32
        if self.pi05_enabled:
            self.time_mlp_in = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)
            self.time_mlp_out = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)
        else:
            self.state_proj = nn.Linear(self.max_state_dim, self.proj_width, dtype=torch.float32)
            self.action_time_mlp_in = nn.Linear(self.proj_width * 2, self.proj_width, dtype=torch.float32)
            self.action_time_mlp_out = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)

        self.action_in_proj = nn.Linear(self.max_action_dim, self.proj_width, dtype=torch.float32)
        self.action_out_proj = nn.Linear(self.proj_width, self.max_action_dim, dtype=torch.float32)

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
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -self.n_action_steps :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
        """Generate Gaussian noise for the action trajectory.

        Args:
            shape: Desired output shape, typically (B, n_action_steps, action_dim).
            device: Target device string or torch.device.

        Returns:
            A float32 tensor of standard normal samples with the given shape.
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed visual and language inputs as the transformer prefix.

        Args:
            images: List of (B, C, H, W) tensors.
            img_masks: List of (B,) boolean masks for image presence.
            lang_tokens: (B, L) token ids.
            lang_masks: (B, L) boolean mask; True indicates valid tokens.

        Returns:
            A tuple of (embs, pad_masks, att_masks):
              - embs: (B, Np, D) concatenated image and language embeddings
              - pad_masks: (B, Np) valid token mask
              - att_masks: (B, Np) attention mask scheme selector
        """
        # Optimize: batch process images and pre-allocate tensors
        num_images = len(images)

        # Stack images and masks for batch processing
        images_stacked = torch.stack(images, dim=0)  # (num_images, bsize, ...)
        img_masks_stacked = torch.stack(img_masks, dim=0)  # (num_images, bsize)

        # Batch embed all images at once
        # Reshape to (num_images * bsize, ...)
        orig_shape = images_stacked.shape
        images_flat = images_stacked.reshape(-1, *orig_shape[2:])
        img_embs_flat = self.paligemma_with_expert.embed_image(images_flat)

        # Reshape back to (num_images, bsize, num_img_embs, emb_dim)
        bsize = orig_shape[1]
        img_embs = img_embs_flat.reshape(num_images, bsize, *img_embs_flat.shape[1:])

        # Normalize image embeddings
        img_emb_dim = img_embs.shape[-1]
        num_img_embs = img_embs.shape[2]

        # Expand masks: (num_images, bsize) -> (num_images, bsize, num_img_embs)
        img_masks_expanded = img_masks_stacked[:, :, None].expand(num_images, bsize, num_img_embs)

        # Reshape to (bsize, num_images * num_img_embs, emb_dim)
        img_embs_concat = img_embs.transpose(0, 1).reshape(bsize, num_images * num_img_embs, img_emb_dim)
        img_masks_concat = img_masks_expanded.transpose(0, 1).reshape(bsize, num_images * num_img_embs)

        # Process language embeddings
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        lang_emb = lang_emb.to(dtype=img_embs_concat.dtype)

        num_lang_embs = lang_emb.shape[1]
        total_seq_len = num_images * num_img_embs + num_lang_embs

        # Pre-allocate final tensors
        embs = torch.empty(
            bsize, total_seq_len, img_emb_dim, dtype=img_embs_concat.dtype, device=img_embs_concat.device
        )
        pad_masks = torch.empty(bsize, total_seq_len, dtype=torch.bool, device=img_embs_concat.device)

        # Fill pre-allocated tensors
        embs[:, : num_images * num_img_embs] = img_embs_concat
        embs[:, num_images * num_img_embs :] = lang_emb
        pad_masks[:, : num_images * num_img_embs] = img_masks_concat
        pad_masks[:, num_images * num_img_embs :] = lang_masks

        # Create attention masks (all zeros for full attention between image and language)
        att_masks = torch.zeros(total_seq_len, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, total_seq_len)

        return embs, pad_masks, att_masks

    def embed_suffix(
        self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed state, action and time tokens as the transformer suffix.

        Args:
            state: (B, state_dim) robot state; ignored when pi05 is enabled.
            noisy_actions: (B, n_action_steps, action_dim) current x_t.
            timestep: (B,) diffusion time in [0, 1].

        Returns:
            (embs, pad_masks, att_masks, adarms_cond) where:
              - embs: (B, Ns, D) suffix embeddings
              - pad_masks: (B, Ns) valid mask
              - att_masks: (B, Ns) causal scheme for suffix
              - adarms_cond: (B, D) AdaRMS conditioning or None
        """
        embs = []
        pad_masks = []
        att_masks = []

        action_emb = self.action_in_proj(noisy_actions)
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        device = action_emb.device

        # Embed state
        if not self.pi05_enabled:
            state_emb = self.state_proj(state)
            embs.append(state_emb[:, None, :])

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        if self.pi05_enabled:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            action_expert_emb = action_emb
            adarms_cond = time_emb
        else:
            # Fuse timestep + action information using an MLP
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)  # swish == silu
            action_time_emb = self.action_time_mlp_out(action_time_emb)
            action_expert_emb = action_time_emb
            adarms_cond = None

        # Add to input tokens
        embs.append(action_expert_emb)

        bsize, action_time_dim = action_expert_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Run the full inference loop to predict an action trajectory.

        Args:
            images: List of (B, C, H, W) image tensors.
            img_masks: List of (B,) boolean masks.
            lang_tokens: (B, L) token ids.
            lang_masks: (B, L) boolean mask for tokens.
            state: (B, state_dim) robot state.
            noise: Optional initial noise; if None, generated internally.

        Returns:
            Predicted actions with shape (B, n_action_steps, action_dim).
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        if noise is None:
            actions_shape = (bsize, self.n_action_steps, self.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.use_cache,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )

        x_t = noise
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)
        for timestep in timesteps:
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep.expand(bsize),
            )
            x_t += dt * v_t

        return x_t

    def denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one denoising step of the noise x_t at a given timestep.

        Args:
            state: (B, state_dim) robot state.
            prefix_pad_masks: (B, Np) prefix pad masks computed from embed_prefix.
            past_key_values: KV cache dict for the prefix (images+language).
            x_t: (B, n_action_steps, action_dim) current noisy actions.
            timestep: (B,) current time in [0, 1].

        Returns:
            v_t prediction with shape (B, n_action_steps, action_dim).
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.n_action_steps :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)
        return v_t
