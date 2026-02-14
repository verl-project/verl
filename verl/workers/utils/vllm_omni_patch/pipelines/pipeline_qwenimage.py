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
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl.workers.utils.diffusers_patch.schedulers import FlowMatchSDEDiscreteScheduler
from verl.workers.utils.vllm_omni_patch.data import DiffusionOutput


class QwenImagePipelineWithLogProb(QwenImagePipeline):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        model = od_config.model
        local_files_only = os.path.exists(model)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

    def _get_qwen_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        drop_idx = self.prompt_template_encode_start_idx
        encoder_hidden_states = self.text_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
    ):
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = (
            attention_mask.unsqueeze(0) if attention_mask is not None and attention_mask.ndim == 1 else attention_mask
        )
        batch_size = prompt_ids.shape[0] if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt_ids, attention_mask=attention_mask)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def diffuse(
        self,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        latents,
        img_shapes,
        txt_seq_lens,
        negative_txt_seq_lens,
        timesteps,
        do_true_cfg,
        guidance,
        true_cfg_scale,
        noise_level,
        sde_window,
        sde_type,
        generator,
    ):
        all_latents = []
        all_log_probs = []
        all_timesteps = []
        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents)
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = t

            # Broadcast timestep to match batch size
            timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            # Forward pass for positive prompt (or unconditional if no CFG)
            self.transformer.do_true_cfg = do_true_cfg
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            # Forward pass for negative prompt (CFG)
            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)
            # compute the previous noisy sample x_t -> x_t-1
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_dict=False,
            )

            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(t)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)

        return latents, all_latents, all_log_probs, all_timesteps

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
        max_sequence_length: int = 512,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
    ) -> DiffusionOutput:
        # # TODO: only support single prompt now
        # if req.prompt is not None:
        #     prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
        prompt_ids = req.prompt_ids if req.prompt_ids is not None else prompt_ids
        prompt_mask = req.prompt_mask if req.prompt_mask is not None else prompt_mask
        negative_prompt_ids = req.negative_prompt_ids if req.negative_prompt_ids is not None else negative_prompt_ids
        negative_prompt_mask = (
            req.negative_prompt_mask if req.negative_prompt_mask is not None else negative_prompt_mask
        )
        height = req.height or self.default_sample_size * self.vae_scale_factor
        width = req.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = req.num_inference_steps or num_inference_steps
        max_sequence_length = req.max_sequence_length or max_sequence_length

        noise_level = req.extra_args.get("noise_level", None) or noise_level
        sde_window_size = req.extra_args.get("sde_window_size", None) or sde_window_size
        sde_window_range = req.extra_args.get("sde_window_range", None) or sde_window_range
        sde_type = req.extra_args.get("sde_type", None) or sde_type

        generator = req.generator or generator
        if generator is None and req.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.seed)
        true_cfg_scale = req.true_cfg_scale or true_cfg_scale
        req_num_outputs = getattr(req, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt_ids is not None:
            if isinstance(prompt_ids, list):
                prompt_ids = torch.tensor(prompt_ids, device=self.device)
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        else:
            batch_size = prompt_embeds.shape[0]

        has_neg_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )
        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        # num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            end = start + sde_window_size
            sde_window = (start, end)
        else:
            sde_window = (0, len(timesteps) - 1)

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            img_shapes,
            txt_seq_lens,
            negative_txt_seq_lens,
            timesteps,
            do_true_cfg,
            guidance,
            true_cfg_scale,
            noise_level,
            sde_window,
            sde_type,
            generator,
        )

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        return DiffusionOutput(
            output=image,
            all_latents=all_latents,
            all_log_probs=all_log_probs,
            all_timesteps=all_timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        )
