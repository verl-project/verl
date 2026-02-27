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

import torch.nn as nn
from diffusers.models.normalization import AdaLayerNormContinuous
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenEmbedLayer3DRope,
    QwenEmbedRope,
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
    QwenTimestepProjEmbeddings,
)


class QwenImageTransformer2DModelFixed(QwenImageTransformer2DModel):
    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ):
        super(QwenImageTransformer2DModel, self).__init__()
        self.parallel_config = od_config.parallel_config
        model_config = od_config.tf_model_config
        self.num_layers = model_config.num_layers
        self.attention_head_dim = model_config.attention_head_dim
        self.num_attention_heads = model_config.num_attention_heads
        self.joint_attention_dim = model_config.joint_attention_dim
        self.in_channels = model_config.in_channels
        self.out_channels = model_config.out_channels or self.in_channels
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.guidance_embeds = model_config.guidance_embeds
        self.axes_dims_rope = model_config.axes_dims_rope

        if not use_layer3d_rope:
            self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(self.axes_dims_rope), scale_rope=True)
        else:
            self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=list(self.axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, use_additional_t_cond=use_additional_t_cond
        )

        self.txt_norm = RMSNorm(self.joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(self.joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t
