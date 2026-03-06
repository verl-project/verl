# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from https://github.com/PRIME-RL/SimpleVLA-RL/blob/main/verl/utils/vla_utils/openvla_oft/
# form https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-trajall/blob/main/

"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions.
Inherits from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained,
but exactly replicate the logic in `prismatic.models.vlms.prismatic.py`.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Optional

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig
from .constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)
from .train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)

# Set up logger
logger = logging.getLogger(__name__)


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    """
    Vision backbone for Prismatic models that handles image feature extraction.

    Supports both single backbone (e.g., SigLIP) and fused backbone (e.g., SigLIP + DINOv2) configurations.
    For fused backbones, features from both models are concatenated along the feature dimension.
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: list[int],
        timm_model_ids: list[str],
        timm_override_act_layers: list[Optional[str]],
    ) -> None:
        """
        Initialize the vision backbone.

        Args:
            use_fused_vision_backbone: Whether to use two backbones and fuse their features
            image_sizes: List of image sizes for each backbone
            timm_model_ids: List of TIMM model IDs to use for each backbone
            timm_override_act_layers: List of activation layer overrides for each backbone
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # Default value, can be overridden later

        # Validate number of (fused) vision backbones
        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        # Create primary featurizer
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # Create secondary featurizer if using fused backbone
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch LayerScale modules for HF compatibility
        self._patch_layer_scales()

    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        Create a TIMM-based featurizer model with appropriate configurations.

        Args:
            model_id: The TIMM model ID to load
            img_size: Input image size for the model
            act_layer: Override for the activation layer type

        Returns:
            A configured featurizer model
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch the forward function to extract the second-to-last layer features
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer

    def _patch_layer_scales(self) -> None:
        """
        Patch all LayerScale modules to be compatible with HF's parameter naming.

        HF Transformers overwrites parameters with names containing 'gamma',
        so we need to rename and modify the forward method.
        """
        # Patch primary featurizer
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # Patch secondary featurizer if it exists
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        """
        Returns the number of vision patches output by the vision backbone.

        Returns:
            Number of patches per image
        """
        return self.featurizer.patch_embed.num_patches

    def get_num_images_in_input(self) -> int:
        """
        Returns the number of input images for the vision backbone.

        Returns:
            Number of images expected in the input
        """
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """
        Sets the number of input images for the vision backbone.

        Args:
            num_images_in_input: Number of images to expect in the input
        """
        self.num_images_in_input = num_images_in_input

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone.

        If `self.use_fused_vision_backbone == True`, uses both SigLIP and DINOv2 transformers to extract visual features
        (otherwise uses SigLIP only). Allows multi-image inputs (but only for fused vision backbone).

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
        """
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # Get patches from both SigLIP and DINOv2 vision transformers
                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)

                # Concatenate SigLIP and DINOv2 patches along the hidden dimension
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # Concatenate all patches along the patch dimension
            return torch.cat(all_patches, dim=1)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[list[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear | nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        """
        Replace embeddings in input_embeddings at positions where all_actions_mask is True
        with embeddings from noisy_action_features, using vectorized operations.

        Args:
            input_embeddings: Tensor of shape (B, S, D)
            all_actions_mask: Boolean tensor of shape (B, S)
            noisy_action_features: Tensor of shape (B, K, D) where K is the number of True values in mask per sample

        Returns:
            Modified input_embeddings tensor
        """
        # Clone input to avoid modifying the original tensor
        new_input_embeddings = input_embeddings.clone()

        # Create a tensor with the same shape of input_embeddings to hold the noisy action features
        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        # Create batch indices for splicing
        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        # Get indices where mask is True for each sample
        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        # Move the noisy action features into their correct positions
        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

        # Combine original input embeddings and noisy action embeddings using the mask
        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )

        return new_input_embeddings

    def _process_action_masks(self, labels):
        """Helper to get action masks from labels"""
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        """Process vision features with optional FiLM conditioning"""
        if use_film:
            # FiLM: Infuse language inputs into visual features
            patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
        else:
            patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

        # Project patch embeddings into language embedding space
        return self.projector(patch_features)

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """Process proprioceptive features and append to vision features"""
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
            # proprio: (bsz, proprio_dim) or (propro_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # For simplicity, just append proprio token to the end of projected vision patch tokens
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """Build multimodal embeddings and attention mask"""
        # Update attention mask
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        return multimodal_embeddings, multimodal_attention_mask

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """Build multimodal labels with IGNORE_INDEX for patch embeddings"""
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    # === Core Prismatic VLM `forward()` Logic ===
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     output_projector_features: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     proprio=None,
    #     proprio_projector=None,
    #     noisy_actions=None,
    #     noisy_action_projector=None,
    #     diffusion_timestep_embeddings=None,
    #     use_film: bool = False,
    # ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
    #     """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     output_projector_features = output_projector_features if output_projector_features is not None else False
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
    #     use_cache = use_cache and not self.training

    #     # Instantiate Placeholder for Projector Features
    #     projected_patch_embeddings = None

    #     # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
    #     if input_ids.shape[1] == 1:
    #         assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
    #         assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
    #         assert labels is None, "Unexpected key `labels` provided during cached generation!"

    #         language_model_output = self.language_model(
    #             input_ids=input_ids,
    #             attention_mask=None,
    #             position_ids=None,
    #             past_key_values=past_key_values,
    #             inputs_embeds=None,
    #             labels=None,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    #     # === Handle Unimodal Forward ===
    #     elif pixel_values is None:
    #         assert (input_ids is not None) and (inputs_embeds is None), \
    #             "Missing `input_ids` in language-only forward!"
    #         assert past_key_values is None, \
    #             "Unexpected key `past_key_values` provided during language-only forward!"

    #         language_model_output = self.language_model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=None,
    #             past_key_values=None,
    #             inputs_embeds=None,
    #             labels=labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    #     # === Handle Multimodal Forward ===
    #     elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
    #         assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

    #         #test
    #
    #         #test end

    #         # Get input embeddings (from language model embeddings)
    #         input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

    #         # Extract action masks
    #         all_actions_mask = self._process_action_masks(labels)

    #         # Extract the language portion of the input embeddings (i.e. remove the action tokens portion)
    #         language_embeddings = input_embeddings[~all_actions_mask].reshape(
    #             input_embeddings.shape[0], -1, input_embeddings.shape[2]
    #         )  # (B, lang_seq_len, llm_dim)

    #         # Get visual features
    #         projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

    #         # Add proprioceptive state if provided
    #         projected_patch_embeddings = self._process_proprio_features(
    #             projected_patch_embeddings, proprio, proprio_projector
    #         )

    #         # [Diffusion] Add diffusion timestep embedding if provided
    #         if diffusion_timestep_embeddings is not None:
    #             # For simplicity, just append diffusion timestep embedding to the end of projected vision patch tokens
    #             projected_patch_embeddings = torch.cat(
    #                 (projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
    #             )

    #         # Process action embeddings
    #         if noisy_actions is not None:
    #             # Get mask corresponding to all action tokens
    #             all_actions_mask = self._process_action_masks(labels)

    #             # Reshape noisy actions into individual action tokens
    #             # noisy_actions: (B, chunk_len, action_dim) -> (B, chunk_len * action_dim, 1)
    #             B = noisy_actions.shape[0]
    #             noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

    #             # Project noisy action tokens into language model embedding space
    #             noisy_action_features = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)

    #             # Replace embeddings of the action tokens with noisy action embeddings
    #             input_embeddings = self._replace_input_embeddings(
    #                 input_embeddings, all_actions_mask, noisy_action_features
    #             )
    #         else:
    #             # Replace the embeddings of the action tokens with zeros
    #             # (Later on, the positional embeddings will be added to them)
    #             all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
    #             input_embeddings = input_embeddings * ~all_actions_mask

    #         # Build multimodal embeddings & attention mask
    #         multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
    #             input_embeddings, projected_patch_embeddings, attention_mask
    #         )

    #         # Build labels for multimodal sequence if needed
    #         multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)

    #         # Dispatch to language model
    #         language_model_output = self.language_model(
    #             input_ids=None,
    #             attention_mask=multimodal_attention_mask,
    #             position_ids=None,
    #             past_key_values=None,
    #             inputs_embeds=multimodal_embeddings,
    #             labels=multimodal_labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    #     # === Otherwise =>> Assume Invalid! ===
    #     elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
    #         raise ValueError("Non-homogenous batch of (text, image) input \
    #                          -- forward() does not support mixed batches!")

    #     else:
    #         raise ValueError(
    #             "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
    #             f"=> `input_ids` = {input_ids is not None}\n"
    #             f"=> `attention_mask` = {attention_mask is not None}\n"
    #             f"=> `pixel_values` = {pixel_values is not None}\n"
    #             f"=> `labels` = {labels is not None}\n"
    #             f"=> `input_embeds` = {inputs_embeds is not None}\n"
    #             f"=> `past_key_values` = {past_key_values is not None}\n"
    #             f"=> `use_cache` = {use_cache}"
    #         )

    #     # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
    #     if not return_dict:
    #         if output_projector_features and (projected_patch_embeddings is not None):
    #             return *language_model_output, projected_patch_embeddings

    #         return language_model_output

    #     return PrismaticCausalLMOutputWithPast(
    #         loss=language_model_output.loss,
    #         logits=language_model_output.logits,
    #         past_key_values=language_model_output.past_key_values,
    #         hidden_states=language_model_output.hidden_states,
    #         attentions=language_model_output.attentions,
    #         projector_features=projected_patch_embeddings,
    #     )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)

    def _prepare_input_for_action_prediction_verl(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction_verl(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        return labels

    def _verl_discrete_compute_logits(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):  # contintue!!!!!
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        # last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        # actions_hidden_states = last_hidden_states[
        #     :,
        #     NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
        #     :,
        # ]  # (B, act_chunk_len, D)

        # Handle different prediction methods
        # if action_head is not None:
        #     # L1 regression prediction
        #     normalized_actions = action_head.predict_action(actions_hidden_states)
        #     normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        #     normalized_actions = normalized_actions.float().cpu().detach().numpy()
        # else:
        # Discrete token-based prediction

        compute_logits = language_model_output.logits[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
        ]

        return compute_logits

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     unnorm_key: Optional[str] = None,
    #     proprio=None,
    #     proprio_projector=None,
    #     action_head=None,
    #     noisy_action_projector=None,
    #     use_film: bool = False,
    #     **kwargs: str,
    # ) :
    #     """Predict actions from input sequence, with options for different prediction methods.

    #     Args:
    #         input_ids: Input token ids
    #         unnorm_key: Key for unnormalization statistics
    #         proprio: Proprioceptive features
    #         proprio_projector: Projector for proprioceptive features
    #         action_head: Optional head for L1 regression or diffusion-based prediction
    #         noisy_action_projector: Projector for noisy actions in diffusion-based prediction
    #         use_film: Whether to use FiLM conditioning
    #         **kwargs: Additional arguments including pixel_values and attention_mask

    #     Returns:
    #         Tuple of (unnormalized_actions, action_hidden_states)
    #     """
    #     # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    #     # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    #     # if not torch.all(input_ids[:, -1] == 29871):
    #     #     input_ids = torch.cat(
    #     #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
    #     #     )
    #     #print("!!!!!!!!!!!!!!Entering forward!!!!!!!!!!")
    #     pixel_values = kwargs["pixel_values"]
    #     attention_mask = kwargs["attention_mask"]

    #     # Create fake labels tensor (needed for action mask)
    #     labels = input_ids.clone()
    #     labels[:] = IGNORE_INDEX

    #     # Get number of tokens in prompt (excluding the start token)
    #     NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

    #     # Prepare inputs by adding necessary tokens
    #     #input_ids, attention_mask = self._prepare_input_for_action_prediction_verl(input_ids, attention_mask)

    #     #test
    #     placeholder_action_token_ids = (
    #         torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
    #     )
    #     input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

    #     # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
    #     stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
    #     input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

    #     # Extend the attention mask to fit the new shape of input
    #     # Note: Only batch size == 1 supported right now
    #     mask_extension = (
    #         torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
    #         .to(attention_mask.device)
    #         .to(attention_mask.dtype)
    #     )
    #     attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

    #     #return input_ids, attention_mask

    #     #test end

    #     # Update labels tensor for action mask computation later
    #     #labels = self._prepare_labels_for_action_prediction_verl(labels, input_ids)
    #     #test

    #     ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
    #     labels_extension = (
    #         torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
    #         * ARBITRARY_ACTION_TOKEN_IDX
    #     )
    #     labels = torch.cat([labels, labels_extension], dim=-1)

    #     # Replace last label token with stop token
    #     labels[:, -1] = STOP_INDEX

    #     #return labels

    #     #test ed

    #     # Get input embeddings and action masks

    #     input_embeddings = self.get_input_embeddings()(input_ids)

    #     #all_actions_mask = self._process_action_masks(labels)
    #     #test
    #     #current_action_mask = get_current_action_mask(labels)
    #     newline_positions = labels != IGNORE_INDEX

    #     # Calculate cumulative sum to identify regions between newlines
    #     cumsum = torch.cumsum(newline_positions, dim=1)

    #     # Create the mask
    #     mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    #     # Extract the action part only
    #     action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
    #     current_action_mask = action_tokens_only_mask * mask

    #     #next_actions_mask = get_next_actions_mask(labels)
    #     newline_positions = labels != IGNORE_INDEX

    #     # Calculate cumulative sum to identify regions between newlines
    #     cumsum = torch.cumsum(newline_positions, dim=1)

    #     # Create the mask
    #     mask = cumsum > ACTION_DIM

    #     # Extract the action part only
    #     action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
    #     next_actions_mask = action_tokens_only_mask * mask

    #     all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)

    #     #test end

    #     # Extract language embeddings
    #     language_embeddings = input_embeddings[~all_actions_mask].reshape(
    #         input_embeddings.shape[0], -1, input_embeddings.shape[2]
    #     )

    #     # Process vision features
    #     #projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)
    #     #test
    #     if use_film:
    #         # FiLM: Infuse language inputs into visual features
    #         raise ValueError
    #         patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
    #     else:
    #         patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

    #     projected_patch_embeddings = self.projector(patch_features)
    #     #test end

    #     # Add proprioceptive features if provided
    #     use_proprio = proprio_projector is not None and proprio is not None
    #     if use_proprio:
    #         proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device,
    #                                            dtype=projected_patch_embeddings.dtype)
    #         projected_patch_embeddings = self._process_proprio_features(
    #             projected_patch_embeddings, proprio, proprio_projector
    #         )

    #     # Use diffusion if provided, otherwise use regression or discrete prediction
    #     use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

    #     # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
    #     NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
    #     if use_proprio:
    #         NUM_PATCHES += 1
    #     if use_diffusion:
    #         NUM_PATCHES += 1

    #     if use_diffusion:
    #         raise ValueError
    #         # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
    #         noise = torch.randn(
    #             size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
    #         )

    #         # Run diffusion-based prediction
    #         normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
    #             input_embeddings,
    #             all_actions_mask,
    #             noise,
    #             action_head,
    #             projected_patch_embeddings,
    #             labels,
    #             attention_mask,
    #             NUM_PATCHES,
    #             NUM_PROMPT_TOKENS,
    #             noisy_action_projector,
    #         )
    #     else:
    #         # Run regression or discrete token-based prediction
    #         # compute_logits = self._verl_discrete_compute_logits(
    #         #     input_embeddings,
    #         #     all_actions_mask,
    #         #     projected_patch_embeddings,
    #         #     attention_mask,
    #         #     labels,
    #         #     NUM_PATCHES,
    #         #     NUM_PROMPT_TOKENS,
    #         #     action_head,
    #         # )

    #         #test

    #         all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
    #         input_embeddings = input_embeddings * ~all_actions_mask

    #         # Build multimodal embeddings and attention mask
    #         # multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
    #         #     input_embeddings, projected_patch_embeddings, attention_mask
    #         # )
    #         #test

    #         projected_patch_attention_mask = None
    #         if attention_mask is not None:
    #             projected_patch_attention_mask = torch.full(
    #                 (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #                 fill_value=True,
    #                 dtype=attention_mask.dtype,
    #                 device=attention_mask.device,
    #             )

    #         # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
    #         multimodal_embeddings = torch.cat(
    #             [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
    #         )

    #         multimodal_attention_mask = None
    #         if attention_mask is not None:
    #             multimodal_attention_mask = torch.cat(
    #                 [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
    #             )

    #         #return multimodal_embeddings, multimodal_attention_mask

    #         #test end

    #         # Forward pass through language model
    #         language_model_output = self.language_model(
    #             input_ids=None,
    #             attention_mask=multimodal_attention_mask,
    #             position_ids=None,
    #             past_key_values=None,
    #             inputs_embeds=multimodal_embeddings,
    #             labels=None,
    #             use_cache=None,
    #             output_attentions=False,
    #             output_hidden_states=False,
    #             return_dict=True,
    #         )

    #         compute_logits = language_model_output.logits[
    #                     :,
    #                     NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + \
    #                     NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
    #                 ]

    #         #test end

    #     return compute_logits

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values=None,
        attention_mask=None,
        # labels=None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ):
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        # if not torch.all(input_ids[:, -1] == 29871):
        #     input_ids = torch.cat(
        #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        #     )

        # pixel_values = kwargs["pixel_values"]
        # attention_mask = kwargs["attention_mask"]

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

        # # Prepare inputs by adding necessary tokens
        # #input_ids, attention_mask = self._prepare_input_for_action_prediction_verl(input_ids, attention_mask)

        # #test
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        # Get input embeddings and action masks

        # NUM_PROMPT_TOKENS = kwargs["num_prompt_tokens"]

        input_embeddings = self.get_input_embeddings()(input_ids)

        # all_actions_mask = self._process_action_masks(labels)
        # test
        # current_action_mask = get_current_action_mask(labels)
        newline_positions = labels != IGNORE_INDEX

        # Calculate cumulative sum to identify regions between newlines
        cumsum = torch.cumsum(newline_positions, dim=1)

        # Create the mask
        mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

        # Extract the action part only
        action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
        current_action_mask = action_tokens_only_mask * mask

        # next_actions_mask = get_next_actions_mask(labels)
        newline_positions = labels != IGNORE_INDEX

        # Calculate cumulative sum to identify regions between newlines
        cumsum = torch.cumsum(newline_positions, dim=1)

        # Create the mask
        mask = cumsum > ACTION_DIM

        # Extract the action part only
        action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
        next_actions_mask = action_tokens_only_mask * mask

        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)

        # test end

        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # Process vision features
        # projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)
        # test
        if use_film:
            # FiLM: Infuse language inputs into visual features
            raise ValueError
            patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
        else:
            patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

        projected_patch_embeddings = self.projector(patch_features)
        # test end

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(
                projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype
            )
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            raise ValueError
            # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # Run diffusion-based prediction
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            # Run regression or discrete token-based prediction
            # compute_logits = self._verl_discrete_compute_logits(
            #     input_embeddings,
            #     all_actions_mask,
            #     projected_patch_embeddings,
            #     attention_mask,
            #     labels,
            #     NUM_PATCHES,
            #     NUM_PROMPT_TOKENS,
            #     action_head,
            # )

            # test

            all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
            input_embeddings = input_embeddings * ~all_actions_mask

            # Build multimodal embeddings and attention mask
            # multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            #     input_embeddings, projected_patch_embeddings, attention_mask
            # )
            # test

            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )

            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # return multimodal_embeddings, multimodal_attention_mask

            # test end

            # Forward pass through language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            compute_logits = language_model_output.logits[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            ]

            # test end

        return compute_logits


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig
    _supports_sdpa = True

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions

    def _run_diffusion_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        noise,
        action_head,
        projected_patch_embeddings,
        labels,
        attention_mask,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        noisy_action_projector,
    ):
        """Run diffusion-based action prediction"""
        # Set diffusion timestep values
        action_head.noise_scheduler.set_timesteps(action_head.num_diffusion_steps)
        # Clone embedding for reuse in each timestep
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise

        # Reverse diffusion: Iteratively denoise to generate action prediction
        for t in action_head.noise_scheduler.timesteps:
            # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action
            # embedding, and diffusion timestep embedding)
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )  # (B, llm_dim)
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

            # [Diffusion] Replace the embeddings of the action tokens with noisy actions
            # (Later on, the positional embeddings will be added to them)

            # For simplicity, append diffusion timestep embedding to the end of projected vision tokens
            projected_patch_embeddings = torch.cat(
                (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

            # Reshape and project noisy actions into language embedding space
            B = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            # Replace action token embeddings with noisy action embeddings
            input_embeddings = self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            # Build multimodal embeddings and attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # Forward pass through language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract hidden states for action portion of response
            last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                :,
            ]  # (B, act_chunk_len, D)

            # Predict noise and update noisy actions: x_t -> x_{t-1}
            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        # Return final actions
        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states

    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]  # (B, act_chunk_len, D)

        # Handle different prediction methods
        if action_head is not None:
            # L1 regression prediction
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # Discrete token-based prediction
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states

    def _verl_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        do_sample=True,
        temperature=1,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        # last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        # actions_hidden_states = last_hidden_states[
        #     :,
        #     NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
        #     :,
        # ]  # (B, act_chunk_len, D)

        # Handle different prediction methods
        # if action_head is not None:
        #     # L1 regression prediction
        #     normalized_actions = action_head.predict_action(actions_hidden_states)
        #     normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        #     normalized_actions = normalized_actions.float().cpu().detach().numpy()
        # else:
        # Discrete token-based prediction

        # test
        # NUM_PROMPT_TOKENS = NUM_PROMPT_TOKENS + NUM_PATCHES
        # j = torch.arange(language_model_output.logits.shape[1], device=NUM_PROMPT_TOKENS.device)
        # start = NUM_PROMPT_TOKENS.unsqueeze(1)
        # end = start + ACTION_DIM * NUM_ACTIONS_CHUNK
        # mask_2d = (j >= start) & (j < end)
        # mask = mask_2d.unsqueeze(-1)
        # actions_masks = mask.expand_as(language_model_output.logits)

        NUM_PROMPT_TOKENS = NUM_PROMPT_TOKENS + NUM_PATCHES
        batch_size = language_model_output.logits.shape[0]
        device = language_model_output.logits.device

        start_indices = NUM_PROMPT_TOKENS.unsqueeze(1)  # [batch_size, 1]
        position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=device).unsqueeze(0)  # [1, seq_length]
        seq_indices = start_indices + position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]
        # test end
        # test add
        # print("language_model_output",language_model_output.logits.shape[-1])
        # print("self.vocab_size",self.vocab_size) 32000
        # topk_values, topk_indices = torch.topk(language_model_output.logits, k=256, dim=-1)
        # print(topk_indices)
        # assert language_model_output.logits.shape[-1] == self.vocab_size
        # test add
        if not do_sample:
            # org
            # reponse_ids = language_model_output.logits[
            #         :,
            #         NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES +\
            #         NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            #     ].argmax(dim=2)
            # reponse_ids = language_model_output.logits[actions_masks].argmax(dim=2)
            # org end

            # padding
            # reponse_ids = language_model_output.logits[
            #     torch.arange(batch_size, device=device).unsqueeze(-1),
            #     seq_indices,
            #     :
            # ].argmax(dim=2)
            # padding end

            # padding + only get last 256 token
            reponse_ids_logits = language_model_output.logits[
                torch.arange(batch_size, device=device).unsqueeze(-1), seq_indices, :
            ]
            start_index = self.vocab_size - 256
            response_last256 = reponse_ids_logits[..., -256 - 64 : -64]  # Shape: [batch_size, seq_len, 256]
            last256_argmax = response_last256.argmax(dim=-1)  # Shape: [batch_size, seq_len]
            reponse_ids = last256_argmax + start_index  # Shape: [batch_size, seq_len]
            # padding + only get last 256 token end

            predicted_action_token_ids = reponse_ids.cpu().numpy()

        else:
            assert temperature > 0
            # org
            # action_logits  = language_model_output.logits[
            #         :,
            #         NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + \
            #         NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            #     ]
            # action_logits = language_model_output.logits[actions_masks]
            # org end

            action_logits = language_model_output.logits[
                torch.arange(batch_size, device=device).unsqueeze(-1), seq_indices, :
            ]
            # padding
            # scaled_logits = action_logits / temperature
            # probs = torch.softmax(scaled_logits, dim=-1)
            # probs_flat = probs.reshape(-1, probs.shape[-1])  # (B*act_chunk_len, vocab_size)
            # sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)  # (B*act_chunk_len, 1)
            # reponse_ids = sampled_indices_flat.view(action_logits.shape[0], -1)
            # padding end

            # padding + only get last 256 token
            action_logits_last256 = action_logits[..., -256 - 64 : -64]
            scaled_logits = action_logits_last256 / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            assert probs.shape[-1] == 256
            probs_flat = probs.reshape(-1, probs.shape[-1])
            sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
            original_ids_flat = sampled_indices_flat + (self.vocab_size - 256)
            reponse_ids = original_ids_flat.view(action_logits.shape[0], -1)
            # padding + only get last 256 token end

            predicted_action_token_ids = reponse_ids.cpu().numpy()

        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        # normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        normalized_actions = normalized_actions.reshape(-1, ACTION_DIM)

        return normalized_actions, reponse_ids
        # return normalized_actions, actions_hidden_states

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # Get input embeddings and action masks
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # Process vision features
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(
                projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype
            )
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # Run diffusion-based prediction
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            # Run regression or discrete token-based prediction
            normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                action_head,
            )

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states

    def generate_action_verl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        # if not torch.all(input_ids[:, -1] == 29871):
        #     input_ids = torch.cat(
        #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        #     )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]
        do_sample = kwargs["do_sample"]
        temperature = kwargs["temperature"]

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        # NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token
        # test
        padding_idx = kwargs["padding_idx"]
        num_prompt_tokens = input_ids.ne(padding_idx).sum(dim=1) - 1
        # test end

        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # here to convert padding from before to last
        # test
        padding_mask = input_ids.ne(padding_idx)
        assert torch.all(padding_mask == attention_mask.ne(0))
        # print("in predict_action padding_mask:", padding_mask)
        padding_mask = padding_mask.int()
        sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
        input_ids = torch.gather(input_ids, 1, sorted_indices)
        attention_mask = torch.gather(attention_mask, 1, sorted_indices)
        labels = torch.gather(labels, 1, sorted_indices)
        assert not use_film
        # test end

        # Get input embeddings and action masks
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # Process vision features
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(
                projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype
            )
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            raise ValueError
            # # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
            # noise = torch.randn(
            #     size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            # )

            # # Run diffusion-based prediction
            # normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
            #     input_embeddings,
            #     all_actions_mask,
            #     noise,
            #     action_head,
            #     projected_patch_embeddings,
            #     labels,
            #     attention_mask,
            #     NUM_PATCHES,
            #     NUM_PROMPT_TOKENS,
            #     noisy_action_projector,
            # )
        else:
            # Run regression or discrete token-based prediction
            normalized_actions, reponse_ids = self._verl_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                num_prompt_tokens,
                action_head,
                do_sample=do_sample,
                temperature=temperature,
            )

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)
        # verl add!
        actions = actions.reshape(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)
        #
        return actions, reponse_ids

    @staticmethod
    def _check_unnorm_key(norm_stats: dict[str, dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
