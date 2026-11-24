# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""RL routing replay around FSDPTurbo's native Kimi K3.1 DSA model."""

from .kimi_k31_r3 import KimiK31ReplayForCausalLM, KimiK31ReplayForConditionalGeneration


class KimiK31ForConditionalGeneration(KimiK31ReplayForConditionalGeneration):
    """Construct native DSA and retain explicit RL route replay and indexer freezing."""

    def __init__(self, config):
        text = config.text_config
        dsa = text.dsa_config
        if dsa is None:
            raise ValueError("Kimi K3.1 requires text_config.dsa_config")
        full_layers = set(text.linear_attn_config["full_attn_layers"])
        if set(dsa["dsa_layers"]) != full_layers:
            raise ValueError("K3.1 DSA layers must match the non-KDA layers")
        if dsa.get("index_topk_freq", 1) != 1:
            raise ValueError("This training adapter requires an indexer on each DSA layer")
        if dsa.get("index_rope_dim", 0) != 0 or dsa.get("indexer_use_fp8", False):
            raise ValueError("The native training path requires a BF16 NoPE indexer")
        super().__init__(config)
        for index, layer in enumerate(self.language_model.model.layers):
            if index + 1 in full_layers:
                # Keep the existing RL policy: no auxiliary indexer loss is
                # added to PPO/GRPO, and indexer weights stay fixed.
                layer.self_attn.indexer.requires_grad_(False)
        self.language_model = KimiK31ReplayForCausalLM(self.language_model)
        self.post_init()
