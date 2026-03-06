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

import math

import pytest

from verl.utils.flops_counter import FlopsCounter

VALID_CONFIG_TYPE = {"llama", "qwen2", "qwen3", "qwen3_moe", "deepseek_v3", "mistral", "gemma3_text", "apertus"}


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


CONFIG = {
    "llama": {
        "config": {  # llama2-7B
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # 6*(vocab*hidden*2+layer*(hidden*(q+k+v+head*head_dim)+ hidden*inter*3))*token_sum +
        # 6*sum(seqlen^2)*layer*head*head_dim
        # 6*(32000*4096*2+32*(4096*4096*4+4096*11008*3))*(512+1024+2048) +
        # 6*(512*512+1024*1024+2048*2048)*32*4096
        # 6*(32000*4096*2+32*(4096*4096*4+4096*11008*3))*(4096+4096+4096) +
        # 6*(4096*4096+4096*4096+4096*4096)*32*4096
        "expected_flops_tuple": (149226491215872 / 1e12, 536372695793664 / 1e12),
    },
    "qwen2": {
        "config": {  # Qwen/Qwen2.5-7B-Instruct
            "model_type": "qwen2",
            "vocab_size": 152064,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # 6*(vocab*hidden*2+layer*(hidden*(q+k+v+head*head_dim)+ hidden*inter*3))*token_sum +
        # 6*sum(seqlen^2)*layer*head*head_dim
        # 6*(152064*3584*2+28*(3584*(3584+512+512+3584)+3584*18944*3))*(512+1024+2048) +
        # 6*(512*512+1024*1024+2048*2048)*28*3584
        # 6*(152064*3584*2+28*(3584*(3584+512+512+3584)+3584*18944*3))*(4096+4096+4096) +
        # 6*(4096*4096+4096*4096+4096*4096)*28*3584
        "expected_flops_tuple": (167073690943488 / 1e12, 591764889010176 / 1e12),
    },
    "qwen3": {
        "config": {  # Qwen/Qwen3-8B
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # 6*(vocab*hidden*2+layer*(hidden*(q+k+v+head*head_dim)+ hidden*inter*3))*token_sum +
        # 6*sum(seqlen^2)*layer*head*head_dim
        # 6*(151936*4096*2+36*(4096*(128*32+128*8*2+128*32)+4096*12288*3))*(512+1024+2048) +
        # 6*(512*512+1024*1024+2048*2048)*36*128*32
        # 6*(151936*4096*2+36*(4096*(128*32+128*8*2+128*32)+4096*12288*3))*(4096+4096+4096) +
        # 6*(4096*4096+4096*4096+4096*4096)*36*128*32
        "expected_flops_tuple": (180997438046208 / 1e12, 648394032807936 / 1e12),
    },
    "qwen3_moe": {
        "config": {  # Qwen/Qwen3-30B-A3B-Base
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "vocab_size": 151936,
            "num_hidden_layers": 48,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "head_dim": 128,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 8,
            "num_experts": 128,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # 6*(vocab*hidden*2+layer*(hidden*(q+k+v+head*head_dim)+hidden*inter*top_k_exp*3 +
        # hidden*num_experts))*token_sum + 6*sum(seqlen^2)*layer*head*head_dim
        # 6*(151936*2048*2+48*(2048*(128*32+128*4*2+128*32)+2048*768*8*3+2048*128))*(512+1024+2048) +
        # 6*(512*512+1024*1024+2048*2048)*48*128*32
        # 6*(151936*2048*2+48*(2048*(128*32+128*4*2+128*32)+2048*768*8*3+2048*128))*(4096+4096+4096) +
        # 6*(4096*4096+4096*4096+4096*4096)*48*128*32
        "expected_flops_tuple": (78593069678592 / 1e12, 306570470621184 / 1e12),
    },
    "deepseek_v3": {
        "config": {  # deepseek-ai/DeepSeek-Prover-V2-671B
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "vocab_size": 129280,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "first_k_dense_replace": 3,
            "num_attention_heads": 128,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 18432,
            "qk_nope_head_dim": 128,
            "q_lora_rank": 1536,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # (1536*7168+128*192*1536+7168*(512+64)+128*(128+128)*512+128*128*7168) = 187105280
        # 6*(129280*7168*2+ 3*(7168*18432*3+187105280)+ 58*(187105280+7168*256+7168*2048*9*3))*(512+1024+2048) +
        # 3*(512*512+1024*1024+2048*2048)*61*(192+128)*128
        # 6*(129280*7168*2+ 3*(7168*18432*3+187105280)+ 58*(187105280+7168*256+7168*2048*9*3))*(4096+4096+4096) +
        # 3*(4096*4096+4096*4096+4096*4096)*61*(192+128)*128
        "expected_flops_tuple": (848766538088448 / 1e12, 3145850406567936 / 1e12),
    },
    "mistral": {
        "config": {  # mistralai/Mistral-Small-24B-Instruct-2501
            "model_type": "mistral",
            "vocab_size": 131072,
            "hidden_size": 5120,
            "intermediate_size": 32768,
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # Mistral uses same architecture as Llama, with GQA
        # 6*(vocab*hidden*2+layer*(hidden*(q+k+v+head*head_dim)+ hidden*inter*3))*token_sum +
        # 12*sum(seqlen^2)*layer*head*head_dim
        # vocab part: 131072*5120*2 = 1342177280
        # attn part per layer: 5120*(128*32+128*8+128*8+128*32) = 5120*10240 = 52428800
        # mlp part per layer: 5120*32768*3 = 503316480
        # total per layer: 52428800 + 503316480 = 555745280
        # all layers: 1342177280 + 40*555745280 = 23571988480
        # For batch [512, 1024, 2048], tokens_sum = 3584:
        # dense flops: 6 * 23571988480 * 3584 = 506892040273920
        # attn flops: 6 * 5505024 * 40 * 128 * 32 = 10823317585920
        # total: 517715357859840 / 1e12 = 517.71535785984
        # For batch [4096, 4096, 4096], tokens_sum = 12288:
        # dense flops: 6 * 23571988480 * 12288 = 1737915566653440
        # attn flops: 6 * 50331648 * 40 * 128 * 32 = 98956046499840
        # total: 1836871613153280 / 1e12 = 1836.87161315328
        "expected_flops_tuple": (512303699066880 / 1e12, 1787393589903360 / 1e12),
    },
    "gemma3_text": {
        "config": {  # Gemma3-12B-IT-TextOnly
            "model_type": "gemma3_text",
            "vocab_size": 262208,
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "sliding_window": 1024,
            "layer_types": None,
            # Will be auto-generated based on sliding_window_pattern
            "sliding_window_pattern": 6,
            # Every 6th layer is full attention
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # Gemma3 has alternating sliding window attention
        # With sliding_window_pattern=6: layers 5,11,17,23,29,35,41,47 use full attention (8 layers)
        # Other 40 layers use sliding window attention with window_size=1024
        #
        # Non-attention FLOPs:
        # vocab part: 262208*3840*2 = 2013757440
        # attn part per layer: 3840*(256*16+256*8+256*8+256*16) = 3840*12288 = 47185920
        # mlp part per layer: 3840*15360*3 = 176947200
        # total per layer: 47185920 + 176947200 = 224133120
        # all layers: 2013757440 + 48*224133120 = 12772147200
        #
        # For batch [512, 1024, 2048], tokens_sum = 3584:
        # dense flops: 6 * 12772147200 * 3584 = 274652253388800
        # seqlen_square_sum: 180355072 (calculated with sliding window logic)
        # attn flops: 6 * 180355072 * 256 * 16 = 8864812498944
        # total: 283517065887744 / 1e12 = 283.517065887744
        #
        # For batch [4096, 4096, 4096], tokens_sum = 12288:
        # dense flops: 6 * 12772147200 * 12288 = 941664868761600
        # seqlen_square_sum: 905969664 (calculated with sliding window logic)
        # attn flops: 6 * 905969664 * 256 * 16 = 44530220924928
        # total: 986195089686528 / 1e12 = 986.195089686528
        "expected_flops_tuple": (279084659638272 / 1e12, 963929979224064 / 1e12),
    },
    "gpt_oss": {
        "config": {
            "model_type": "gpt_oss",
            "vocab_size": 201088,
            "hidden_size": 2880,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "intermediate_size": 2880,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "sliding_window": 128,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # GPT-OSS has alternating sliding / full attention
        # Even layers (12 layers) use sliding window attention with window_size = 128
        # Odd layers  (12 layers) use full attention
        #
        # Non-attention FLOPs:
        # vocab part: 201088 * 2880 * 2 = 1158266880
        # attn linear part per layer:
        #   Q: 2880 * (64 * 64) = 11796480
        #   K: 2880 * (8  * 64) = 1474560
        #   V: 2880 * (8  * 64) = 1474560
        #   O: (64 * 64) * 2880 = 11796480
        #   attn linear total = 26542080
        # mlp (MoE, SwiGLU) part per layer:
        #   gate: 2880 * 32 = 92160
        #   active experts: 3 * 2880 * 2880 * 4 = 99532800
        #   mlp total = 99624960
        # total per layer: 26542080 + 99624960 = 126167040
        # all layers:
        #   126167040 * 24 = 3028008960
        # total dense params:
        #   3028008960 + 1158266880 = 4186275840
        #
        # For batch [512, 1024, 2048], tokens_sum = 3584:
        # dense flops: 6 * 4186275840 * 3584 = 90021675663360
        # seqlen_square_sum: 71565312 (calculated with sliding window logic)
        # attn flops: 6 * 71565312 * 64 * 64 = 3517578215424
        # total: 93539253878784 / 1e12 = 93.539253878784
        #
        # For batch [4096, 4096, 4096], tokens_sum = 12288:
        # dense flops: 6 * 4186275840 * 12288 = 308646629068800
        # seqlen_square_sum: 622854144 (calculated with sliding window logic)
        # attn flops: 6 * 622854144 * 64 * 64 = 30613642948608
        # total: 339260272017408 / 1e12 = 339.260272017408
        "expected_flops_tuple": (91780464771072 / 1e12, 323953008574464 / 1e12),
    },
    "apertus": {
        "config": {  # swiss-ai/Apertus-8B
            "model_type": "apertus",
            "vocab_size": 131072,
            "hidden_size": 4096,
            "intermediate_size": 21504,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "hidden_act": "xielu",
            # head_dim will be derived as 4096 / 32 = 128
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # Calculation for Apertus (hidden_act="xielu" -> MLP uses [k_mlp=2]*H*I params; qk_norm=True -> [k_qkn=2]*H):
        # V=131072, H=4096, I=21504, L=32, k_mlp=2 (XIELU), k_qkn=2 (QK norm), S=6
        # S*(2*V*H + L*(4*H**2 + k_mlp*H*I + k_qkn*H)) * (SUM[seqlen]) + 6*SUM[seqlen**2]*L*H
        "expected_flops_tuple": (194825353691136 / 1e12, 692711652851712 / 1e12),
    },
    "qwen3_vl": {
        "config": {  # Qwen/Qwen3-VL-8B
            "model_type": "qwen3_vl",
            # -------- Text config --------
            "text_config": {
                "vocab_size": 151936,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
            },
            # -------- Vision config (ViT) --------
            "vision_config": {
                "deepstack_visual_indexes": [8, 16, 24],
                "num_heads": 16,
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 4096,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
            },
        },
        "batch_seqlens_tuple": (
            [512, 1024, 2048],
            [4096, 4096, 4096],
        ),
        "images_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # -----Text-----
        # 6*(vocab*hidden*2
        #   + layer*(hidden*(q+k+v+o) + hidden*inter*3)
        # )*token_sum
        # + 6*sum(seqlen^2)*layer*hidden
        #
        # -----ViT-----
        # patch_embed_N =hidden*temporal_patch_size*in_channels* patch_size^2
        # attn_linear_N =hidden*(4*hidden)
        # mlp_N =hidden*inter*2
        # merger_N =((o+hidden*spatial_merge_size^2) * (hidden*spatial_merge_size^2))
        # deepstack_merger_N =merger_N * 3
        # dense_N =patch_embed_N + (attn_linear_N + mlp_N) * 27 + deepstack_merger_N + merger_N
        #
        # 6*(151936*4096*2
        #   + 36*(4096*(4096+1024+1024+4096) + 4096*12288*3)
        # )*(512+1024+2048)
        # + 12*(512*512+1024*1024+2048*2048)*36*4096
        # + 6 * dense_N * (512 + 1024 + 2048)
        # + 12 * (512**2 + 1024**2 + 2048**2) * 27 * 16 * 72
        #
        # 6*(151936*4096*2
        #   + 36*(4096*(4096+1024+1024+4096) + 4096*12288*3)
        # )*(4096+4096+4096)
        # + 12*(4096*4096+4096*4096+4096*4096)*36*4096
        # + 6 * dense_N * (4096 + 4096 + 2048)
        # + 12 * (4096**2 + 4096**2 + 4096**2) * 27 * 16 * 72
        "expected_flops_tuple": (
            195379819708416 / 1e12,
            709446422495232 / 1e12,
        ),
    },
    "qwen3_vl_moe": {
        "config": {  # Qwen/Qwen3-VL-30B-A3B
            "model_type": "qwen3_vl_moe",
            # -------- Text config --------
            "text_config": {
                "vocab_size": 151936,
                "hidden_size": 2048,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "moe_intermediate_size": 768,
                "num_experts": 128,
                "num_experts_per_tok": 8,
            },
            # -------- Vision config (ViT) --------
            "vision_config": {
                "deepstack_visual_indexes": [8, 16, 24],
                "num_heads": 16,
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 4096,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
            },
        },
        "batch_seqlens_tuple": (
            [512, 1024, 2048],
            [4096, 4096, 4096],
        ),
        "images_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # -----Text-----
        # 6*(vocab*hidden*2
        #   + layer*(hidden*(q+k+v+head*head_dim)+hidden*inter*top_k_exp*3+hidden*num_experts)
        # )*token_sum
        # + 6*sum(seqlen^2)*layer*hidden
        #
        # -----ViT-----
        # patch_embed_N =hidden*temporal_patch_size*in_channels* patch_size^2
        # attn_linear_N =hidden*(4*hidden)
        # mlp_N =hidden*inter*2
        # merger_N =((o+hidden*spatial_merge_size^2) * (hidden*spatial_merge_size^2))
        # deepstack_merger_N =merger_N * 3
        # dense_N =patch_embed_N + (attn_linear_N + mlp_N) * 27 + deepstack_merger_N + merger_N
        #
        # 6*(151936*2048*2
        #   + 48*(2048*(128*32+128*4*2+128*32)+2048*768*8*3+2048*128)
        # )*(512+1024+2048)
        # + 12*(512*512+1024*1024+2048*2048)*48*4096
        # + 6 * dense_N * (512 + 1024 + 2048)
        # + 12 * (512**2 + 1024**2 + 2048**2) * 27 * 16 * 72
        #
        # 6*(151936*2048*2
        #   48*(2048*(128*32+128*4*2+128*32)+2048*768*8*3+2048*128)
        # )*(4096+4096+4096)
        # + 12*(4096*4096+4096*4096+4096*4096)*48*4096
        # + 6 * dense_N * (4096 + 4096 + 2048)
        # + 12 * (4096**2 + 4096**2 + 4096**2) * 27 * 16 * 72
        "expected_flops_tuple": (
            92975451340800 / 1e12,
            367622860308480 / 1e12,
        ),
    },
}


@pytest.mark.parametrize(
    "config_type",
    [
        "llama",
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "deepseek_v3",
        "mistral",
        "gemma3_text",
        "apertus",
        "gpt_oss",
        "qwen3_vl",
        "qwen3_vl_moe",
    ],
)
def test_flops_counter(config_type: str):
    test_config = CONFIG[config_type]
    config = Config(test_config["config"])
    flops_counter = FlopsCounter(config)
    if "images_seqlens_tuple" in test_config:
        for batch_seqlens, images_seqlens, expected_flops in zip(
            test_config["batch_seqlens_tuple"],
            test_config["images_seqlens_tuple"],
            test_config["expected_flops_tuple"],
            strict=True,
        ):
            # set delta time to 1 to get the flops
            counted_flops, _ = flops_counter.estimate_flops(batch_seqlens, 1, images_seqlens=images_seqlens)
            print(f"Expect flops for {test_config['config']} is {expected_flops}, but get {counted_flops}")
            assert math.isclose(counted_flops, expected_flops), (
                f"Expect flops for {test_config['config']} is {expected_flops}, but get {counted_flops}"
            )
    else:
        for batch_seqlens, expected_flops in zip(
            test_config["batch_seqlens_tuple"], test_config["expected_flops_tuple"], strict=True
        ):
            # set delta time to 1 to get the flops
            counted_flops, _ = flops_counter.estimate_flops(batch_seqlens, 1)
            print(f"Expect flops for {test_config['config']} is {expected_flops}, but get {counted_flops}")
            assert math.isclose(counted_flops, expected_flops), (
                f"Expect flops for {test_config['config']} is {expected_flops}, but get {counted_flops}"
            )
