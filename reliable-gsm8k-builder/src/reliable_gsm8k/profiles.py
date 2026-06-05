from __future__ import annotations

from copy import deepcopy


MODEL_PROFILES: dict[str, dict] = {
    "qwen25_3b": {
        "backend": "transformers_causal_lm",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "tokenizer_name": "Qwen/Qwen2.5-3B-Instruct",
        "trust_remote_code": True,
        "local_files_only": False,
        "tokenizer_use_fast": True,
        "padding_side": "left",
        "device_map": "auto",
        "dtype": "float16",
        "attn_implementation": None,
        "enforce_attn_implementation": True,
    },
    "qwen25_3b_sglang": {
        "backend": "sglang",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "api_base": "http://127.0.0.1:30000/v1",
        "api_key": "EMPTY",
        "timeout_seconds": 120,
        "max_retries": 8,
        "retry_base_seconds": 1.0,
        "retry_max_seconds": 30.0,
    },
    "gpt4o_mini": {
        "backend": "openai_compatible",
        "model_name": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 120,
        "max_retries": 8,
        "retry_base_seconds": 1.0,
        "retry_max_seconds": 30.0,
    },
    "judgelm_7b": {
        "backend": "transformers_causal_lm",
        "model_name": "BAAI/JudgeLM-7B-v1.0",
        "tokenizer_name": "BAAI/JudgeLM-7B-v1.0",
        "trust_remote_code": True,
        "local_files_only": False,
        "tokenizer_use_fast": True,
        "device_map": "auto",
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "enforce_attn_implementation": True,
    },
    "gpt41_mini": {
        "backend": "openai_compatible",
        "model_name": "gpt-4.1-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 120,
        "max_retries": 8,
        "retry_base_seconds": 1.0,
        "retry_max_seconds": 30.0,
    },
}


INFERENCE_PROFILES: dict[str, dict] = {
    "greedy": {
        "max_new_tokens": 512,
        "max_length": 2048,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "use_chat_template": True,
        "system_prompt": "You are a helpful assistant.",
    },
    "sample_balanced": {
        "max_new_tokens": 512,
        "max_length": 2048,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_chat_template": True,
        "system_prompt": "You are a helpful assistant.",
    },
    "sample_creative": {
        "max_new_tokens": 512,
        "max_length": 2048,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 0.95,
        "use_chat_template": True,
        "system_prompt": "You are a helpful assistant.",
    },
    "greedy_different_system_prompt_test": {
        "max_new_tokens": 512,
        "max_length": 2048,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "use_chat_template": True,
        "system_prompt": "You are an extremely capable mathematician that has won many prices",
    },
    "greedy_bad_math_system_prompt_test": {
        "max_new_tokens": 512,
        "max_length": 2048,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "use_chat_template": True,
        "system_prompt": "You are very bad at math and you always make mistakes",
    },
    "debug_case_1": {
        "max_new_tokens": 512,
        "max_length": 2048,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "use_chat_template": True,
        "system_prompt": "You always answer 1 or a",
    },
    "debug_case_2": {
        "max_new_tokens": 512,
        "max_length": 2048,
        "do_sample": False,
        "repetition_penalty": 0.1,
        "temperature": 1000.0,
        "top_p": 0.1,
        "use_chat_template": True,
        "system_prompt": "You are a helpful assistant.",
    },
}

def get_generator_profile(name: str) -> dict:
    if name not in MODEL_PROFILES:
        raise KeyError(f"Unknown generator profile: {name}")
    return deepcopy(MODEL_PROFILES[name])


def get_inference_profile(name: str) -> dict:
    if name not in INFERENCE_PROFILES:
        raise KeyError(f"Unknown inference profile: {name}")
    return deepcopy(INFERENCE_PROFILES[name])


def get_judge_profile(name: str) -> dict:
    if name not in MODEL_PROFILES:
        raise KeyError(f"Unknown judge profile: {name}")
    return deepcopy(MODEL_PROFILES[name])
