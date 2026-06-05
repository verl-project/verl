from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from reliable_gsm8k.hf_runtime import disable_hf_transfer_if_unavailable


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    raw_response: dict[str, Any]


class TextGenerationBackend(Protocol):
    def generate(self, *, prompt: str, sampling: dict[str, Any], metadata: dict[str, Any] | None = None) -> list[GenerationResponse]:
        ...


_RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRY_DELAY_MS_RE = re.compile(r"try again in\s+(\d+(?:\.\d+)?)ms", re.IGNORECASE)
_RETRY_DELAY_SECONDS_RE = re.compile(r"try again in\s+(\d+(?:\.\d+)?)s", re.IGNORECASE)
_TRANSFORMERS_MODEL_CACHE: dict[tuple[Any, ...], tuple[Any, Any, str | None, dict[str, Any]]] = {}


def _parse_retry_after_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        delay = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, delay)


def _extract_retry_delay_seconds(
    *,
    response_body: str,
    headers: Any,
    attempt_index: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> float:
    header_value = None
    if headers is not None:
        try:
            header_value = headers.get("Retry-After")
        except AttributeError:
            header_value = None
    delay = _parse_retry_after_seconds(header_value)
    if delay is None:
        match = _RETRY_DELAY_MS_RE.search(response_body)
        if match:
            delay = float(match.group(1)) / 1000.0
    if delay is None:
        match = _RETRY_DELAY_SECONDS_RE.search(response_body)
        if match:
            delay = float(match.group(1))
    if delay is None:
        delay = retry_base_seconds * (2**attempt_index)
    return min(max(0.0, delay), retry_max_seconds)


def resolve_api_key_from_config(config: dict[str, Any]) -> str | None:
    api_key_env = config.get("api_key_env")
    api_key = config.get("api_key")
    if api_key_env:
        api_key = os.environ.get(str(api_key_env))
        if not api_key:
            raise ValueError(f"Missing required environment variable for API key: {api_key_env}")
    if api_key is None:
        return None
    return str(api_key)


def describe_model_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": config.get("backend"),
        "model_name": config.get("model_name"),
        "api_base": config.get("api_base"),
        "timeout_seconds": int(config.get("timeout_seconds", 120)),
        "device": config.get("device"),
        "device_map": config.get("device_map"),
        "torch_dtype": config.get("torch_dtype"),
        "attn_implementation": config.get("attn_implementation"),
        "padding_side": config.get("padding_side"),
    }


def get_backend_runtime_metadata(backend: TextGenerationBackend, config: dict[str, Any]) -> dict[str, Any]:
    runtime_info = getattr(backend, "runtime_info", None)
    if isinstance(runtime_info, dict):
        return runtime_info
    return describe_model_config(config)


def _resolve_torch_dtype(torch_module: Any, dtype_name: str | None) -> Any:
    if dtype_name is None or dtype_name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[dtype_name]


def _normalize_device_target(target: Any) -> str | None:
    if target is None:
        return None
    if isinstance(target, int):
        return f"cuda:{target}"
    target_str = str(target)
    if target_str.isdigit():
        return f"cuda:{target_str}"
    return target_str


def _infer_input_device(model: Any) -> str | None:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for target in device_map.values():
            target_str = _normalize_device_target(target)
            if target_str not in {"cpu", "disk", "meta"}:
                return target_str
    device = getattr(model, "device", None)
    return _normalize_device_target(device)


def _freeze_max_memory(max_memory: Any) -> Any:
    if not isinstance(max_memory, dict):
        return max_memory
    return tuple(sorted((str(key), str(value)) for key, value in max_memory.items()))


def _build_chat_messages(*, prompt: str, use_chat_template: bool, system_prompt: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if use_chat_template:
        prompt_text = "" if system_prompt is None else str(system_prompt)
        if prompt_text.strip():
            messages.append({"role": "system", "content": prompt_text})
    messages.append({"role": "user", "content": prompt})
    return messages


@dataclass
class OpenAICompatibleBackend:
    model_name: str
    api_base: str
    api_key: str
    timeout_seconds: int = 120
    max_retries: int = 8
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 30.0

    def generate(self, *, prompt: str, sampling: dict[str, Any], metadata: dict[str, Any] | None = None) -> list[GenerationResponse]:
        temperature = float(sampling.get("temperature", 0.0))
        top_p = float(sampling.get("top_p", 1.0))
        do_sample = sampling.get("do_sample")
        use_chat_template = bool(sampling.get("use_chat_template", True))
        system_prompt = sampling.get("system_prompt", "You are a helpful assistant.")
        if do_sample is None:
            do_sample = temperature > 0.0 or top_p < 1.0
        if not bool(do_sample):
            # OpenAI-style APIs do not expose `do_sample`; force deterministic settings.
            temperature = 0.0
            top_p = 1.0
        payload = {
            "model": self.model_name,
            "messages": _build_chat_messages(
                prompt=prompt,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
            ),
            "temperature": temperature,
            "max_tokens": sampling.get("max_tokens", 512),
            "top_p": top_p,
            "n": sampling.get("n", 1),
            "seed": sampling.get("seed"),
        }
        endpoint = self.api_base.rstrip("/") + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        decoded = None
        for attempt_index in range(self.max_retries + 1):
            request = urllib.request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                response_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in _RETRYABLE_HTTP_STATUS_CODES and attempt_index < self.max_retries:
                    delay_seconds = _extract_retry_delay_seconds(
                        response_body=response_body,
                        headers=getattr(exc, "headers", None),
                        attempt_index=attempt_index,
                        retry_base_seconds=self.retry_base_seconds,
                        retry_max_seconds=self.retry_max_seconds,
                    )
                    time.sleep(delay_seconds)
                    continue
                raise RuntimeError(
                    f"Generation request failed for model '{self.model_name}' at '{endpoint}' "
                    f"with HTTP {exc.code}: {response_body}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Generation request failed for model '{self.model_name}' at '{endpoint}': {exc}. "
                    "Verify outbound network access and API key settings."
                ) from exc
        if decoded is None:
            raise RuntimeError("request retries exhausted without a decoded response")
        outputs: list[GenerationResponse] = []
        for index, choice in enumerate(decoded.get("choices", [])):
            message = choice.get("message", {})
            outputs.append(
                GenerationResponse(
                    text=message.get("content", ""),
                    raw_response={
                        "backend": "openai_compatible",
                        "choice_index": index,
                        "choice": choice,
                        "metadata": metadata or {},
                    },
                )
            )
        return outputs


@dataclass
class SGLangBackend:
    model_name: str
    api_base: str
    api_key: str | None = None
    timeout_seconds: int = 120
    max_retries: int = 8
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.runtime_info = {
            "backend": "sglang",
            "model_name": self.model_name,
            "api_base": self.api_base,
            "timeout_seconds": self.timeout_seconds,
        }

    def generate(self, *, prompt: str, sampling: dict[str, Any], metadata: dict[str, Any] | None = None) -> list[GenerationResponse]:
        temperature = float(sampling.get("temperature", 0.0))
        top_p = float(sampling.get("top_p", 1.0))
        do_sample = sampling.get("do_sample")
        use_chat_template = bool(sampling.get("use_chat_template", True))
        system_prompt = sampling.get("system_prompt", "You are a helpful assistant.")
        if do_sample is None:
            do_sample = temperature > 0.0 or top_p < 1.0
        if not bool(do_sample):
            temperature = 0.0
            top_p = 1.0
        payload = {
            "model": self.model_name,
            "messages": _build_chat_messages(
                prompt=prompt,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
            ),
            "temperature": temperature,
            "max_tokens": sampling.get("max_tokens", 512),
            "top_p": top_p,
            "n": sampling.get("n", 1),
            "seed": sampling.get("seed"),
        }
        endpoint = self.api_base.rstrip("/") + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        decoded = None
        for attempt_index in range(self.max_retries + 1):
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(
                endpoint,
                data=body,
                method="POST",
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                response_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in _RETRYABLE_HTTP_STATUS_CODES and attempt_index < self.max_retries:
                    delay_seconds = _extract_retry_delay_seconds(
                        response_body=response_body,
                        headers=getattr(exc, "headers", None),
                        attempt_index=attempt_index,
                        retry_base_seconds=self.retry_base_seconds,
                        retry_max_seconds=self.retry_max_seconds,
                    )
                    time.sleep(delay_seconds)
                    continue
                raise RuntimeError(
                    f"SGLang generation request failed for model '{self.model_name}' at '{endpoint}' "
                    f"with HTTP {exc.code}: {response_body}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"SGLang generation request failed for model '{self.model_name}' at '{endpoint}': {exc}. "
                    "Verify the SGLang server is running and reachable."
                ) from exc
        if decoded is None:
            raise RuntimeError("request retries exhausted without a decoded response")
        outputs: list[GenerationResponse] = []
        for index, choice in enumerate(decoded.get("choices", [])):
            message = choice.get("message", {})
            outputs.append(
                GenerationResponse(
                    text=message.get("content", ""),
                    raw_response={
                        "backend": "sglang",
                        "choice_index": index,
                        "choice": choice,
                        "metadata": metadata or {},
                    },
                )
            )
        return outputs


@dataclass
class TransformersCausalLMBackend:
    model_name: str
    tokenizer_name: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    tokenizer_use_fast: bool = True
    padding_side: str | None = None
    device: str | None = None
    device_map: str | None = "auto"
    torch_dtype: str | None = "auto"
    attn_implementation: str | None = None
    enforce_attn_implementation: bool = False
    low_cpu_mem_usage: bool = True
    max_memory: dict[str, Any] | None = None
    offload_folder: str | None = None
    offload_state_dict: bool = False
    generation_use_cache: bool = True
    clear_cuda_cache_after_generate: bool = False

    def __post_init__(self) -> None:
        tokenizer, model, input_device, runtime_info = self._load_or_reuse_model()
        self._tokenizer = tokenizer
        self._model = model
        self._input_device = input_device
        self.runtime_info = runtime_info

    def _load_or_reuse_model(self) -> tuple[Any, Any, str | None, dict[str, Any]]:
        cache_key = (
            self.model_name,
            self.tokenizer_name,
            self.trust_remote_code,
            self.local_files_only,
            self.tokenizer_use_fast,
            self.padding_side,
            self.device,
            self.device_map,
            self.torch_dtype,
            self.attn_implementation,
            self.enforce_attn_implementation,
            self.low_cpu_mem_usage,
            _freeze_max_memory(self.max_memory),
            self.offload_folder,
            self.offload_state_dict,
            self.generation_use_cache,
            self.clear_cuda_cache_after_generate,
        )
        cached = _TRANSFORMERS_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        disable_hf_transfer_if_unavailable()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name or self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            use_fast=self.tokenizer_use_fast,
        )
        if self.padding_side is not None:
            tokenizer.padding_side = self.padding_side
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
            "torch_dtype": _resolve_torch_dtype(torch, self.torch_dtype),
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        if self.max_memory is not None:
            model_kwargs["max_memory"] = self.max_memory
        if self.offload_folder:
            Path(self.offload_folder).expanduser().mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(Path(self.offload_folder).expanduser())
        if self.offload_state_dict:
            model_kwargs["offload_state_dict"] = True

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        normalized_device = _normalize_device_target(self.device)
        if normalized_device and self.device_map is None:
            model = model.to(normalized_device)
        model.eval()

        loaded_attn = getattr(model.config, "_attn_implementation", None) or getattr(model.config, "attn_implementation", None)
        if self.attn_implementation and self.enforce_attn_implementation and loaded_attn != self.attn_implementation:
            raise RuntimeError(
                f"Requested attn_implementation '{self.attn_implementation}' for '{self.model_name}', "
                f"but the loaded model reports '{loaded_attn}'."
            )

        input_device = normalized_device or _infer_input_device(model)
        runtime_info = {
            "backend": "transformers_causal_lm",
            "model_name": self.model_name,
            "device": input_device,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "attn_implementation": self.attn_implementation,
            "padding_side": getattr(self._tokenizer if hasattr(self, "_tokenizer") else tokenizer, "padding_side", None),
            "loaded_attn_implementation": loaded_attn,
        }
        cached = (tokenizer, model, input_device, runtime_info)
        _TRANSFORMERS_MODEL_CACHE[cache_key] = cached
        return cached

    def _render_prompt(self, *, prompt: str, sampling: dict[str, Any]) -> str:
        use_chat_template = bool(sampling.get("use_chat_template", True))
        if not use_chat_template:
            return prompt
        chat_template = getattr(self._tokenizer, "chat_template", None)
        if chat_template:
            system_prompt = sampling.get("system_prompt", "You are a helpful assistant.")
            return self._tokenizer.apply_chat_template(
                _build_chat_messages(
                    prompt=prompt,
                    use_chat_template=True,
                    system_prompt=system_prompt,
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, *, prompt: str, sampling: dict[str, Any], metadata: dict[str, Any] | None = None) -> list[GenerationResponse]:
        import torch

        rendered_prompt = self._render_prompt(prompt=prompt, sampling=sampling)
        max_length_raw = sampling.get("max_length")
        tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if max_length_raw is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = int(max_length_raw)
        encoded_inputs = self._tokenizer(rendered_prompt, **tokenizer_kwargs)
        if self._input_device:
            encoded_inputs = {key: value.to(self._input_device) for key, value in encoded_inputs.items()}

        outputs: list[GenerationResponse] = []
        count = int(sampling.get("n", 1))
        max_new_tokens = int(sampling.get("max_tokens", 512))
        temperature = float(sampling.get("temperature", 0.0))
        top_p = float(sampling.get("top_p", 1.0))
        repetition_penalty = float(sampling.get("repetition_penalty", 1.0))
        base_seed = sampling.get("seed")
        do_sample = sampling.get("do_sample")
        if do_sample is None:
            do_sample = temperature > 0.0 or top_p < 1.0
        do_sample = bool(do_sample)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "use_cache": self.generation_use_cache,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        for sample_index in range(count):
            if base_seed is not None:
                sample_seed = int(base_seed) + sample_index
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
            else:
                sample_seed = None
            with torch.inference_mode():
                generated = self._model.generate(**encoded_inputs, **generation_kwargs)
            input_length = encoded_inputs["input_ids"].shape[-1]
            generated_tokens = generated[0][input_length:]
            text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            del generated
            del generated_tokens
            if self.clear_cuda_cache_after_generate and torch.cuda.is_available():
                torch.cuda.empty_cache()
            outputs.append(
                GenerationResponse(
                    text=text,
                    raw_response={"backend": "transformers_causal_lm", "sample_index": sample_index, "seed": sample_seed, "metadata": metadata or {}},
                )
            )
        return outputs


def create_generation_backend(config: dict[str, Any]) -> TextGenerationBackend:
    backend_type = config.get("backend")
    if backend_type == "openai_compatible":
        api_key = resolve_api_key_from_config(config)
        return OpenAICompatibleBackend(
            model_name=config["model_name"],
            api_base=config["api_base"],
            api_key=api_key or "EMPTY",
            timeout_seconds=int(config.get("timeout_seconds", 120)),
            max_retries=int(config.get("max_retries", 8)),
            retry_base_seconds=float(config.get("retry_base_seconds", 1.0)),
            retry_max_seconds=float(config.get("retry_max_seconds", 30.0)),
        )
    if backend_type == "sglang":
        api_base = config.get("api_base")
        if not api_base:
            raise ValueError("SGLang backend requires 'api_base' in the model profile.")
        api_key = resolve_api_key_from_config(config)
        return SGLangBackend(
            model_name=config["model_name"],
            api_base=str(api_base),
            api_key=api_key,
            timeout_seconds=int(config.get("timeout_seconds", 120)),
            max_retries=int(config.get("max_retries", 8)),
            retry_base_seconds=float(config.get("retry_base_seconds", 1.0)),
            retry_max_seconds=float(config.get("retry_max_seconds", 30.0)),
        )
    if backend_type == "transformers_causal_lm":
        return TransformersCausalLMBackend(
            model_name=config["model_name"],
            tokenizer_name=config.get("tokenizer_name"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            local_files_only=bool(config.get("local_files_only", False)),
            tokenizer_use_fast=bool(config.get("tokenizer_use_fast", True)),
            padding_side=config.get("padding_side"),
            device=config.get("device"),
            device_map=config.get("device_map", "auto"),
            torch_dtype=config.get("torch_dtype", "auto"),
            attn_implementation=config.get("attn_implementation"),
            enforce_attn_implementation=bool(config.get("enforce_attn_implementation", False)),
            low_cpu_mem_usage=bool(config.get("low_cpu_mem_usage", True)),
            max_memory=config.get("max_memory"),
            offload_folder=config.get("offload_folder"),
            offload_state_dict=bool(config.get("offload_state_dict", False)),
            generation_use_cache=bool(config.get("generation_use_cache", True)),
            clear_cuda_cache_after_generate=bool(config.get("clear_cuda_cache_after_generate", False)),
        )
    raise ValueError(f"Unsupported generation backend: {backend_type}")
