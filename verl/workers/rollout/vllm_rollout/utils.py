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
import ctypes
import json
import logging
import os
import platform
import signal
import threading
from collections.abc import Mapping
from types import MethodType
from typing import Any, Literal, Optional, get_args

import torch
from vllm.outputs import RequestOutput

from verl.plugin.platform import get_platform
from verl.utils.device import is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights
from verl.workers.rollout.vllm_rollout.weight_update_utils import apply_buffer_updates, split_buffer_updates

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}


def _resolve_vllm_weight_sync_local_rank(worker_local_rank: int, parallel_config: Any) -> int:
    worker_local_rank = int(worker_local_rank)
    if parallel_config is None:
        return worker_local_rank

    tp_size = max(int(getattr(parallel_config, "tensor_parallel_size", 1) or 1), 1)
    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    dp_local_size = int(getattr(parallel_config, "data_parallel_size_local", 1) or 1)
    if dp_size <= 1 and dp_local_size <= 1:
        return worker_local_rank

    dp_local_rank = getattr(parallel_config, "data_parallel_rank_local", None)
    if dp_local_rank is None:
        dp_rank = getattr(parallel_config, "data_parallel_rank", None)
        if dp_rank is None:
            dp_rank = getattr(parallel_config, "data_parallel_index", None)
        if dp_rank is not None and dp_local_size > 0:
            dp_local_rank = int(dp_rank) % dp_local_size

    if dp_local_rank is None:
        return worker_local_rank

    tp_rank = worker_local_rank % tp_size
    return int(dp_local_rank) * tp_size + tp_rank


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
            npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
            return "NPU-" + npu_visible_devices[device_id]
        else:
            return f"NPU-{device_id}"
    else:
        try:
            return current_platform.get_device_uuid(device_id)
        except Exception:
            return get_platform().get_device_uuid(device_id=device_id)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        if os.environ.get("VERL_FULL_DETERMINISM", "0") == "1":
            from verl.workers.engine.utils import enable_full_determinism

            # VERL_SEED is set by vLLMHttpServer.__init__ only when the
            # rollout config has full_determinism=true.  Worker sub-processes
            # inherit their parent's env, so rollout workers will see it but
            # RM workers (whose parent vLLMHttpServer does not set it) won't.
            # If VERL_SEED is missing, skip — RM doesn't need the determinism
            # patch, only rollout does.
            verl_seed = os.environ.get("VERL_SEED")
            if verl_seed is not None:
                enable_full_determinism(seed=int(verl_seed))

        # 1. patch for Lora
        VLLMHijack.hijack()
        # 2. patch online fp8 quant
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        vllm_config = kwargs.get("vllm_config")
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        _is_modelopt_qat = type(quant_config).__name__ == "ModelOptNvFp4Config"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
        elif _is_modelopt_qat:
            from verl.utils.modelopt import apply_modelopt_nvfp4_patches

            apply_modelopt_nvfp4_patches()
            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        instance._is_modelopt_qat = _is_modelopt_qat
        return instance

    def _get_drafter_model(self):
        """Return the drafter's model object, or None if unavailable."""
        drafter = getattr(self.model_runner, "drafter", None)
        return drafter.model if drafter is not None and hasattr(drafter, "model") else None

    def _get_draft_model_config(self):
        """Return the draft model config from speculative_config, or None."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec.draft_model_config if spec is not None and spec.draft_model_config is not None else None

    def _use_mtp_drafter_weight_sync(self):
        """Return whether the vLLM MTP drafter should receive actor weights."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec is not None and spec.method == "mtp" and self._get_drafter_model() is not None

    def _iter_all_models(self):
        """Yield models that need weight updates.

        Only vLLM MTP drafter sync is supported for now. Independent non-MTP
        draft models are not compatible with actor weight loading through this path.
        """
        yield self.model_runner.model
        if self._use_mtp_drafter_weight_sync():
            yield self._get_drafter_model()

    def _iter_all_models_with_config(self):
        """Yield (model, model_config) for models that need post-processing."""
        yield self.model_runner.model, self.model_runner.vllm_config.model_config
        if self._use_mtp_drafter_weight_sync():
            draft_cfg = self._get_draft_model_config()
            if draft_cfg is not None:
                yield self._get_drafter_model(), draft_cfg

    def monkey_patch_model(self, vocab_size: int):
        for model in self._iter_all_models():
            # patch compute_logits to avoid sampling OOV token
            monkey_patch_compute_logits(model, vocab_size)
            # patch weight loader to support MoE model
            patch_vllm_moe_model_weight_loader(model)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        use_standard_weight_load = not (peft_config and base_sync_done) and not is_fp8_model(
            self.model_runner.vllm_config
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            for model in self._iter_all_models():
                prepare_qat_for_load_weights(model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif use_standard_weight_load:
            # Re-apply here because async IPC weight sync can happen long after init and lose MoE weight_loader attrs.
            for model in self._iter_all_models():
                patch_vllm_moe_model_weight_loader(model)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            for model in self._iter_all_models():
                manual_process_weights_after_loading(model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            for model, model_config in self._iter_all_models_with_config():
                process_weights_after_loading(model, model_config, self.device)

    def _apply_buffer_updates_all_models(self, buffer_updates, main_named_buffers):
        """Apply buffer updates to the main model and any synced MTP drafter.

        The main model (yielded first) reuses the prebuilt ``named_buffers`` map;
        the drafter builds its own. Returns buffers applied to the main model.
        """
        models = list(self._iter_all_models())
        loaded = apply_buffer_updates(models[0], buffer_updates, named_buffers=main_named_buffers)
        for model in models[1:]:
            apply_buffer_updates(model, buffer_updates)
        return loaded

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            # Clone out of the receiver's reused IPC bucket buffer: add_lora keeps these tensors
            # past this callback, so views into the freed/overwritten buffer crash later (#6454).
            weights = {name: tensor.clone() for name, tensor in weights}
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            param_updates, buffer_updates, named_buffers = split_buffer_updates(self.model_runner.model, weights)
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(param_updates, self.model_runner) if param_updates else []
                # Keep the draft model in sync when present.
                if self._use_mtp_drafter_weight_sync() and param_updates:
                    load_quanted_weights(param_updates, self.model_runner, is_drafter=True)
                loaded_buffers = self._apply_buffer_updates_all_models(buffer_updates, named_buffers)
                logger.info(
                    f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}, loaded_buffers: {loaded_buffers}"
                )
            else:
                if param_updates:
                    for model in self._iter_all_models():
                        model.load_weights(param_updates)
                loaded_buffers = self._apply_buffer_updates_all_models(buffer_updates, named_buffers)
                logger.info(
                    f"Loading standard weights (non-FP8, async), "
                    f"loaded_params: {len(param_updates)}, loaded_buffers: {loaded_buffers}"
                )

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.

        Uses Ray job id + replica_rank + rollout-local rank to match the sender
        side and avoid cross-job collisions on shared hosts.
        In PD mode, each engine actor's local ranks start at 0; the optional
        VERL_ZMQ_BASE_TRAINER_RANK offset maps them back to trainer ranks.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        vllm_config = getattr(self.model_runner, "vllm_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        local_rank = _resolve_vllm_weight_sync_local_rank(self.local_rank, parallel_config)
        trainer_rank_base = os.environ.get("VERL_ZMQ_BASE_TRAINER_RANK")
        trainer_rank = int(trainer_rank_base) + local_rank if trainer_rank_base is not None else local_rank
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{trainer_rank}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


def build_mtp_speculative_config(
    method: str, num_speculative_tokens: int, engine_speculative_config: Any = None
) -> dict[str, Any]:
    """Build vLLM's MTP speculative config, applying rollout engine overrides."""
    if engine_speculative_config is None:
        engine_speculative_config = {}
    if isinstance(engine_speculative_config, str):
        engine_speculative_config = json.loads(engine_speculative_config)
    if not isinstance(engine_speculative_config, Mapping):
        raise TypeError("rollout.engine_kwargs.vllm.speculative_config must be a mapping when MTP rollout is enabled")

    return {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        **{key: val for key, val in engine_speculative_config.items() if val is not None},
    }


def extract_prompt_logprobs(output: RequestOutput, num_prompt_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None:
        return

    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls
