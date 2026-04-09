# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import argparse
import logging
from dataclasses import asdict
from typing import Any, Optional

import ray
import torchvision.transforms as T
import vllm_omni.entrypoints.cli.serve
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DiffusionModelConfig, DiffusionRolloutConfig
from verl.workers.rollout.replica import DiffusionOutput
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _init_model_config(self, model_config):
        """Use DiffusionModelConfig instead of HFModelConfig."""
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        """No-op: diffusion models don't have max_position_embeddings."""
        pass

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Omni-specific post-init: create PIL→tensor converter, then log."""
        self._to_tensor = T.PILToTensor()
        super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _get_override_generation_config(self) -> dict:
        """Diffusion models have no LLM sampling params; return empty dict."""
        return {}

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        # custom_pipeline is passed directly to run_server; not supported via CLI yet
        engine_kwargs.pop("custom_pipeline", None)
        engine_kwargs.pop("stage_configs_path", None)

    def _get_worker_extension_cls(self) -> str:
        return "verl.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM-Omni CLI"

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    async def run_server(self, args: argparse.Namespace):
        vllm_omni_kwargs = self.config.engine_kwargs.get("vllm_omni", {})
        stage_configs_path = vllm_omni_kwargs.get("stage_configs_path")

        if stage_configs_path is not None:
            # Multi-stage (e.g. BAGEL): per-stage engine args live in the YAML.
            engine_args = {
                "stage_configs_path": stage_configs_path,
                "worker_extension_cls": self._get_worker_extension_cls(),
            }
        else:
            # Single-stage (Qwen-Image): CLI args path.
            engine_args = asdict(OmniEngineArgs.from_cli_args(args))

        # TODO (mike): read custom_pipeline from CLI
        custom_pipeline = vllm_omni_kwargs.get("custom_pipeline")
        if custom_pipeline is not None:
            engine_args["enable_dummy_pipeline"] = True
            engine_args["custom_pipeline_args"] = {"pipeline_class": custom_pipeline}

        if stage_configs_path is not None:
            engine_args["model"] = self.model_config.path

        # TODO (mike): support parsing engine config from CLI
        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        # TODO (mike): support multi node
        raise NotImplementedError("vLLM-Omni headless mode is not implemented yet.")

    # -----------------------------------------------------------------------
    # wake_up hook: Omni does not restore KV cache on wake-up
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        priority: int = 0,
        lora_request: Optional[LoRARequest] = None,
        lora_scale: float = 1.0,
    ) -> DiffusionOutput:
        """Generate sequence with token-in-image-out."""
        prompt_ids = normalize_token_ids(prompt_ids)
        default_params_list = getattr(self.engine, "default_sampling_params_list", [])
        is_multi_stage = len(default_params_list) > 1

        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        # Add lora request (caller-supplied lora_request takes precedence)
        if lora_request is None and self.lora_as_adapter:
            # For multi-stage, only the diffusion stage (last) has LoRA.
            # Target it specifically to avoid errors on the LLM stage.
            if is_multi_stage:
                diffusion_stage_id = len(default_params_list) - 1
                results = await self.engine.collective_rpc("list_loras", stage_ids=[diffusion_stage_id])
                loaded_ids = results[0]
            else:
                loaded_ids = await self.engine.list_loras()
            lora_loaded = VLLM_LORA_INT_ID in loaded_ids
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        # Build OmniCustomPrompt with pre-tokenized IDs
        custom_prompt: OmniCustomPrompt = {"prompt_token_ids": prompt_ids}
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}
        if is_multi_stage:
            custom_prompt["modalities"] = ["image"]

        # Build OmniDiffusionSamplingParams from the incoming dict
        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for k, v in sampling_params.items():
            if hasattr(OmniDiffusionSamplingParams, k):
                sampling_kwargs[k] = v
            else:
                extra_args[k] = v
        sampling_kwargs["extra_args"] = extra_args
        if lora_request is not None:
            sampling_kwargs["lora_request"] = lora_request
            sampling_kwargs["lora_scale"] = lora_scale
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)

        # Build sampling params list: multi-stage models use defaults for non-diffusion stages
        if is_multi_stage:
            # Multi-stage (e.g. BAGEL): use defaults for earlier stages, override diffusion stage
            sampling_params_list = list(default_params_list)
            sampling_params_list[-1] = diffusion_sampling_params
        else:
            sampling_params_list = [diffusion_sampling_params]

        # Call AsyncOmni.generate() with the correct API
        generator = self.engine.generate(
            prompt=custom_prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        )

        # Get final response
        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        diffusion_output = self._to_tensor(final_res.images[0]).float() / 255.0

        # Extract extra data from custom_output (populated by DiffusionEngine)
        mm_output = final_res.custom_output or {}

        # Single-stage pipelines (Qwen-Image) batch outputs with a leading batch
        # dim that should be stripped. Multi-stage (BAGEL) returns un-batched tensors.
        def _unbatch(v):
            if v is None or is_multi_stage:
                return v
            return v[0]

        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")
            log_probs = _unbatch(all_log_probs)
        else:
            log_probs = None

        all_latents = mm_output.get("all_latents")
        all_timesteps = mm_output.get("all_timesteps")
        prompt_embeds = mm_output.get("prompt_embeds")
        prompt_embeds_mask = mm_output.get("prompt_embeds_mask")
        negative_prompt_embeds = mm_output.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = mm_output.get("negative_prompt_embeds_mask")

        extra_fields = {
            "all_latents": _unbatch(all_latents),
            "all_timesteps": _unbatch(all_timesteps),
            "prompt_embeds": _unbatch(prompt_embeds),
            "prompt_embeds_mask": _unbatch(prompt_embeds_mask),
            "negative_prompt_embeds": _unbatch(negative_prompt_embeds),
            "negative_prompt_embeds_mask": _unbatch(negative_prompt_embeds_mask),
            "global_steps": self.global_steps,
        }

        # Determine stop reason from finish_reason
        if final_res.request_output is not None and hasattr(final_res.request_output, "finish_reason"):
            finish_reason = final_res.request_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        num_preempted = None
        if final_res.request_output is not None and hasattr(final_res.request_output, "num_preempted"):
            num_preempted = final_res.request_output.num_preempted

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def wait_for_requests_to_drain(self):
        # TODO (mike): implement this once DP is supported.
        pass


class vLLMOmniReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig,
        model_config: DiffusionModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def _validate_launch_requirements(self) -> None:
        """No-op: the parent check validates vllm.__version__ which is
        irrelevant for vllm-omni (a separate package)."""
        pass

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"
