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
import argparse
import asyncio
import logging
from dataclasses import asdict
from typing import Any, Optional

import ray
import torchvision.transforms as T
import vllm_omni.entrypoints.cli.serve
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DiffusersModelConfig, DiffusionRolloutConfig
from verl.workers.rollout.replica import ImageOutput, RolloutMode
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_base_async_server import BaseVLLMHttpServer, BaseVLLMReplica

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMOmniHttpServer(BaseVLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Omni-specific post-init: create PIL→tensor converter, then log."""
        self._to_tensor = T.PILToTensor()
        super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        # custom_pipeline is passed directly to run_server; not supported via CLI yet
        engine_kwargs.pop("custom_pipeline", None)

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
        engine_args = AsyncOmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        # TODO (mike): read custom_pipeline from CLI
        custom_pipeline = self.config.engine_kwargs.get("vllm_omni", {}).get("custom_pipeline", None)
        if custom_pipeline is not None:
            engine_args["enable_dummy_pipeline"] = True
            engine_args["custom_pipeline_args"] = {"pipeline_class": custom_pipeline}

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
    ) -> ImageOutput:
        """Generate sequence with token-in-image-out."""
        prompt_ids = normalize_token_ids(prompt_ids)

        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        # Add lora request
        lora_request = None
        if self.lora_as_adapter:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        # Build OmniCustomPrompt with pre-tokenized IDs
        custom_prompt: OmniCustomPrompt = {"prompt_ids": prompt_ids}
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

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
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)

        # Call AsyncOmni.generate() with the correct API
        generator = self.engine.generate(
            prompt=custom_prompt,
            request_id=request_id,
            sampling_params_list=[diffusion_sampling_params],
        )

        # Get final response
        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        image = (self._to_tensor(final_res.images[0]) / 255.0).tolist()

        # Extract extra data from custom_output (populated by DiffusionEngine)
        mm_output = final_res.custom_output or {}

        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")
            log_probs = all_log_probs[0].tolist() if all_log_probs is not None else None
        else:
            log_probs = None

        all_latents = mm_output.get("all_latents")
        all_timesteps = mm_output.get("all_timesteps")
        prompt_embeds = mm_output.get("prompt_embeds")
        prompt_embeds_mask = mm_output.get("prompt_embeds_mask")
        negative_prompt_embeds = mm_output.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = mm_output.get("negative_prompt_embeds_mask")

        extra_info = {
            "all_latents": all_latents[0] if all_latents is not None else None,
            "all_timesteps": all_timesteps[0] if all_timesteps is not None else None,
            "prompt_embeds": prompt_embeds[0] if prompt_embeds is not None else None,
            "prompt_embeds_mask": prompt_embeds_mask[0] if prompt_embeds_mask is not None else None,
            "negative_prompt_embeds": negative_prompt_embeds[0] if negative_prompt_embeds is not None else None,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask[0]
            if negative_prompt_embeds_mask is not None
            else None,
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

        return ImageOutput(
            image=image,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_info=extra_info,
        )

    # -----------------------------------------------------------------------
    # Abort / lifecycle overrides
    # -----------------------------------------------------------------------

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all ongoing generation requests.

        Returns:
            dict[str, Any]: Dictionary containing:
                - aborted_count: Number of requests aborted
                - request_ids: List of aborted request IDs
        """
        try:
            # Take an atomic snapshot to avoid race conditions with the vLLM engine thread
            request_states_snapshot = list(self.engine.output_processor.request_states.items())
            request_ids = [req_id for req_id, _ in request_states_snapshot]

            if not request_ids:
                return {"aborted_count": 0, "request_ids": []}

            # For each request, create an abort output and put it to its queue
            # This allows the generator to receive the aborted result
            from vllm.v1.engine import FinishReason

            for _, req_state in request_states_snapshot:
                request_output = req_state.make_request_output(
                    [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
                )
                req_state.queue.put(request_output)

            # Abort requests in the output processor and engine core
            self.engine.output_processor.abort_requests(request_ids)
            await self.engine.engine_core.abort_requests_async(request_ids)

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def resume_generation(self):
        """Resume generation after abort_all_requests (pause_generation).

        # TODO (mike): no usage now
        """
        if self.node_rank != 0:
            return

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}


class vLLMOmniReplica(BaseVLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig,
        model_config: DiffusersModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"