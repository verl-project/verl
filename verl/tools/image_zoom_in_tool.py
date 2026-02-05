# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from math import ceil, floor
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
from qwen_vl_utils import fetch_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisualExecutionWorker:
    """Worker for executing visual processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing visual processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visual_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visual execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisualExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class ImageZoomInTool(BaseTool):
    """A tool for zooming in on an image by cropping it based on a bounding box.

    This tool provides a zoom-in functionality by cropping a region from an image,
    with rate limiting and concurrent execution support through Ray.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the zoom-in operation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    MIN_DIMENSION = 28

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": (
                    "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an "
                    "optional object label."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items":{"type":"number"},
                            "minItems":4,
                            "maxItems":4,
                            "description": (
                                "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is "
                                "the top-left corner and (x2, y2) is the bottom-right corner."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box (optional).",
                        },
                    },
                    "required": ["bbox_2d"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_visual_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        logger.info(f"Initialized ImageZoomInTool with config: {config}")

    def _validate_bbox(self, left: float, top: float, right: float, bottom: float) -> bool:
        """Validate the bounding box dimensions and aspect ratio."""
        try:
            if not (left < right and top < bottom):
                logger.warning(f"Invalid bbox shape: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            height = bottom - top
            width = right - left

            # Prevent division by zero for zero-sized boxes
            if min(height, width) == 0:
                logger.warning(f"Bbox has zero width or height: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            if max(height, width) / min(height, width) > 100:
                logger.warning(f"Bbox aspect ratio > 100: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            return True
        except Exception as e:
            logger.warning(f"Bbox validation error: {e}")
            return False

    def _maybe_resize_bbox(self, bbox_2d: list[float], image_width: int, image_height: int) -> Optional[list[float]]:
        """
        Clamp, validate, and potentially resize a bounding box.

        This function ensures the final bounding box is within image bounds and meets the minimum
        dimension requirements. If the initial box is too small, it attempts to expand it
        from its center. It performs a final check to guarantee the output dimensions are valid.

        Returns:
            A valid bounding box as a list of coordinates, or None if validation fails.
        """
        left, top, right, bottom = bbox_2d

        # 1. Clamp the initial bounding box to the image dimensions.
        left = max(0.0, float(left))
        top = max(0.0, float(top))
        right = min(float(image_width), float(right))
        bottom = min(float(image_height), float(bottom))

        # 2. If clamped bbox is invalid, return immediately.
        if not self._validate_bbox(left, top, right, bottom):
            return None

        current_bbox = [left, top, right, bottom]
        height = bottom - top
        width = right - left

        # 3. If the box is too small, attempt to resize it.
        if height < self.MIN_DIMENSION or width < self.MIN_DIMENSION:
            logger.info(f"Bbox {width}x{height} is smaller than {self.MIN_DIMENSION}, attempting resize.")
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0

            min_dim = min(height, width)
            if min_dim == 0:  # Safeguard for zero-area boxes
                return None

            # 1. Calculate the target dimensions to make the smallest side MIN_DIMENSION.
            ratio = self.MIN_DIMENSION / min_dim
            target_width = width * ratio
            target_height = height * ratio

            # 2. If the target size is larger than the image, scale it down to fit.
            #    This preserves the aspect ratio while respecting image boundaries.
            if target_width > image_width:
                scale_down = image_width / target_width
                target_width = image_width
                target_height *= scale_down

            if target_height > image_height:
                scale_down = image_height / target_height
                target_height = image_height
                target_width *= scale_down

            # 3. Determine the coordinates for the box centered on the original center.
            new_half_width = target_width / 2.0
            new_half_height = target_height / 2.0
            new_left = center_x - new_half_width
            new_top = center_y - new_half_height

            # 4. Shift the box if it extends beyond the image boundaries to keep its size.
            if new_left < 0:
                new_left = 0
            if new_top < 0:
                new_top = 0
            if new_left + target_width > image_width:
                new_left = image_width - target_width
            if new_top + target_height > image_height:
                new_top = image_height - target_height

            new_right = new_left + target_width
            new_bottom = new_top + target_height

            # Use floor and ceil for final integer coordinates.
            current_bbox = [floor(new_left), floor(new_top), ceil(new_right), ceil(new_bottom)]

        # 4. Final validation on the resulting bounding box (either original or resized).
        final_left, final_top, final_right, final_bottom = current_bbox
        if not self._validate_bbox(final_left, final_top, final_right, final_bottom):
            logger.warning(f"Final bbox is invalid after processing: {current_bbox}")
            return None

        final_height = floor(final_bottom) - floor(final_top)
        final_width = floor(final_right) - floor(final_left)

        if final_height < self.MIN_DIMENSION or final_width < self.MIN_DIMENSION:
            logger.warning(
                f"Final bbox size ({final_width}x{final_height}) are still smaller than minimum ({self.MIN_DIMENSION})."
                f"Original bbox: {bbox_2d}, original image size: {image_width}x{image_height}"
            )
            return None

        return current_bbox

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for image zoom-in tool.

        This method initializes a new session for an image, which can then be used
        for operations like zooming. It fetches the image from various sources
        and stores it internally.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            **kwargs: Should contain 'image' key with image data, or 'create_kwargs'
                containing {'image': image_data}. Image can be one of the following:
                - A PIL.Image.Image object.
                - A string containing an HTTP or HTTPS URL.
                - A string containing a local file path.
                - A string containing a file URI (e.g., "file:///path/to/image.jpg").
                - A string containing a base64-encoded image in the format of "data:image/jpeg;base64,..."

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs
        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        img = fetch_image({"image": image})
        self._instance_dict[instance_id] = {
            "image": img,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        bbox_2d = parameters.get("bbox_2d")
        label = parameters.get("label", "")

        if not bbox_2d or len(bbox_2d) != 4:
            return (
                ToolResponse(text="Error: bbox_2d parameter is missing or not a list of 4 numbers."),
                -0.05,
                {"success": False},
            )

        instance_data = self._instance_dict[instance_id]
        image = instance_data["image"]
        image_width, image_height = image.size

        try:
            resized_bbox = self._maybe_resize_bbox(bbox_2d, image_width=image_width, image_height=image_height)

            if resized_bbox is None:
                error_msg = (
                    f"Error: The specified bounding box {bbox_2d} is invalid or results in a crop smaller than "
                    f"the minimum size of {self.MIN_DIMENSION}x{self.MIN_DIMENSION}."
                )
                logger.warning(f"Tool execution failed: {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            cropped_image = image.crop(resized_bbox)
            logger.info(f"Cropped image size: {cropped_image.size}")
        except Exception as e:
            logger.error(f"Error processing image zoom-in: {e}")
            return ToolResponse(text=f"Error processing image zoom-in: {e}"), -0.05, {"success": False}

        response_text = f"Zoomed in on the image to the region {bbox_2d}."
        if label:
            response_text = f"Zoomed in on the image to the region {bbox_2d} with label {label}."

        return (
            ToolResponse(
                image=[cropped_image],
                text=response_text,
            ),
            0.0,
            {"success": True},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
