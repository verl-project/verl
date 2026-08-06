# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class OpenAIFunctionPropertySchema(BaseModel):
    """The schema of a parameter in OpenAI format."""

    # Union's type is list[str], e.g. ["integer", "number"] for int | float unions.
    type: str | list[str]
    description: str | None = None
    # JSON Schema's ``enum`` accepts any JSON value, not just strings, so
    # ``Literal[1, 2, 3]`` -> ``enum: [1, 2, 3]`` is a valid schema.
    enum: list[Any] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    """The schema of parameters in OpenAI format."""

    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    # ``required`` can be omitted when no parameter is required.
    required: list[str] = Field(default_factory=list)


class OpenAIFunctionSchema(BaseModel):
    """The schema of a function in OpenAI format."""

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema = Field(
        default_factory=lambda: OpenAIFunctionParametersSchema(type="object", properties={}, required=[])
    )
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    """The schema of a tool in OpenAI format."""

    type: str
    function: OpenAIFunctionSchema


class OpenAIFunctionParsedSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(
        parsed_schema: OpenAIFunctionParsedSchema,
    ) -> tuple["OpenAIFunctionCallSchema", bool]:
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        # If the arguments is not a dict, it means the arguments is not a valid JSON string
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments), has_decode_error


class OpenAIFunctionToolCall(BaseModel):
    """The tool call in OpenAI format."""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema


class ToolResponse(BaseModel):
    """The response from a tool execution."""

    text: str | None = None
    image: list[Any] | None = None
    video: list[Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if "image" in values and not isinstance(values["image"], list):
            raise ValueError(
                f"Image must be a list, but got {type(values['image'])}. Please check the tool.execute(). "
                f"For single images, wrap in a list: [image]. "
                f"Example: {{'image': [img1]}} or {{'image': [img1, img2, ...]}}."
            )
        if "video" in values and not isinstance(values["video"], list):
            raise ValueError(
                f"Video must be a list, but got {type(values['video'])}. Please check the tool.execute(). "
                f"For single videos, wrap in a list: [video]. "
                f"Example: {{'video': [video1]}} or {{'video': [video1, video2, ...]}}."
            )

        return values

    def is_empty(self) -> bool:
        return not self.text and not self.image and not self.video

    def is_text_only(self) -> bool:
        return self.text and not self.image and not self.video
