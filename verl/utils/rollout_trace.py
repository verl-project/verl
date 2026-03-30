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

import contextlib
import functools
import inspect
import os
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel

from verl.utils.ray_utils import get_event_loop

_trace_enabled: ContextVar[bool] = ContextVar("_trace_enabled", default=True)


class RolloutTraceConfig:
    """Configuration for rollout tracing with various backends.

    Singleton configuration class for managing rollout trace settings across different
    tracing backends like Weave and MLflow.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', or None).
        client (Optional[object]): Client instance for the selected backend.
        token2text (bool): Whether to convert tokens to text in traces. Defaults to False.
        project_name (str): Name of the project for tracing.
        experiment_name (str): Name of the experiment for tracing.
        max_samples_per_step_per_worker (Optional[int]): Maximum number of unique samples to trace
            per worker per step. If None, all samples are traced. If set, each worker will randomly
            select up to this many unique samples to trace (including all their rollouts for GRPO).
            Total traces = max_samples_per_step_per_worker * num_workers * n_rollouts_per_sample.
    """

    _instance: Optional["RolloutTraceConfig"] = None
    backend: Optional[str] = None
    client: Optional[object] = None
    token2text: bool = False
    _initialized: bool = False
    project_name: str = None
    experiment_name: str = None
    max_samples_per_step_per_worker: Optional[int] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RolloutTraceConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init(
        cls,
        project_name: str,
        experiment_name: str,
        backend: str,
        token2text: bool = False,
        max_samples_per_step_per_worker: Optional[int] = None,
    ):
        config = cls.get_instance()
        if config._initialized:
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.max_samples_per_step_per_worker = max_samples_per_step_per_worker

        if backend == "weave":
            import warnings

            import weave

            # Suppress weave's use of deprecated Pydantic V1 __fields__ attribute
            warnings.filterwarnings("ignore", message=".*__fields__.*deprecated", category=DeprecationWarning)

            config.client = weave.init(project_name)
        elif backend == "mlflow":
            import mlflow

            mlflow.config.enable_async_logging()
            config.client = mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            mlflow.set_experiment(project_name)
        else:
            config.client = None

        config._initialized = True

    @classmethod
    def get_backend(cls) -> Optional[str]:
        return cls.get_instance().backend

    @classmethod
    def get_client(cls) -> Optional[object]:
        return cls.get_instance().client

    @classmethod
    def enable_token2text(cls) -> Optional[bool]:
        return cls.get_instance().token2text

    @classmethod
    def reset(cls):
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(
    sample_index=None, step=None, rollout_n=None, name="rollout_trace", validate=False, trace: bool = True
):
    """A context manager to add attributes to a trace for the configured backend.

    Args:
        sample_index: Sample index for the trace.
        step: Training step number.
        rollout_n: Rollout number (for GRPO with multiple rollouts per sample).
        name: Name for the trace span (used by mlflow backend).
        validate: Whether this is a validation run.
        trace: If False, disables tracing for the duration of the context.
    """
    backend = RolloutTraceConfig.get_backend()

    should_skip = backend is not None and not trace

    if should_skip:
        token = _trace_enabled.set(False)
        try:
            yield
        finally:
            _trace_enabled.reset(token)
        return

    # Build attributes for the trace
    attributes = {}
    if backend:
        if sample_index is not None:
            attributes["sample_index"] = sample_index
        if step is not None:
            attributes["step"] = step
        if rollout_n is not None:
            attributes["rollout_n"] = rollout_n
        attributes["validate"] = validate
        attributes["experiment_name"] = RolloutTraceConfig.get_instance().experiment_name

    if not attributes or backend is None:
        yield
        return

    if backend == "weave":
        import weave

        with weave.attributes(attributes):
            yield
    elif backend == "mlflow":
        import mlflow

        with mlflow.start_span(name=name) as span:
            trace_id = span.trace_id
            for key, value in attributes.items():
                mlflow.set_trace_tag(trace_id, str(key), str(value))
            yield
    else:
        yield


def _convert_pil_to_weave_images(obj):
    """Recursively convert Pydantic models to dicts while preserving PIL images.

    Weave natively renders PIL.Image.Image objects in traces, but Pydantic's
    model_dump() converts them to metadata dicts. This function walks the
    output structure and serializes BaseModel fields manually so PIL images
    are kept as-is for weave to render.
    """
    from PIL import Image as PILImage

    if isinstance(obj, PILImage.Image):
        return obj
    elif isinstance(obj, dict):
        return {k: _convert_pil_to_weave_images(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        converted = [_convert_pil_to_weave_images(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    elif isinstance(obj, BaseModel):
        # Walk fields manually instead of model_dump() to preserve PIL images
        return {k: _convert_pil_to_weave_images(v) for k, v in obj.__dict__.items()}
    return obj


def rollout_trace_op(func):
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return await func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        enable_token2text = RolloutTraceConfig.enable_token2text()
        if backend is None:
            return await func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        async def _decode_with_image_collapse(self, token_ids):
            """Decode token IDs, collapsing image token spans to [image]."""
            loop = get_event_loop()
            image_token_id = None
            if hasattr(self, "processor") and self.processor is not None:
                image_token_id = getattr(self.processor, "image_token_id", None)
            if image_token_id is not None:
                filtered_ids = []
                in_image_span = False
                for tid in token_ids:
                    if tid == image_token_id:
                        if not in_image_span:
                            in_image_span = True
                            filtered_ids.append(tid)
                    else:
                        in_image_span = False
                        filtered_ids.append(tid)
                text = await loop.run_in_executor(None, self.tokenizer.decode, filtered_ids)
                image_token_text = await loop.run_in_executor(None, self.tokenizer.decode, [image_token_id])
                if image_token_text:
                    text = text.replace(image_token_text, "[image]")
                return text
            return await loop.run_in_executor(None, self.tokenizer.decode, token_ids)

        async def add_token2text_single(self, result):
            """Add token2text for a single result object that has prompt_ids/response_ids/token_ids."""
            if isinstance(result, BaseModel):
                if backend == "mlflow":
                    _result = result.model_dump()
                else:
                    _result = dict(vars(result))
            else:
                _result = dict(vars(result))

            if hasattr(result, "prompt_ids"):
                _result["prompt_text"] = await _decode_with_image_collapse(self, result.prompt_ids)

            if hasattr(result, "response_ids"):
                _result["response_text"] = await _decode_with_image_collapse(self, result.response_ids)

            # TokenOutput uses token_ids instead of response_ids
            if hasattr(result, "token_ids") and not hasattr(result, "response_ids"):
                _result["response_text"] = await _decode_with_image_collapse(self, result.token_ids)
            return _result

        async def enrich_inputs(self, inputs):
            """Decode token IDs and preserve PIL images in trace inputs."""
            if not hasattr(self, "tokenizer") or not hasattr(self.tokenizer, "decode"):
                return inputs
            enriched = dict(inputs)
            if "prompt_ids" in enriched and isinstance(enriched["prompt_ids"], list):
                enriched["prompt_text"] = await _decode_with_image_collapse(self, enriched["prompt_ids"])
            return enriched

        async def add_token2text(self, result):
            if not hasattr(self, "tokenizer") or not hasattr(self.tokenizer, "decode"):
                return result

            if hasattr(result, "prompt_ids") or hasattr(result, "token_ids"):
                return await add_token2text_single(self, result)

            # Handle grouped outputs (e.g. AgentLoopGroupOutput) where trajectories
            # contain the per-turn prompt_ids/response_ids
            if hasattr(result, "trajectories"):
                if isinstance(result, BaseModel):
                    if backend == "mlflow":
                        _result = result.model_dump()
                    else:
                        _result = dict(vars(result))
                else:
                    _result = dict(vars(result))
                _result["trajectories"] = [
                    await add_token2text_single(self, traj)
                    if hasattr(traj, "prompt_ids")
                    else (dict(vars(traj)) if isinstance(traj, BaseModel) else traj)
                    for traj in result.trajectories
                ]
                return _result

            return result

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            if enable_token2text:
                trace_inputs = await enrich_inputs(self, inputs)
                trace_inputs = _convert_pil_to_weave_images(trace_inputs)
            else:
                trace_inputs = _convert_pil_to_weave_images(inputs)
            call = tracer.create_call(op=func.__qualname__, inputs=trace_inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)

                if enable_token2text:
                    _result = await add_token2text(self, result)
                    _output = _convert_pil_to_weave_images(_result)
                else:
                    _output = _convert_pil_to_weave_images(result)
                tracer.finish_call(call, output=_output)

                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                if enable_token2text:
                    trace_inputs = await enrich_inputs(self, inputs)
                    span.set_inputs(trace_inputs)
                else:
                    span.set_inputs(inputs)
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await add_token2text(self, result)
                    span.set_outputs(_result)
                else:
                    span.set_outputs(result)

            return result

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        if backend is None:
            return func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = func(self, *args, **kwargs)
                _output = _convert_pil_to_weave_images(result)
                tracer.finish_call(call, output=_output)
                return result
            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            return mlflow.trace(func)(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
