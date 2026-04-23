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

import base64
import contextlib
import functools
import inspect
import io
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
            import weave

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


def _is_pil_image(obj) -> bool:
    """Check whether ``obj`` is a PIL.Image.Image without requiring PIL imported eagerly."""
    try:
        from PIL import Image as _PILImage

        return isinstance(obj, _PILImage.Image)
    except ImportError:
        return False


def _pil_to_data_uri(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _collect_images_as_content_parts(obj, _depth=0, _parts=None):
    """Walk ``obj`` and collect every PIL image as an OpenAI ``image_url`` part.

    MLflow's trace UI renders images inline only in the "Chat tab" and only
    when inputs/outputs follow the OpenAI chat-messages schema with
    ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``
    content parts (see
    https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/multimodal/).

    We do a *shallow flatten*: any PIL image encountered at any depth is
    turned into one ``image_url`` content part, in traversal order. Scalar
    PIL inputs also work thanks to the initial scalar branch.
    """
    if _parts is None:
        _parts = []
    if _depth > 10:
        return _parts
    if _is_pil_image(obj):
        _parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _pil_to_data_uri(obj)},
            }
        )
        return _parts
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_images_as_content_parts(v, _depth + 1, _parts)
        return _parts
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_images_as_content_parts(v, _depth + 1, _parts)
        return _parts
    return _parts


def _to_mlflow_chat_messages(
    payload: dict, *, role: str, default_name: str = "payload"
) -> dict:
    """Convert a free-form dict payload into MLflow's OpenAI-style chat schema.

    The resulting ``{"messages": [...]}`` structure is rendered in the MLflow
    trace UI's Chat tab with images shown inline. Non-image content is kept
    as a single JSON text part so nothing is lost.

    Args:
        payload: Serialized dict payload (PIL images already converted to
            data URIs by :func:`_serialize_for_trace`, OR still raw PIL —
            both are handled).
        role: ``"user"`` for inputs, ``"assistant"`` for outputs.
        default_name: Fallback top-level field name when images are attached
            directly (not inside a dict).
    """
    image_parts = _collect_images_as_content_parts(payload)

    # Text part: everything except PIL images (stripped) rendered as JSON.
    def _strip_images(obj, _depth=0):
        if _depth > 10:
            return repr(obj)
        if _is_pil_image(obj):
            return None
        if isinstance(obj, dict):
            return {k: _strip_images(v, _depth + 1) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            stripped = [_strip_images(v, _depth + 1) for v in obj]
            return type(obj)(stripped) if isinstance(obj, tuple) else stripped
        return obj

    stripped = _strip_images(payload)
    try:
        import json as _json

        text = _json.dumps(stripped, default=repr, ensure_ascii=False, indent=2)
    except Exception:
        text = repr(stripped)

    content_parts: list = [{"type": "text", "text": text}]
    content_parts.extend(image_parts)

    return {"messages": [{"role": role, "content": content_parts}]}


def _serialize_for_trace(obj, _depth=0, *, keep_pil: bool = False):
    """Recursively prepare ``obj`` for tracing backends.

    Two transformations happen here:

    1. ``PIL.Image.Image`` objects — MLflow cannot serialize them and only
       stores the ``repr`` string. We convert to ``data:image/png;base64,…``
       data URIs by default. Weave's native type handler DOES render PIL
       images inline, so pass ``keep_pil=True`` on the weave backend.

    2. Lists/tuples containing PIL images — Weave's auto-renderer is
       documented only for plain PIL values, not "list of PIL inside a dict".
       To maximise the chance the UI shows them inline, we split any such
       container into flat ``image_0 / image_1 / ...`` keys when it lives
       inside a dict, and fall back to a dict with the same shape otherwise.

    Args:
        obj: The object to serialize.
        _depth: Recursion depth (internal guard).
        keep_pil: If True, leave PIL images as raw objects (weave backend).
            If False, convert to base64 data URI strings (mlflow backend).
    """
    if _depth > 10:
        return repr(obj)

    # Scalar PIL image.
    if _is_pil_image(obj):
        if keep_pil:
            return obj
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if (
                isinstance(v, (list, tuple))
                and v
                and any(_is_pil_image(x) for x in v)
            ):
                # Flatten ``list[PIL | other]`` into ``{k}_0, {k}_1, ...``
                # so that Weave's UI renders each element as an image slot.
                non_image_items: list = []
                for i, item in enumerate(v):
                    if _is_pil_image(item):
                        out[f"{k}_{i}"] = _serialize_for_trace(
                            item, _depth + 1, keep_pil=keep_pil
                        )
                    else:
                        non_image_items.append(
                            _serialize_for_trace(item, _depth + 1, keep_pil=keep_pil)
                        )
                if non_image_items:
                    out[f"{k}_other"] = non_image_items
            else:
                out[k] = _serialize_for_trace(v, _depth + 1, keep_pil=keep_pil)
        return out

    if isinstance(obj, (list, tuple)):
        serialized = [
            _serialize_for_trace(v, _depth + 1, keep_pil=keep_pil) for v in obj
        ]
        return type(obj)(serialized) if isinstance(obj, tuple) else serialized

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

        async def add_token2text(self, result):
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None or not hasattr(tokenizer, "decode"):
                return result

            if isinstance(result, BaseModel):
                _result = result.model_dump()
            elif hasattr(result, "__dict__"):
                _result = dict(vars(result))
            else:
                return result

            loop = get_event_loop()

            # AgentLoopOutput: prompt_ids + response_ids
            if hasattr(result, "prompt_ids"):
                prompt_text = await loop.run_in_executor(None, tokenizer.decode, result.prompt_ids)
                _result["prompt_text"] = prompt_text
            if hasattr(result, "response_ids"):
                response_text = await loop.run_in_executor(None, tokenizer.decode, result.response_ids)
                _result["response_text"] = response_text

            # TokenOutput (generate): token_ids holds the response tokens
            if hasattr(result, "token_ids") and not hasattr(result, "prompt_ids"):
                response_text = await loop.run_in_executor(None, tokenizer.decode, result.token_ids)
                _result["response_text"] = response_text

            return _result

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            # Keep PIL.Image objects intact so Weave auto-renders them as
            # inline images in the trace UI (base64 strings would display
            # as raw text only).
            weave_inputs = _serialize_for_trace(inputs, keep_pil=True)
            call = tracer.create_call(op=func.__qualname__, inputs=weave_inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)

                if enable_token2text:
                    _result = await add_token2text(self, result)
                    tracer.finish_call(call, output=_serialize_for_trace(_result, keep_pil=True))
                else:
                    tracer.finish_call(call, output=_serialize_for_trace(result, keep_pil=True))

                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                # MLflow renders images inline only when inputs/outputs follow
                # the OpenAI chat-messages schema. Wrap our structured payload
                # into a single chat message whose content parts include one
                # ``image_url`` per PIL image plus a JSON text blob for the
                # rest of the fields.
                span.set_inputs(
                    _to_mlflow_chat_messages(inputs, role="user")
                )
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await add_token2text(self, result)
                    span.set_outputs(
                        _to_mlflow_chat_messages(_result, role="assistant")
                    )
                else:
                    span.set_outputs(
                        _to_mlflow_chat_messages(
                            result if isinstance(result, dict) else {"output": result},
                            role="assistant",
                        )
                    )

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
                tracer.finish_call(call, output=result)
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
