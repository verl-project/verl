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
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from verl.utils.ray_utils import get_event_loop

_trace_enabled: ContextVar[bool] = ContextVar("_trace_enabled", default=True)


@dataclass
class Token2TextField:
    """Configuration for a single token-to-text field mapping in rollout tracing.

    When ``token2text`` is enabled globally via :class:`RolloutTraceConfig`,
    each :class:`Token2TextField` tells the :func:`rollout_trace_op` decorator
    where to find a list of token IDs and what name to give the decoded text in
    the trace output.

    Args:
        source: Where to find the token IDs -- ``"input"`` to read from the
            decorated function's arguments, ``"output"`` to read from its
            return value.
        field: Name of the field containing the token IDs.
        decode_to: Name of the target field for the decoded text that will
            appear in the trace output.
    """

    source: str  # "input" or "output"
    field: str
    decode_to: str


# Default fields used when ``token2text_fields`` is not explicitly provided to
# ``rollout_trace_op``.  This preserves backward compatibility with the
# original hard-coded behaviour that looked for ``result.prompt_ids`` and
# ``result.response_ids``.
_DEFAULT_TOKEN2TEXT_FIELDS = [
    Token2TextField(source="output", field="prompt_ids", decode_to="prompt_text"),
    Token2TextField(source="output", field="response_ids", decode_to="response_text"),
]


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
    tokenizer: Optional[object] = None

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
        tokenizer: Optional[object] = None,
    ):
        config = cls.get_instance()
        if config._initialized:
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.max_samples_per_step_per_worker = max_samples_per_step_per_worker
        config.tokenizer = tokenizer

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
    def get_tokenizer(cls) -> Optional[object]:
        return cls.get_instance().tokenizer

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


def _resolve_tokenizer(self):
    """Resolve the tokenizer to use for token2text decoding.

    Resolution order:
    1. ``self.tokenizer`` — per-instance tokenizer (supports multi-model setups
       where different traced objects use different tokenizers).
    2. ``RolloutTraceConfig.get_tokenizer()`` — global fallback set once per
       worker at init time (for classes that don't carry their own tokenizer,
       e.g. ``AsyncLLMServerManager``).

    Returns:
        A tokenizer object with a ``decode`` method, or ``None``.
    """
    tokenizer = getattr(self, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer
    tokenizer = RolloutTraceConfig.get_tokenizer()
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer
    return None


async def _add_token2text(self, result, inputs, token2text_fields):
    """Decode token ID fields into text and return an enriched copy for tracing.

    This is used internally by :func:`rollout_trace_op` to convert token IDs
    (from either the function's inputs or outputs) into human-readable text
    that gets recorded in the trace.

    Args:
        self: The class instance.  If it has a ``tokenizer`` attribute with a
            ``decode`` method, that tokenizer is used; otherwise falls back to
            the global tokenizer stored in :class:`RolloutTraceConfig`.
        result: The return value of the decorated function.
        inputs: Dict of the decorated function's bound arguments (excluding
            ``self``).
        token2text_fields: List of :class:`Token2TextField` describing which
            fields to decode.  If ``None``, falls back to
            :data:`_DEFAULT_TOKEN2TEXT_FIELDS`.

    Returns:
        A dict copy of *result* enriched with decoded text fields, or the
        original *result* unchanged when no fields could be decoded.
    """
    tokenizer = _resolve_tokenizer(self)
    if tokenizer is None:
        return result

    effective_fields = token2text_fields if token2text_fields is not None else _DEFAULT_TOKEN2TEXT_FIELDS

    loop = get_event_loop()
    decoded: dict[str, str] = {}

    for field_cfg in effective_fields:
        token_ids = None
        if field_cfg.source == "input":
            token_ids = inputs.get(field_cfg.field)
        elif field_cfg.source == "output":
            if hasattr(result, field_cfg.field):
                token_ids = getattr(result, field_cfg.field)
            elif isinstance(result, dict):
                token_ids = result.get(field_cfg.field)

        if token_ids is not None:
            text = await loop.run_in_executor(None, tokenizer.decode, token_ids)
            decoded[field_cfg.decode_to] = text

    if not decoded:
        return result

    # Create a mutable dict copy of the result and add decoded fields.
    # Use model_dump() for Pydantic models to get a proper copy;
    # otherwise vars() returns a reference to internal __dict__ which
    # can cause serialization issues with MLflow.
    if isinstance(result, BaseModel):
        _result = result.model_dump()
    elif isinstance(result, dict):
        _result = dict(result)
    else:
        _result = dict(vars(result))

    _result.update(decoded)
    return _result


def rollout_trace_op(func=None, *, token2text_fields=None):
    """Decorator that traces function calls with the configured tracing backend.

    Can be used in two forms:

    1. Without arguments (backward compatible)::

        @rollout_trace_op
        async def run(self, ...): ...

    2. With explicit token-to-text field mappings::

        @rollout_trace_op(token2text_fields=[
            Token2TextField(source="input", field="prompt_ids", decode_to="prompt_text"),
            Token2TextField(source="output", field="token_ids", decode_to="response_text"),
        ])
        async def generate(self, ...): ...

    When ``token2text`` is enabled globally (via :class:`RolloutTraceConfig`)
    and ``token2text_fields`` is provided, the decorator decodes the specified
    token ID fields and includes the decoded text in the trace output.  When
    ``token2text_fields`` is ``None`` (the default), it falls back to the
    legacy behaviour of looking for ``result.prompt_ids`` and
    ``result.response_ids``.

    Args:
        func: The function being decorated (set automatically when used
            without parentheses).
        token2text_fields: Optional list of :class:`Token2TextField`
            specifying which fields to decode.  ``None`` means use defaults.
    """

    def _decorator(fn):
        fields = token2text_fields  # capture in closure

        @functools.wraps(fn)
        async def async_wrapper(self, *args, **kwargs):
            if not _trace_enabled.get():
                return await fn(self, *args, **kwargs)

            backend = RolloutTraceConfig.get_backend()
            enable_token2text = RolloutTraceConfig.enable_token2text()
            if backend is None:
                return await fn(self, *args, **kwargs)

            sig = inspect.signature(fn)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            inputs = dict(bound_args.arguments)
            del inputs["self"]

            if backend == "weave":
                tracer = RolloutTraceConfig.get_client()
                from weave.trace.context import call_context

                cur_attributes = {**call_context.call_attributes.get()}
                call = tracer.create_call(op=fn.__qualname__, inputs=inputs, attributes=cur_attributes)
                try:
                    result = await fn(self, *args, **kwargs)

                    if enable_token2text:
                        _result = await _add_token2text(self, result, inputs, fields)
                        tracer.finish_call(call, output=_result)
                    else:
                        tracer.finish_call(call, output=result)

                    return result

                except Exception as e:
                    tracer.finish_call(call, exception=e)
                    raise e
            elif backend == "mlflow":
                import mlflow

                with mlflow.start_span(name=fn.__qualname__) as span:
                    span.set_inputs(inputs)
                    result = await fn(self, *args, **kwargs)
                    if enable_token2text:
                        _result = await _add_token2text(self, result, inputs, fields)
                        span.set_outputs(_result)
                    else:
                        span.set_outputs(result)

                return result

            else:
                return await fn(self, *args, **kwargs)

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not _trace_enabled.get():
                return fn(self, *args, **kwargs)

            backend = RolloutTraceConfig.get_backend()
            if backend is None:
                return fn(self, *args, **kwargs)

            sig = inspect.signature(fn)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            inputs = dict(bound_args.arguments)
            del inputs["self"]

            if backend == "weave":
                tracer = RolloutTraceConfig.get_client()
                from weave.trace.context import call_context

                cur_attributes = {**call_context.call_attributes.get()}
                call = tracer.create_call(op=fn.__qualname__, inputs=inputs, attributes=cur_attributes)
                try:
                    result = fn(self, *args, **kwargs)
                    tracer.finish_call(call, output=result)
                    return result
                except Exception as e:
                    tracer.finish_call(call, exception=e)
                    raise e
            elif backend == "mlflow":
                import mlflow

                return mlflow.trace(fn)(self, *args, **kwargs)
            else:
                return fn(self, *args, **kwargs)

        return async_wrapper if inspect.iscoroutinefunction(fn) else wrapper

    if func is not None:
        # Called as @rollout_trace_op (without parentheses)
        return _decorator(func)
    # Called as @rollout_trace_op(...) (with parentheses)
    return _decorator
