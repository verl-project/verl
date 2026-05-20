from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any
from uuid import uuid4

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from verl.agent.framework.types import SessionHandle, Trajectory
from verl.agent.gateway.types import GatewaySessionState, SessionPhase, TrajectoryBuffer
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.chat_template import apply_chat_template as _apply_chat_template, initialize_system_prompt
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.utils import run_uvicorn


class MalformedRequestError(ValueError):
    pass


_DEFAULT_ALLOWED_REQUEST_SAMPLING_PARAM_KEYS = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
})


# Map backend stop_reason values to OpenAI-spec finish_reason values.
# OpenAI Chat Completions spec defines finish_reason ∈
# {"stop", "length", "tool_calls", "content_filter", "function_call"}.
#
# Note on vLLM information loss: the vLLM rollout adapter
# (verl/workers/rollout/vllm_rollout/vllm_async_server.py:538-545)
# collapses vLLM's raw finish_reason "stop" and "length" into a single
# "completed" stop_reason before the gateway sees it. As a result,
# mapping "completed" -> "stop" here cannot recover whether generation
# actually hit max_tokens; recovering that distinction requires the
# vLLM adapter to preserve the raw finish_reason on TokenOutput.
# TODO(phase-c): preserve raw backend finish_reason on TokenOutput
# (e.g. a new TokenOutput.finish_reason field or
# extra_fields["finish_reason"]) so the gateway can distinguish vLLM
# "length" from "stop" instead of mapping both to "stop".
_FINISH_REASON_MAP = {
    "completed": "stop",
    "stop": "stop",
    "matched_stop": "stop",
    "eos": "stop",
    "length": "length",
    "max_tokens": "length",
    "aborted": "stop",
    "abort": "stop",
}


# TODO: double-check if all these validations/normalization are necessary
# Make sure they don't alter messages in unexpected ways.
def _normalize_message_content(content: Any) -> Any:
    """Normalize message content: coerce None to empty string, validate type."""
    if isinstance(content, list | dict | str):
        return content
    if content is None:
        return ""
    raise MalformedRequestError(f"Unsupported content type: {type(content).__name__}")


def _validate_tool_calls(tool_calls: Any) -> None:
    """Validate tool_calls structure. Does not modify content."""
    if not isinstance(tool_calls, list):
        raise MalformedRequestError("tool_calls must be a list")
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise MalformedRequestError("tool_calls entries must be objects")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise MalformedRequestError("tool_call.function must be an object")


def _normalize_message(message: Any) -> dict[str, Any]:
    """Normalize a single message: validate structure, coerce types, filter to known fields.

    Constructs a new dict with only role/content/tool_calls/tool_call_id.
    This ensures prefix comparison is not affected by extraneous fields.
    """
    if not isinstance(message, dict):
        raise MalformedRequestError("messages entries must be objects")

    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise MalformedRequestError("message.role must be a non-empty string")

    normalized: dict[str, Any] = {
        "role": role,
        "content": _normalize_message_content(message.get("content", "")),
    }
    if "name" in message:
        name = message["name"]
        if not isinstance(name, str):
            raise MalformedRequestError("message.name must be a string")
        normalized["name"] = name
    if "tool_calls" in message:
        _validate_tool_calls(message["tool_calls"])
        normalized["tool_calls"] = list(message["tool_calls"])
    if "tool_call_id" in message:
        normalized["tool_call_id"] = str(message["tool_call_id"])
    return normalized


def _validate_tools(tools: Any) -> list[Any] | None:
    """Validate tools structure. Does not modify content."""
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")
    return tools


def _normalize_request_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate the request payload, extracting messages and tools.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise MalformedRequestError("messages must be non-empty")
    return {
        "messages": [_normalize_message(message) for message in messages],
        "tools": _validate_tools(payload.get("tools")),
    }


def _build_sampling_params(
    payload: dict[str, Any],
    *,
    base_sampling_params: dict[str, Any],
    allowed_request_sampling_param_keys: frozenset[str],
) -> dict[str, Any]:
    sampling_params = dict(base_sampling_params)
    for key in allowed_request_sampling_param_keys:
        if key in payload:
            sampling_params[key] = payload[key]
    return sampling_params


def _canonicalize_tool_arguments_for_comparison(arguments: Any) -> tuple[str, Any]:
    if isinstance(arguments, dict | list):
        return ("json", arguments)
    if isinstance(arguments, str):
        try:
            return ("json", json.loads(arguments))
        except json.JSONDecodeError:
            return ("raw", arguments)
    return ("raw", arguments)


def _canonicalize_message_for_prefix_comparison(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    tool_calls = normalized.get("tool_calls")
    if not isinstance(tool_calls, list):
        return normalized

    normalized_tool_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        normalized_tool_call = dict(tool_call)
        function = normalized_tool_call.get("function")
        if isinstance(function, dict) and "arguments" in function:
            normalized_function = dict(function)
            normalized_function["arguments"] = _canonicalize_tool_arguments_for_comparison(function["arguments"])
            normalized_tool_call["function"] = normalized_function
        normalized_tool_calls.append(normalized_tool_call)
    normalized["tool_calls"] = normalized_tool_calls
    return normalized


def _is_message_prefix(prefix: list[dict[str, Any]], messages: list[dict[str, Any]]) -> bool:
    if len(prefix) > len(messages):
        return False
    return [
        _canonicalize_message_for_prefix_comparison(message)
        for message in prefix
    ] == [
        _canonicalize_message_for_prefix_comparison(message)
        for message in messages[: len(prefix)]
    ]


def _is_request_context_prefix(
    *,
    session: GatewaySessionState,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> bool:
    if session.request_tools != tools:
        return False
    # TODO: dict equality is not token-level equivalent — two tool schemas with
    # different key order compare equal in Python but may tokenize differently.
    # This could cause a false prefix match on the tools path.  Low practical
    # risk (same agent rarely reorders keys within a session), but worth noting.
    #TODO: need to improve the prefix check logic, e.g.,how to handle tool lists and multimodal data
    return _is_message_prefix(session.message_history, messages)


def _copy_trajectory_buffer(buffer: TrajectoryBuffer | None) -> TrajectoryBuffer | None:
    if buffer is None:
        return None
    return TrajectoryBuffer(
        prompt_ids=list(buffer.prompt_ids),
        response_ids=list(buffer.response_ids),
        response_mask=list(buffer.response_mask),
        response_logprobs=list(buffer.response_logprobs),
    )


def _count_chat_turns(message_history: list[dict[str, Any]]) -> int:
    """Count chat turns consistent with ToolAgentLoop semantics.

    ToolAgentLoop computes: user_turns + assistant_turns + 1 (the +1 accounts
    for the initial prompt).  We count user + assistant role messages and add 1.
    System and tool messages are excluded.
    """
    return sum(1 for m in message_history if m.get("role") in ("user", "assistant")) + 1


def _materialize_response_logprobs(buffer: TrajectoryBuffer) -> list[float] | None:
    if not buffer.response_logprobs:
        return None
    return list(buffer.response_logprobs)


def _build_multi_modal_trajectory_data(
    image_data: list[Any] | None,
    video_data: list[Any] | None,
) -> dict[str, Any] | None:
    multi_modal_data: dict[str, Any] = {}
    if image_data:
        multi_modal_data["images"] = list(image_data)
    if video_data:
        multi_modal_data["videos"] = list(video_data)
    return multi_modal_data or None


class _GatewayActor:
    def __init__(
        self,
        tokenizer,
        backend,
        *,
        processor=None,
        vision_info_extractor=None,
        vision_info_extractor_kwargs: dict[str, Any] | None = None,
        tool_parser_name: str | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        base_sampling_params: dict[str, Any] | None = None,
        allowed_request_sampling_param_keys: set[str] | frozenset[str] | None = None,
    ):
        # Same pattern as vllm_async_server.py / async_sglang_server.py:
        # use the node's routable IP for both bind and URL.
        self._server_address = ray.util.get_node_ip_address()
        self._tokenizer = tokenizer
        self._processor = processor
        self._backend = backend
        self._vision_info_extractor = vision_info_extractor or self._default_vision_info_extractor
        self._vision_info_extractor_kwargs = dict(vision_info_extractor_kwargs or {})
        self._apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self._base_sampling_params = dict(base_sampling_params or {})
        allowed_keys = (
            _DEFAULT_ALLOWED_REQUEST_SAMPLING_PARAM_KEYS
            if allowed_request_sampling_param_keys is None
            else frozenset(allowed_request_sampling_param_keys)
        )
        self._allowed_request_sampling_param_keys = allowed_keys
        self._system_prompt = initialize_system_prompt(
            tokenizer,
            **self._apply_chat_template_kwargs,
        )
        self._tool_parser = (
            ToolParser.get_tool_parser(tool_parser_name, tokenizer) if tool_parser_name else None
        )
        self._sessions: dict[str, GatewaySessionState] = {}
        self._app = FastAPI()
        self._server_port: int | None = None
        self._server_task: asyncio.Task | None = None
        self._server_base_url: str | None = None
        self._register_routes()

    def _register_routes(self) -> None:
        @self._app.post("/sessions/{session_id}/v1/chat/completions")
        async def _chat_completions(session_id: str, request: Request):
            payload = await request.json()
            return await self._handle_chat_completions(session_id=session_id, payload=payload)

        @self._app.post("/sessions/{session_id}/complete")
        async def _complete(session_id: str, request: Request):
            payload = await request.json()
            reward_info = payload.get("reward_info")
            try:
                await self.complete_session(session_id=session_id, reward_info=reward_info)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return JSONResponse({"status": "ok"})

    def _require_started(self) -> None:
        if self._server_base_url is None:
            raise RuntimeError("GatewayActor.start() must be called before session creation")

    def _get_session(self, session_id: str) -> GatewaySessionState:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session

    def _set_phase(self, session: GatewaySessionState, phase: SessionPhase) -> None:
        session.phase = phase
        self._touch_session(session)

    def _touch_session(self, session: GatewaySessionState) -> None:
        session.updated_at = time.time()

    def _materialize_active_trajectory(self, session: GatewaySessionState) -> None:
        active = session.active_trajectory
        if active is None:
            return

        self._touch_session(session)
        session.trajectories.append(
            self._build_materialized_trajectory(
                session=session,
                active=active,
            )
        )
        session.active_trajectory = None

    def _build_materialized_trajectory(
        self,
        *,
        session: GatewaySessionState,
        active: TrajectoryBuffer,
    ) -> Trajectory:
        return Trajectory(
            prompt_ids=list(active.prompt_ids),
            response_ids=list(active.response_ids),
            response_mask=list(active.response_mask),
            response_logprobs=_materialize_response_logprobs(active),
            reward_info={},
            num_turns=_count_chat_turns(session.message_history),
            multi_modal_data=_build_multi_modal_trajectory_data(session.image_data, session.video_data),
        )

    async def _default_vision_info_extractor(
        self,
        messages: list[dict[str, Any]],
        *,
        image_patch_size: int,
    ) -> tuple[list[Any] | None, list[Any] | None]:
        # Keep the dataset dependency lazy so custom extractors do not pay for
        # RLHFDataset imports unless they actually use the default path.
        from verl.utils.dataset.rl_dataset import RLHFDataset

        return await RLHFDataset.process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            config=self._vision_info_extractor_kwargs.get("config"),
        )

    async def _extract_multi_modal_data(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[Any] | None, list[Any] | None]:
        if self._processor is None:
            return None, None

        has_multi_modal_blocks = False
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"image", "image_url", "video", "video_url"}:
                    has_multi_modal_blocks = True
                    break
            if has_multi_modal_blocks:
                break

        if not has_multi_modal_blocks:
            return None, None

        return await self._vision_info_extractor(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            **self._vision_info_extractor_kwargs,
        )

    def _encode_full(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> list[int]:
        """Encode a full conversation for a new trajectory (includes system prompt + generation prompt)."""
        if self._processor is not None:
            raw_prompt = _apply_chat_template(
                self._processor,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **self._apply_chat_template_kwargs,
            )
            videos = video_data
            video_metadata = None
            if videos is not None:
                videos, video_metadata = zip(*videos, strict=False)
                videos, video_metadata = list(videos), list(video_metadata)
            model_inputs = self._processor(
                text=[raw_prompt],
                images=image_data,
                videos=videos,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_sample_frames=False,
            )
            return normalize_token_ids(model_inputs["input_ids"])

        return normalize_token_ids(
            _apply_chat_template(
                self._tokenizer, messages, tools=tools, add_generation_prompt=True,
                **self._apply_chat_template_kwargs,
            )
        )
    # TODO: check if delta tokenization is better than remove_system_prompt
    def _encode_incremental(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> list[int]:
        """Encode incremental messages (tool results, user follow-ups) for a continuation turn.

        Uses the remove_system_prompt pattern from ToolAgentLoop: encode the new messages
        alone (which prepends a system prompt), then strip the known system_prompt prefix.
        No tools parameter — tool schema is already in the initial prompt_ids.
        """
        if self._processor is not None:
            raw_prompt = _apply_chat_template(
                self._processor,
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self._apply_chat_template_kwargs,
            )
            videos = video_data
            video_metadata = None
            if videos is not None:
                videos, video_metadata = zip(*videos, strict=False)
                videos, video_metadata = list(videos), list(video_metadata)
            model_inputs = self._processor(
                text=[raw_prompt],
                images=image_data,
                videos=videos,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_sample_frames=False,
            )
            ids = normalize_token_ids(model_inputs["input_ids"])
        else:
            ids = normalize_token_ids(
                _apply_chat_template(
                    self._tokenizer, messages, add_generation_prompt=True,
                    **self._apply_chat_template_kwargs,
                )
            )
        return ids[len(self._system_prompt):]

    async def _decode_response(
        self, response_ids: list[int], *, tools: list[dict[str, Any]] | None = None,
        stop_reason: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Decode model output tokens into an OpenAI-compatible assistant message.

        Returns:
            message: OpenAI-compatible assistant message.
            finish_reason: "tool_calls" when tool calls are present, else the
                OpenAI-spec-normalized stop_reason (see _FINISH_REASON_MAP).
        """
        if self._tool_parser is not None and tools:
            parsed_tools = None
            try:
                from verl.tools.schemas import OpenAIFunctionToolSchema
                parsed_tools = [
                    OpenAIFunctionToolSchema(**t) if isinstance(t, dict) else t
                    for t in tools
                ]
            except Exception:
                pass
            content, function_calls = await self._tool_parser.extract_tool_calls(response_ids, parsed_tools)
            if function_calls:
                tool_calls = [
                    {
                        "id": f"call_{uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": fc.name, "arguments": fc.arguments},
                    }
                    for fc in function_calls
                ]
                message = {
                    "role": "assistant",
                    # Use "" instead of None so that prefix comparison with
                    # _normalize_message_content (which also coerces None → "")
                    # stays consistent.  Both must agree on the None policy.
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
                return message, "tool_calls"
        response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        finish_reason = _FINISH_REASON_MAP.get(stop_reason, stop_reason) if stop_reason else "stop"
        return {"role": "assistant", "content": response_text}, finish_reason

    async def _handle_chat_completions(self, session_id: str, payload: dict[str, Any]) -> JSONResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

        try:
            request_context = _normalize_request_context(payload)
        except MalformedRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async with session.generation_lock:
            if session.phase != SessionPhase.ACTIVE:
                raise HTTPException(status_code=409, detail=f"Session {session_id} is {session.phase.value.lower()}")

            async with session.request_lock:
                if session.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409, detail=f"Session {session_id} is {session.phase.value.lower()}"
                    )

                self._touch_session(session)
                messages = request_context["messages"]
                tools = request_context["tools"]
                materialized_trajectory = None
                image_data = None
                video_data = None

                if session.active_trajectory is None:
                    image_data, video_data = await self._extract_multi_modal_data(messages)
                    active_trajectory = TrajectoryBuffer(
                        prompt_ids=self._encode_full(
                            messages, tools=tools, image_data=image_data, video_data=video_data
                        )
                    )
                elif _is_request_context_prefix(session=session, messages=messages, tools=tools):
                    active_trajectory = _copy_trajectory_buffer(session.active_trajectory)
                    image_data = list(session.image_data) if session.image_data is not None else None
                    video_data = list(session.video_data) if session.video_data is not None else None
                    incremental_messages = messages[len(session.message_history) :]
                    if incremental_messages:
                        new_image_data, new_video_data = await self._extract_multi_modal_data(incremental_messages)
                        if new_image_data:
                            if image_data is None:
                                image_data = []
                            image_data.extend(new_image_data)
                        if new_video_data:
                            if video_data is None:
                                video_data = []
                            video_data.extend(new_video_data)
                        incremental_ids = self._encode_incremental(
                            incremental_messages,
                            image_data=new_image_data,
                            video_data=new_video_data,
                        )
                        active_trajectory.response_ids.extend(incremental_ids)
                        active_trajectory.response_mask.extend([0] * len(incremental_ids))
                        if active_trajectory.response_logprobs:
                            active_trajectory.response_logprobs.extend([0.0] * len(incremental_ids))
                else:
                    materialized_trajectory = self._build_materialized_trajectory(
                        session=session,
                        active=session.active_trajectory,
                    )
                    image_data, video_data = await self._extract_multi_modal_data(messages)
                    active_trajectory = TrajectoryBuffer(
                        prompt_ids=self._encode_full(
                            messages, tools=tools, image_data=image_data, video_data=video_data
                        )
                    )

                generation_context_ids = active_trajectory.prompt_ids + active_trajectory.response_ids
                sampling_params = _build_sampling_params(
                    payload,
                    base_sampling_params=self._base_sampling_params,
                    allowed_request_sampling_param_keys=self._allowed_request_sampling_param_keys,
                )

            output = await self._backend.generate(
                request_id=session_id,
                prompt_ids=generation_context_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )

            response_ids = list(output.token_ids)
            active_trajectory.response_ids.extend(response_ids)
            active_trajectory.response_mask.extend([1] * len(response_ids))
            if output.log_probs is not None:
                active_trajectory.response_logprobs.extend(list(output.log_probs))

            assistant_msg, finish_reason = await self._decode_response(
                response_ids, tools=tools, stop_reason=output.stop_reason,
            )
            async with session.request_lock:
                if session.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409, detail=f"Session {session_id} is {session.phase.value.lower()}"
                    )

                if materialized_trajectory is not None:
                    session.trajectories.append(materialized_trajectory)
                session.active_trajectory = active_trajectory
                session.image_data = list(image_data) if image_data is not None else None
                session.video_data = list(video_data) if video_data is not None else None
                session.message_history = messages + [assistant_msg]
                session.request_tools = tools
                self._touch_session(session)

                return JSONResponse(
                    {
                        "id": f"chatcmpl-{uuid4().hex}",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": assistant_msg,
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(generation_context_ids),
                            "completion_tokens": len(response_ids),
                            "total_tokens": len(generation_context_ids) + len(response_ids),
                        },
                    }
                )

    async def start(self) -> None:
        if self._server_task is not None:
            return
        self._server_port, self._server_task = await run_uvicorn(self._app, None, self._server_address)
        self._server_base_url = f"http://{self._server_address}:{self._server_port}"

    async def shutdown(self) -> None:
        if self._server_task is None:
            return
        self._server_task.cancel()
        try:
            await self._server_task
        except asyncio.CancelledError:
            pass
        self._server_task = None
        self._server_port = None
        self._server_base_url = None

    async def create_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> SessionHandle:
        self._require_started()
        if session_id in self._sessions:
            raise RuntimeError(f"Session {session_id} already exists")

        handle = SessionHandle(
            session_id=session_id,
            base_url=f"{self._server_base_url}/sessions/{session_id}/v1",
        )
        self._sessions[session_id] = GatewaySessionState(handle=handle, metadata=dict(metadata or {}))
        return handle

    async def complete_session(self, session_id: str, reward_info: dict[str, Any] | None = None) -> None:
        session = self._get_session(session_id)
        async with session.request_lock:
            # Accommodate retry attempts
            if session.phase not in {SessionPhase.COMPLETED, SessionPhase.ACTIVE}:
                raise RuntimeError(f"Session {session_id} is {session.phase.value.lower()}")

            if reward_info is not None:
                session.reward_info = dict(reward_info)

            self._set_phase(session, SessionPhase.COMPLETED)
            session.completed.set()

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            # Already finalized or aborted by a concurrent caller — nothing to wait for.
            return
        if session.phase == SessionPhase.COMPLETED:
            # Fast path: agent already called /complete, no need to wait.
            return

        await asyncio.wait_for(session.completed.wait(), timeout=timeout)

        # Post-await: the session may have been aborted during the wait.
        # The local reference is still valid even if the session was removed from _sessions.
        if session.phase == SessionPhase.ABORTED:
            raise RuntimeError(f"Session {session_id} is aborted")

    async def finalize_session(self, session_id: str) -> list[Trajectory]:
        session = self._get_session(session_id)
        async with session.request_lock:
            if session.phase == SessionPhase.ABORTED:
                raise RuntimeError(f"Session {session_id} is aborted")
            if session.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {session_id} is finalized")

            self._touch_session(session)
            self._materialize_active_trajectory(session)
            self._set_phase(session, SessionPhase.FINALIZED)
            session.completed.set()
            trajectories = [replace(trajectory, reward_info=dict(session.reward_info)) for trajectory in session.trajectories]
            self._sessions.pop(session_id, None)
            return trajectories

    async def abort_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return  # Already finalized or aborted — treat as idempotent.
        async with session.request_lock:
            if session.phase == SessionPhase.ABORTED:
                return  # Concurrent abort — idempotent.
            if session.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {session_id} is finalized")

            self._set_phase(session, SessionPhase.ABORTED)
            session.completed.set()
            self._sessions.pop(session_id, None)

    async def get_session_state(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        return {
            "session_id": session.handle.session_id,
            "metadata": dict(session.metadata),
            "phase": session.phase.value,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "num_trajectories": len(session.trajectories),
            "has_active_trajectory": session.active_trajectory is not None,
        }


GatewayActor = ray.remote(_GatewayActor)
