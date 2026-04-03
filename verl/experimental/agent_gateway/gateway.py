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

from verl.experimental.agent_framework.types import SessionHandle, Trajectory
from verl.experimental.agent_gateway.types import GatewaySessionState, SessionPhase, TrajectoryBuffer
from verl.workers.rollout.utils import run_uvicorn


class MalformedRequestError(ValueError):
    pass


class UnsupportedRequestError(ValueError):
    pass


def _canonicalize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize_json_like(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_json_like(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise MalformedRequestError(f"Unsupported JSON-like value: {type(value).__name__}")


def _normalize_message_content(content: Any) -> Any:
    if isinstance(content, list):
        return [_canonicalize_json_like(part) for part in content]
    if isinstance(content, dict):
        return _canonicalize_json_like(content)
    if content is None or isinstance(content, str):
        return "" if content is None else content
    raise MalformedRequestError(f"Unsupported content type: {type(content).__name__}")


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        raise MalformedRequestError("tool_calls must be a list")

    normalized_tool_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise MalformedRequestError("tool_calls entries must be objects")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise MalformedRequestError("tool_call.function must be an object")
        arguments = function.get("arguments", "")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise MalformedRequestError("tool_call.function.arguments must decode to a JSON object") from exc
        if not isinstance(arguments, dict):
            raise MalformedRequestError("tool_call.function.arguments must decode to a JSON object")
        normalized_tool_calls.append(
            {
                "id": str(tool_call.get("id", "")),
                "type": str(tool_call.get("type", "")),
                "function": {
                    "name": str(function.get("name", "")),
                    "arguments": _canonicalize_json_like(arguments),
                },
            }
        )
    return normalized_tool_calls


def _normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise MalformedRequestError("messages entries must be objects")
    if "name" in message:
        raise UnsupportedRequestError("message.name is not supported in PR1")

    normalized_message: dict[str, Any] = {
        "role": str(message.get("role", "")),
        "content": _normalize_message_content(message.get("content", "")),
    }
    if "tool_calls" in message:
        normalized_message["tool_calls"] = _normalize_tool_calls(message["tool_calls"])
    if "tool_call_id" in message:
        normalized_message["tool_call_id"] = str(message["tool_call_id"])
    return normalized_message


def _normalize_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")
    return [_canonicalize_json_like(tool) for tool in tools]


def _normalize_request_context(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise MalformedRequestError("messages must be non-empty")
    return {
        "messages": [_normalize_message(message) for message in messages],
        "tools": _normalize_tools(payload.get("tools")),
    }


def _is_message_prefix(prefix: list[dict[str, Any]], messages: list[dict[str, Any]]) -> bool:
    if len(prefix) > len(messages):
        return False
    return prefix == messages[: len(prefix)]


def _is_request_context_prefix(
    *,
    session: GatewaySessionState,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> bool:
    if session.request_tools != tools:
        return False
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


class _GatewayActor:
    def __init__(self, tokenizer, backend, host: str = "127.0.0.1"):
        self._tokenizer = tokenizer
        self._backend = backend
        self._host = host
        self._sessions: dict[str, GatewaySessionState] = {}
        self._app = FastAPI()
        self._server_port: int | None = None
        self._server_task: asyncio.Task | None = None
        self._server_base_url: str | None = None
        self._terminal_session_phases: dict[str, SessionPhase] = {}
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
            await self.complete_session(session_id=session_id, reward_info=reward_info)
            return JSONResponse({"status": "ok"})

    def _require_started(self) -> None:
        if self._server_base_url is None:
            raise RuntimeError("GatewayActor.start() must be called before session creation")

    def _get_session(self, session_id: str) -> GatewaySessionState:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session

    def _get_terminal_phase(self, session_id: str) -> SessionPhase | None:
        return self._terminal_session_phases.get(session_id)

    def _raise_if_terminal(self, session_id: str) -> None:
        terminal_phase = self._get_terminal_phase(session_id)
        if terminal_phase is not None:
            raise RuntimeError(f"Session {session_id} is {terminal_phase.value.lower()}")

    def _set_phase(self, session: GatewaySessionState, phase: SessionPhase) -> None:
        session.phase = phase
        session.completed_flag = phase == SessionPhase.COMPLETED
        session.aborted_flag = phase == SessionPhase.ABORTED
        self._touch_session(session)

    def _mark_terminal(self, session_id: str, phase: SessionPhase) -> None:
        self._terminal_session_phases[session_id] = phase

    def _touch_session(self, session: GatewaySessionState) -> None:
        session.updated_at = time.time()

    def _materialize_active_trajectory(self, session: GatewaySessionState) -> None:
        active = session.active_trajectory
        if active is None:
            return

        self._touch_session(session)
        session.trajectories.append(
            Trajectory(
                uid=session.handle.session_id,
                session_id=session.handle.session_id,
                trajectory_id=session.next_trajectory_id,
                prompt_ids=list(active.prompt_ids),
                response_ids=list(active.response_ids),
                response_mask=list(active.response_mask),
                response_logprobs=list(active.response_logprobs),
                reward_info={},
                num_turns=len(session.message_history),
            )
        )
        session.next_trajectory_id += 1
        session.active_trajectory = None

    def _build_materialized_trajectory(
        self,
        *,
        session: GatewaySessionState,
        active: TrajectoryBuffer,
        trajectory_id: int,
    ) -> Trajectory:
        return Trajectory(
            uid=session.handle.session_id,
            session_id=session.handle.session_id,
            trajectory_id=trajectory_id,
            prompt_ids=list(active.prompt_ids),
            response_ids=list(active.response_ids),
            response_mask=list(active.response_mask),
            response_logprobs=list(active.response_logprobs),
            reward_info={},
            num_turns=len(session.message_history),
        )

    def _start_new_trajectory(self, session: GatewaySessionState, messages: list[dict[str, Any]]) -> None:
        prompt_ids = list(self._tokenizer.encode_messages(messages, add_generation_prompt=True))
        session.active_trajectory = TrajectoryBuffer(prompt_ids=prompt_ids)

    async def _handle_chat_completions(self, session_id: str, payload: dict[str, Any]) -> JSONResponse:
        try:
            session = self._get_session(session_id)
        except KeyError as exc:
            terminal_phase = self._get_terminal_phase(session_id)
            if terminal_phase is not None:
                raise HTTPException(status_code=409, detail=f"Session {session_id} is {terminal_phase.value.lower()}") from exc
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async with session.request_lock:
            if session.phase != SessionPhase.ACTIVE:
                raise HTTPException(status_code=409, detail=f"Session {session_id} is {session.phase.value.lower()}")

            self._touch_session(session)
            try:
                request_context = _normalize_request_context(payload)
            except MalformedRequestError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except UnsupportedRequestError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            messages = request_context["messages"]
            tools = request_context["tools"]
            tentative_trajectories = list(session.trajectories)
            tentative_next_trajectory_id = session.next_trajectory_id

            if session.active_trajectory is None:
                tentative_active = TrajectoryBuffer(
                    prompt_ids=list(self._tokenizer.encode_messages(messages, add_generation_prompt=True))
                )
            elif _is_request_context_prefix(session=session, messages=messages, tools=tools):
                tentative_active = _copy_trajectory_buffer(session.active_trajectory)
                incremental_messages = messages[len(session.message_history) :]
                if incremental_messages:
                    incremental_ids = list(
                        self._tokenizer.encode_messages(incremental_messages, add_generation_prompt=True)
                    )
                    tentative_active.response_ids.extend(incremental_ids)
                    tentative_active.response_mask.extend([0] * len(incremental_ids))
                    tentative_active.response_logprobs.extend([0.0] * len(incremental_ids))
            else:
                tentative_trajectories.append(
                    self._build_materialized_trajectory(
                        session=session,
                        active=session.active_trajectory,
                        trajectory_id=tentative_next_trajectory_id,
                    )
                )
                tentative_next_trajectory_id += 1
                tentative_active = TrajectoryBuffer(
                    prompt_ids=list(self._tokenizer.encode_messages(messages, add_generation_prompt=True))
                )

            prompt_ids = tentative_active.prompt_ids + tentative_active.response_ids
            sampling_params = dict(payload)
            sampling_params.pop("messages", None)
            sampling_params.pop("model", None)
            sampling_params.pop("tools", None)
            output = await self._backend.generate(
                request_id=session_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

            response_ids = list(output.token_ids)
            tentative_active.response_ids.extend(response_ids)
            tentative_active.response_mask.extend([1] * len(response_ids))
            if output.log_probs is not None:
                tentative_active.response_logprobs.extend(list(output.log_probs))
            else:
                tentative_active.response_logprobs.extend([0.0] * len(response_ids))

            response_text = self._tokenizer.decode(response_ids)
            session.trajectories = tentative_trajectories
            session.next_trajectory_id = tentative_next_trajectory_id
            session.active_trajectory = tentative_active
            session.message_history = messages + [{"role": "assistant", "content": response_text}]
            session.request_tools = tools
            self._touch_session(session)

            return JSONResponse(
                {
                    "id": f"chatcmpl-{uuid4().hex}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": output.stop_reason or "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(response_ids),
                        "total_tokens": len(prompt_ids) + len(response_ids),
                    },
                }
            )

    async def start(self) -> None:
        if self._server_task is not None:
            return
        self._server_port, self._server_task = await run_uvicorn(self._app, None, self._host)
        self._server_base_url = f"http://{self._host}:{self._server_port}"

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
        if session_id in self._sessions or session_id in self._terminal_session_phases:
            raise RuntimeError(f"Session {session_id} already exists")

        handle = SessionHandle(
            session_id=session_id,
            base_url=f"{self._server_base_url}/sessions/{session_id}/v1",
        )
        self._sessions[session_id] = GatewaySessionState(handle=handle, metadata=dict(metadata or {}))
        return handle

    async def complete_session(self, session_id: str, reward_info: dict[str, Any] | None = None) -> None:
        self._raise_if_terminal(session_id)
        session = self._get_session(session_id)
        async with session.request_lock:
            if session.phase == SessionPhase.COMPLETED:
                if reward_info is not None:
                    session.reward_info = dict(reward_info)
                self._touch_session(session)
                session.completed.set()
                return
            if session.phase != SessionPhase.ACTIVE:
                raise RuntimeError(f"Session {session_id} is {session.phase.value.lower()}")

            if reward_info is not None:
                session.reward_info = dict(reward_info)
            self._set_phase(session, SessionPhase.COMPLETED)
            session.completed.set()

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        terminal_phase = self._get_terminal_phase(session_id)
        if terminal_phase == SessionPhase.FINALIZED:
            return
        if terminal_phase == SessionPhase.ABORTED:
            raise RuntimeError(f"Session {session_id} is aborted")

        session = self._get_session(session_id)
        if session.phase in {SessionPhase.COMPLETED, SessionPhase.FINALIZED}:
            return
        if session.phase == SessionPhase.ABORTED:
            raise RuntimeError(f"Session {session_id} is aborted")

        if timeout is None:
            await session.completed.wait()
        else:
            await asyncio.wait_for(session.completed.wait(), timeout=timeout)

        terminal_phase = self._get_terminal_phase(session_id)
        if terminal_phase == SessionPhase.FINALIZED:
            return
        if terminal_phase == SessionPhase.ABORTED:
            raise RuntimeError(f"Session {session_id} is aborted")

        if session.phase == SessionPhase.COMPLETED:
            return
        if session.phase == SessionPhase.ABORTED:
            raise RuntimeError(f"Session {session_id} is aborted")

    async def finalize_session(self, session_id: str) -> list[Trajectory]:
        self._raise_if_terminal(session_id)
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
            self._mark_terminal(session_id, SessionPhase.FINALIZED)
            return trajectories

    async def abort_session(self, session_id: str) -> None:
        terminal_phase = self._get_terminal_phase(session_id)
        if terminal_phase == SessionPhase.ABORTED:
            return
        if terminal_phase == SessionPhase.FINALIZED:
            raise RuntimeError(f"Session {session_id} is finalized")

        session = self._get_session(session_id)
        async with session.request_lock:
            if session.phase == SessionPhase.ABORTED:
                return
            if session.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {session_id} is finalized")

            self._set_phase(session, SessionPhase.ABORTED)
            session.completed.set()
            self._sessions.pop(session_id, None)
            self._mark_terminal(session_id, SessionPhase.ABORTED)

    async def get_session_state(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        return {
            "session_id": session.handle.session_id,
            "metadata": dict(session.metadata),
            "phase": session.phase.value,
            "completed_flag": session.completed_flag,
            "aborted_flag": session.aborted_flag,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "num_trajectories": len(session.trajectories),
            "has_active_trajectory": session.active_trajectory is not None,
        }


GatewayActor = ray.remote(_GatewayActor)
