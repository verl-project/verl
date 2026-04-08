import asyncio

import ray

from verl.experimental.agent_framework.types import SessionHandle, Trajectory
from verl.workers.rollout.replica import TokenOutput


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
        parts = []
        for message in messages:
            parts.append(f"{message['role']}:{self._normalize_content(message.get('content', ''))}\n")
        if add_generation_prompt:
            parts.append("assistant:")
        text = "".join(parts)
        if tokenize:
            return [ord(char) for char in text]
        return text

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def _normalize_content(self, content):
        if isinstance(content, list):
            return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if content is None:
            return ""
        return str(content)


class QueuedBackend:
    def __init__(self, responses):
        self._responses = list(responses)

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        text = self._responses.pop(0)
        token_ids = [ord(char) for char in text]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
        )


class NoLogprobBackend:
    def __init__(self, response_text: str = "OK"):
        self.response_text = response_text

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        token_ids = [ord(char) for char in self.response_text]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=None,
            stop_reason="completed",
        )


class RejectToolsSamplingParamsBackend:
    def __init__(self, response_text: str = "OK"):
        self.response_text = response_text

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        if "tools" in sampling_params:
            raise RuntimeError("tools leaked into sampling_params")
        token_ids = [ord(char) for char in self.response_text]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
        )


class FailingBackend:
    def __init__(self, error_message: str = "backend failure"):
        self.error_message = error_message
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
            }
        )
        raise RuntimeError(self.error_message)


class SequencedBackend:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
            }
        )
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        token_ids = [ord(char) for char in step]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
        )


@ray.remote
class RecordingLoadBalancer:
    def __init__(self, server_id: str = "server-0"):
        self.server_id = server_id
        self.acquire_calls = []
        self.release_calls = []

    def acquire_server(self, request_id: str) -> str:
        self.acquire_calls.append(request_id)
        return self.server_id

    def release_server(self, server_id: str) -> None:
        self.release_calls.append(server_id)

    def stats(self):
        return {
            "acquire_calls": list(self.acquire_calls),
            "release_calls": list(self.release_calls),
        }


@ray.remote
class RecordingRolloutServer:
    def __init__(self, response_text: str = "OWNER"):
        self.response_text = response_text
        self.calls = []

    async def generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
    ):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
            }
        )
        token_ids = [ord(char) for char in self.response_text]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
        )

    def get_calls(self):
        return list(self.calls)


class RejectConcurrentSessionBackend:
    def __init__(self, responses, delay: float = 0.05):
        self._responses = list(responses)
        self._delay = delay
        self._active_request_ids: set[str] = set()
        self.call_windows = []

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        if request_id in self._active_request_ids:
            raise RuntimeError(f"concurrent request for session {request_id}")

        self._active_request_ids.add(request_id)
        started_at = asyncio.get_running_loop().time()
        try:
            await asyncio.sleep(self._delay)
            text = self._responses.pop(0)
            token_ids = [ord(char) for char in text]
            return TokenOutput(
                token_ids=token_ids,
                log_probs=[-0.1] * len(token_ids),
                stop_reason="completed",
            )
        finally:
            finished_at = asyncio.get_running_loop().time()
            self.call_windows.append((request_id, started_at, finished_at))
            self._active_request_ids.remove(request_id)


@ray.remote
class TrackingGatewayActor:
    def __init__(self, name: str):
        self.name = name
        self.sessions = {}
        self.created = []
        self.finalized = []
        self.aborted = []

    async def start(self):
        return None

    async def shutdown(self):
        return None

    async def create_session(self, session_id: str, metadata: dict | None = None):
        handle = SessionHandle(session_id=session_id, base_url=f"http://{self.name}/{session_id}/v1")
        self.sessions[session_id] = {"metadata": metadata or {}}
        self.created.append(session_id)
        return handle

    async def finalize_session(self, session_id: str):
        self.finalized.append(session_id)
        self.sessions.pop(session_id, None)
        return [
            Trajectory(
                uid=session_id,
                session_id=session_id,
                trajectory_id=0,
                prompt_ids=[1],
                response_ids=[2],
                response_mask=[1],
            )
        ]

    async def abort_session(self, session_id: str):
        self.aborted.append(session_id)
        self.sessions.pop(session_id, None)
        return None

    async def wait_for_completion(self, session_id: str, timeout: float | None = None):
        return None

    async def stats(self):
        return {
            "name": self.name,
            "created": list(self.created),
            "finalized": list(self.finalized),
            "aborted": list(self.aborted),
        }
