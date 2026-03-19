from __future__ import annotations

import logging
from typing import Any

import ray

from sglang.srt.utils import get_zmq_socket, kill_itself_when_parent_died

from verl.workers.rollout.decoupled_spec_rollout.draft_proxy import DraftProxy
from verl.workers.rollout.decoupled_spec_rollout.protocol import (
    DraftProxyIpcConfig,
    DraftProxyMessage,
    DraftProxyMessageType,
    DraftResult,
    DraftServerEndpoint,
    DraftStatus,
)

logger = logging.getLogger(__name__)


def _maybe_init_ray():
    if ray.is_initialized():
        return
    try:
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False, logging_level=logging.ERROR)
    except Exception:
        logger.exception("Failed to initialize Ray in DraftProxy subprocess")


def _resolve_actor_handle(endpoint: DraftServerEndpoint):
    if endpoint.actor_handle is not None:
        return endpoint.actor_handle
    if endpoint.actor_name is not None:
        return ray.get_actor(endpoint.actor_name)
    raise ValueError(f"Draft endpoint {endpoint.replica_rank} does not provide actor handle or actor name")


def _handle_proxy_message(proxy: DraftProxy, message: DraftProxyMessage) -> DraftProxyMessage | None:
    if message.message_type == DraftProxyMessageType.DRAFT_REQUEST and message.request is not None:
        route = proxy.submit_request(message.request)
        endpoint = proxy.endpoint_by_replica[route.draft_replica_rank]
        try:
            actor_handle = _resolve_actor_handle(endpoint)
            result = ray.get(actor_handle.handle_draft_request.remote(message.request))
            result.metadata.setdefault("draft_replica_rank", route.draft_replica_rank)
            result.metadata.setdefault("draft_server_address", route.draft_server_address)
        except Exception as exc:
            logger.exception("DraftProxy failed to get draft result")
            result = DraftResult(
                request_id=message.request.request_id,
                session_id=message.request.session_id,
                status=DraftStatus.FAILED,
                metadata={
                    "error": str(exc),
                    "draft_replica_rank": route.draft_replica_rank,
                    "draft_server_address": route.draft_server_address,
                },
            )
        proxy.complete_request(message.request.request_id, result)
        return DraftProxyMessage.from_draft_result(result)

    if message.message_type == DraftProxyMessageType.VERIFY_RESULT and message.verify_result is not None:
        proxy.notify_verify_result(message.verify_result)
        return None

    if message.message_type == DraftProxyMessageType.SHUTDOWN:
        return DraftProxyMessage.shutdown()

    if message.message_type == DraftProxyMessageType.ERROR:
        return message

    return DraftProxyMessage.error_message(f"Unsupported DraftProxy message type: {message.message_type}")


class DraftProxyManager:
    """Detokenizer-like manager process for DraftProxy."""

    def __init__(
        self,
        *,
        verify_replica_rank: int,
        num_speculative_steps: int,
        draft_endpoints: list[DraftServerEndpoint],
        ipc_config: DraftProxyIpcConfig,
    ):
        self.proxy = DraftProxy(
            verify_replica_rank=verify_replica_rank,
            num_speculative_steps=num_speculative_steps,
            draft_endpoints=draft_endpoints,
        )
        self.ipc_config = ipc_config
        self._resolved_endpoints = [
            DraftServerEndpoint(
                replica_rank=endpoint.replica_rank,
                server_address=endpoint.server_address,
                actor_handle=_resolve_actor_handle(endpoint),
                node_rank=endpoint.node_rank,
                actor_name=endpoint.actor_name,
            )
            for endpoint in draft_endpoints
        ]
        self.proxy.register_drafters(self._resolved_endpoints)
        self.init_ipc_channels()

    def init_ipc_channels(self):
        import zmq

        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context,
            zmq.PULL,
            self.ipc_config.scheduler_to_proxy_ipc_name,
            True,
        )
        self.send_to_scheduler = get_zmq_socket(
            context,
            zmq.PUSH,
            self.ipc_config.proxy_to_scheduler_ipc_name,
            True,
        )

    def event_loop(self):
        while True:
            message = self.recv_from_scheduler.recv_pyobj()
            response = _handle_proxy_message(self.proxy, message)
            if response is not None:
                self.send_to_scheduler.send_pyobj(response)
            if response is not None and response.message_type == DraftProxyMessageType.SHUTDOWN:
                break


def run_draftproxy_subprocess(
    *,
    verify_replica_rank: int,
    num_speculative_steps: int,
    draft_endpoints: list[dict[str, Any]],
    ipc_config: DraftProxyIpcConfig,
    ready_event,
):
    kill_itself_when_parent_died()
    _maybe_init_ray()
    manager = DraftProxyManager(
        verify_replica_rank=verify_replica_rank,
        num_speculative_steps=num_speculative_steps,
        draft_endpoints=[DraftServerEndpoint.from_metadata(item) for item in draft_endpoints],
        ipc_config=ipc_config,
    )
    ready_event.set()
    manager.event_loop()
