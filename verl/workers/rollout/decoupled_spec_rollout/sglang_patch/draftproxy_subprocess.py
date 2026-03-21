from __future__ import annotations

import logging
import signal
from typing import Any

import psutil
import ray
import setproctitle
import zmq

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, get_zmq_socket, kill_itself_when_parent_died
from sglang.utils import get_exception_traceback

from verl.workers.rollout.decoupled_spec_rollout.draft_proxy import DraftProxy
from verl.workers.rollout.decoupled_spec_rollout.protocol import (
    DraftProxyIpcConfig,
    DraftProxyMessage,
    DraftProxyMessageType,
    DraftResult,
    get_num_speculative_steps_from_env,
    get_verify_replica_rank_from_env,
)

logger = logging.getLogger(__name__)


def _maybe_init_ray():
    if ray.is_initialized():
        return
    try:
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False, logging_level=logging.ERROR)
    except Exception:
        logger.exception("Failed to initialize Ray in DraftProxy subprocess")


def _resolve_target_dp_rank(source_dp_rank: int, requested_dp_rank: int | None) -> int:
    return source_dp_rank if requested_dp_rank is None else int(requested_dp_rank)


def _handle_proxy_message(
    proxy: DraftProxy, message: DraftProxyMessage, source_dp_rank: int
) -> tuple[DraftProxyMessage | None, int | None]:
    if message.message_type == DraftProxyMessageType.DRAFT_REQUEST and message.request is not None:
        route = proxy.submit_request(message.request)
        try:
            actor_handle = proxy.draft_actor_handles[route.draft_index]
            result = ray.get(actor_handle.handle_draft_request.remote(message.request))
        except Exception as exc:
            logger.exception("DraftProxy failed to get draft result")
            result = DraftResult(
                request_id=message.request.request_id,
                draft_token_ids=[],
            )
        proxy.complete_request(message.request.request_id, result)
        return (
            DraftProxyMessage.from_draft_result(result),
            _resolve_target_dp_rank(source_dp_rank, message.request.scheduler_dp_rank),
        )

    if message.message_type == DraftProxyMessageType.VERIFY_RESULT and message.request is not None:
        proxy.notify_verify_request(message.request)
        return None, None

    if message.message_type == DraftProxyMessageType.REQUEST_TERMINATE and message.terminate is not None:
        proxy.terminate_request(message.terminate)
        return None, None

    if message.message_type == DraftProxyMessageType.SHUTDOWN:
        return DraftProxyMessage.shutdown(), source_dp_rank

    if message.message_type == DraftProxyMessageType.ERROR:
        return message, source_dp_rank

    return (
        DraftProxyMessage.error_message(f"Unsupported DraftProxy message type: {message.message_type}"),
        source_dp_rank,
    )


class DraftProxyManager:
    """Detokenizer-like manager process for DraftProxy."""

    def __init__(
        self,
        *,
        verify_replica_rank: int,
        num_speculative_steps: int,
        draft_actor_handles: list[Any],
        ipc_config: DraftProxyIpcConfig,
    ):
        self.proxy = DraftProxy(
            verify_replica_rank=verify_replica_rank,
            num_speculative_steps=num_speculative_steps,
            draft_actor_handles=draft_actor_handles,
        )
        self.ipc_config = ipc_config
        self.init_ipc_channels()

    def init_ipc_channels(self):
        self.context = zmq.Context(1 + 2 * len(self.ipc_config.dp_ipc_endpoints))
        self.recv_from_scheduler = {}
        self.send_to_scheduler = {}
        self.poller = zmq.Poller()

        for dp_rank, endpoints in sorted(self.ipc_config.dp_ipc_endpoints.items()):
            recv_socket = get_zmq_socket(
                self.context,
                zmq.PULL,
                endpoints.scheduler_to_proxy_ipc_name,
                True,
            )
            send_socket = get_zmq_socket(
                self.context,
                zmq.PUSH,
                endpoints.proxy_to_scheduler_ipc_name,
                True,
            )
            self.recv_from_scheduler[dp_rank] = recv_socket
            self.send_to_scheduler[dp_rank] = send_socket
            self.poller.register(recv_socket, zmq.POLLIN)

    def close(self):
        for socket_map_name in ("recv_from_scheduler", "send_to_scheduler"):
            socket_map = getattr(self, socket_map_name, None) or {}
            for socket in socket_map.values():
                socket.close(linger=0)
        if getattr(self, "context", None) is not None:
            self.context.term()

    def event_loop(self):
        try:
            while True:
                events = dict(self.poller.poll())
                for dp_rank, recv_socket in self.recv_from_scheduler.items():
                    if events.get(recv_socket) != zmq.POLLIN:
                        continue
                    message = recv_socket.recv_pyobj()
                    response, target_dp_rank = _handle_proxy_message(self.proxy, message, dp_rank)
                    if response is not None and target_dp_rank is not None:
                        target_socket = self.send_to_scheduler.get(target_dp_rank)
                        if target_socket is None:
                            logger.error("Missing DraftProxy response socket for dp_rank=%s", target_dp_rank)
                        else:
                            target_socket.send_pyobj(response)
                    if response is not None and response.message_type == DraftProxyMessageType.SHUTDOWN:
                        return
        finally:
            self.close()


def run_draftproxy_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    draft_actor_handles: list[Any] | None = None,
    draftproxy_manager_class=DraftProxyManager,
):
    _ = port_args
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::draftproxy")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        _maybe_init_ray()
        ipc_config = DraftProxyIpcConfig.from_env()
        if ipc_config is None:
            raise ValueError("DraftProxy IPC config is not configured")
        if not draft_actor_handles:
            raise ValueError("DraftProxy draft_actor_handles are not configured")
        manager = draftproxy_manager_class(
            verify_replica_rank=get_verify_replica_rank_from_env(),
            num_speculative_steps=get_num_speculative_steps_from_env(),
            draft_actor_handles=draft_actor_handles,
            ipc_config=ipc_config,
        )
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DraftProxyManager hit an exception: {traceback}")
        if parent_process is not None:
            parent_process.send_signal(signal.SIGQUIT)
