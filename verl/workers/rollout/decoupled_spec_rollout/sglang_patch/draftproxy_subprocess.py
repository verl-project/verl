from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
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
    DraftPollWaitMode,
    DraftProxyIpcConfig,
    DraftProxyMessage,
    DraftProxyMessageType,
    DraftRequest,
    DraftResult,
    PollDraftResultsRequest,
    PollDraftResultsResponse,
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


def _resolve_actor_handles(actor_names: list[str], namespace: str | None = None) -> list[Any]:
    return [ray.get_actor(actor_name, namespace=namespace) for actor_name in actor_names]


def _resolve_target_dp_rank(source_dp_rank: int, requested_dp_rank: int | None) -> int:
    return source_dp_rank if requested_dp_rank is None else int(requested_dp_rank)


@dataclass
class PendingPoll:
    request: PollDraftResultsRequest
    target_dp_rank: int
    deadline_monotonic: float | None


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
        init_start = time.perf_counter()
        self.proxy = DraftProxy(
            verify_replica_rank=verify_replica_rank,
            num_speculative_steps=num_speculative_steps,
            draft_actor_handles=draft_actor_handles,
        )
        self.ipc_config = ipc_config
        self.pending_polls: dict[str, PendingPoll] = {}
        self.init_ipc_channels()
        print(
            "[decoupled_spec][draftproxy] manager_init_done "
            f"verify_replica_rank={verify_replica_rank} num_speculative_steps={num_speculative_steps} "
            f"num_drafters={len(draft_actor_handles)} dp_endpoints={len(self.ipc_config.dp_ipc_endpoints)} "
            f"elapsed_s={time.perf_counter() - init_start:.6f}"
        )

    def init_ipc_channels(self):
        init_start = time.perf_counter()
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
        print(
            "[decoupled_spec][draftproxy] init_ipc_channels_done "
            f"verify_replica_rank={self.proxy.verify_replica_rank} "
            f"dp_ranks={sorted(self.ipc_config.dp_ipc_endpoints.keys())} "
            f"elapsed_s={time.perf_counter() - init_start:.6f}"
        )

    def close(self):
        for socket_map_name in ("recv_from_scheduler", "send_to_scheduler"):
            socket_map = getattr(self, socket_map_name, None) or {}
            for socket in socket_map.values():
                socket.close(linger=0)
        if getattr(self, "context", None) is not None:
            self.context.term()

    def _submit_draft_request(self, request: DraftRequest) -> None:
        submit_start = time.perf_counter()
        route = self.proxy.acquire_route(request.request_id)
        actor_handle = self.proxy.draft_actor_handles[route.draft_index]
        object_ref = actor_handle.handle_draft_request.remote(request) # 提交一个 draft 请求，不阻塞等待结果
        self.proxy.submit_request(request, object_ref)
        print(
            "[decoupled_spec][draftproxy] submit_to_drafter_done "
            f"verify_replica_rank={self.proxy.verify_replica_rank} request_id={request.request_id} "
            f"draft_round_id={request.draft_round_id} source_dp_rank={request.scheduler_dp_rank} "
            f"draft_index={route.draft_index} elapsed_s={time.perf_counter() - submit_start:.6f}"
        )

    def _collect_completed_drafts(self) -> None:
        # (request_id, draft_round_id) -> (draft_index, object_ref)
        inflight_items = list(self.proxy.inflight_requests.items())

        if not inflight_items:
            return

        object_refs = [inflight.object_ref for _, inflight in inflight_items]

        # 非阻塞 挑选出已完成的 DraftRequest
        ready_refs, _ = ray.wait(
            object_refs,
            num_returns=len(object_refs),
            timeout=0,
        )

        if not ready_refs:
            return

        ref_to_key = {inflight.object_ref: key for key, inflight in inflight_items} # 维护 ref_to_key 是为了出错的时候能输出信息

        for object_ref in ready_refs:
            key = ref_to_key.get(object_ref)
            if key is None:
                continue

            try:
                result_start = time.perf_counter()
                result = ray.get(object_ref)
                assert isinstance(result, DraftResult)
                assert result.request_id == key.request_id
                assert result.draft_round_id == key.draft_round_id
            except Exception:
                logger.exception("DraftProxy failed to get draft result")
                assert False
            self.proxy.complete_request(key, result)
            print(
                "[decoupled_spec][draftproxy] collect_completed_draft_done "
                f"verify_replica_rank={self.proxy.verify_replica_rank} request_id={key.request_id} "
                f"draft_round_id={key.draft_round_id} draft_tokens={len(result.draft_token_ids)} "
                f"elapsed_s={time.perf_counter() - result_start:.6f}"
            )

    def _send_poll_response(
        self,
        pending_poll: PendingPoll,
        *,
        timed_out: bool,
    ) -> None:
        send_start = time.perf_counter()
        ready_results, missing_keys = self.proxy.peek_ready_results(pending_poll.request.keys)
        if not ready_results and missing_keys and not timed_out:
            return

        keys_to_pop = [result.key for result in ready_results]
        popped_results = self.proxy.pop_ready_results(keys_to_pop)
        response = PollDraftResultsResponse(
            poll_id=pending_poll.request.poll_id,
            results=popped_results,
            missing_keys=missing_keys,
            timed_out=timed_out,
        )
        target_socket = self.send_to_scheduler.get(pending_poll.target_dp_rank)
        if target_socket is None:
            logger.error(
                "Missing DraftProxy response socket for dp_rank=%s",
                pending_poll.target_dp_rank,
            )
            return
        target_socket.send_pyobj(DraftProxyMessage.from_poll_response(response))
        print(
            "[decoupled_spec][draftproxy] send_poll_response_done "
            f"verify_replica_rank={self.proxy.verify_replica_rank} poll_id={pending_poll.request.poll_id} "
            f"target_dp_rank={pending_poll.target_dp_rank} results={len(popped_results)} "
            f"missing_keys={len(missing_keys)} timed_out={timed_out} "
            f"elapsed_s={time.perf_counter() - send_start:.6f}"
        )

    def _flush_pending_polls(self) -> None:
        now = time.monotonic()
        for poll_id, pending_poll in list(self.pending_polls.items()):
            _, missing_keys = self.proxy.peek_ready_results(pending_poll.request.keys)
            if not missing_keys:
                self._send_poll_response(pending_poll, timed_out=False)
                self.pending_polls.pop(poll_id, None)
                continue

            if (
                pending_poll.deadline_monotonic is not None
                and now >= pending_poll.deadline_monotonic
            ):
                self._send_poll_response(pending_poll, timed_out=True)
                self.pending_polls.pop(poll_id, None)

    def _handle_poll_request(
        self,
        poll_request: PollDraftResultsRequest,
        source_dp_rank: int,
    ) -> None:
        handle_start = time.perf_counter()
        if poll_request.wait_mode != DraftPollWaitMode.ALL:
            raise RuntimeError(f"Unsupported poll wait mode: {poll_request.wait_mode}")

        pending_poll = PendingPoll(
            request=poll_request,
            target_dp_rank=_resolve_target_dp_rank(source_dp_rank, poll_request.scheduler_dp_rank),
            deadline_monotonic=(
                None
                if poll_request.timeout_ms is None
                else time.monotonic() + max(0, poll_request.timeout_ms) / 1000.0
            ),
        )
        _, missing_keys = self.proxy.peek_ready_results(poll_request.keys)
        if not missing_keys:
            self._send_poll_response(pending_poll, timed_out=False)
            print(
                "[decoupled_spec][draftproxy] handle_poll_request_immediate "
                f"verify_replica_rank={self.proxy.verify_replica_rank} poll_id={poll_request.poll_id} "
                f"source_dp_rank={source_dp_rank} keys={len(poll_request.keys)} "
                f"elapsed_s={time.perf_counter() - handle_start:.6f}"
            )
            return

        self.pending_polls[poll_request.poll_id] = pending_poll
        print(
            "[decoupled_spec][draftproxy] handle_poll_request_pending "
            f"verify_replica_rank={self.proxy.verify_replica_rank} poll_id={poll_request.poll_id} "
            f"source_dp_rank={source_dp_rank} keys={len(poll_request.keys)} "
            f"missing_keys={len(missing_keys)} pending_polls={len(self.pending_polls)} "
            f"elapsed_s={time.perf_counter() - handle_start:.6f}"
        )

    def _handle_proxy_message(self, message: DraftProxyMessage, source_dp_rank: int) -> None:
        print(
            "[decoupled_spec][draftproxy] handle_proxy_message "
            f"verify_replica_rank={self.proxy.verify_replica_rank} source_dp_rank={source_dp_rank} "
            f"message_type={message.message_type}"
        )
        if message.message_type == DraftProxyMessageType.SUBMIT_DRAFT and message.request is not None:
            self._submit_draft_request(message.request) # verifier 向 DraftProxy 提交一个 DraftRequest->转发给 drafter
            return

        if (
            message.message_type == DraftProxyMessageType.POLL_DRAFT_RESULTS
            and message.poll_request is not None
        ):
            self._handle_poll_request(message.poll_request, source_dp_rank)
            return

        if message.message_type == DraftProxyMessageType.REQUEST_TERMINATE and message.terminate is not None:
            self.proxy.terminate_request(message.terminate)
            return

        raise RuntimeError(f"Unsupported DraftProxy message type: {message.message_type}")

    def event_loop(self):
        print(
            "[decoupled_spec][draftproxy] event_loop_start "
            f"verify_replica_rank={self.proxy.verify_replica_rank}"
        )
        try:
            while True:
                self._collect_completed_drafts()
                self._flush_pending_polls()

                events = dict(self.poller.poll(timeout=50))
                for dp_rank, recv_socket in self.recv_from_scheduler.items():
                    if events.get(recv_socket) != zmq.POLLIN:
                        continue
                    message = recv_socket.recv_pyobj()
                    self._handle_proxy_message(message, dp_rank)

                self._collect_completed_drafts()
                self._flush_pending_polls()
        finally:
            self.close()


def run_draftproxy_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    draft_actor_names: list[str] | None = None,
    draft_actor_namespace: str | None = None,
    draftproxy_manager_class=DraftProxyManager,
):
    _ = port_args
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::draftproxy")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()
    process_start = time.perf_counter()

    try:
        _maybe_init_ray()
        ipc_config = DraftProxyIpcConfig.from_env()
        if ipc_config is None:
            raise ValueError("DraftProxy IPC config is not configured")
        if not draft_actor_names:
            raise ValueError("DraftProxy draft_actor_names are not configured")
        draft_actor_handles = _resolve_actor_handles(draft_actor_names, namespace=draft_actor_namespace)
        manager = draftproxy_manager_class(
            verify_replica_rank=get_verify_replica_rank_from_env(),
            num_speculative_steps=get_num_speculative_steps_from_env(),
            draft_actor_handles=draft_actor_handles,
            ipc_config=ipc_config,
        )
        print(
            "[decoupled_spec][draftproxy] process_ready "
            f"verify_replica_rank={get_verify_replica_rank_from_env()} "
            f"num_drafters={len(draft_actor_handles)} namespace={draft_actor_namespace!r} "
            f"elapsed_s={time.perf_counter() - process_start:.6f}"
        )
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DraftProxyManager hit an exception: {traceback}")
        if parent_process is not None:
            parent_process.send_signal(signal.SIGQUIT)
