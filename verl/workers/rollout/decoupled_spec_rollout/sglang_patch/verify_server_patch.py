from __future__ import annotations

from collections import deque
import logging
import multiprocessing as mp
import os
import time

import zmq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.utils import broadcast_pyobj, get_zmq_socket
from sglang.srt.managers.io_struct import BatchTokenizedGenerateReqInput, TokenizedGenerateReqInput

from verl.workers.rollout.decoupled_spec_rollout.protocol import (
    DraftPollWaitMode,
    DraftProxyIpcConfig,
    DraftLookupKey,
    DraftProxyMessage,
    DraftProxyMessageType,
    DraftRequest,
    DraftResult,
    PollDraftResultsRequest,
    PollDraftResultsResponse,
    RequestTerminateMessage,
    RequestTerminateReason,
    get_num_speculative_steps_from_env,
    get_verify_replica_rank_from_env,
)
from verl.workers.rollout.decoupled_spec_rollout.sglang_patch.decoupled_spec_verify_patch import (
    patch_speculative_worker_factory,
)

# Set by SGLangHttpServer.launch_server immediately before verify launch_subprocesses runs.
_pending_draft_actor_names: list[str] | None = None
_pending_draft_actor_namespace: str | None = None


def set_pending_draft_actor_names(names: list[str] | None) -> None:
    """Pass drafter Ray actor names from the verify server process into launch_subprocesses."""
    global _pending_draft_actor_names
    _pending_draft_actor_names = list(names) if names else None
    


def set_pending_draft_actor_namespace(namespace: str | None) -> None:
    global _pending_draft_actor_namespace
    _pending_draft_actor_namespace = namespace
    

logger = logging.getLogger(__name__)


def init_tokenizer_manager(*args, **kwargs):
    from sglang.srt.entrypoints.engine import init_tokenizer_manager as upstream

    return upstream(*args, **kwargs)


def _build_sampling_params_dict(sampling_params) -> dict:
    params = {
        "max_new_tokens": sampling_params.max_new_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "frequency_penalty": sampling_params.frequency_penalty,
        "presence_penalty": sampling_params.presence_penalty,
        "repetition_penalty": sampling_params.repetition_penalty,
        "min_new_tokens": sampling_params.min_new_tokens,
        "n": sampling_params.n,
        "ignore_eos": sampling_params.ignore_eos,
        "skip_special_tokens": sampling_params.skip_special_tokens,
        "spaces_between_special_tokens": sampling_params.spaces_between_special_tokens,
        "no_stop_trim": sampling_params.no_stop_trim,
        "custom_params": sampling_params.custom_params,
        "stream_interval": sampling_params.stream_interval,
        "logit_bias": sampling_params.logit_bias,
        "sampling_seed": sampling_params.sampling_seed,
    }
    if sampling_params.stop_strs is not None:
        params["stop"] = sampling_params.stop_strs
    if sampling_params.stop_token_ids is not None:
        params["stop_token_ids"] = list(sampling_params.stop_token_ids)
    if sampling_params.stop_regex_strs is not None:
        params["stop_regex"] = sampling_params.stop_regex_strs
    if sampling_params.json_schema is not None:
        params["json_schema"] = sampling_params.json_schema
    if sampling_params.regex is not None:
        params["regex"] = sampling_params.regex
    if sampling_params.ebnf is not None:
        params["ebnf"] = sampling_params.ebnf
    if sampling_params.structural_tag is not None:
        params["structural_tag"] = sampling_params.structural_tag
    return params


def _is_draftproxy_enabled(self: Scheduler) -> bool:
    return getattr(self, "_send_to_draftproxy", None) is not None and getattr(self, "_recv_from_draftproxy", None) is not None


def _is_verify_scheduler_entry_rank(self: Scheduler) -> bool:
    return self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0


def _is_verify_scheduler_external_draft_leader(self: Scheduler) -> bool:
    return _is_verify_scheduler_entry_rank(self) and _is_draftproxy_enabled(self)


def _require_draftproxy_on_entry_rank(self: Scheduler) -> None:
    if _is_verify_scheduler_entry_rank(self) and getattr(self, "_decoupled_draftproxy_expected", False):
        assert _is_draftproxy_enabled(self), (
            "DraftProxy for VerifySGLangHttpServer is expected on entry scheduler "
            f"dp_rank={_get_scheduler_dp_rank(self)}"
        )


def _iter_live_batch_reqs(batch) -> list:
    return [req for req in batch.reqs if not req.is_retracted and not req.finished()]


def _iter_live_reqs(reqs) -> list:
    return [req for req in reqs if not req.is_retracted and not req.finished()]


def _get_scheduler_dp_rank(self: Scheduler) -> int:
    dp_rank = getattr(self, "dp_rank", None)
    return 0 if dp_rank is None else int(dp_rank)


def _next_draft_round_id(self: Scheduler, request_id: str) -> int:
    next_round_id = int(self._decoupled_next_draft_round_by_rid.get(request_id, 0))
    self._decoupled_next_draft_round_by_rid[request_id] = next_round_id + 1
    return next_round_id


def _make_poll_id(self: Scheduler) -> str:
    self._decoupled_poll_seq += 1
    return f"dp{_get_scheduler_dp_rank(self)}-poll-{self._decoupled_poll_seq}"


# 请求完成prefill之后需要发送一次DraftRequest，这个DraftRequest的结果应该在第二轮（而非第一轮）decode中被使用，这种情况称为"warmup"
def _get_prefill_warmup_request_ids(batch) -> set[str]:
    decoding_rids = {req.rid for req in (batch.decoding_reqs or [])}
    warmup_request_ids = set()
    for req in _iter_live_batch_reqs(batch):
        if req.rid in decoding_rids or req.is_chunked > 0:
            continue
        warmup_request_ids.add(req.rid)
    return warmup_request_ids


def _get_decode_reqs_for_poll(self: Scheduler, batch) -> list:
    return [
        req
        for req in _iter_live_batch_reqs(batch)
        if req.rid not in self._decoupled_needs_warmup_decode_rids
    ]



def _get_waiting_draft_queue(
    self: Scheduler,
    request_id: str,
    *,
    create: bool = False,
) -> deque[DraftLookupKey] | None:
    waiting_keys = self._decoupled_waiting_draft_keys.get(request_id)
    if waiting_keys is None and create:
        waiting_keys = deque()
        self._decoupled_waiting_draft_keys[request_id] = waiting_keys
    return waiting_keys


def _peek_waiting_draft_key(self: Scheduler, request_id: str) -> DraftLookupKey | None:
    waiting_keys = _get_waiting_draft_queue(self, request_id)
    if not waiting_keys:
        return None
    return waiting_keys[0]


def _append_waiting_draft_key(
    self: Scheduler,
    request_id: str,
    waiting_key: DraftLookupKey,
) -> None:
    waiting_keys = _get_waiting_draft_queue(self, request_id, create=True)
    waiting_keys.append(waiting_key)


def _pop_waiting_draft_key(self: Scheduler, request_id: str) -> DraftLookupKey | None:
    waiting_keys = _get_waiting_draft_queue(self, request_id)
    
    assert waiting_keys is not None

    waiting_key = waiting_keys.popleft()
    if not waiting_keys:
        self._decoupled_waiting_draft_keys.pop(request_id, None)
    return waiting_key


def _clear_local_request_draft_state(self: Scheduler, request_id: str) -> None:
    self._decoupled_waiting_draft_keys.pop(request_id, None)
    self._decoupled_next_draft_round_by_rid.pop(request_id, None)
    self._decoupled_needs_warmup_decode_rids.discard(request_id)
    for key in list(self._decoupled_pending_draft_results):
        if key.request_id == request_id:
            self._decoupled_pending_draft_results.pop(key, None)


def _build_draft_request_from_req(
    self: Scheduler,
    req,
    draft_round_id: int,
) -> DraftRequest:
    return DraftRequest(
        request_id=req.rid,
        verify_replica_rank=get_verify_replica_rank_from_env(),
        draft_round_id=draft_round_id,
        scheduler_dp_rank=_get_scheduler_dp_rank(self),
        prompt_token_ids=list(req.origin_input_ids),
        committed_token_ids=list(req.output_ids),
        num_speculative_steps=get_num_speculative_steps_from_env(),
        sampling_params=_build_sampling_params_dict(req.sampling_params),
    )


def _submit_draft_request(
    self: Scheduler,
    req,
    *,
    needs_warmup_decode: bool = False,
) -> None:
    submit_start = time.perf_counter()
    draft_round_id = _next_draft_round_id(self, req.rid)
    draft_request = _build_draft_request_from_req(self, req, draft_round_id=draft_round_id)
    
    if _is_verify_scheduler_external_draft_leader(self):
        self._send_to_draftproxy.send_pyobj(DraftProxyMessage.from_submit_draft(draft_request))
    _append_waiting_draft_key(self, req.rid, draft_request.key)
    if needs_warmup_decode:
        self._decoupled_needs_warmup_decode_rids.add(req.rid)
    setattr(req, "decoupled_spec_is_warmup_decode", bool(needs_warmup_decode))
    setattr(req, "decoupled_spec_draft_result", None)
    waiting_queue = _get_waiting_draft_queue(self, req.rid)
    


def _send_draft_requests(
    self: Scheduler,
    batch,
    target_reqs=None,
    warmup_request_ids: set[str] | None = None,
) -> None:
    _require_draftproxy_on_entry_rank(self)

    reqs = _iter_live_reqs(target_reqs) if target_reqs is not None else _iter_live_batch_reqs(batch)
    for req in reqs:
        if req.is_chunked > 0:
            continue
        _submit_draft_request(
            self,
            req,
            needs_warmup_decode=bool(warmup_request_ids and req.rid in warmup_request_ids),
        )


def _recv_poll_response(
    self: Scheduler,
    poll_id: str,
) -> PollDraftResultsResponse:
    recv_start = time.perf_counter()
    cached_response = self._decoupled_pending_poll_responses.pop(poll_id, None)
    if cached_response is not None:
        
        return cached_response

    while True:
        if not self._recv_from_draftproxy.poll():
            continue

        response = self._recv_from_draftproxy.recv_pyobj()
        if (
            response.message_type != DraftProxyMessageType.POLL_RESPONSE
            or response.poll_response is None
        ):
            raise RuntimeError(f"Unexpected DraftProxy response: {response.message_type}")

        poll_response = response.poll_response
        if poll_response.poll_id == poll_id:
            return poll_response
        self._decoupled_pending_poll_responses[poll_response.poll_id] = poll_response
        


def _sync_draft_results_across_schedulers(
    self: Scheduler,
    leader_results: list[DraftResult] | None,
) -> list[DraftResult]:
    # Mirror SGLang's control-plane fanout: only the entry scheduler receives
    # external messages, then synchronizes small Python objects to peer TP/CP
    # schedulers via torch.distributed on the CPU group.
    source_payload = list(leader_results or []) if _is_verify_scheduler_entry_rank(self) else []

    if self.tp_size == 1:
        return source_payload

    synced_results = broadcast_pyobj(
        source_payload,
        self.tp_group.rank,
        self.tp_cpu_group,
        src=self.tp_group.ranks[0],
    )
    return list(synced_results)


def _bind_draft_results_to_reqs(self: Scheduler, live_reqs: list) -> None:
    for req in live_reqs:
        waiting_key = _peek_waiting_draft_key(self, req.rid)
        if waiting_key is None:
            setattr(req, "decoupled_spec_draft_result", None)
            continue

        draft_result = self._decoupled_pending_draft_results.pop(waiting_key, None)
        if draft_result is None:
            raise RuntimeError(f"Draft result missing for request {req.rid} round {waiting_key.draft_round_id}")

        _pop_waiting_draft_key(self, req.rid)
        setattr(req, "decoupled_spec_draft_result", draft_result)
        


def _wait_for_draft_results(self: Scheduler, batch, target_reqs=None) -> None:
    _require_draftproxy_on_entry_rank(self)

    wait_start = time.perf_counter()
    live_reqs = _iter_live_reqs(target_reqs) if target_reqs is not None else _iter_live_batch_reqs(batch)
    if not live_reqs:
        return

    missing_keys = []
    seen_missing_keys = set()
    for req in live_reqs:
        waiting_key = _peek_waiting_draft_key(self, req.rid)
        if waiting_key is None:
            continue
        if (
            waiting_key not in self._decoupled_pending_draft_results
            and waiting_key not in seen_missing_keys
        ):
            missing_keys.append(waiting_key)
            seen_missing_keys.add(waiting_key)

    leader_results: list[DraftResult] | None = None
    if _is_verify_scheduler_external_draft_leader(self) and missing_keys:
        poll_request = PollDraftResultsRequest(
            poll_id=_make_poll_id(self),
            scheduler_dp_rank=_get_scheduler_dp_rank(self),
            keys=missing_keys,
            timeout_ms=None,
            wait_mode=DraftPollWaitMode.ALL,
        )
        
        self._send_to_draftproxy.send_pyobj(DraftProxyMessage.from_poll_request(poll_request))
        poll_response = _recv_poll_response(self, poll_request.poll_id)
        leader_results = list(poll_response.results)
        

    synced_results = _sync_draft_results_across_schedulers(self, leader_results)
    for result in synced_results:
        self._decoupled_pending_draft_results[result.key] = result

    _bind_draft_results_to_reqs(self, live_reqs)
    


def _build_request_terminate_message(req, reason: RequestTerminateReason) -> RequestTerminateMessage:
    return RequestTerminateMessage(request_id=req.rid, reason=reason)


def _advance_decode_round_and_submit_drafts(self: Scheduler, batch) -> None:
    _require_draftproxy_on_entry_rank(self)

    advance_start = time.perf_counter()
    requests_to_send = []
    terminate_messages = []
    for req in batch.reqs:
        if req.is_retracted:
            terminate_messages.append(
                _build_request_terminate_message(req, RequestTerminateReason.ABORT)
            )
            _clear_local_request_draft_state(self, req.rid)
            continue

        if req.finished():
            finish_reason = (
                getattr(req, "finished_reason", None)
                or getattr(req, "finish_reason", None)
                or getattr(req, "stop_reason", None)
            )
            finish_reason_type = (
                finish_reason.get("type")
                if isinstance(finish_reason, dict)
                else getattr(finish_reason, "type", None)
                or getattr(finish_reason, "value", None)
                or getattr(finish_reason, "name", None)
                or finish_reason
            )
            finish_reason_text = None if finish_reason_type is None else str(finish_reason_type).lower()
            if finish_reason_text is not None and (
                "max_model_len" in finish_reason_text
                or "max_length" in finish_reason_text
                or "length" in finish_reason_text
            ):
                print(
                    "[decoupled_spec][verify_scheduler] request_terminated_by_max_model_len "
                    f"dp_rank={_get_scheduler_dp_rank(self)} request_id={req.rid} "
                    f"finish_reason={finish_reason_type}"
                )
            elif finish_reason_text is not None and (
                "eos" in finish_reason_text or finish_reason_text == "stop"
            ):
                print(
                    "[decoupled_spec][verify_scheduler] request_terminated_by_eos_token "
                    f"dp_rank={_get_scheduler_dp_rank(self)} request_id={req.rid} "
                    f"finish_reason={finish_reason_type}"
                )
            terminate_messages.append(
                _build_request_terminate_message(req, RequestTerminateReason.FINISHED)
            )
            _clear_local_request_draft_state(self, req.rid)
            continue

        if req.rid in self._decoupled_needs_warmup_decode_rids:
            # warmup 集合内的 request，经过一次 decode 之后（也就是这里），就 warmup 完毕了
            self._decoupled_needs_warmup_decode_rids.discard(req.rid)
            setattr(req, "decoupled_spec_is_warmup_decode", False)

        requests_to_send.append(req)

    for req in requests_to_send:
        _submit_draft_request(self, req)

    for terminate_message in terminate_messages:
        
        if _is_verify_scheduler_external_draft_leader(self):
            self._send_to_draftproxy.send_pyobj(DraftProxyMessage.from_request_terminate(terminate_message))
    


def _patch_verify_scheduler():
    if getattr(Scheduler, "_verl_decoupled_spec_patched", False):
        return

    original_init_ipc_channels = Scheduler.init_ipc_channels
    original_recv_requests = Scheduler.recv_requests
    original_run_batch = Scheduler.run_batch
    original_process_batch_result = Scheduler.process_batch_result

    def patched_init_ipc_channels(self, port_args):
        init_start = time.perf_counter()
        original_init_ipc_channels(self, port_args)
        self._decoupled_pending_draft_results = {}
        self._decoupled_waiting_draft_keys = {}  # request_id -> deque[DraftLookupKey]
        self._decoupled_next_draft_round_by_rid = {} # request_id -> 该 request 下一次发送 DraftRequest 的 round
        self._decoupled_needs_warmup_decode_rids = set()
        self._decoupled_pending_poll_responses = {}
        self._decoupled_poll_seq = 0
        self._send_to_draftproxy = None
        self._recv_from_draftproxy = None
        ipc_config = DraftProxyIpcConfig.from_env()
        self._decoupled_draftproxy_expected = ipc_config is not None
        if ipc_config is None:
            return
        if _is_verify_scheduler_entry_rank(self):
            endpoints = ipc_config.get_endpoints(_get_scheduler_dp_rank(self))
            draftproxy_context = zmq.Context(2)
            self._draftproxy_context = draftproxy_context
            self._send_to_draftproxy = get_zmq_socket(
                draftproxy_context,
                zmq.PUSH,
                endpoints.scheduler_to_proxy_ipc_name,
                False,
            )
            self._recv_from_draftproxy = get_zmq_socket(
                draftproxy_context,
                zmq.PULL,
                endpoints.proxy_to_scheduler_ipc_name,
                False,
            )
            print(
                "[decoupled_spec][verify_scheduler] init_ipc_channels_done "
                f"dp_rank={_get_scheduler_dp_rank(self)} "
                f"scheduler_to_proxy={endpoints.scheduler_to_proxy_ipc_name} "
                f"proxy_to_scheduler={endpoints.proxy_to_scheduler_ipc_name} "
                f"elapsed_s={time.perf_counter() - init_start:.6f}"
            )

    def patched_recv_requests(self):
        recv_reqs = original_recv_requests(self)
        if not _is_verify_scheduler_entry_rank(self) or not recv_reqs:
            return recv_reqs

        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateReqInput):
                print(
                    "[decoupled_spec][verify_scheduler] recv_generate_request "
                    f"dp_rank={_get_scheduler_dp_rank(self)} rid={recv_req.rid} "
                    f"input_len={len(recv_req.input_ids) if recv_req.input_ids is not None else 0} "
                    f"max_new_tokens={getattr(recv_req.sampling_params, 'max_new_tokens', None)} "
                    f"stream={getattr(recv_req, 'stream', None)}"
                )
            elif isinstance(recv_req, BatchTokenizedGenerateReqInput):
                batch_rids = [req.rid for req in recv_req.batch]
                print(
                    "[decoupled_spec][verify_scheduler] recv_batch_generate_request "
                    f"dp_rank={_get_scheduler_dp_rank(self)} batch_size={len(recv_req.batch)} "
                    f"rids={batch_rids}"
                )
        return recv_reqs

    def patched_run_batch(self, batch, pp_proxy_tensors=None):
        if batch is not None and batch.forward_mode.is_decode():
            run_batch_wait_start = time.perf_counter()
            target_reqs = _get_decode_reqs_for_poll(self, batch) # 这个 batch 中，需要 poll DraftResult 的 req 列表
            _wait_for_draft_results(self, batch, target_reqs=target_reqs)


        # 注：original_run_batch 内部会调用 self.model_worker.forward_batch_generation，而 model_worker 的工厂方法已经被patch
        run_batch_start = time.perf_counter()
        result = original_run_batch(self, batch, pp_proxy_tensors)
        
        return result

    def patched_process_batch_result(self, batch, result):
        process_start = time.perf_counter()
        original_process_batch_result(self, batch, result)
        if not self.is_generation:
            return

        if batch.forward_mode.is_extend() and not batch.is_dllm():
            _send_draft_requests(
                self,
                batch,
                warmup_request_ids=_get_prefill_warmup_request_ids(batch),
            )
        elif batch.forward_mode.is_decode():
            _advance_decode_round_and_submit_drafts(self, batch)
        

    Scheduler.init_ipc_channels = patched_init_ipc_channels
    Scheduler.recv_requests = patched_recv_requests
    Scheduler.run_batch = patched_run_batch
    Scheduler.process_batch_result = patched_process_batch_result
    Scheduler._verl_decoupled_spec_patched = True


def run_scheduler_process(*args, **kwargs):
    from sglang.srt.entrypoints.engine import run_scheduler_process as upstream

    patch_speculative_worker_factory()
    _patch_verify_scheduler()
    return upstream(*args, **kwargs)


def launch_subprocesses(
    server_args,
    init_tokenizer_manager_func,
    run_scheduler_process_func,
    run_detokenizer_process_func,
    port_args=None,
):
    launch_start = time.perf_counter()
    from sglang.srt.entrypoints.engine import (
        _launch_scheduler_processes,
        _set_envs_and_config,
        _wait_for_scheduler_ready,
    )
    from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import configure_logger, launch_dummy_health_check_server
    from verl.workers.rollout.decoupled_spec_rollout.sglang_patch.draftproxy_subprocess import (
        run_draftproxy_process,
    )

    configure_logger(server_args)
    _set_envs_and_config(server_args)
    server_args.check_server_args()
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
    logger.info(f"{server_args=}")
    

    scheduler_launch_start = time.perf_counter()
    scheduler_procs, scheduler_pipe_readers = _launch_scheduler_processes(
        server_args=server_args,
        port_args=port_args,
        run_scheduler_process_func=run_scheduler_process_func,
    )
    

    if server_args.node_rank >= 1:
        scheduler_infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            return None, None, scheduler_infos, port_args
        launch_dummy_health_check_server(server_args.host, server_args.port, server_args.enable_metrics)
        for proc in scheduler_procs:
            proc.join()
            logger.error(f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}")
        return None, None, scheduler_infos, port_args

    detoken_proc = mp.Process(
        target=run_detokenizer_process_func,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    global _pending_draft_actor_names, _pending_draft_actor_namespace
    draft_actor_names = _pending_draft_actor_names
    _pending_draft_actor_names = None
    draft_actor_namespace = _pending_draft_actor_namespace
    _pending_draft_actor_namespace = None

    ipc_config = DraftProxyIpcConfig.from_env()
    if ipc_config is not None and draft_actor_names:
        draftproxy_launch_start = time.perf_counter()
        mp.Process(
            target=run_draftproxy_process,
            args=(
                server_args,
                port_args,
                draft_actor_names,
                draft_actor_namespace,
            ),
        ).start()
        

    if server_args.tokenizer_worker_num == 1:
        tokenizer_manager, template_manager = init_tokenizer_manager_func(server_args, port_args)
    else:
        tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
        template_manager = None

    scheduler_infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
    tokenizer_manager.max_req_input_len = scheduler_infos[0]["max_req_input_len"]
    return tokenizer_manager, template_manager, scheduler_infos, port_args


def run_detokenizer_process(*args, **kwargs):
    from sglang.srt.entrypoints.engine import run_detokenizer_process as upstream

    return upstream(*args, **kwargs)
