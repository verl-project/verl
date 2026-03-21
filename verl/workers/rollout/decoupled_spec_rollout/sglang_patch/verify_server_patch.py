from __future__ import annotations

import logging
import multiprocessing as mp
import os

import zmq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.utils import get_zmq_socket

from verl.workers.rollout.decoupled_spec_rollout.protocol import (
    DraftProxyIpcConfig,
    DraftProxyMessage,
    DraftProxyMessageType,
    DraftRequest,
    DraftRequestKind,
    VerifyResult,
    get_num_speculative_steps_from_env,
    get_verify_replica_rank_from_env,
)

# Set by SGLangHttpServer.launch_server immediately before verify launch_subprocesses runs.
_pending_draft_actor_handles: list | None = None


def set_pending_draft_actor_handles(handles: list | None) -> None:
    """Pass drafter Ray actor handles from the verify server process into launch_subprocesses (cannot use env JSON)."""
    global _pending_draft_actor_handles
    _pending_draft_actor_handles = list(handles) if handles else None

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


def _iter_live_batch_reqs(batch) -> list:
    return [req for req in batch.reqs if not req.is_retracted and not req.finished()]


def _get_scheduler_dp_rank(self: Scheduler) -> int:
    dp_rank = getattr(self, "dp_rank", None)
    return 0 if dp_rank is None else int(dp_rank)


def _build_draft_request_from_req(
    self: Scheduler,
    req,
    *,
    request_kind: DraftRequestKind | None = None,
) -> DraftRequest:
    request_kind = request_kind or (
        DraftRequestKind.PREFILL if req.decode_batch_idx == 0 else DraftRequestKind.DECODE
    )
    committed_token_ids = list(req.output_ids)
    return DraftRequest(
        request_id=req.rid,
        session_id=req.session_id,
        verify_replica_rank=get_verify_replica_rank_from_env(),
        scheduler_dp_rank=_get_scheduler_dp_rank(self),
        prompt_token_ids=list(req.origin_input_ids),
        committed_token_ids=committed_token_ids,
        target_position=len(req.origin_input_ids) + len(committed_token_ids),
        num_speculative_steps=get_num_speculative_steps_from_env(),
        request_kind=request_kind,
        sampling_params=_build_sampling_params_dict(req.sampling_params),
        metadata={
            "priority": req.priority,
            "routing_key": req.routing_key,
            "decode_batch_idx": req.decode_batch_idx,
            "kv_committed_len": req.kv_committed_len,
        },
    )


def _send_draft_requests(self: Scheduler, batch) -> None:

    assert _is_draftproxy_enabled(self), "DraftProxy for SGLangHttpServer dp_rank=%s is not enabled" % _get_scheduler_dp_rank(self)

    for req in _iter_live_batch_reqs(batch):
        if req.is_chunked > 0:
            continue
        draft_request = _build_draft_request_from_req(self, req)
        self._send_to_draftproxy.send_pyobj(DraftProxyMessage.from_draft_request(draft_request))
        self._decoupled_waiting_draft_rids.add(draft_request.request_id)
        setattr(req, "decoupled_spec_draft_result", None)


def _wait_for_draft_results(self: Scheduler, batch, target_reqs=None) -> None:
    assert _is_draftproxy_enabled(self), "DraftProxy for VerifySGLangHttpServer dp_rank=%s is not enabled" % _get_scheduler_dp_rank(self)

    live_reqs = [
        req
        for req in (target_reqs if target_reqs is not None else batch.reqs)
        if not req.is_retracted and not req.finished()
    ]
    if not live_reqs:
        return

    missing_ids = [
        req.rid
        for req in live_reqs
        if req.rid in self._decoupled_waiting_draft_rids and req.rid not in self._decoupled_pending_draft_results
    ]

    while missing_ids:
        if not self._recv_from_draftproxy.poll(timeout=10000):
            raise TimeoutError(
                f"Timed out waiting for DraftProxy results for request ids: {missing_ids}"
            )

        response = self._recv_from_draftproxy.recv_pyobj()
        if response.message_type == DraftProxyMessageType.DRAFT_RESULT and response.result is not None:
            self._decoupled_pending_draft_results[response.result.request_id] = response.result
        elif response.message_type == DraftProxyMessageType.ERROR:
            request_id = response.request.request_id if response.request is not None else None
            raise RuntimeError(
                f"DraftProxy returned error for request {request_id}: {response.error or 'Unknown DraftProxy error'}"
            )
        else:
            raise RuntimeError(f"Unexpected DraftProxy batch response: {response.message_type}")

        missing_ids = [
            req.rid
            for req in live_reqs
            if req.rid in self._decoupled_waiting_draft_rids
            and req.rid not in self._decoupled_pending_draft_results
        ]

    for req in live_reqs:
        if req.rid not in self._decoupled_waiting_draft_rids:
            setattr(req, "decoupled_spec_draft_result", None)
            continue

        draft_result = self._decoupled_pending_draft_results.pop(req.rid, None)
        if draft_result is None:
            raise RuntimeError(f"Draft result missing for request {req.rid}")

        self._decoupled_waiting_draft_rids.discard(req.rid)
        setattr(req, "decoupled_spec_draft_result", draft_result)


def _build_verify_result_from_req(self: Scheduler, req) -> VerifyResult:
    draft_result = getattr(req, "decoupled_spec_draft_result", None)
    accepted_token_ids = list(req.output_ids[-1:]) if req.output_ids else []
    return VerifyResult(
        request_id=req.rid,
        session_id=req.session_id,
        scheduler_dp_rank=_get_scheduler_dp_rank(self),
        accepted_token_ids=accepted_token_ids,
        rollback_to=None,
        finished=req.finished(),
        metadata={
            "draft_status": draft_result.status.value if draft_result is not None else None,
            "draft_token_ids": list(draft_result.draft_token_ids) if draft_result is not None else [],
            "accepted_prefix_len": draft_result.accepted_prefix_len if draft_result is not None else 0,
            "decode_batch_idx": req.decode_batch_idx,
        },
    )


def _send_verify_results_and_trigger_draft(self: Scheduler, batch) -> None:
    if not _is_draftproxy_enabled(self):
        return

    verify_results = []
    next_round_requests = []
    for req in [req for req in batch.reqs if not req.is_retracted]:
        verify_result = _build_verify_result_from_req(self, req)
        verify_results.append(verify_result)
        if not req.finished():
            next_round_requests.append(
                _build_draft_request_from_req(self, req, request_kind=DraftRequestKind.DECODE)
            )
        else:
            self._decoupled_pending_draft_results.pop(req.rid, None)
            self._decoupled_waiting_draft_rids.discard(req.rid)

    for verify_result in verify_results:
        self._send_to_draftproxy.send_pyobj(DraftProxyMessage.from_verify_result(verify_result))

    for draft_request in next_round_requests:
        self._send_to_draftproxy.send_pyobj(DraftProxyMessage.from_draft_request(draft_request))
        self._decoupled_waiting_draft_rids.add(draft_request.request_id)


def _patch_verify_scheduler():
    if getattr(Scheduler, "_verl_decoupled_spec_patched", False):
        return

    original_init_ipc_channels = Scheduler.init_ipc_channels
    original_run_batch = Scheduler.run_batch
    original_process_batch_result_prefill = Scheduler.process_batch_result_prefill
    original_process_batch_result_decode = Scheduler.process_batch_result_decode

    def patched_init_ipc_channels(self, port_args):
        original_init_ipc_channels(self, port_args)
        self._decoupled_pending_draft_results = {}
        self._decoupled_waiting_draft_rids = set()
        self._send_to_draftproxy = None
        self._recv_from_draftproxy = None
        ipc_config = DraftProxyIpcConfig.from_env()
        if ipc_config is None:
            return
        if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
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

    def patched_run_batch(self, batch, pp_proxy_tensors=None):
        if batch is not None:
            if batch.forward_mode.is_decode():
                _wait_for_draft_results(self, batch)
            elif batch.forward_mode.is_extend() and batch.decoding_reqs:
                _wait_for_draft_results(self, batch, target_reqs=batch.decoding_reqs)
        return original_run_batch(self, batch, pp_proxy_tensors)

    def patched_process_batch_result_prefill(self, batch, result): # 对于 prefill 批次的处理逻辑
        original_process_batch_result_prefill(self, batch, result)
        if self.is_generation:
            _send_draft_requests(self, batch)

    def patched_process_batch_result_decode(self, batch, result): # 对于 decode 批次的处理逻辑
        original_process_batch_result_decode(self, batch, result)
        if self.is_generation:
            _send_verify_results_and_trigger_draft(self, batch)

    Scheduler.init_ipc_channels = patched_init_ipc_channels
    Scheduler.run_batch = patched_run_batch
    Scheduler.process_batch_result_prefill = patched_process_batch_result_prefill
    Scheduler.process_batch_result_decode = patched_process_batch_result_decode
    Scheduler._verl_decoupled_spec_patched = True


def run_scheduler_process(*args, **kwargs):
    from sglang.srt.entrypoints.engine import run_scheduler_process as upstream

    _patch_verify_scheduler()
    return upstream(*args, **kwargs)


def launch_subprocesses(
    server_args,
    init_tokenizer_manager_func,
    run_scheduler_process_func,
    run_detokenizer_process_func,
    port_args=None,
):
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

    global _pending_draft_actor_handles
    draft_actor_handles = _pending_draft_actor_handles
    _pending_draft_actor_handles = None

    ipc_config = DraftProxyIpcConfig.from_env()
    if ipc_config is not None and draft_actor_handles:
        mp.Process(
            target=run_draftproxy_process,
            args=(
                server_args,
                port_args,
                draft_actor_handles,
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
