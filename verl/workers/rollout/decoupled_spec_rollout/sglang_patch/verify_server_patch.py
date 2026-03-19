from __future__ import annotations

import logging
import multiprocessing as mp
import os

import zmq
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.utils import get_zmq_socket

from verl.workers.rollout.decoupled_spec_rollout.protocol import (
    DraftProxyIpcConfig,
    DraftProxyMessage,
    DraftProxyMessageType,
    DraftRequest,
    DraftRequestKind,
    DraftResult,
    DraftStatus,
    get_draft_endpoints_from_env,
    get_num_speculative_steps_from_env,
    get_verify_replica_rank_from_env,
)

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


def _build_draft_request(self: Scheduler, recv_req: TokenizedGenerateReqInput) -> DraftRequest | None:
    if recv_req.input_ids is None:
        return None

    session_id = recv_req.session_params.id if recv_req.session_params is not None else None
    request_kind = DraftRequestKind.PREFILL
    full_prompt_token_ids = list(recv_req.input_ids)

    if (
        recv_req.session_params is not None
        and recv_req.session_params.id is not None
        and recv_req.session_params.id in self.sessions
        and recv_req.session_params.rid is not None
    ):
        session = self.sessions[recv_req.session_params.id]
        request_kind = DraftRequestKind.DECODE
        last_req_node = session.req_nodes.get(recv_req.session_params.rid)
        if last_req_node is not None:
            last_req = last_req_node.req
            full_prompt_token_ids = (
                list(last_req.origin_input_ids)
                + list(last_req.output_ids[: last_req.sampling_params.max_new_tokens])
                + list(recv_req.input_ids)
            )

    return DraftRequest(
        request_id=recv_req.rid,
        session_id=session_id,
        verify_replica_rank=get_verify_replica_rank_from_env(),
        prompt_token_ids=full_prompt_token_ids,
        committed_token_ids=[],
        target_position=len(full_prompt_token_ids),
        num_speculative_steps=get_num_speculative_steps_from_env(),
        request_kind=request_kind,
        sampling_params=_build_sampling_params_dict(recv_req.sampling_params),
        metadata={
            "priority": recv_req.priority,
            "routing_key": recv_req.routing_key,
        },
    )


def _request_draft_result(self: Scheduler, recv_req: TokenizedGenerateReqInput) -> DraftResult | None:
    if getattr(self, "_send_to_draftproxy", None) is None or getattr(self, "_recv_from_draftproxy", None) is None:
        return None

    draft_request = _build_draft_request(self, recv_req)
    if draft_request is None:
        return None

    self._send_to_draftproxy.send_pyobj(DraftProxyMessage.from_draft_request(draft_request))
    if not self._recv_from_draftproxy.poll(timeout=10000):
        return DraftResult(
            request_id=draft_request.request_id,
            session_id=draft_request.session_id,
            status=DraftStatus.FAILED,
            metadata={"error": "Timed out waiting for DraftProxy response"},
        )

    response = self._recv_from_draftproxy.recv_pyobj()
    if response.message_type == DraftProxyMessageType.DRAFT_RESULT and response.result is not None:
        return response.result
    if response.message_type == DraftProxyMessageType.ERROR:
        return DraftResult(
            request_id=draft_request.request_id,
            session_id=draft_request.session_id,
            status=DraftStatus.FAILED,
            metadata={"error": response.error or "Unknown DraftProxy error"},
        )
    return DraftResult(
        request_id=draft_request.request_id,
        session_id=draft_request.session_id,
        status=DraftStatus.FAILED,
        metadata={"error": f"Unexpected DraftProxy response: {response.message_type}"},
    )


def _patch_verify_scheduler():
    if getattr(Scheduler, "_verl_decoupled_spec_patched", False):
        return

    original_init_ipc_channels = Scheduler.init_ipc_channels
    original_handle_generate_request = Scheduler.handle_generate_request
    original_add_request_to_queue = Scheduler._add_request_to_queue

    def patched_init_ipc_channels(self, port_args):
        original_init_ipc_channels(self, port_args)
        self._decoupled_pending_draft_results = {}
        self._send_to_draftproxy = None
        self._recv_from_draftproxy = None
        ipc_config = DraftProxyIpcConfig.from_env()
        if ipc_config is None:
            return
        if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
            draftproxy_context = zmq.Context(2)
            self._draftproxy_context = draftproxy_context
            self._send_to_draftproxy = get_zmq_socket(
                draftproxy_context,
                zmq.PUSH,
                ipc_config.scheduler_to_proxy_ipc_name,
                False,
            )
            self._recv_from_draftproxy = get_zmq_socket(
                draftproxy_context,
                zmq.PULL,
                ipc_config.proxy_to_scheduler_ipc_name,
                False,
            )

    def patched_handle_generate_request(self, recv_req):
        draft_result = _request_draft_result(self, recv_req)
        if draft_result is not None:
            self._decoupled_pending_draft_results[recv_req.rid] = draft_result
            setattr(recv_req, "decoupled_spec_draft_result", draft_result)
        return original_handle_generate_request(self, recv_req)

    def patched_add_request_to_queue(self, req):
        draft_result = self._decoupled_pending_draft_results.pop(req.rid, None)
        if draft_result is not None:
            setattr(req, "decoupled_spec_draft_result", draft_result)
        return original_add_request_to_queue(self, req)

    Scheduler.init_ipc_channels = patched_init_ipc_channels
    Scheduler.handle_generate_request = patched_handle_generate_request
    Scheduler._add_request_to_queue = patched_add_request_to_queue
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

    ipc_config = DraftProxyIpcConfig.from_env()
    draft_endpoints = get_draft_endpoints_from_env()
    if ipc_config is not None and draft_endpoints:
        mp.Process(
            target=run_draftproxy_process,
            args=(
                server_args,
                port_args,
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
