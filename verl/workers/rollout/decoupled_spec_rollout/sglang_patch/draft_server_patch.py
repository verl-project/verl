from __future__ import annotations

from copy import deepcopy

from sglang.srt.managers.io_struct import GenerateReqInput

from verl.workers.rollout.decoupled_spec_rollout.protocol import DraftRequest


def init_tokenizer_manager(*args, **kwargs):
    from sglang.srt.entrypoints.engine import init_tokenizer_manager as upstream

    return upstream(*args, **kwargs)


def run_scheduler_process(*args, **kwargs):
    from sglang.srt.entrypoints.engine import run_scheduler_process as upstream

    return upstream(*args, **kwargs)


def run_detokenizer_process(*args, **kwargs):
    from sglang.srt.entrypoints.engine import run_detokenizer_process as upstream

    return upstream(*args, **kwargs)


def build_generate_req_from_draft_request(
    draft_request: DraftRequest,
    *,
    request_id: str,
    max_new_tokens: int,
) -> GenerateReqInput:
    sampling_params = deepcopy(draft_request.sampling_params)
    sampling_params.pop("max_tokens", None)
    sampling_params["max_new_tokens"] = max_new_tokens
    sampling_params["ignore_eos"] = True

    return GenerateReqInput(
        rid=request_id,
        input_ids=draft_request.full_token_ids,
        sampling_params=sampling_params,
        return_logprob=False,
        stream=False,
    )
