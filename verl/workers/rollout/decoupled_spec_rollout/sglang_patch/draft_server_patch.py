from __future__ import annotations

from copy import deepcopy
import time
from uuid import uuid4

from sglang.srt.managers.io_struct import GenerateReqInput

from verl.workers.rollout.decoupled_spec_rollout.protocol import DraftRequest, DraftResult


def init_tokenizer_manager(*args, **kwargs):
    from sglang.srt.entrypoints.engine import init_tokenizer_manager as upstream

    return upstream(*args, **kwargs)


def run_scheduler_process(*args, **kwargs):
    from sglang.srt.entrypoints.engine import run_scheduler_process as upstream

    return upstream(*args, **kwargs)


def run_detokenizer_process(*args, **kwargs):
    from sglang.srt.entrypoints.engine import run_detokenizer_process as upstream

    return upstream(*args, **kwargs)


async def handle_draft_request(self, draft_request):
    if self.server_role != "draft":
        raise ValueError("handle_draft_request is only supported on draft servers")

    prompt_ids = draft_request.full_token_ids
    max_possible_tokens = self.config.max_model_len - len(prompt_ids)
    if max_possible_tokens <= 0:
        
        return DraftResult(
            request_id=draft_request.request_id,
            draft_round_id=draft_request.draft_round_id,
            draft_token_ids=[],
        )

    max_new_tokens = max(0, min(draft_request.num_speculative_steps + 1, max_possible_tokens)) # drafter 生成 draft token 时需要多生成一个，用于 verify 时对齐
    generate_request = build_generate_req_from_draft_request(
        draft_request,
        request_id=f"draft-{draft_request.request_id}-{uuid4().hex[:8]}",
        max_new_tokens=max_new_tokens,
    )
    try:
        final_output_ids = []
        async for output in self.tokenizer_manager.generate_request(generate_request, None):
            final_output_ids = list(output.get("output_ids", []))
            print(f"from drafter: Draft request {draft_request.request_id} in round {draft_request.draft_round_id} got output ids: {final_output_ids}")
            if output.get("meta_info", {}).get("finish_reason") is not None:
                return DraftResult(
                    request_id=draft_request.request_id,
                    draft_round_id=draft_request.draft_round_id,
                    draft_token_ids=final_output_ids,
                )
    except Exception:
        assert False, "generate_request for draft request failed"

    assert False, "generate_request for draft request finished without finish_reason"


def install_draft_server_patches(server_cls) -> None:
    if getattr(server_cls, "_verl_decoupled_draft_patched", False):
        return

    server_cls.handle_draft_request = handle_draft_request
    server_cls._verl_decoupled_draft_patched = True


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
