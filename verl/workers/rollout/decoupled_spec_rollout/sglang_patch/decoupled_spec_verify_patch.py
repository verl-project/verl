from __future__ import annotations

import logging
import os

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

_EXTERNAL_VERIFY_ENV = "VERL_SGLANG_EXTERNAL_DRAFT_VERIFY"


def set_external_draft_verify_env() -> None:
    os.environ[_EXTERNAL_VERIFY_ENV] = "1"


def _is_external_draft_verify_enabled() -> bool:
    return os.getenv(_EXTERNAL_VERIFY_ENV) == "1"


def _get_req_tail_token_id(req) -> int:
    if req.output_ids:
        return int(req.output_ids[-1])
    if req.origin_input_ids:
        return int(req.origin_input_ids[-1])
    raise RuntimeError(f"Request {req.rid} has no committed token to anchor external draft verification.")


class ExternalDraftVerifyWorker:
    """Verify-only speculative worker that consumes external linear draft tokens."""

    verify = EAGLEWorker.verify
    _mamba_verify_update = EAGLEWorker._mamba_verify_update

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int | None,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        del gpu_id, tp_rank, dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.page_size = server_args.page_size
        self.topk = 1
        self.speculative_num_steps = int(server_args.speculative_num_steps)
        self.speculative_num_draft_tokens = int(server_args.speculative_num_draft_tokens)
        self.enable_nan_detection = bool(server_args.enable_nan_detection)
        self.device = self.model_runner.device

    def clear_cache_pool(self):
        # No local draft-side KV/state exists for the external verify worker.
        return

    def _get_verify_buffers(self, draft_token_num: int):
        if draft_token_num != self.speculative_num_draft_tokens:
            return None, None

        attn_backend = getattr(self.target_worker.model_runner, "attn_backend", None)
        if attn_backend is None:
            return None, None

        get_buffers = getattr(attn_backend, "get_verify_buffers_to_fill_after_draft", None)
        if get_buffers is None:
            return None, None

        try:
            return get_buffers()
        except Exception as exc:  # pragma: no cover - defensive fallback only
            logger.debug("Falling back to eager verify buffers: %s", exc)
            return None, None

    def _build_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        full_draft_tokens_by_req: list[list[int]] = []
        for req in batch.reqs:
            draft_result = getattr(req, "decoupled_spec_draft_result", None)
            if draft_result is None:
                raise RuntimeError(f"Missing external draft tokens for request {req.rid}.")

            proposal_tokens = list(getattr(draft_result, "draft_token_ids", []) or [])
            if not proposal_tokens:
                raise RuntimeError(
                    f"External draft verification requires at least one proposed token for request {req.rid}."
                )

            tail_token = _get_req_tail_token_id(req)
            if bool(getattr(draft_result, "includes_committed_tail", False)):
                full_draft_tokens_by_req.append(proposal_tokens)
            else:
                # The current draft server returns pure proposal tokens. Prepend
                # the latest committed token so the linear chain matches
                # SGLang's topk=1 verify layout: [root, draft_1, ..., draft_k].
                full_draft_tokens_by_req.append([tail_token] + proposal_tokens)

        draft_token_num = min(
            min(len(tokens) for tokens in full_draft_tokens_by_req),
            self.speculative_num_draft_tokens,
        )
        if draft_token_num < 2:
            raise RuntimeError("External draft verification requires at least one draft token per request.")

        spec_steps = draft_token_num - 1
        truncated_tokens = [tokens[:draft_token_num] for tokens in full_draft_tokens_by_req]
        verified_id = torch.tensor(
            [tokens[0] for tokens in truncated_tokens],
            dtype=torch.long,
            device=batch.device,
        )
        draft_tokens = torch.tensor(
            [tokens[1:] for tokens in truncated_tokens],
            dtype=torch.long,
            device=batch.device,
        )

        batch_size = batch.batch_size()
        seq_lens_sum = int(torch.sum(batch.seq_lens).item())
        selected_index = torch.arange(
            draft_token_num - 1, dtype=torch.long, device=batch.device
        ).expand(batch_size, -1).contiguous()
        parent_list = torch.zeros(
            (batch_size, spec_steps), dtype=torch.long, device=batch.device
        )
        if spec_steps > 1:
            parent_list[:, 1:] = torch.arange(
                spec_steps - 1, dtype=torch.long, device=batch.device
            )

        tree_mask_buf, position_buf = self._get_verify_buffers(draft_token_num)
        (
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            flat_draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=selected_index,
            draft_tokens=draft_tokens,
            seq_lens=batch.seq_lens,
            seq_lens_sum=seq_lens_sum,
            topk=1,
            spec_steps=spec_steps,
            num_verify_tokens=draft_token_num,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            tree_mask_buf=tree_mask_buf,
            position_buf=position_buf,
        )

        return EagleVerifyInput(
            draft_token=flat_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=spec_steps,
            topk=1,
            draft_token_num=draft_token_num,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            return self.target_worker.forward_batch_generation(model_worker_batch)

        spec_info = self._build_verify_input(batch)
        can_use_full_graph_path = spec_info.draft_token_num == self.speculative_num_draft_tokens
        logits_output, verify_output, _, can_run_cuda_graph = self.verify(batch, spec_info)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph and can_use_full_graph_path,
        )


def patch_speculative_worker_factory() -> None:
    if getattr(SpeculativeAlgorithm, "_verl_external_draft_verify_patched", False):
        return

    original_create_worker = SpeculativeAlgorithm.create_worker

    def patched_create_worker(self, server_args: ServerArgs):
        if (
            _is_external_draft_verify_enabled()
            and server_args.disable_overlap_schedule
            and (self.is_eagle() or self.is_standalone())
        ):
            return ExternalDraftVerifyWorker
        return original_create_worker(self, server_args)

    SpeculativeAlgorithm.create_worker = patched_create_worker
    SpeculativeAlgorithm._verl_external_draft_verify_patched = True
