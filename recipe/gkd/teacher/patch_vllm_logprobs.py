# patch_vllm_logprobs.py
# === Why we avoid F.log_softmax over the full vocabulary (teacher side) ===
#
# Background:
#   log_softmax(z)_i = z_i - logsumexp(z)
# i.e., for each position, log-probabilities are just logits minus a single
# normalization scalar (logsumexp) shared across the vocabulary.
#
# --- Teacher side (vLLM / inference-time prompt_logprobs) ---
# Goal:
#   Return only top-k logprobs per position for distillation.
# Key idea:
#   The ranking of tokens is identical under logits and logprobs because
#   log_softmax is a monotone transform per element (subtracting the same
#   logsumexp constant for all tokens). Therefore:
#     1) topk_vals, topk_idx = topk(logits, k)          # [N, k]
#     2) lse = logsumexp(logits, dim=-1)                # [N]
#     3) topk_logp = topk_vals - lse[:, None]           # [N, k]
# This avoids materializing a huge [N, V] fp32 log_softmax tensor (OOM source).
import torch

_patched = False
_orig_compute_logprobs = None
_orig_gather_logprobs = None


def apply_patch():
    """
    Patch vLLM v1 Sampler to avoid allocating full [N, V] fp32 log_softmax.

    Works with the sampler.py you provided:
      - vllm.v1.sample.sampler.Sampler.compute_logprobs (staticmethod)
      - vllm.v1.sample.sampler.Sampler.gather_logprobs  (staticmethod)
    """
    global _patched, _orig_compute_logprobs, _orig_gather_logprobs
    if _patched:
        print("[patch_vllm_logprobs] already patched")
        return True

    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.sample.ops.logprobs import batched_count_greater_than

    _orig_compute_logprobs = Sampler.compute_logprobs
    _orig_gather_logprobs = Sampler.gather_logprobs

    # 1) Do NOT materialize full log_softmax. Just pass logits through.
    @staticmethod
    def compute_logprobs_no_full_softmax(logits: torch.Tensor) -> torch.Tensor:
        # Keep original dtype (bf16/fp16) to avoid huge fp32 tensor.
        return logits

    # 2) Compute only top-k logprobs (and token logprob) via topk + logsumexp.
    @staticmethod
    def gather_logprobs_from_logits(
        scores: torch.Tensor,      # actually logits
        num_logprobs: int,
        token_ids: torch.Tensor,   # int64 [N]
    ) -> LogprobsTensors:
        assert token_ids.dtype == torch.int64
        logits = scores  # rename

        # top-k on logits => [N,k]
        topk_vals, topk_indices = torch.topk(logits, num_logprobs, dim=-1)

        # token logit => [N,1]
        token_ids_col = token_ids.unsqueeze(-1)
        token_vals = logits.gather(-1, token_ids_col)

        # rank: compare logits is equivalent to compare logprobs (monotonic transform)
        token_ranks = batched_count_greater_than(logits, token_vals)

        # logsumexp over vocab => [N] (no [N,V] output tensor)
        lse = torch.logsumexp(logits, dim=-1).to(torch.float32).unsqueeze(-1)

        # small tensors -> fp32 for stable subtraction
        topk_logprobs = topk_vals.to(torch.float32) - lse      # [N,k]
        token_logprobs = token_vals.to(torch.float32) - lse    # [N,1]

        # concat: token first then topk (keeps vLLM's expected layout)
        indices = torch.cat((token_ids_col, topk_indices), dim=1).to(torch.int32)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)  # fp32 [N,k+1]

        return LogprobsTensors(indices, logprobs, token_ranks)

    Sampler.compute_logprobs = compute_logprobs_no_full_softmax
    Sampler.gather_logprobs = gather_logprobs_from_logits

    _patched = True
    print("[patch_vllm_logprobs] patched Sampler.compute_logprobs + Sampler.gather_logprobs (no full [N,V] log_softmax)")
    return True


def remove_patch():
    global _patched, _orig_compute_logprobs, _orig_gather_logprobs
    if not _patched:
        return
    from vllm.v1.sample.sampler import Sampler
    Sampler.compute_logprobs = _orig_compute_logprobs
    Sampler.gather_logprobs = _orig_gather_logprobs
    _patched = False
    print("[patch_vllm_logprobs] removed patch")