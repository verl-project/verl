"""Kimi output-token support shared with the rollout logits mask."""
import torch


def is_kimi_model(hf_config):
    return getattr(hf_config, "model_type", "") in {"kimi_k3", "kimi_k31"}


def mask_kimi_generation_logits(logits: torch.Tensor, hf_config):
    if not is_kimi_model(hf_config):
        return logits
    token_id = getattr(hf_config, "media_placeholder_token_id", None)
    if token_id is None or not 0 <= token_id < logits.shape[-1]:
        raise ValueError("Kimi generation mask requires a valid media_placeholder_token_id")
    if hasattr(logits, "full_tensor"):
        logits = logits.full_tensor()
    # Match vLLM's get_vision_placeholder_token_ids/monkey_patch_compute_logits.
    # The placeholder has image features on input, but cannot be generated.
    mask = torch.arange(logits.shape[-1], device=logits.device) == token_id
    return logits.masked_fill(mask, -torch.inf)
