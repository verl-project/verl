#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import os
import random
import time

# Must be set before importing torch / first CUDA use
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib import pyplot as plt
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional, used only when --engine vllm
try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None


SUFFIX = "Let's think step by step and answer in \\boxed{}."


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def run_with_optional_profiler(trace_path: str, target_fn, *args, **kwargs):
    """
    Wrap `target_fn(*args, **kwargs)` in a torch.profiler context when `trace_path` is provided.
    Produces a Chrome/Perfetto-compatible JSON trace at the requested location and otherwise
    acts as a direct pass-through.
    """
    if not trace_path:
        return target_fn(*args, **kwargs)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    trace_dir = os.path.dirname(trace_path)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        result = target_fn(*args, **kwargs)
        prof.step()

    prof.export_chrome_trace(trace_path)
    print(f"[torch.profiler] Trace written to {trace_path}")
    return result


def _get_base_module_and_head(model: AutoModelForCausalLM):
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    base = None
    base_prefix = getattr(model, "base_model_prefix", None)
    if isinstance(base_prefix, str) and hasattr(model, base_prefix):
        base = getattr(model, base_prefix)
    for name in ["model", "transformer", "language_model", "backbone", "base_model"]:
        if base is None and hasattr(model, name):
            base = getattr(model, name)
    if base is None or head is None:
        raise RuntimeError("Could not locate base transformer or lm_head on the HF model.")
    return base, head


@torch.no_grad()
def _chunked_token_logprobs_from_hidden(
    hiddens: torch.Tensor,
    head_weight: torch.Tensor,
    targets: torch.Tensor,
    time_chunk: int = 256,
    head_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    hiddens: [B, L, H] (on CPU)
    head_weight: [V, H] on its own (likely CUDA) device
    targets: [B, L-1] (on CPU)
    returns lp: [B, L-1] (float32, on CPU)
    """
    B, L, H = hiddens.shape
    T = L - 1
    V, Hw = head_weight.shape
    assert H == Hw, f"Hidden size mismatch: {H} vs {Hw}"

    out_device = hiddens.device  # CPU
    weight_device = head_weight.device  # e.g., cuda:0/1

    # Rely on explicit .to(device) calls instead of changing the global CUDA device.

    # Project in fp32 (matches HF `_Fp32LmHeadLinear`) to avoid bf16 precision loss.
    W = head_weight.to(dtype=torch.float32, device=weight_device)
    b = head_bias.to(dtype=torch.float32, device=weight_device) if head_bias is not None else None

    lp = torch.empty((B, T), dtype=torch.float32, device=out_device)

    t = 0
    cur_chunk = max(1, int(time_chunk))

    while t < T:
        cur = min(cur_chunk, T - t)
        try:
            # Slice, make contiguous, then copy CPU->CUDA (blocking copy for stability)
            h = hiddens[:, t : t + cur, :].contiguous().view(-1, H).to(weight_device, non_blocking=False)
            y = targets[:, t : t + cur].contiguous().view(-1).to(weight_device, non_blocking=False)

            logits = F.linear(h.to(torch.float32), W, b)  # [B*cur, V] on weight_device, fp32 for parity

            # Numerically stable log softmax for chosen tokens
            m = logits.max(dim=-1).values
            lse = torch.logsumexp((logits - m.unsqueeze(1)).to(torch.float32), dim=-1) + m.to(torch.float32)
            chosen = logits.gather(1, y.unsqueeze(1)).squeeze(1).to(torch.float32)

            lp_chunk = (chosen - lse).view(B, cur).to(out_device, non_blocking=False)
            lp[:, t : t + cur] = lp_chunk

            # cleanup
            del h, y, logits, m, lse, chosen, lp_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t += cur
            cur_chunk = time_chunk
        except RuntimeError as e:
            # Back off time chunk on OOM
            if "out of memory" in str(e).lower() and cur > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cur_chunk = max(1, cur // 2)
                continue
            raise
    return lp


def infer_log_probs_batch(model: AutoModelForCausalLM, sequences, dtype, device: str, time_chunk: int = 256):
    """
    Run base forward on single GPU, offload [B,L,H] to CPU, then do chunked head on the head device.
    """
    lens = [len(s) for s in sequences]
    Lm = max(lens) if lens else 0
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else (model.config.eos_token_id or 0)

    inp = torch.full((len(sequences), Lm), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(sequences):
        if len(s) > 0:
            inp[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    attn = (inp != pad_id).long()

    try:
        model.config.use_cache = False
    except Exception:
        pass

    base, head = _get_base_module_and_head(model)

    with torch.inference_mode():
        outputs = base(
            input_ids=inp,
            attention_mask=attn,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        hidden_states = outputs[0]  # [B, Lm, H]
        del outputs
    # Offload activations to CPU, then make contiguous + (optional) pin
    if hidden_states.is_cuda:
        hidden_states = hidden_states.to("cpu", non_blocking=False)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    hidden_states = hidden_states.contiguous()
    try:
        hidden_states = hidden_states.pin_memory()
    except Exception:
        pass

    # Move targets to CPU for chunked processing
    tgt = inp[:, 1:].to("cpu")

    lp = _chunked_token_logprobs_from_hidden(
        hiddens=hidden_states,
        head_weight=head.weight,
        head_bias=getattr(head, "bias", None),
        targets=tgt,
        time_chunk=time_chunk,
    )

    del hidden_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lp, lens


def infer_log_probs_batch_vllm(model: LLM, sequences, batch_size: int = 64):
    """
    Use vLLM to compute logprobs for full sequences via prompt_logprobs.
    Returns logprobs for tokens [1:] (shifted by one, matching HF's format).

    Returns:
        lp: torch.Tensor of shape [B, L-1] with logprobs for each sequence
        lens: list of sequence lengths
    """
    assert LLM is not None, "vLLM is not installed."

    lens = [len(s) for s in sequences]
    Lm = max(lens) if lens else 0
    B = len(sequences)

    # Use prompt_logprobs to get logprobs for all positions in the sequence
    # We need max_tokens >= 1 to get prompt_logprobs, but we'll ignore the generated token
    sp = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,  # Minimum to trigger prompt_logprobs
        prompt_logprobs=8,
        logprobs=0,  # Don't need sample logprobs
    )

    # Pre-allocate output tensor [B, Lm-1] filled with NaN (will be masked by actual lengths)
    lp = torch.full((B, Lm - 1), float("nan"), dtype=torch.float32)

    for start in tqdm(range(0, len(sequences), batch_size), desc="vLLM prefill/scoring"):
        batch_seqs = sequences[start : start + batch_size]
        batch_inputs = [{"prompt_token_ids": seq} for seq in batch_seqs]

        outs = model.generate(batch_inputs, sampling_params=sp)

        for batch_idx_offset, (seq, o) in enumerate(zip(batch_seqs, outs, strict=False)):
            idx = start + batch_idx_offset
            seq_len = len(seq)

            # Extract prompt logprobs (logprobs for all positions in the prompt)
            prompt_logprobs = o.prompt_logprobs

            if prompt_logprobs is None:
                raise RuntimeError("vLLM returned no prompt_logprobs; set prompt_logprobs >= 1.")

            # prompt_logprobs is a dict mapping position -> dict of token_id -> Logprob
            # Or it might be a list/None for some positions
            # We want logprobs for tokens at positions [1, 2, ..., len(seq)-1]
            # which correspond to the "next token" predictions
            seq_lp = []
            for pos in range(1, seq_len):  # Skip position 0 (no prediction)
                token_id = seq[pos]

                # Handle different possible structures of prompt_logprobs
                if isinstance(prompt_logprobs, dict):
                    lp_dict = prompt_logprobs.get(pos)
                elif isinstance(prompt_logprobs, list):
                    # If it's a list, it might be indexed by position
                    if pos < len(prompt_logprobs):
                        lp_dict = prompt_logprobs[pos]
                    else:
                        lp_dict = None
                else:
                    lp_dict = None

                if lp_dict is None:
                    raise RuntimeError(f"No logprob dict at position {pos} for sequence of length {seq_len}")

                # lp_dict should be a dict mapping token_id -> Logprob object
                if isinstance(lp_dict, dict):
                    lp_obj = lp_dict.get(token_id)
                    if lp_obj is None:
                        # Token might not be in top-k, need to compute from full distribution or use approximation
                        # For now, raise an error - could fall back to HF if needed
                        message = (
                            f"Token {token_id} not in prompt_logprobs at position {pos} "
                            "(increase prompt_logprobs or token not in top-k)"
                        )
                        raise RuntimeError(message)

                    # Extract logprob value (could be Logprob object with .logprob attribute or just a float)
                    if hasattr(lp_obj, "logprob"):
                        seq_lp.append(float(lp_obj.logprob))
                    else:
                        seq_lp.append(float(lp_obj))
                else:
                    raise RuntimeError(f"Unexpected prompt_logprobs format at position {pos}: {type(lp_dict)}")

            # Store logprobs in output tensor
            if seq_lp:
                seq_lp_tensor = torch.tensor(seq_lp, dtype=torch.float32)
                lp[idx, : len(seq_lp_tensor)] = seq_lp_tensor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return lp, lens


def _init_single_process_dist():
    """
    Initialize a 1-rank torch.distributed group if not already initialized.
    Megatron-Core expects process groups, even for single-node inference.
    """
    import torch.distributed as dist

    if not dist.is_available() or dist.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=0, world_size=1)


def build_megatron_prefill_model(model_name: str, dtype: torch.dtype, seed: int | None = None):
    """
    Build a Megatron-LM module (via Megatron-Bridge) for prefill/scoring.
    Returns a dict with the model, HF config, forward fn, and bridge/provider.
    """
    _init_single_process_dist()

    try:
        from megatron.core import parallel_state as mpu
        from megatron.core import tensor_parallel
        from transformers import AutoConfig

        # Prefer Megatron-Bridge; fall back to vanilla mbridge if provider API is missing.
        try:
            from verl.models.mcore.bridge import AutoBridge

            _has_provider = True
        except Exception:
            from verl.models.mcore.mbridge import AutoBridge

            _has_provider = False

        from verl.models.mcore.registry import get_mcore_forward_no_padding_fn
        from verl.utils.megatron_utils import (
            McoreModuleWrapperConfig,
            load_megatron_model_to_gpu,
            make_megatron_module,
        )
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Megatron-Bridge/mbridge and Megatron-Core are required for --prefill-engine megatron"
        ) from exc

    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Megatron prefill supports only float16/bfloat16 dtype")

    if not mpu.is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            nccl_communicator_config_path=None,
        )
        # Megatron requires model-parallel RNG to be initialized before module creation.
        tensor_parallel.random.model_parallel_cuda_manual_seed(seed or 1234)

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if hasattr(AutoBridge, "from_hf_pretrained"):
        bridge = AutoBridge.from_hf_pretrained(
            model_name,
            trust_remote_code=True)
    elif hasattr(AutoBridge, "from_pretrained"):
        bridge = AutoBridge.from_pretrained(
            model_name,
            trust_remote_code=True)
    else:
        bridge = AutoBridge.from_config(hf_config, dtype=dtype)

    provider = None
    tf_config = None
    if _has_provider and hasattr(bridge, "to_megatron_provider"):
        from megatron.core.transformer.enums import AttnBackend

        provider = bridge.to_megatron_provider(load_weights=False)
        provider.params_dtype = dtype
        provider.tensor_model_parallel_size = 4
        provider.pipeline_model_parallel_size = 1
        provider.expert_model_parallel_size = 1
        provider.expert_tensor_parallel_size = 1
        provider.virtual_pipeline_model_parallel_size = None
        provider.context_parallel_size = 1
        provider.sequence_parallel = False
        provider.variable_seq_lengths = True
        try:
            provider.attention_backend = AttnBackend.flash
        except Exception:
            pass
        provider.finalize()
    else:
        tf_config = getattr(bridge, "config", None)

    wrap_config = McoreModuleWrapperConfig(
        is_value_model=False,
        share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
        wrap_with_ddp=False,
        use_distributed_optimizer=False,
    )

    module, _ = make_megatron_module(
        wrap_config=wrap_config,
        tf_config=tf_config,
        hf_config=hf_config,
        bridge=bridge,
        provider=provider,
        override_model_config={},
        override_ddp_config=None,
        peft_cls=None,
        peft_config=None,
    )

    if hasattr(bridge, "load_hf_weights"):
        bridge.load_hf_weights(module, model_name, allowed_mismatched_params=[])
    elif hasattr(bridge, "load_weights"):
        bridge.load_weights(module, model_name)
    else:  # pragma: no cover
        raise RuntimeError("Bridge object lacks weight loading APIs")
    load_megatron_model_to_gpu(module, load_grad=False)

    forward_fn = get_mcore_forward_no_padding_fn(hf_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "module": module,
        "hf_config": hf_config,
        "forward_fn": forward_fn,
        "bridge": bridge,
        "provider": provider,
        "device": device,
    }


def infer_log_probs_batch_megatron(state: dict, sequences, pad_id: int):
    """
    Compute logprobs with Megatron-LM using Megatron-Bridge.
    Returns padded logprobs tensor and per-sequence lengths.
    """
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    if not sequences:
        return torch.empty((0, 0), dtype=torch.float32), []

    module = state["module"]
    if isinstance(module, list):
        module = module[0]
    forward_fn = state["forward_fn"]
    device = state["device"]

    nested_ids = sequences if hasattr(sequences, "offsets") else torch.nested.nested_tensor(
        [torch.tensor(seq, dtype=torch.long, device=device) for seq in sequences],
        layout=torch.jagged,
    )
    # Per-token temperature to match expected nested layout
    temperature = torch.nested.nested_tensor(
        [torch.ones(len(seq), device=device, dtype=torch.float32) for seq in sequences],
        layout=torch.jagged,
    )

    def logits_processor(logits, label, temperature):
        temp = temperature.to(logits.dtype)
        temp = torch.clamp(temp, min=1e-8)
        logits = logits.float()
        logits.div_(temp.unsqueeze(dim=-1))
        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
        return {"log_probs": log_probs}

    outputs = forward_fn(
        model=module,
        input_ids=nested_ids,
        multi_modal_inputs={},
        logits_processor=logits_processor,
        logits_processor_args={"label": nested_ids, "temperature": temperature},
        pad_token_id=pad_id,
        data_format="bshd",
    )

    log_probs = outputs["log_probs"]
    lens = log_probs.offsets().diff().tolist()
    lp_padded = log_probs.to_padded_tensor(float("nan"))
    return lp_padded, lens


def cleanup_megatron_prefill(state: dict | None):
    if not state:
        return
    try:
        del state["module"]
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------- plotting -------------------------


def plot_correlation(eng_logp, prefill_logp, out_png, log_space=False):
    p_min, p_max = (-40, 0) if log_space else (0, 1)
    X = (eng_logp if log_space else eng_logp.exp()).float().cpu().numpy()
    Y = (prefill_logp if log_space else prefill_logp.exp()).float().cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [4, 2]})
    axes[0].set_aspect("equal")
    axes[0].set_xlim(p_min, p_max)
    axes[0].set_ylim(p_min, p_max)
    axes[1].set_xlim(p_min, p_max)
    hist, xe, ye = np.histogram2d(X, Y, bins=100, range=[[p_min, p_max], [p_min, p_max]], density=False)
    hist = np.log(hist + 1e-10)
    Xm, Ym = np.meshgrid(xe[:-1], ye[:-1])
    im = axes[0].pcolormesh(Xm, Ym, hist.T, shading="auto")
    axes[0].plot([p_min, p_max], [p_min, p_max], linestyle="--", linewidth=1)
    axes[0].set_xlabel("Engine " + ("log-prob" if log_space else "probability"))
    axes[0].set_ylabel("Prefill " + ("log-prob" if log_space else "probability"))
    fig.colorbar(im, ax=axes[0], label="Log Frequency")
    hx, xe1 = np.histogram(X, bins=100, range=[p_min, p_max], density=True)
    hy, ye1 = np.histogram(Y, bins=100, range=[p_min, p_max], density=True)
    axes[1].plot(xe1[:-1], np.log(hx + 1e-12), label="Engine")
    axes[1].plot(ye1[:-1], np.log(hy + 1e-12), label="Prefill")
    axes[1].legend()
    axes[1].set_ylabel("Log Density")
    axes[1].set_xlabel("log-prob" if log_space else "probability")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_sample_prob_diff(prefill_logp, eng_logp, out_png):
    diff = prefill_logp.exp() - eng_logp.exp()
    xs = np.arange(len(diff))
    plt.figure(figsize=(10, 4))
    plt.plot(xs, diff.cpu().numpy())
    plt.xlabel("Response token index")
    plt.ylabel("Δ prob (Prefill - Engine)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ------------------------- engines -------------------------


def run_vllm(
    prompt_ids_list,
    model,
    batch_size,
    max_new_tokens,
    seed=0,
    use_inductor=False,
    enforce_eager=False,
    return_routing=False,
    dtype=torch.bfloat16,
):
    # NOTE: script not valid for temp != 1.0, vllm needs a patch bc logprobs are returned before sampling
    sp = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=int(max_new_tokens),
        logprobs=0,
        detokenize=True,
        seed=seed,
        # return_expert_routing=True,
    )

    # TODO: enable routing again
    # routing = []
    prompt_ids, gen_ids, gen_logp, texts = [], [], [], []
    for i in tqdm(range(0, len(prompt_ids_list), batch_size), desc="vLLM generation"):
        batch_prompts = prompt_ids_list[i : i + batch_size]
        # vLLM expects a list of PromptInputs; use tokenized prompts schema
        batch_inputs = [{"prompt_token_ids": ids} for ids in batch_prompts]
        outs = model.generate(batch_inputs, sampling_params=sp)
        # breakpoint()
        for pid_sent, o in zip(batch_prompts, outs, strict=False):
            sample = o.outputs[0]
            # breakpoint()
            p_ids = list(pid_sent)
            g_ids = list(sample.token_ids)

            if sample.logprobs is None:
                raise RuntimeError("vLLM returned no logprobs; set SamplingParams.logprobs >= 1.")

            chosen_lp = []
            for t, tok_id in enumerate(g_ids):
                lp_dict = sample.logprobs[t]  # dict[token_id -> Logprob]
                lp_obj = lp_dict.get(tok_id)
                if lp_obj is None:
                    raise RuntimeError("Chosen token not in returned top-k logprobs (???)")
                chosen_lp.append(float(lp_obj.logprob))
            g_lp = torch.tensor(chosen_lp, dtype=torch.float32)

            prompt_ids.append(p_ids)
            gen_ids.append(g_ids)
            gen_logp.append(g_lp.cpu())
            texts.append(sample.text)

            # Disabled for now
            # routing.append(sample.expert_choices)
    return prompt_ids, gen_ids, gen_logp, texts  # , routing


def run_sglang(prompt_ids_list, model, batch_size, max_new_tokens, dtype=torch.bfloat16):
    from sglang.srt.entrypoints.engine import Engine

    # map torch dtype to sglang dtype string
    if dtype == torch.bfloat16:
        s_dtype = "bfloat16"
    elif dtype == torch.float16:
        s_dtype = "float16"
    elif dtype == torch.float32:
        s_dtype = "float32"
    else:
        s_dtype = "bfloat16"

    engine = Engine(
        model_path=model,
        dtype=s_dtype,
        tp_size=1,
        trust_remote_code=True,
        load_format="auto",
        log_level="INFO",
        max_running_requests=1024,
    )
    prompt_ids, gen_ids, gen_logp, texts = [], [], [], []
    for i in tqdm(range(0, len(prompt_ids_list), batch_size), desc="SGLang generation"):
        batch_ids = prompt_ids_list[i : i + batch_size]
        sp = {
            "n": 1,
            "max_new_tokens": int(max_new_tokens),
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "ignore_eos": False,
            "min_new_tokens": 0,
            "skip_special_tokens": True,
            "spaces_between_special_tokens": True,
        }
        outs = engine.generate(
            prompt=None, sampling_params=sp, return_logprob=True, input_ids=batch_ids, image_data=None
        )
        for pid, out in zip(batch_ids, outs, strict=False):
            tpl = out["meta_info"]["output_token_logprobs"]
            g_lp = torch.tensor([float(t[0]) for t in tpl], dtype=torch.float32)
            g_ii = [int(t[1]) for t in tpl]
            prompt_ids.append(pid)
            gen_ids.append(g_ii)
            gen_logp.append(g_lp)
            texts.append(out["text"])
    try:
        engine.shutdown()
    except Exception:
        pass
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt_ids, gen_ids, gen_logp, texts


def run_hf(prompt_ids_list, model, tokenizer, batch_size, max_new_tokens, seed=0, dtype=torch.bfloat16):
    prompt_ids, gen_ids, gen_logp, texts = [], [], [], []
    print(f"Running HF generation with batch size {batch_size}, num prompts {len(prompt_ids_list)}...")

    for start in tqdm(range(0, len(prompt_ids_list), batch_size), desc="HF generation"):
        batch_prompts = prompt_ids_list[start : start + batch_size]

        batch = tokenizer.pad({"input_ids": batch_prompts}, return_tensors="pt", padding_side="left")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        prompt_len = batch["input_ids"].shape[1]

        pad_id = tokenizer.pad_token_id

        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )

            sequences = out.sequences
            generations = sequences[:, prompt_len:]
            scores = out.scores

            attn_mask = generations != pad_id
            generation_lengths = torch.sum(attn_mask, dim=1)

            step_logprobs = []
            for step_logits, step_tokens in zip(scores, generations.T, strict=False):
                logsumexp = torch.logsumexp(step_logits, dim=-1)
                chosen = step_logits.gather(1, step_tokens.unsqueeze(1)).squeeze(1)
                step_logprobs.append(chosen - logsumexp)
                step_logprobs[-1] = step_logprobs[-1].cpu()

            batch_logprobs = torch.stack(step_logprobs, dim=1)

        for idx, (prompt, gen, lp_vec) in enumerate(
            zip(batch_prompts, generations.tolist(), batch_logprobs, strict=False)
        ):
            gen_len = int(generation_lengths[idx])
            prompt_ids.append(list(prompt))
            gen_ids.append(gen[:gen_len])
            gen_logp.append(lp_vec.cpu()[:gen_len])
            texts.append(tokenizer.decode(gen[:gen_len], skip_special_tokens=True))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gc.collect()
    return prompt_ids, gen_ids, gen_logp, texts


def build_hf_model(model_name, device, dtype=torch.bfloat16, attention_impl="flash_attention_2"):
    # Choose attention implementation conservatively for non-bf16
    if dtype == torch.float32:
        print("WARNING: using eager attention impl for HF because dtype == float32")
        attention_impl = "eager"

    print("using dtype: ", dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=attention_impl,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    return model


# ------------------------- main -------------------------


def to_list_ids(x):
    # Normalize to python list[int]
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, tuple | list):
        # Sometimes we get a list/tuple of arrays
        if x and hasattr(x[0], "__iter__"):
            return list(x[0])
        return list(x)
    return list(x)


def build_inputs(tokenizer, dataset, num_prompts, n):
    use_chat = hasattr(tokenizer, "apply_chat_template") and bool(getattr(tokenizer, "chat_template", None))
    user_texts = None
    ids_list = None

    if use_chat:
        user_texts = [f"{row['problem']}\n\n{SUFFIX}" for row in dataset]

        if num_prompts != -1:
            user_texts = user_texts[:num_prompts]
        user_texts = user_texts * n

        # Try chat template first; if it errors (e.g., broken template), fall back
        try:
            ids_list = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors=None,  # ensures python lists, not tensors
                )
                for text in user_texts
            ]
        except Exception:
            # Fallback to non chat-templated tokenization
            ids_list = [tokenizer(text, add_special_tokens=True)["input_ids"] for text in user_texts]
    else:
        user_texts = [f"Q: {row['problem']}\nA: " for row in dataset]

        if num_prompts != -1:
            user_texts = user_texts[:num_prompts]
        user_texts = user_texts * n

        # No chat template available → use vanilla tokenization
        ids_list = [tokenizer(text, add_special_tokens=True)["input_ids"] for text in user_texts]

    print(user_texts[0])
    print()

    return [to_list_ids(ids) for ids in ids_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang", "hf"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--prefill-batch-size",
        type=int,
        default=64,
        help="Batch size for prefill/scoring (keep small to avoid OOM, esp for HF)",
    )
    ap.add_argument("--time-chunk-size", type=int, default=128, help="Time chunk (tokens) for head projection")
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--vllm-tp", type=int, default=1, help="tensor parallel size for vLLM")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dataset", default="AI-MO/aimo-validation-aime")  # or MathArena/aime_2025
    ap.add_argument("--vllm-use-inductor", action="store_true")
    ap.add_argument("--vllm-enforce-eager", action="store_true", help="Disable CUDA graphs in vLLM (use eager mode)")
    ap.add_argument("--num-prompts", type=int, default=-1)
    ap.add_argument("--vllm-gpu-util", type=float, default=0.65, help="gpu util arg for vllm engine allocation")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument(
        "--profile-trace",
        action="store_false",
        help="Enable torch.profiler tracing; trace saved alongside other outputs.",
    )
    ap.add_argument(
        "--prefill-engine",
        choices=["hf", "vllm", "megatron"],
        default="hf",
        help="Engine to use for prefill/scoring forward pass (default: hf)",
    )
    ap.add_argument(
        "--hf-attention-impl",
        choices=["flash_attention_2", "eager"],
        default="flash_attention_2",
        help="attention implementation to use for transformers hf model (default: flash_attention_2)",
    )
    args = ap.parse_args()

    # Start end-to-end timer
    time_start_e2e = time.perf_counter()

    set_seed(args.seed)
    ensure_dir(args.out)

    # Resolve dtype from string aliases
    if args.dtype in {"f32", "fp32", "float32", "torch.float32"}:
        dtype = torch.float32
    elif args.dtype in {"f16", "fp16", "float16", "torch.float16"}:
        dtype = torch.float16
    elif args.dtype in {"bf16", "bfp16", "bfloat16", "torch.bfloat16"}:
        dtype = torch.bfloat16
    else:
        raise Exception(f"Unrecognized dtype {args.dtype}")

    # Use absolute out dir and pre-compute optional profiler trace path
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "profiler_trace.json") if args.profile_trace else None

    if args.vllm_use_inductor or args.vllm_enforce_eager:
        assert args.engine == "vllm", "vLLM options require --engine vllm"

    try:
        ds = load_dataset(args.dataset, split="train")
    except Exception:
        print("'train' split not found, using 'test'")
        ds = load_dataset(args.dataset, split="test")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build chat prompts using the model's chat template (if it has one, otherwise just rawdog it)
    chat_prompt_ids = build_inputs(tokenizer=tokenizer, dataset=ds, n=args.n, num_prompts=args.num_prompts)

    # ------------------------------ Generation ------------------------------

    llm_model = None
    hf_model = None
    megatron_prefill_state = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        assert torch.cuda.device_count() >= 1, "No visible CUDA devices"

    if args.engine == "vllm":
        # device defaults to CUDA_VISIBLE_DEVICES if available
        llm_kwargs = {
            "model": args.model,
            "dtype": dtype,
            "tensor_parallel_size": args.vllm_tp,
            "trust_remote_code": True,
            "enforce_eager": args.vllm_enforce_eager,
            # "max_seq_len_to_capture": args.max_new_tokens,
            "gpu_memory_utilization": args.vllm_gpu_util,
            "max_model_len": args.max_new_tokens + 1 + max(len(p) for p in chat_prompt_ids) if chat_prompt_ids else 0,
        }
        if args.vllm_use_inductor:
            llm_kwargs["compilation_config"] = {"use_inductor": True}

        try:
            llm_model = LLM(**llm_kwargs)
        except StopIteration:
            # Older vLLM versions don't support compilation_config
            if "compilation_config" in llm_kwargs:
                llm_kwargs.pop("compilation_config", None)
                llm_model = LLM(**llm_kwargs)
            else:
                raise

    if args.engine == "hf":
        hf_model = build_hf_model(args.model, dtype=dtype, device=device, attention_impl=args.hf_attention_impl)

    # Start generation timer
    time_start_gen = time.perf_counter()

    if args.engine == "vllm":
        p_ids, g_ids, g_lp, texts = run_with_optional_profiler(
            trace_path,
            run_vllm,
            chat_prompt_ids,
            llm_model,
            args.batch_size,
            args.max_new_tokens,
            seed=args.seed,
            use_inductor=args.vllm_use_inductor,
            enforce_eager=args.vllm_enforce_eager,
            dtype=dtype,
        )
    elif args.engine == "sglang":
        p_ids, g_ids, g_lp, texts = run_with_optional_profiler(
            trace_path, run_sglang, chat_prompt_ids, args.model, args.batch_size, args.max_new_tokens, dtype=dtype
        )
    elif args.engine == "hf":
        p_ids, g_ids, g_lp, texts = run_with_optional_profiler(
            trace_path,
            run_hf,
            chat_prompt_ids,
            hf_model,
            tokenizer,
            args.batch_size,
            args.max_new_tokens,
            seed=args.seed,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Invalid engine: {args.engine}")

    # End generation timer
    time_end_gen = time.perf_counter()
    generation_duration = time_end_gen - time_start_gen
    print(f"\n[Timing] Generation duration: {generation_duration:.2f} seconds")

    # Clear up engine memory if not using vllm anymore
    if args.engine == "vllm" and args.prefill_engine != "vllm":
        print("DEALLOCATING vllm engine")
        del llm_model

    if args.engine == "hf" and args.prefill_engine != "hf":
        print("DEALLOCATING hf engine")
        del hf_model

    gc.collect()

    # somewhat safe to call regardless of reuse
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sequences = [pi + gi for pi, gi in zip(p_ids, g_ids, strict=False)]

    # ---------- Prefill/scoring (length-bucket to cut padding) ----------

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # initialize hf prefill model if needed
    if args.prefill_engine == "hf" and hf_model is None:
        hf_model = build_hf_model(model_name=args.model, dtype=dtype, device=device)

    # initialize vllm prefill model if needed
    if args.prefill_engine == "vllm" and llm_model is None:
        llm_model = LLM(
            model=args.model,
            dtype=dtype,
            tensor_parallel_size=args.vllm_tp,
            trust_remote_code=True,
            enforce_eager=args.vllm_enforce_eager,
            max_seq_len_to_capture=args.max_new_tokens,
            gpu_memory_utilization=args.vllm_gpu_util,
            max_model_len=args.max_new_tokens + 1 + max(len(p) for p in chat_prompt_ids) if chat_prompt_ids else 0,
            compilation_config={
                "use_inductor": args.vllm_use_inductor,
            },
        )
    if args.prefill_engine == "megatron" and megatron_prefill_state is None:
        megatron_prefill_state = build_megatron_prefill_model(model_name=args.model, dtype=dtype, seed=args.seed)

    # Sanity checks for consistency between engines
    if args.prefill_engine == "hf":
        V = hf_model.get_input_embeddings().weight.size(0)
        for idx, (pi, gi, elp) in enumerate(zip(p_ids, g_ids, g_lp, strict=False)):
            assert len(gi) == len(elp), (
                f"Engine IDs/logprobs length mismatch at sample {idx} (len(ids)={len(gi)} vs len(lp)={len(elp)})"
            )
            if (pi and max(pi) >= V) or (gi and max(gi) >= V):
                raise ValueError(f"Token id out of range at sample {idx}: vocab={V}, max_id={max(pi + gi)}")

        max_pos = getattr(hf_model.config, "max_position_embeddings", None)
        if max_pos is not None:
            too_long = [i for i, s in enumerate(sequences) if len(s) > max_pos]
            if len(too_long) > 0:
                warning = (
                    f"Warning: {len(too_long)} sequences exceed model "
                    f"max_position_embeddings={max_pos} and may be truncated or slow."
                )
                print(warning)

    # Start prefill/scoring timer
    time_start_prefill = time.perf_counter()

    idxs = list(range(len(sequences)))
    idxs.sort(key=lambda i: len(sequences[i]))  # shortest -> longest
    prefill_rows = [None] * len(sequences)

    if args.prefill_engine == "vllm":
        # Process in batches
        for start in range(0, len(sequences), args.prefill_batch_size):
            batch_idx = idxs[start : start + args.prefill_batch_size]
            seq_batch = [sequences[i] for i in batch_idx]

            lp_batch, lens_batch = infer_log_probs_batch_vllm(llm_model, sequences=seq_batch, batch_size=len(batch_idx))

            for j, (i_orig, L) in enumerate(zip(batch_idx, lens_batch, strict=False)):
                prefill_rows[i_orig] = lp_batch[j, : L - 1].detach().cpu()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elif args.prefill_engine == "hf":
        # Use HF for prefill/scoring (default)
        for start in tqdm(range(0, len(sequences), args.prefill_batch_size), desc="HF prefill/scoring"):
            batch_idx = idxs[start : start + args.prefill_batch_size]
            seq_batch = [sequences[i] for i in batch_idx]

            lp_batch, lens_batch = infer_log_probs_batch(
                hf_model, sequences=seq_batch, device=device, dtype=dtype, time_chunk=args.time_chunk_size
            )
            for j, (i_orig, L) in enumerate(zip(batch_idx, lens_batch, strict=False)):
                prefill_rows[i_orig] = lp_batch[j, : L - 1].detach().cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif args.prefill_engine == "megatron":
        for start in tqdm(range(0, len(sequences), args.prefill_batch_size), desc="Megatron prefill/scoring"):
            batch_idx = idxs[start : start + args.prefill_batch_size]
            seq_batch = [sequences[i] for i in batch_idx]

            lp_batch, lens_batch = infer_log_probs_batch_megatron(
                megatron_prefill_state, sequences=seq_batch, pad_id=pad_id
            )
            for j, (i_orig, L) in enumerate(zip(batch_idx, lens_batch, strict=False)):
                prefill_rows[i_orig] = lp_batch[j, : L - 1].detach().cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Invalid prefill engine: {args.prefill_engine}")

    # Clean up reused engines
    if args.prefill_engine == "vllm":
        del llm_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.prefill_engine == "hf" and args.engine != "hf":
        del hf_model
    if args.prefill_engine == "megatron":
        cleanup_megatron_prefill(megatron_prefill_state)

    # End prefill/scoring timer
    time_end_prefill = time.perf_counter()
    prefill_duration = time_end_prefill - time_start_prefill
    print(f"[Timing] Prefill/scoring duration: {prefill_duration:.2f} seconds")

    # Align engine vs HF on generated tokens
    eng_slices, prefill_slices, slice_indices = [], [], []
    for idx, (pi, gi, eng_lp) in enumerate(zip(p_ids, g_ids, g_lp, strict=False)):
        Lp, Lg = len(pi), len(gi)
        prefill_row = prefill_rows[idx]
        start = max(Lp - 1, 0)
        prefill_slice = prefill_row[start : start + Lg]
        m = min(len(eng_lp), len(prefill_slice))
        if m > 0:
            eng_slices.append(eng_lp[:m])
            prefill_slices.append(prefill_slice[:m])
            slice_indices.append(idx)

    eng_all = torch.cat(eng_slices) if eng_slices else torch.empty(0)
    prefill_all = torch.cat(prefill_slices) if prefill_slices else torch.empty(0)

    # Save raw per-item
    # Re-ensure output directory and use absolute path to avoid cwd ambiguity
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "engine_outputs.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for pi, gi, elp, txt in zip(p_ids, g_ids, g_lp, texts, strict=False):
            f.write(
                json.dumps(
                    {"prompt_ids": pi, "gen_ids": gi, "gen_logprobs": [float(x) for x in elp.tolist()], "text": txt}
                )
                + "\n"
            )

    # Summary metrics + plots
    if len(eng_all) > 0:
        e = eng_all.float().cpu().numpy()
        h = prefill_all.float().cpu().numpy()
        mae = float(np.mean(np.abs(h - e)))
        rmse = float(np.sqrt(np.mean((h - e) ** 2)))
        corr = float(np.corrcoef(h, e)[0, 1]) if len(h) > 1 else float("nan")

        # Additional metrics in probability space
        lnp = np.clip(h.astype(np.float64), -80.0, 0.0)
        lnq = np.clip(e.astype(np.float64), -80.0, 0.0)
        p_raw = np.exp(lnp)
        q_raw = np.exp(lnq)
        diff = p_raw - q_raw
        rollout_probs_diff_mean = float(np.mean(diff))
        rollout_probs_diff_std = float(np.std(diff))

        # Bernoulli KL between chosen-token probabilities (clip to avoid log(0))
        eps = 1e-12
        p = np.clip(p_raw, eps, 1.0 - eps)
        q = np.clip(q_raw, eps, 1.0 - eps)
        kl_vals = p * (np.log(p) - np.log(q)) + (1.0 - p) * (np.log1p(-p) - np.log1p(-q))
        kl_divergence = float(np.mean(kl_vals))

        # Completion length stats (in tokens)
        completion_lengths = np.array([len(gen_ids) for gen_ids in g_ids], dtype=np.int64)
        avg_completion_length = float(np.mean(completion_lengths)) if completion_lengths.size > 0 else 0.0
        min_completion_length = int(np.min(completion_lengths)) if completion_lengths.size > 0 else 0
        max_completion_length = int(np.max(completion_lengths)) if completion_lengths.size > 0 else 0

        with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mae_logprob": mae,
                    "rmse_logprob": rmse,
                    "pearson_r": corr,
                    "kl_divergence": kl_divergence,
                    "rollout_probs_diff_mean": rollout_probs_diff_mean,
                    "rollout_probs_diff_std": rollout_probs_diff_std,
                    "avg_completion_length": avg_completion_length,
                    "min_completion_length": min_completion_length,
                    "max_completion_length": max_completion_length,
                    "n_tokens": int(len(h)),
                    "timing": {
                        "generation_seconds": round(generation_duration, 2),
                        "prefill_scoring_seconds": round(prefill_duration, 2),
                        "end_to_end_seconds": 0.0,  # Will be updated below
                    },
                },
                f,
                indent=2,
            )

        plot_correlation(eng_all, prefill_all, os.path.join(out_dir, "diff_raw.png"), log_space=False)
        plot_correlation(eng_all, prefill_all, os.path.join(out_dir, "diff_log.png"), log_space=True)

        j = len(eng_slices) // 2 if eng_slices else 0
        if eng_slices:
            plot_sample_prob_diff(prefill_slices[j], eng_slices[j], os.path.join(out_dir, "sample_prob_diff.png"))
            orig_j = slice_indices[j]
            with open(os.path.join(out_dir, "sample_completion.txt"), "w", encoding="utf-8") as f:
                f.write(texts[orig_j])
            toks = tokenizer.convert_ids_to_tokens(g_ids[orig_j][: len(eng_slices[j])])
            with open(os.path.join(out_dir, "sample_token_diffs.csv"), "w", newline="", encoding="utf-8") as cf:
                w = csv.writer(cf)
                w.writerow(["idx", "token", "prob_hf", "prob_engine", "delta"])
                for i, (hlp, elp, t) in enumerate(zip(prefill_slices[j], eng_slices[j], toks, strict=False)):
                    ph, pe = float(hlp.exp()), float(elp.exp())
                    w.writerow([i, t, ph, pe, ph - pe])

            j_longest = max(range(len(slice_indices)), key=lambda k: len(g_ids[slice_indices[k]]))
            orig_longest = slice_indices[j_longest]
            plot_sample_prob_diff(
                prefill_slices[j_longest], eng_slices[j_longest], os.path.join(out_dir, "longest_prob_diff.png")
            )
            with open(os.path.join(out_dir, "longest_completion.txt"), "w", encoding="utf-8") as f:
                f.write(texts[orig_longest])
            toks = tokenizer.convert_ids_to_tokens(g_ids[orig_longest][: len(eng_slices[j_longest])])
            with open(os.path.join(out_dir, "longest_token_diffs.csv"), "w", newline="", encoding="utf-8") as cf:
                w = csv.writer(cf)
                w.writerow(["idx", "token", "prob_hf", "prob_engine", "delta"])
                for i, (hlp, elp, t) in enumerate(
                    zip(prefill_slices[j_longest], eng_slices[j_longest], toks, strict=False)
                ):
                    ph, pe = float(hlp.exp()), float(elp.exp())
                    w.writerow([i, t, ph, pe, ph - pe])

    # End end-to-end timer and report all timings
    time_end_e2e = time.perf_counter()
    e2e_duration = time_end_e2e - time_start_e2e
    print(f"[Timing] End-to-end duration: {e2e_duration:.2f} seconds")

    # Update timing in summary_metrics.json if it exists
    summary_metrics_path = os.path.join(out_dir, "summary_metrics.json")
    if os.path.exists(summary_metrics_path):
        with open(summary_metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["timing"]["end_to_end_seconds"] = round(e2e_duration, 2)
        with open(summary_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # Clean up torch.distributed process group if initialized (avoids NCCL warning)
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
