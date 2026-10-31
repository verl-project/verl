# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# there is some bug in mcore 0.12, so we need to patch it
# 1. `get_query_key_value_tensors` in `multi_latent_attention.py` works wrong when packed_seq_params is not None


def pure_torch_hadamard_transform(x, scale=1.0):
    """Fast Walsh-Hadamard transform along the last dim (size must be 2**k).

    Vectorized butterfly accumulated in fp32, numerically equivalent to
    ``F.linear(x, scipy.linalg.hadamard(dim)) * scale`` and bit-for-bit identical to
    the Dao-AILab CUDA kernel it stands in for.
    """
    import torch

    n = x.shape[-1]
    if n < 1 or n & (n - 1) != 0:
        raise ValueError(f"hadamard_transform requires last dim to be a power of 2, got {n}")
    orig_dtype = x.dtype
    orig_shape = x.shape
    # Accumulate in fp32 for numerical stability, cast back at the end.
    y = x.to(torch.float32).reshape(-1, n)
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    y = y * scale
    return y.reshape(orig_shape).to(orig_dtype)


def apply_fast_hadamard_transform_shim():
    """Provide a pure-torch ``fast_hadamard_transform`` when the CUDA package is absent.

    DeepSeek-V4 / V3.2 sparse-attention (DSA) in Megatron-LM does
    ``from fast_hadamard_transform import hadamard_transform`` at import time and
    asserts it is not None inside the indexer's ``rotate_activation``. The upstream
    Dao-AILab package only ships nvcc kernels and cannot be built on ROCm, so that
    import yields ``None`` there and every DSA forward crashes.

    Register ``pure_torch_hadamard_transform`` under the real import name so any
    importer picks it up, and back-fill modules that already captured ``None``
    (e.g. Megatron's ``dsa`` module at import time). This is a no-op once the name
    imports, be it the real package or the stub an earlier call installed.
    """
    import importlib
    import logging
    import sys
    import types

    # Catch every exception, not just ImportError: a CUDA extension built against
    # another toolkit fails to load with OSError. An import that goes through means
    # every importer of the name resolves to that same module, so none of them can
    # be left holding a None binding.
    try:
        importlib.import_module("fast_hadamard_transform")
        return
    except Exception:
        pass

    module = types.ModuleType("fast_hadamard_transform")
    module.hadamard_transform = pure_torch_hadamard_transform
    sys.modules["fast_hadamard_transform"] = module

    # Back-fill modules that already ran `from fast_hadamard_transform import
    # hadamard_transform` and captured None (e.g. Megatron's dsa.py at import).
    # Read `__dict__` directly: `getattr` would trigger PEP-562 module-level
    # `__getattr__` hooks, which can raise or force lazy imports.
    for mod in list(sys.modules.values()):
        mod_dict = getattr(mod, "__dict__", None)
        if mod_dict is not None and mod_dict.get("hadamard_transform", "keep") is None:
            mod_dict["hadamard_transform"] = pure_torch_hadamard_transform

    logging.getLogger(__name__).warning(
        "fast_hadamard_transform is unavailable; falling back to a pure-torch Fast "
        "Walsh-Hadamard transform. Results match the CUDA kernel but DSA forward will be slower."
    )


class _MissingFlashAttnCute:
    """Meta-path finder that reports ``flash_attn.cute`` as an absent module."""

    _PREFIX = "flash_attn.cute"

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._PREFIX or fullname.startswith(self._PREFIX + "."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def neutralize_broken_flash_attn_cute():
    """Hide ``flash_attn.cute`` when importing it raises anything but ImportError.

    Megatron-LM probes FA4 with ``from flash_attn.cute import flash_attn_varlen_func``
    guarded by ``except ImportError``. flash-attn 2.8.3 ships that subpackage but
    declares no ``nvidia-cutlass-dsl`` dependency, and the code targets the 4.5.x
    API, so against the 4.6.x sglang pins it raises ``AttributeError`` on
    ``cute.core.ThrMma``. That escapes the guard and aborts
    ``import megatron.core.transformer.attention`` — i.e. every mcore entry point.

    Probe once and only install the finder when the subpackage is actually broken,
    so a build whose FA4 does import keeps it. Must run before anything imports
    megatron's GPT layer specs.
    """
    import importlib
    import importlib.util
    import logging
    import sys

    if any(isinstance(finder, _MissingFlashAttnCute) for finder in sys.meta_path):
        return
    try:
        if importlib.util.find_spec("flash_attn.cute") is None:
            return
    except Exception:
        return  # flash-attn absent or unimportable; the probe already sees ImportError

    try:
        importlib.import_module("flash_attn.cute")
    except ImportError:
        return  # already the exception every probe expects
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
    else:
        return  # FA4 imports here; leave it usable

    # A failed import can leave half-initialized submodules behind; drop them so
    # the finder below is what every later import reaches.
    for name in [n for n in sys.modules if n == "flash_attn.cute" or n.startswith("flash_attn.cute.")]:
        del sys.modules[name]
    sys.meta_path.insert(0, _MissingFlashAttnCute())

    logging.getLogger(__name__).warning(
        "flash_attn.cute (FlashAttention-4) is unusable and is being hidden so optional "
        "FA4 probes fail with ImportError as they expect; FlashAttention-2 is unaffected. "
        "Cause: %s",
        reason,
    )


def apply_patch():
    # DeepSeek sparse-attention (DSA) needs ``fast_hadamard_transform``, which
    # cannot be built on ROCm (its setup requires nvcc). Install a pure-torch
    # fallback from the central mcore patch entry so every DSA importer picks it
    # up without engine-specific wiring. megatron_utils.make_megatron_module
    # calls this after model creation, so the shim also back-fills importers.
    apply_fast_hadamard_transform_shim()

    import megatron.core
    import torch
    import torch.nn.functional as F
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.multi_latent_attention import (
        MLASelfAttention,
        MultiLatentAttention,
        apply_rotary_pos_emb,
        deprecate_inference_params,
        gather_from_sequence_parallel_region,
        gather_from_tensor_model_parallel_region,
        scatter_to_sequence_parallel_region,
    )
    from packaging import version

    mcore_ge_013 = version.parse(megatron.core.__version__) >= version.parse("0.13.0")
    mcore_ge_0162 = version.parse(megatron.core.__version__) >= version.parse("0.16.2")

    def patch_get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        if self.config.rope_type == "rope":
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            try:
                # In case of TypeError: RotaryEmbedding.forward() got an unexpected keyword argument 'packed_seq'
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
            except TypeError:
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len)

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)

            q_compressed = self.q_layernorm(q_compressed)
        else:
            q_compressed = hidden_states

        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================
        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # hidden_states:[s, b, 2048], q: [s, b, n * 192]
                q, _ = self.linear_q_proj(q_compressed)

            q_len, bsz, _ = q.size()

            # q: [s, b, n, 192]
            q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

            # kv: [s, b, 2048]
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            # kv: [s, b, n, 256]
            kv = kv.view(
                q_len,
                bsz,
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            cp_size = parallel_state.get_context_parallel_world_size()
            if inference_context is not None:
                # add offset to the sequence start for inference
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            elif packed_seq_params is None or cp_size == 1:
                # Shorten rotary_pos_emb to the sequence length when inference_params
                # is not provided. This makes sure we can run forward directly with
                # any sequence length. During training, the sequence length is always
                # the full rotary_pos_emb length, except for sequence packing + CP.
                # When sequence packing and context parallel are both enabled, the
                # position embedding will not split rotary_pos_emb, so it may exceed
                # the sequence length on this CP rank, but we need the full rotary_pos_emb
                # to cover the full sequence, so we do not shorten it here.
                rotary_pos_emb = rotary_pos_emb[0:q_len]

            # [s, b, 64] -> [s, b, 1, 64]
            k_pos_emb = torch.unsqueeze(k_pos_emb, 2)

            # q: [s, b, n, 128], q_pos_emb: [s, b, n, 64]
            q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

            # k_no_pe: [s, b, n, 128], value: [s, b, n, 128]
            k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                q_pos_emb = q_pos_emb.squeeze(1)
                k_pos_emb = k_pos_emb.squeeze(1)
                q_no_pe = q_no_pe.squeeze(1)
                k_no_pe = k_no_pe.squeeze(1)
                value = value.squeeze(1)
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            # q_pos_emb: [s, b, n, 64], k_pos_emb:[s, b, 1, 64]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=mscale,
            )
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
            )

            # query: [s, b, n, 192]
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
            if packed_seq_params is not None:
                k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
            else:
                # key: [s, b, n, 192]
                k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return query, key, value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)

        return query, key, value

    def patch_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """Forward pass for multi-latent attention"""
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert rotary_pos_cos is None and rotary_pos_sin is None, "MLA does not support Flash Decoding"

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        qkv = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )
        query, key, value = qkv[:3]
        q_compressed = None
        # kv_compressed = None
        if len(qkv) > 4:
            q_compressed = qkv[3]
            # kv_compressed = qkv[4]

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        if mcore_ge_013:
            query, key, value, _, attn_mask_type, _ = self._adjust_key_value_for_inference(
                inference_context, query, key, value, rotary_pos_emb=None
            )
        else:
            query, key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
                inference_context, query, key, value, rotary_pos_emb=None
            )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        orig_v_dim = value.shape[-1] if value is not None else None
        thd_packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        need_v_pad = (
            thd_packed_seq
            and getattr(self.config, "experimental_attention_variant", None) is None
            and value is not None
            and query.shape[-1] != orig_v_dim
        )
        if need_v_pad:
            # Pad V so THD attention can run when Q/V head dims differ.
            value = F.pad(value, [0, query.shape[-1] - orig_v_dim])
            self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            extra_kwargs = {}
            if getattr(self.config, "experimental_attention_variant", None) == "dsa":
                # For dsa we need to pass in the original hidden states and the compressed
                # query representation.
                extra_kwargs["x"] = hidden_states
                extra_kwargs["qr"] = q_compressed
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
                **extra_kwargs,
            )
        if thd_packed_seq:
            if need_v_pad:
                if core_attn_out.ndim == 2:
                    core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-1], -1, value.shape[-1])
                core_attn_out = core_attn_out[..., :orig_v_dim]
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias

    # This patch targets mcore 0.12 MLA behavior only.
    # For newer mcore, upstream MLA already has packed-seq + CP handling and
    # overriding it with the legacy implementation can break RoPE shapes.
    if not mcore_ge_013:
        MLASelfAttention.get_query_key_value_tensors = patch_get_query_key_value_tensors

    if not mcore_ge_0162:
        MultiLatentAttention.forward = patch_forward


def apply_patch_megatron_v012_with_torch_v28_v29() -> None:
    # Error due to missing serialization_format in _write_item of megatron v012;
    # resolved by using megatron v013's implementation.
    import inspect
    import logging
    import os
    from pathlib import Path

    import megatron.core
    import torch
    from megatron.core.dist_checkpointing.strategies.async_utils import _disable_gc
    from megatron.core.dist_checkpointing.strategies.filesystem_async import _process_memory
    from packaging import version
    from torch import multiprocessing as mp
    from torch.distributed.checkpoint.filesystem import _write_item

    if (
        version.parse(torch.__version__).base_version not in ("2.8.0", "2.9.0")
        or version.parse(megatron.core.__version__).base_version != "0.12.1"
    ):
        return

    WriteBucket = tuple[Path, str, tuple[list, list]]

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_patch(
        transform_list,
        local_proc_idx: int,
        write_bucket: WriteBucket,
        results_queue: mp.SimpleQueue,
        count_queue: mp.JoinableQueue,
        use_fsync: bool,
        **kwargs,
    ) -> None:
        """
        Performs actual data saving to storage.

        Args:
            local_proc_idx (int): index of a local process that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (mp.Queue): queue to return the write results
                to the proxy checkpoint process.
            count_queue (mp.JoinableQueue): queue to marks worker task as completed
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None, the write result are put into the `queue`
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"{local_proc_idx} started")
        mem_before = _process_memory()
        use_msc = kwargs.get("use_msc", False)
        local_results = []
        try:
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            extra_kwargs = {}
            if "serialization_format" in inspect.signature(_write_item).parameters:
                from torch.distributed.checkpoint.filesystem import SerializationFormat

                extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE
            if use_msc:
                import multistorageclient as msc

                open_file = msc.open
            else:
                open_file = open
            with open_file(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    local_results.append(
                        _write_item(*transform_list, stream, data, write_item, storage_key, **extra_kwargs)
                    )

                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu
                    local_results.append(
                        _write_item(*transform_list, stream, tensor, write_item, storage_key, **extra_kwargs)
                    )

                if use_fsync:
                    if use_msc:
                        stream.fsync()
                    else:
                        os.fsync(stream.fileno())
            local_output = (local_proc_idx, local_results)
        except Exception as e:
            logger.debug(f"{local_proc_idx} failed")
            local_output = (local_proc_idx, e)  # type: ignore[assignment]

        results_queue.put(local_output)
        # Signal this process is done.
        count_queue.get()
        count_queue.task_done()

        mem_after = _process_memory()
        logger.debug(f"{local_proc_idx} consumed: {mem_after - mem_before}, before: {mem_before}, after: {mem_after}")

    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

    FileSystemWriterAsync.write_preloaded_data = write_preloaded_data_patch


def apply_mtp_inference_patch():
    from megatron.core.models.gpt.gpt_model import GPTModel

    _original_postprocess = GPTModel._postprocess

    def _patched(self, *args, **kwargs):
        original_mtp_num_layers = self.config.mtp_num_layers
        if not self.config.mtp_num_layers:
            self.config.mtp_num_layers = None
        try:
            return _original_postprocess(self, *args, **kwargs)
        finally:
            self.config.mtp_num_layers = original_mtp_num_layers

    GPTModel._postprocess = _patched


# When using checkpoint + MoE models (like Qwen3-30B-A3B and Qwen3-VL-30B-A3B),
# input tensors and their grads will stay in gpu memory after forward_backward completes.
# see https://github.com/NVIDIA/Megatron-LM/pull/3267
def apply_patch_megatron_recomputation_backward():
    import megatron.core.tensor_parallel.random as rd
    import torch

    _fork_rng = rd._fork_rng
    _set_all_rng_states = rd._set_all_rng_states
    detach_variable = rd.detach_variable
    gather_split_1d_tensor = rd.gather_split_1d_tensor
    safely_set_viewless_tensor_data = rd.safely_set_viewless_tensor_data

    @staticmethod
    def patch_backward(ctx, *args):
        """Backward pass."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))

        with _fork_rng():
            # Set the states to what it used to be before the forward pass.
            _set_all_rng_states(*ctx.rng_states)

            # Compute the forward pass.
            detached_inputs = detach_variable(inputs)

            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # filter out non tensor outputs for backward pass
        outputs, args = zip(
            *filter(lambda x: torch.is_tensor(x[0]) and x[0].requires_grad, zip(outputs, args, strict=False)),
            strict=False,
        )
        torch.autograd.backward(outputs, args)
        # Clone grads to return
        grads = tuple(
            inp.grad.clone()
            if isinstance(inp, torch.Tensor) and inp.grad is not None
            else inp.grad
            if isinstance(inp, torch.Tensor)
            else inp
            for inp in detached_inputs
        )
        cur_stream = torch.cuda.current_stream()
        # Release saved input/grad tensors (MoE residual-memory leak fix, commit 04df110c).
        # Skip MTP layer checkpoints: their saved ``hidden_states`` aliases the decoder
        # output (a ``torch.chunk`` view), and MTP backward runs before the decoder
        # backward, so ``resize_(0)`` would truncate storage the decoder still needs
        # -> async CUDA illegal-memory-access.
        is_mtp_checkpoint = False
        run_fn = getattr(ctx, "run_function", None)
        for cell in getattr(run_fn, "__closure__", None) or ():
            try:
                obj = cell.cell_contents
            except ValueError:
                continue
            if obj.__class__.__name__ == "MultiTokenPredictionLayer":
                is_mtp_checkpoint = True
                break
        if not is_mtp_checkpoint:
            for t in detached_inputs:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    t.record_stream(cur_stream)
                    t.untyped_storage().resize_(0)
                    if t.grad is not None:
                        t.grad.record_stream(cur_stream)
                        t.grad.untyped_storage().resize_(0)
        # ctx.saved_tensors = None
        return (None, None) + grads

    rd.CheckpointFunction.backward = patch_backward


# ---------------------------------------------------------------------------
# Megatron-Core optimizer compatibility for quantized checkpoints with the
# precision-aware optimizer (DeepSeek-V4 and similar FP8 / MXFP4 models).
#
# Two upstream fixes are pending; until they land, verl applies runtime
# equivalents once per process (set VERL_MEGATRON_OPTIMIZER_COMPAT=0 to skip):
#
# 1. https://github.com/NVIDIA/Megatron-LM/pull/8245 -- for quantized model
#    params without ``get_high_precision_init_val``, ``DistributedOptimizer``
#    builds the fp32 main-param shard as ``model_param.float().view(-1)[a:b]``
#    under autograd, i.e. a non-leaf view with ``requires_grad=True``.  The
#    default optimizer path never validates it, but ``optimizer_cpu_offload=True``
#    rebuilds the HybridDeviceOptimizer through ``torch.optim.Optimizer.__init__``,
#    whose ``add_param_group`` raises ``can't optimize a non-leaf Tensor``.
#    Building the groups under ``torch.no_grad()`` yields grad_fn-free shards;
#    the recorded grad_fn was never used (Megatron copies grads into the main
#    params by hand).
# 2. https://github.com/NVIDIA/Megatron-LM/pull/7279 -- ``get_grad_norm_fp32`` /
#    ``clip_grad_by_total_norm_fp32`` hand one list of gradients to a single
#    multi-tensor kernel that is dispatched on the dtype of the first tensor.
#    With ``use_precision_aware_optimizer`` + bf16 main grads, bf16 params expose
#    bf16 decoupled grads while the model's native fp32 params (DSV4's hc
#    alpha/bias, ape, attn_sink) keep fp32 grads: the kernel reads past the
#    smaller elements and dies with an illegal memory access.  The replacement
#    runs the kernel once per dtype group and combines the partial norms; a
#    single-dtype list is forwarded to the original function unchanged.
# ---------------------------------------------------------------------------


def wrap_no_grad(cls, name: str, marker: str) -> bool:
    """Wrap ``cls.<name>`` so it runs under ``torch.no_grad()``.

    Handles plain functions, classmethods and staticmethods.  Returns False when
    the attribute is missing or already wrapped (``marker`` set on ``cls``).
    """
    import logging

    import torch

    if getattr(cls, marker, False):
        return False
    if name not in cls.__dict__:
        # Inherited (or renamed) upstream: nothing to wrap here.  Say so, because a silent skip would
        # look like a successful patch to anyone reading the logs.
        logging.getLogger(__name__).warning(
            "%s.%s is not defined on the class itself (inherited or renamed); no-grad wrap skipped", cls.__name__, name
        )
        return False
    raw = cls.__dict__[name]
    kind = type(raw) if isinstance(raw, (classmethod, staticmethod)) else None
    func = raw.__func__ if kind is not None else raw

    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    wrapped.__name__ = getattr(func, "__name__", name)
    wrapped.__doc__ = getattr(func, "__doc__", None)
    setattr(cls, name, kind(wrapped) if kind is not None else wrapped)
    setattr(cls, marker, True)
    return True


def apply_distributed_optimizer_no_grad_main_params_patch() -> bool:
    """Build ``DistributedOptimizer`` main-param groups under ``no_grad`` (Megatron-LM #8245)."""
    import logging

    try:
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    except ImportError:
        return False
    applied = wrap_no_grad(
        DistributedOptimizer, "_build_model_and_main_param_groups", "_verl_no_grad_main_params_patch"
    )
    if applied:
        # WARNING on purpose: Ray workers do not surface INFO, and users need to see that the patch is live.
        logging.getLogger(__name__).warning(
            "Patched DistributedOptimizer._build_model_and_main_param_groups to run under no_grad "
            "(Megatron-LM #8245: non-leaf fp32 main-param shards for quantized params)"
        )
    return applied


def make_dtype_safe_grad_fns(cg):
    """Return per-dtype-group versions of ``get_grad_norm_fp32`` / ``clip_grad_by_total_norm_fp32``.

    ``cg`` is ``megatron.core.optimizer.clip_grads`` (or any object exposing the same
    names); it is a parameter so the logic can be unit-tested without a GPU.
    """
    import logging

    import torch

    logger = logging.getLogger(__name__)
    orig_norm = cg.get_grad_norm_fp32
    orig_clip = cg.clip_grad_by_total_norm_fp32
    announced = {"norm": False, "clip": False}

    def _group_by_dtype(tensors):
        groups = {}
        for t in tensors:
            groups.setdefault(t.dtype, []).append(t)
        return groups

    def get_grad_norm_fp32(grads_for_norm, norm_type=2, grad_stats_parallel_group=None):
        if isinstance(grads_for_norm, torch.Tensor):
            grads_for_norm = [grads_for_norm]
        groups = _group_by_dtype(grads_for_norm)
        if len(groups) <= 1 or float(norm_type) != 2.0:
            return orig_norm(grads_for_norm, norm_type, grad_stats_parallel_group)
        if not announced["norm"]:
            logger.warning("dtype-safe grad norm active: groups=%s", {str(k): len(v) for k, v in groups.items()})
            announced["norm"] = True
        data_parallel_group = None
        for g in grads_for_norm:
            data_parallel_group = cg.get_data_parallel_group_if_dtensor(g, data_parallel_group)
        total = None
        for gs in groups.values():
            gs = [cg.to_local_if_dtensor(g) for g in gs]
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=gs[0].device)
            grad_norm, _ = cg.multi_tensor_applier(cg.l2_norm_impl, dummy_overflow_buf, [gs], False)
            sq = grad_norm.float() ** 2.0
            total = sq if total is None else total + sq
        if data_parallel_group:
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group)
        if cg.multi_tensor_scale_tensor_impl is not None:
            return total.pow(0.5)
        return total.item() ** 0.5

    def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad=False):
        grads = []
        for param in parameters:
            if use_decoupled_grad:
                if getattr(param, "decoupled_grad", None) is not None:
                    assert param.decoupled_grad.dtype in (torch.float32, torch.bfloat16)
                    grads.append(cg.to_local_if_dtensor(param.decoupled_grad).detach())
            elif param.grad is not None:
                grads.append(cg.to_local_if_dtensor(param.grad).detach())
        groups = _group_by_dtype(grads)
        if len(groups) <= 1:
            return orig_clip(parameters, max_norm, total_norm, use_decoupled_grad)
        if not announced["clip"]:
            logger.warning("dtype-safe grad clip active: groups=%s", {str(k): len(v) for k, v in groups.items()})
            announced["clip"] = True
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if isinstance(clip_coeff, torch.Tensor):
            clip_coeff.clamp_max_(1.0)
            assert cg.multi_tensor_scale_tensor_impl is not None, (
                "clip_coeff is a tensor but multi_tensor_scale_tensor is unavailable"
            )
            for gs in groups.values():
                dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=gs[0].device)
                cg.multi_tensor_applier(cg.multi_tensor_scale_tensor_impl, dummy_overflow_buf, [gs, gs], clip_coeff)
        elif clip_coeff < 1.0:
            for gs in groups.values():
                dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=gs[0].device)
                cg.multi_tensor_applier(cg.multi_tensor_scale_impl, dummy_overflow_buf, [gs, gs], clip_coeff)

    return get_grad_norm_fp32, clip_grad_by_total_norm_fp32


def apply_dtype_safe_grad_norm_patch() -> bool:
    """Install per-dtype-group grad norm / clipping in Megatron-Core (Megatron-LM #7279)."""
    import logging

    try:
        from megatron.core.optimizer import clip_grads as cg
        from megatron.core.optimizer import optimizer as opt_mod
    except ImportError:
        return False
    if getattr(cg, "_verl_dtype_safe_grad_patch", False):
        return False
    get_grad_norm_fp32, clip_grad_by_total_norm_fp32 = make_dtype_safe_grad_fns(cg)
    cg.get_grad_norm_fp32 = get_grad_norm_fp32
    cg.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32
    opt_mod.get_grad_norm_fp32 = get_grad_norm_fp32
    opt_mod.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32
    try:
        from megatron.core.optimizer import layer_wise_optimizer as lwo

        lwo.get_grad_norm_fp32 = get_grad_norm_fp32
    except ImportError:
        pass
    cg._verl_dtype_safe_grad_patch = True
    logging.getLogger(__name__).warning(
        "Installed dtype-safe get_grad_norm_fp32 / clip_grad_by_total_norm_fp32 (Megatron-LM #7279)"
    )
    return True


def apply_megatron_optimizer_compat_patches() -> None:
    """Apply both optimizer compatibility patches unless VERL_MEGATRON_OPTIMIZER_COMPAT=0."""
    import os

    if os.getenv("VERL_MEGATRON_OPTIMIZER_COMPAT", "1") == "0":
        return
    apply_distributed_optimizer_no_grad_main_params_patch()
    apply_dtype_safe_grad_norm_patch()
