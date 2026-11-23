"""Training RMSNorm and raw NoPE channels with a page-stride-aware writer."""
import torch
import torch_npu

def write_raw_nope_cache(impl, kv_no_split, slots, kv_cache):
    assert impl.qk_rope_head_dim == 64 and impl.kv_lora_rank == 512
    assert kv_no_split.dtype == torch.bfloat16
    assert not impl.enable_sparse_sfa_c8 and not impl.enable_sparse_sfa_turboquant
    latent, extra = kv_no_split.split([512, 64], dim=-1)
    latent = impl.kv_a_layernorm(latent)
    slots = slots[:latent.shape[0]]
    for values, cache in ((latent, kv_cache[0]), (extra, kv_cache[1])):
        assert cache.ndim == 4 and cache.shape[2] == 1 and cache.dtype == values.dtype
        # Write through a contiguous physical-storage view, with addresses
        # derived from the real plane offset and page/token strides. Flattening
        # the logical plane can copy a padded cache and lose in-place writes.
        tile = 64
        width = cache.shape[-1]
        assert cache.stride(-1) == 1 and width % tile == 0
        assert all(x % tile == 0 for x in (cache.storage_offset(), cache.stride(0), cache.stride(1)))
        assert cache.stride(1) == width
        storage_elements = cache.untyped_storage().nbytes() // cache.element_size()
        assert storage_elements % tile == 0
        storage = cache.as_strided((storage_elements // tile, tile), (tile, 1), storage_offset=0)
        logical = slots.to(torch.int64)
        first = (cache.storage_offset() // tile + (logical // cache.shape[1]) * (cache.stride(0) // tile)
                 + (logical % cache.shape[1]) * (cache.stride(1) // tile))
        addresses = first[:, None] + torch.arange(width // tile, device=slots.device, dtype=torch.int64)
        addresses = torch.where(logical[:, None] >= 0, addresses, -1).reshape(-1, 1)
        torch_npu.npu_scatter_nd_update_(storage, addresses, values.contiguous().reshape(-1, tile))
    return None, None
