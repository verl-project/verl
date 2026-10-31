"""Shim: the unified image has no flash-attn wheel; verl only imports bert_padding (pure torch).
Re-export verl's own torch implementation of those helpers. Attention kernels are NOT provided."""
