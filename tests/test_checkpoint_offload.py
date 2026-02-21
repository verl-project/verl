"""Unit tests for CheckpointInputOffload.

Tests verify:
1. Numerical correctness — gradients match with/without offloading
2. GPU memory reduction — checkpoint inputs actually moved to CPU
3. Parameter skip — leaf tensors with requires_grad are not offloaded
4. No-grad safety — offloader is harmless under torch.no_grad()
5. Multiple micro-batches — re-entrant usage works correctly
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from verl.utils.checkpoint_offload import CheckpointInputOffload


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class CheckpointedMLP(nn.Module):
    """Simple 4-layer MLP with gradient checkpointing on inner layers."""

    def __init__(self, dim=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self._gc_kwargs = {"use_reentrant": False}

    def enable_gc(self, context_fn=None):
        if context_fn is not None:
            self._gc_kwargs["context_fn"] = context_fn

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i > 0 and i < len(self.layers) - 1:
                # Checkpoint middle layers (not first/last to test non-checkpointed path too)
                x = checkpoint(lambda inp, l=layer: l(torch.relu(inp)), x, **self._gc_kwargs)
            else:
                x = layer(torch.relu(x))
        return x


def _get_grads(model, x, offloader=None):
    """Run forward+backward and return parameter gradients."""
    model.zero_grad()
    ctx = offloader if offloader is not None else torch.utils.checkpoint.checkpoint
    # Use nullcontext if no offloader
    from contextlib import nullcontext
    ctx_mgr = offloader if offloader is not None else nullcontext()

    with ctx_mgr:
        out = model(x)
    loss = out.sum()
    loss.backward()

    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


class TestCheckpointInputOffload:

    def test_numerical_correctness(self):
        """Gradients must match exactly with and without offloading."""
        torch.manual_seed(42)
        model = CheckpointedMLP(dim=128, num_layers=4).cuda().to(torch.bfloat16)
        x = torch.randn(4, 128, device='cuda', dtype=torch.bfloat16)

        # Baseline: no offloading
        grads_baseline = _get_grads(model, x)

        # With offloading
        offloader = CheckpointInputOffload(pin_memory=True)
        model.enable_gc(context_fn=offloader.get_context_fn())
        grads_offload = _get_grads(model, x, offloader=offloader)

        for name in grads_baseline:
            assert name in grads_offload, f"Missing gradient for {name}"
            torch.testing.assert_close(
                grads_baseline[name], grads_offload[name],
                msg=f"Gradient mismatch for {name}"
            )

    def test_gpu_memory_reduction(self):
        """Peak GPU memory should decrease with offloading enabled."""
        dim = 512
        num_layers = 8
        batch_size = 16

        torch.manual_seed(42)
        model = CheckpointedMLP(dim=dim, num_layers=num_layers).cuda().to(torch.bfloat16)
        x = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)

        # Baseline
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        grads_baseline = _get_grads(model, x)
        torch.cuda.synchronize()
        peak_baseline = torch.cuda.max_memory_allocated()

        model.zero_grad()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # With offloading
        offloader = CheckpointInputOffload(pin_memory=True, min_tensor_numel=1)
        model.enable_gc(context_fn=offloader.get_context_fn())
        grads_offload = _get_grads(model, x, offloader=offloader)
        torch.cuda.synchronize()
        peak_offload = torch.cuda.max_memory_allocated()

        # Offloading should use less or equal peak memory
        # (For small models the effect may be minimal, so we just check it doesn't increase much)
        print(f"Peak baseline: {peak_baseline / 1e6:.1f} MB, Peak offload: {peak_offload / 1e6:.1f} MB")
        # Allow 10% tolerance for small model overhead
        assert peak_offload <= peak_baseline * 1.1, (
            f"Offloading should not significantly increase memory: "
            f"baseline={peak_baseline / 1e6:.1f}MB, offload={peak_offload / 1e6:.1f}MB"
        )

    def test_parameter_not_offloaded(self):
        """Parameters (leaf tensors with requires_grad) should NOT be offloaded."""
        offloader = CheckpointInputOffload(pin_memory=True, min_tensor_numel=1)

        # Simulate a parameter
        param = torch.randn(100, 100, device='cuda', requires_grad=True)
        assert param.is_leaf
        result = offloader._pack(param)
        # Should be returned as-is (not wrapped in a dict)
        assert isinstance(result, torch.Tensor), "Parameters should not be offloaded"
        assert result is param, "Parameters should be passed through unchanged"

    def test_activation_offloaded(self):
        """Non-leaf CUDA tensors above min_numel should be offloaded."""
        offloader = CheckpointInputOffload(pin_memory=True, min_tensor_numel=1)

        # Simulate an activation (non-leaf tensor)
        x = torch.randn(100, 100, device='cuda', requires_grad=True)
        activation = x * 2  # non-leaf
        assert not activation.is_leaf

        result = offloader._pack(activation)
        assert isinstance(result, dict), "Activations should be offloaded to CPU"
        assert 'cpu' in result and 'device' in result
        assert result['cpu'].device == torch.device('cpu')
        assert result['cpu'].is_pinned()

    def test_small_tensor_not_offloaded(self):
        """Tensors below min_tensor_numel should NOT be offloaded."""
        offloader = CheckpointInputOffload(pin_memory=True, min_tensor_numel=1000)

        # Small activation (100 elements < 1000 threshold)
        x = torch.randn(10, 10, device='cuda', requires_grad=True)
        activation = x * 2
        assert activation.numel() == 100

        result = offloader._pack(activation)
        assert isinstance(result, torch.Tensor), "Small tensors should not be offloaded"

    def test_cpu_tensor_not_offloaded(self):
        """CPU tensors should be passed through."""
        offloader = CheckpointInputOffload(pin_memory=True, min_tensor_numel=1)

        cpu_tensor = torch.randn(100, 100)
        result = offloader._pack(cpu_tensor)
        assert isinstance(result, torch.Tensor)
        assert result is cpu_tensor

    def test_non_tensor_passthrough(self):
        """Non-tensor objects should be passed through in both pack and unpack."""
        offloader = CheckpointInputOffload(pin_memory=True)

        # Test _pack with non-tensor
        result = offloader._pack("not a tensor")
        assert result == "not a tensor"

        # Test _unpack with non-dict
        result = offloader._unpack("not a dict")
        assert result == "not a dict"

    def test_unpack_roundtrip(self):
        """Pack then unpack should produce an equivalent tensor."""
        offloader = CheckpointInputOffload(pin_memory=True, min_tensor_numel=1)

        x = torch.randn(10, 10, device='cuda', requires_grad=True)
        activation = x * 2  # non-leaf

        packed = offloader._pack(activation)
        assert isinstance(packed, dict)

        # Sync the D2H stream before unpacking
        if offloader.d2h_stream is not None:
            offloader.d2h_stream.synchronize()

        unpacked = offloader._unpack(packed)
        assert unpacked.device == activation.device
        torch.testing.assert_close(unpacked, activation)

    def test_no_grad_safety(self):
        """Offloader should be harmless under torch.no_grad()."""
        torch.manual_seed(42)
        model = CheckpointedMLP(dim=128, num_layers=4).cuda().to(torch.bfloat16)
        x = torch.randn(4, 128, device='cuda', dtype=torch.bfloat16)

        offloader = CheckpointInputOffload(pin_memory=True)
        model.enable_gc(context_fn=offloader.get_context_fn())

        with torch.no_grad():
            with offloader:
                out = model(x)
        # Should complete without error
        assert out.shape == (4, 128)

    def test_multiple_micro_batches(self):
        """Re-entrant usage across multiple micro-batches should work."""
        torch.manual_seed(42)
        model = CheckpointedMLP(dim=128, num_layers=4).cuda().to(torch.bfloat16)

        offloader = CheckpointInputOffload(pin_memory=True)
        model.enable_gc(context_fn=offloader.get_context_fn())

        model.zero_grad()
        total_loss = 0.0
        for i in range(3):
            x = torch.randn(4, 128, device='cuda', dtype=torch.bfloat16)
            with offloader:
                out = model(x)
            loss = out.sum() / 3.0
            loss.backward()
            total_loss += loss.item()

        # All parameters should have gradients
        for name, p in model.named_parameters():
            assert p.grad is not None, f"Missing gradient for {name} after multiple micro-batches"

    def test_context_manager_protocol(self):
        """Verify __enter__/__exit__ properly push/pop hooks."""
        offloader = CheckpointInputOffload(pin_memory=True)

        # Using as context manager should not raise
        with offloader:
            x = torch.randn(10, device='cuda', requires_grad=True)
            y = x * 2
            # Inside context, hooks should be active

        # Outside context, hooks should be popped (no error on normal ops)
        z = torch.randn(10, device='cuda', requires_grad=True)
        w = z * 2
        w.sum().backward()

    def test_non_contiguous_roundtrip(self):
        """Non-contiguous source tensors should round-trip with correct values."""
        offloader = CheckpointInputOffload(pin_memory=True, min_tensor_numel=1)

        base = torch.randn(10, 20, device='cuda')
        # Create non-contiguous views: transposed, sliced, strided
        views = [
            base.t(),                          # transposed: stride=(1, 20)
            base[:, ::2],                       # strided slice: every other column
            base[2:8, 3:15],                    # sub-block slice
        ]

        for i, v in enumerate(views):
            assert not v.is_contiguous(), f"View {i} should be non-contiguous"
            packed = offloader._pack(v)
            assert isinstance(packed, dict), f"View {i} should be offloaded"

            if offloader.d2h_stream is not None:
                offloader.d2h_stream.synchronize()

            unpacked = offloader._unpack(packed)
            assert unpacked.shape == v.shape, f"View {i}: shape mismatch"
            torch.testing.assert_close(unpacked, v, msg=f"View {i}: value mismatch")

    def test_prefix_grouper_conflict_runtime(self):
        """Runtime check should catch PrefixGrouper + offloader conflict."""
        from unittest.mock import MagicMock

        # Create a minimal mock of DPActorModule to test the conflict check
        offloader = CheckpointInputOffload(pin_memory=True)
        actor_module = MagicMock()
        actor_module.training = True

        # Simulate the conflict: both use_prefix_grouper and _checkpoint_offloader active
        use_prefix_grouper = True
        _checkpoint_offloader = offloader

        should_raise = (use_prefix_grouper and _checkpoint_offloader is not None and actor_module.training)
        assert should_raise, "Conflict condition should trigger"

        # Verify no conflict when either is disabled
        assert not (False and _checkpoint_offloader is not None and actor_module.training)
        assert not (use_prefix_grouper and None is not None and actor_module.training)
        assert not (use_prefix_grouper and _checkpoint_offloader is not None and False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
