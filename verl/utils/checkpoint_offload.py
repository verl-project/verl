"""Offload gradient checkpoint saved tensors to CPU pinned memory.

Exploits PyTorch's "innermost wins" hook nesting to selectively offload
only checkpoint inputs while leaving intermediates and recomputed tensors
untouched. See docs/perf/checkpoint_input_offload.md for architecture details.
"""

import time
import torch
from contextlib import nullcontext


class CheckpointInputOffload:
    """Offload gradient checkpoint saved tensors to CPU pinned memory.

    Uses OUTER saved_tensors_hooks to intercept checkpoint inputs that are
    saved outside ``_checkpoint_hook`` scope. PyTorch's "innermost wins"
    hook nesting ensures intermediates (inside ``_checkpoint_hook``) are
    NOT intercepted by our hooks.

    Usage::

        offloader = CheckpointInputOffload()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": offloader.get_context_fn(),
        })
        # In training loop:
        with offloader:
            output = model(input_ids, ...)
        loss.backward()  # unpack hooks fire from captured references

    How it works:

    1. ``_NoopSaveInputs.apply()`` (checkpoint.py:1616) saves checkpoint
       inputs BEFORE ``_checkpoint_hook`` scope. At this point, our outer
       hooks are the innermost → ``_pack`` intercepts → async D2H to CPU.

    2. Inside ``_checkpoint_hook`` scope (checkpoint.py:1623), the inner
       hooks become innermost. Our hooks are INACTIVE → intermediates are
       replaced with ``_Holder`` objects by ``_checkpoint_hook``, not offloaded.

    3. During backward recomputation, ``_recomputation_hook`` becomes
       innermost. Our hooks are again INACTIVE → recomputed tensors stay
       on GPU.

    4. When backward accesses checkpoint inputs, the captured ``_unpack``
       fires → sync H2D transfer from CPU pinned memory back to GPU.
    """

    def __init__(self, pin_memory=True, min_tensor_numel=1024):
        self.pin_memory = pin_memory
        self.min_tensor_numel = min_tensor_numel
        self.d2h_stream = None  # lazy init per-device
        # Optional callbacks for external profiling (set externally)
        self._on_pack_cb = None   # Callable[[int], None] — nbytes packed
        self._on_unpack_cb = None  # Callable[[int], None] — nbytes unpacked
        # Diagnostic counters (per forward pass, reset in __enter__)
        self._diag_pack_count = 0
        self._diag_pack_bytes = 0
        self._diag_skip_param = 0
        self._diag_skip_small = 0
        self._diag_skip_noncuda = 0
        self._diag_unpack_count = 0

    def _pack(self, tensor):
        """Pack hook: async D2H for eligible CUDA tensors, pass-through otherwise."""
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            self._diag_skip_noncuda += 1
            return tensor
        if tensor.numel() < self.min_tensor_numel:
            self._diag_skip_small += 1
            return tensor
        # Skip parameters — they persist on GPU and would be wastefully round-tripped
        if tensor.requires_grad and tensor.is_leaf:
            self._diag_skip_param += 1
            return tensor

        if self.d2h_stream is None:
            self.d2h_stream = torch.cuda.Stream(tensor.device)

        cpu_tensor = torch.empty(
            tensor.shape, dtype=tensor.dtype,
            layout=tensor.layout, device='cpu',
            pin_memory=self.pin_memory,
        )
        # Ensure default stream's computation is visible to d2h_stream
        # Without this, d2h_stream may read stale GPU data (race condition)
        self.d2h_stream.wait_stream(torch.cuda.current_stream(tensor.device))
        with torch.cuda.stream(self.d2h_stream):
            cpu_tensor.copy_(tensor, non_blocking=True)
        # Prevent GPU tensor reuse before D2H completes
        tensor.record_stream(self.d2h_stream)

        self._diag_pack_count += 1
        nbytes = tensor.nelement() * tensor.element_size()
        self._diag_pack_bytes += nbytes
        if self._on_pack_cb is not None:
            self._on_pack_cb(nbytes)

        return {'device': tensor.device, 'cpu': cpu_tensor}

    def _unpack(self, packed):
        """Unpack hook: H2D copy back to GPU."""
        if not isinstance(packed, dict):
            return packed

        self._diag_unpack_count += 1
        cpu_tensor = packed['cpu']
        if self._on_unpack_cb is not None:
            self._on_unpack_cb(cpu_tensor.nelement() * cpu_tensor.element_size())
        # D2H guaranteed complete by __exit__ sync (forward already finished)
        return cpu_tensor.to(packed['device'], non_blocking=self.pin_memory)

    def get_context_fn(self):
        """Returns context_fn for gradient_checkpointing_enable.

        Currently returns no-op contexts. The "innermost wins" nesting
        already prevents our hooks from firing during recomputation.
        Kept as a hook point for future instrumentation (e.g., logging
        offload stats during recomputation).
        """
        def context_fn():
            return nullcontext(), nullcontext()
        return context_fn

    def _reset_diag(self):
        self._diag_pack_count = 0
        self._diag_pack_bytes = 0
        self._diag_skip_param = 0
        self._diag_skip_small = 0
        self._diag_skip_noncuda = 0
        self._diag_unpack_count = 0

    def get_diag_summary(self):
        """Return diagnostic summary dict for the last forward pass."""
        return {
            'pack_count': self._diag_pack_count,
            'pack_bytes_mb': self._diag_pack_bytes / (1024 * 1024),
            'skip_param': self._diag_skip_param,
            'skip_small': self._diag_skip_small,
            'skip_noncuda': self._diag_skip_noncuda,
            'unpack_count': self._diag_unpack_count,
        }

    def __enter__(self):
        self._reset_diag()
        self._enter_time = time.monotonic()
        torch._C._autograd._push_saved_tensors_default_hooks(self._pack, self._unpack)
        return self

    def __exit__(self, *args):
        torch._C._autograd._pop_saved_tensors_default_hooks()
        if self.d2h_stream is not None:
            self.d2h_stream.synchronize()
        self._exit_time = time.monotonic()
