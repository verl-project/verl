# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Deferred host scalars: keep per-micro-batch metric and loss scalars on the device and read them once.

On a lazily executing accelerator (e.g. torch_tpu's DEFER_AND_FUSE eager mode) every ``Tensor.item()`` and every
``if tensor:`` is a device->host round trip: the host waits until the device has run everything queued before it. A
policy loss with diagnostics does dozens of them per micro-batch, all before ``loss.backward()`` is dispatched.

Inside ``with defer_host_scalars() as scope:``

* ``host_scalar(t)`` returns a :class:`DeferredScalar` placeholder instead of ``t.item()``; ``host_scalar(a, b, fn=f)``
  stands for ``f(a.item(), b.item())`` (the host-side arithmetic runs later, in Python, on the same numbers);
* ``check(cond, exc_type, message)`` records the condition instead of raising ``exc_type(message)`` on ``not cond``;
* ``host_copy(t)`` returns an uninitialized host tensor (NaN-filled if floating) instead of ``t.cpu()``, filled in
  place by ``resolve()`` (for host copies that nothing reads before the scope resolves, e.g. per-micro-batch outputs);
* ``scope.resolve(tree, ...)`` reads every pending tensor with one stacked device->host copy per (device, dtype), runs
  the placeholders' host functions, raises the first failed check (same exception type and message), and replaces the
  placeholders inside ``tree`` (dicts, lists, ``Metric.values``) by the plain Python numbers.

Outside a scope the helpers are the eager code they replace (``t.item()``; ``if not cond: raise``; ``t.cpu()``), in the
same order.

Exactness: a placeholder resolves to the element ``t.item()`` reads (``torch.stack`` copies, ``tolist()`` converts like
``item()``) and its host function sees the same Python numbers, so every value is bitwise the eager one. A tensor
modified in place between ``host_scalar()`` and ``resolve()`` would change the value read; ``resolve()`` compares every
tensor's version counter and refuses (RuntimeError) instead.

``materialize`` (optional, per scope): called on each deferred tensor, one at a time and in argument order, at the point
where it is deferred. A lazy backend can pass its non-blocking materialization (torch_tpu:
``synchronize(t, wait=False)``), which keeps the deferred-graph cut points exactly where ``item()`` / ``bool()`` had
them, minus the host wait and the copy.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Callable, Iterator, Optional

import torch

__all__ = [
    "DeferredScalar",
    "HostScalarScope",
    "active_scope",
    "check",
    "defer_host_scalars",
    "host_copy",
    "host_scalar",
]

_SCOPE: contextvars.ContextVar[Optional[HostScalarScope]] = contextvars.ContextVar(
    "verl_host_scalar_scope", default=None
)


class DeferredScalar:
    """A host number that will be read from the device by :meth:`HostScalarScope.resolve`.

    Any numeric use before it is resolved raises, so a placeholder can never be mistaken for a value.
    """

    __slots__ = ("_tensors", "_versions", "_fn", "_value", "_resolved")

    def __init__(self, tensors: tuple[torch.Tensor, ...], fn: Optional[Callable[..., Any]]):
        self._tensors = tensors
        self._versions = tuple(t._version for t in tensors)
        self._fn = fn
        self._value: Any = None
        self._resolved = False

    @property
    def resolved(self) -> bool:
        return self._resolved

    @property
    def value(self) -> Any:
        if not self._resolved:
            raise RuntimeError("DeferredScalar read before HostScalarScope.resolve()")
        return self._value

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __index__(self) -> int:
        return self.value.__index__()

    def __repr__(self) -> str:
        return f"DeferredScalar({self._value!r})" if self._resolved else "DeferredScalar(<pending>)"


class HostScalarScope:
    """The placeholders and checks of one :func:`defer_host_scalars` block (see the module docstring)."""

    def __init__(self, materialize: Optional[Callable[[torch.Tensor], Any]] = None):
        self.materialize = materialize
        self.pending: list[DeferredScalar] = []
        self.checks: list[tuple[torch.Tensor, int, type, str]] = []
        self.copies: list[tuple[torch.Tensor, int, torch.Tensor]] = []
        self.reads = 0
        self.resolved = False

    def _take(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if self.resolved:
            raise RuntimeError("HostScalarScope already resolved")
        out = []
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"host_scalar expects tensors, got {type(t).__name__}")
            if t.numel() != 1:
                raise ValueError(f"host_scalar expects one-element tensors, got shape {tuple(t.shape)}")
            out.append(t.detach())
        if self.materialize is not None:
            for t in out:  # one at a time, in order: where the eager code's item() / bool() calls were
                self.materialize(t)
        return tuple(out)

    def defer(self, tensors: tuple[torch.Tensor, ...], fn: Optional[Callable[..., Any]] = None) -> DeferredScalar:
        d = DeferredScalar(self._take(tensors), fn)
        self.pending.append(d)
        return d

    def check(self, cond: torch.Tensor, exc_type: type, message: str) -> None:
        (t,) = self._take((cond,))
        self.checks.append((t, t._version, exc_type, message))

    def defer_copy(self, t: torch.Tensor) -> torch.Tensor:
        if self.resolved:
            raise RuntimeError("HostScalarScope already resolved")
        src = t.detach()
        if self.materialize is not None:
            self.materialize(src)
        dest = torch.empty(src.shape, dtype=src.dtype, device="cpu")
        if dest.is_floating_point() or dest.is_complex():
            dest.fill_(float("nan"))  # a premature read shows NaN, never stale memory
        self.copies.append((src, src._version, dest))
        return dest

    def _read(self) -> dict[int, Any]:
        """id(tensor) -> its Python value, with one stacked device->host copy per (device, dtype)."""
        tensors: dict[int, torch.Tensor] = {}
        for d in self.pending:
            for t in d._tensors:
                tensors.setdefault(id(t), t)
        for t, *_ in self.checks:
            tensors.setdefault(id(t), t)
        groups: dict[tuple, list[int]] = {}
        for k, t in tensors.items():
            groups.setdefault((str(t.device), t.dtype), []).append(k)
        values: dict[int, Any] = {}
        for keys in groups.values():
            stacked = torch.stack([tensors[k].reshape(()) for k in keys])
            host = stacked.cpu().tolist()
            self.reads += 1
            values.update(zip(keys, host, strict=False))
        return values

    def resolve(self, *trees: Any) -> int:
        """Reads every pending value, raises the first failed check, then fills the placeholders inside ``trees``.

        Returns the number of placeholders replaced in ``trees``.
        """
        if self.resolved:
            raise RuntimeError("HostScalarScope already resolved")
        self.resolved = True
        for d in self.pending:
            if tuple(t._version for t in d._tensors) != d._versions:
                raise RuntimeError("a tensor passed to host_scalar() was modified in place before it was read")
        for t, version, _, _ in self.checks:
            if t._version != version:
                raise RuntimeError("a tensor passed to check() was modified in place before it was read")
        for src, version, _ in self.copies:
            if src._version != version:
                raise RuntimeError("a tensor passed to host_copy() was modified in place before it was copied")
        values = self._read() if (self.pending or self.checks) else {}
        for t, _, exc_type, message in self.checks:
            if not values[id(t)]:
                raise exc_type(message)
        for d in self.pending:
            vals = [values[id(t)] for t in d._tensors]
            d._value = d._fn(*vals) if d._fn is not None else vals[0]
            d._resolved = True
            d._tensors = ()
        for src, _, dest in self.copies:
            dest.copy_(src)
        self.copies = []
        return sum(_fill(tree) for tree in trees)


def _fill(obj: Any) -> int:
    """Replaces resolved placeholders in place inside dicts, lists and objects with a ``values`` list (``Metric``)."""
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, DeferredScalar):
                obj[k] = v.value
                n += 1
            else:
                n += _fill(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, DeferredScalar):
                obj[i] = v.value
                n += 1
            else:
                n += _fill(v)
    elif isinstance(obj, tuple):
        if any(isinstance(v, DeferredScalar) for v in obj):
            raise TypeError("cannot fill a DeferredScalar inside a tuple; store it in a list or dict")
        n += sum(_fill(v) for v in obj)
    elif isinstance(getattr(obj, "values", None), list) and not isinstance(obj, torch.Tensor):
        n += _fill(obj.values)
    return n


def active_scope() -> Optional[HostScalarScope]:
    """The innermost active scope, or None."""
    return _SCOPE.get()


@contextlib.contextmanager
def defer_host_scalars(materialize: Optional[Callable[[torch.Tensor], Any]] = None) -> Iterator[HostScalarScope]:
    """Defers :func:`host_scalar` reads and :func:`check` conditions until ``scope.resolve()`` (not nestable)."""
    if _SCOPE.get() is not None:
        raise RuntimeError("defer_host_scalars() is not nestable")
    scope = HostScalarScope(materialize)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def host_scalar(t: torch.Tensor, *more: torch.Tensor, fn: Optional[Callable[..., Any]] = None) -> Any:
    """``t.item()`` (or ``fn(t.item(), *[m.item() for m in more])``), deferred inside :func:`defer_host_scalars`."""
    scope = _SCOPE.get()
    if scope is not None:
        return scope.defer((t, *more), fn)
    vals = [x.item() for x in (t, *more)]
    return fn(*vals) if fn is not None else vals[0]


def host_copy(t: torch.Tensor) -> torch.Tensor:
    """``t.cpu()``; inside :func:`defer_host_scalars` a host tensor that ``resolve()`` fills in place."""
    scope = _SCOPE.get()
    if scope is not None:
        return scope.defer_copy(t)
    return t.cpu()


def check(cond: torch.Tensor, exc_type: type, message: str) -> None:
    """``if not cond: raise exc_type(message)``, deferred to ``resolve()`` inside :func:`defer_host_scalars`."""
    scope = _SCOPE.get()
    if scope is not None:
        scope.check(cond, exc_type, message)
    elif not cond:
        raise exc_type(message)
