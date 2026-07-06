"""Registry — lazy registry for transport + decoder combinations.

Built-in entries are registered as ``collection_name → (transport_path, decoder_path)``
strings.  Classes are imported on first lookup and cached.  A ``Collector``
is created by combining the resolved Transport and Decoder instances.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from verl.workers.rollout.llm_router.collectors.collector import Collector

logger = logging.getLogger(__name__)


class Registry:
    """Lazy registry — maps collection_name to transport + decoder combination.

    ``register_entry(name, transport_path, decoder_path)`` stores
    ``"module.path:ClassName"`` strings; classes are imported on first
    ``get_collector`` call and cached in-place.

    ``get_collector(name, **kwargs)`` resolves both classes, instantiates
    them, and returns a ``Collector(transport, decoder)`` instance.

    ``get_store(name)`` resolves the decoder and reads its ``store_cls``.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, str | type]] = {}

    def register_entry(
        self,
        name: str,
        transport_path: str,
        decoder_path: str,
    ) -> None:
        """Register a lazy entry-point combination.

        Paths use ``"module.path:ClassName"`` format (setuptools entry-point style).
        """
        self._entries[name] = {
            "transport": transport_path,
            "decoder": decoder_path,
        }

    def get_collector(self, name: str, **kwargs: Any) -> Collector:
        """Create a Collector by combining resolved Transport + Decoder.

        Args:
            name: Collection name key.
            **kwargs: Keyword arguments forwarded to the Transport
                constructor.  Decoders are always constructed with no
                arguments.

        Returns:
            A ``Collector`` instance with transport + decoder composed.
        """
        entry = self._entries.get(name)
        if entry is None:
            raise ValueError(f"Unknown collector: '{name}'. Registered: {sorted(self._entries.keys())}")

        transport_cls = self._resolve(entry["transport"], "transport", name)
        decoder_cls = self._resolve(entry["decoder"], "decoder", name)

        transport = transport_cls(**kwargs)
        decoder = decoder_cls()

        return Collector(transport, decoder)

    def get_store(self, name: str) -> type:
        """Look up the store class for a collection name.

        Derived from the decoder's ``store_cls`` attribute.
        """
        entry = self._entries.get(name)
        if entry is None:
            raise ValueError(f"Unknown store: '{name}'. Registered: {sorted(self._entries.keys())}")
        decoder_cls = self._resolve(entry["decoder"], "decoder", name)
        return decoder_cls.store_cls

    def _resolve(self, entry: str | type, role: str, name: str) -> type:
        """Resolve a lazy entry — import from string or return cached class."""
        if isinstance(entry, type):
            return entry
        # Lazy import — "module.path:ClassName"
        module_path, class_name = entry.rsplit(":", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        # Cache-in-place in the entry dict
        self._entries[name][role] = cls
        logger.info("Lazy-imported %s for '%s': %s", role, name, cls.__qualname__)
        return cls


# ── Module-level built-in registration (lazy) ─────────────────────────

BUILTIN_REGISTRY = Registry()
BUILTIN_REGISTRY.register_entry(
    "vllm_metrics",
    transport_path="verl.workers.rollout.llm_router.collectors.transport.http:HTTPTransport",
    decoder_path="verl.workers.rollout.llm_router.collectors.decoder.vllm.metrics:VLLMMetricsDecoder",
)
BUILTIN_REGISTRY.register_entry(
    "vllm_zmq",
    transport_path="verl.workers.rollout.llm_router.collectors.transport.zmq:ZMQTransport",
    decoder_path="verl.workers.rollout.llm_router.collectors.decoder.vllm.kv:VLLMKVDecoder",
)
