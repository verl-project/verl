"""Decoder — abstract base for data decoding and store writing.

A Decoder receives raw data from a Transport and decodes it into
structured updates, writing results to its associated Store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Decoder(ABC):
    """Abstract base for data decoders.

    Subclasses implement ``decode()`` with their backend-specific
    parsing logic and declare ``store_cls`` as a class attribute
    to indicate which Store they write to.
    """

    store_cls: type  # Store class this decoder writes to

    @abstractmethod
    def decode(self, raw_data: bytes | str, node_id: str) -> None:
        """Decode raw data and write results to the store.

        Args:
            raw_data: Raw payload — ``bytes`` (from ZMQ) or ``str``
                (from HTTP response text).
            node_id: Source replica/node identifier.
        """
