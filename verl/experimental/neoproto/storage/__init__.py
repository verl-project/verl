# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Storage submodule for :mod:`verl.experimental.neoproto`."""

from verl.experimental.neoproto.storage.default import (
    DefaultStorageEngine,
    InMemoryStorageEngine,
    get_default_storage_engine,
    get_engine_for_backend,
    set_default_storage_engine,
)
from verl.experimental.neoproto.storage.engine import (
    FieldSpec,
    LocalRef,
    Ref,
    RefTable,
    SliceSpec,
    StorageEngine,
    StorageRef,
    compose_slice,
    new_uid,
)

__all__ = [
    "FieldSpec",
    "Ref",
    "LocalRef",
    "StorageRef",
    "RefTable",
    "SliceSpec",
    "StorageEngine",
    "compose_slice",
    "new_uid",
    "DefaultStorageEngine",
    "InMemoryStorageEngine",
    "get_default_storage_engine",
    "get_engine_for_backend",
    "set_default_storage_engine",
]
