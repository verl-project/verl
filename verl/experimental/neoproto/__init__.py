# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Experimental NeoProto data plane for the v0 trainer."""

from .dispatch import enable_neo_dispatch, is_neo_batch, is_neo_dispatch_enabled, maybe_attach_neo_ref_tables
from .neo import IndexView, NeoProto
from .storage import (
    DefaultStorageEngine,
    FieldSpec,
    InMemoryStorageEngine,
    LocalRef,
    Ref,
    RefTable,
    StorageEngine,
    StorageRef,
    get_default_storage_engine,
    get_engine_for_backend,
    set_default_storage_engine,
)
from .views import DataProto, DataProtoItem

# Ray-oriented alias used by existing verl tests / smoke scripts.
RayStorageEngine = DefaultStorageEngine

__all__ = [
    "DataProto",
    "DataProtoItem",
    "DefaultStorageEngine",
    "FieldSpec",
    "IndexView",
    "InMemoryStorageEngine",
    "LocalRef",
    "NeoProto",
    "RayStorageEngine",
    "Ref",
    "RefTable",
    "StorageEngine",
    "StorageRef",
    "enable_neo_dispatch",
    "get_default_storage_engine",
    "get_engine_for_backend",
    "is_neo_batch",
    "is_neo_dispatch_enabled",
    "maybe_attach_neo_ref_tables",
    "set_default_storage_engine",
]
