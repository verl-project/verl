# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""NeoProto ref/index batch implementation for the v0 trainer."""

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
    "get_default_storage_engine",
    "get_engine_for_backend",
    "set_default_storage_engine",
]
