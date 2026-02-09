# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import ray
import torch
from tensordict import TensorDict


@dataclass
class KVBatchMeta:
    partition_id: str
    """Partition id of the batch."""
    keys: list[str]
    """Keys of the sampled items."""
    tags: list[dict]
    """Tags of the sampled items."""
    extra_info: dict = dataclass_field(default_factory=dict)
    """Meta info of the batch."""

    def __len__(self):
        return len(self.keys)

    def reorder(self, indices: list[int]):
        """Reorder the batch items by the given indices."""
        self.keys = [self.keys[i] for i in indices]
        self.tags = [self.tags[i] for i in indices]


@ray.remote(name="transfer_queue_controller")
class TransferQueueController:
    def __init__(self) -> None:
        self.tags: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
        self.storage_unit: dict[str, dict] = defaultdict(lambda: defaultdict(dict))

    def put(self, partition_id: str, key: str, fields: dict = None, tags: dict = None):
        if tags is not None:
            self.tags[partition_id][key].update(tags)
        if fields is not None:
            self.storage_unit[partition_id][key].update(fields)

    def batch_put(self, partition_id: str, keys: list[str], fields: list[dict], tags: list[dict] = None):
        if tags is None:
            tags = [{}] * len(keys)
        for key, value, tag in zip(keys, fields, tags, strict=True):
            self.put(partition_id, key, value, tag)

    def batch_get(self, batch: KVBatchMeta, fields: list[str]) -> TensorDict:
        tensors = {}
        for field in fields:
            values = []
            for key in batch.keys:
                values.append(self.storage_unit[batch.partition_id][key][field])
            tensors[field] = torch.nested.nested_tensor(values, layout=torch.jagged)
        return TensorDict(tensors, batch_size=len(batch))

    def kv_list(self) -> dict[str, dict]:
        return self.tags


_TQ_HANDLE = None


def init(config=None):
    try:
        transfer_queue = ray.get_actor("transfer_queue_controller")
    except ValueError:
        transfer_queue = TransferQueueController.remote()

    global _TQ_HANDLE
    _TQ_HANDLE = transfer_queue


# ======================================= ASYNC API =======================================


async def async_kv_put(partition_id: str, key: str, fields: dict = None, tags: dict = None):
    await _TQ_HANDLE.put.remote(partition_id, key, fields, tags)


async def async_kv_batch_put(partition_id: str, keys: list[str], fields: list[dict], tags: list[dict] = None):
    await _TQ_HANDLE.batch_put.remote(partition_id, keys, fields, tags)


# ======================================= SYNC API =======================================


def kv_list():
    return ray.get(_TQ_HANDLE.kv_list.remote())


def kv_batch_get(batch: KVBatchMeta, fields: list[str]) -> TensorDict:
    return ray.get(_TQ_HANDLE.batch_get.remote(batch, fields))


def kv_batch_put(partition_id: str, keys: list[str], fields: list[dict], tags: list[dict] = None):
    return ray.get(_TQ_HANDLE.batch_put.remote(partition_id, keys, fields, tags))
