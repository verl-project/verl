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

"""Utilities for DataLoader lifecycle management."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def shutdown_dataloader_workers(dataloader: Any) -> bool:
    """Best-effort shutdown for multiprocessing DataLoader iterators.

    PyTorch and torchdata keep worker processes on the active iterator. There
    is no public close API, so this helper uses the same private shutdown hook
    that the iterator destructor calls, but does it while the trainer is still
    in a normal execution context instead of during Python/Ray teardown.
    """
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is None:
        return False

    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    try:
        if callable(shutdown_workers):
            shutdown_workers()
            return True
        return False
    except Exception:
        logger.warning("Failed to shut down DataLoader workers", exc_info=True)
        return False
    finally:
        if getattr(dataloader, "_iterator", None) is iterator:
            dataloader._iterator = None


def shutdown_dataloaders(*dataloaders: Any) -> None:
    """Shut down active worker iterators for any initialized dataloaders."""
    for dataloader in dataloaders:
        if dataloader is None:
            continue
        shutdown_dataloader_workers(dataloader)
