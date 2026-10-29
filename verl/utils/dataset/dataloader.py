# Copyright 2026 Individual Contributor
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

"""Small construction and batch contracts for custom PPO dataloaders."""

import logging
import sys
from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Any, Protocol

from hydra.errors import InstantiationException
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

RUNTIME_ARGUMENTS = frozenset({"_data_config", "_split", "_batch_size", "_tokenizer", "_processor", "_collate_fn"})


class PromptDataLoader(Protocol):
    """Collated prompt batches; validation needs repeatable, finite iteration."""

    def __iter__(self) -> Iterator[dict[str, Any]]: ...


class StatefulPromptDataLoader(PromptDataLoader, Protocol):
    """Optional resume contract; implementations need not inherit this class."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class DataLoaderExhausted(Exception):
    """A custom training source ended in step-based mode."""


def create_custom_dataloader(
    data_config: DictConfig, *, split: str, batch_size: int, tokenizer: Any, processor: Any, collate_fn: Any
) -> PromptDataLoader:
    """Construct one role's loader without recursively instantiating its configuration."""
    config = data_config.get("custom_dataloader")
    if not isinstance(config, (DictConfig | Mapping)) or not isinstance(config.get("_target_"), str):
        raise ValueError("data.custom_dataloader must contain a class _target_.")
    reserved = RUNTIME_ARGUMENTS | {"_args_", "_partial_", "_recursive_", "_convert_"}
    collisions = reserved.intersection(config)
    if collisions:
        raise ValueError(f"data.custom_dataloader contains reserved arguments: {sorted(collisions)}")
    custom_cls = data_config.get("custom_cls")
    if custom_cls and (custom_cls.get("path") is not None or custom_cls.get("name") is not None):
        raise ValueError("data.custom_dataloader and data.custom_cls cannot be combined.")
    if split not in ("train", "val"):
        raise ValueError(f"Unsupported dataloader split: {split!r}")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError(f"Custom {split} dataloader batch size must be a positive integer.")

    context = deepcopy(data_config)
    OmegaConf.set_readonly(context, True)
    try:
        get_class(config["_target_"])  # Reject function targets, including factories returning shared instances.
        # Bind user configuration first so Hydra cannot reinterpret live runtime inputs as configs.
        constructor = instantiate(config, _partial_=True, _recursive_=False, _convert_="none")
        logger.info("Constructing custom %s dataloader %s", split, config["_target_"])
        loader = constructor(
            _data_config=context,
            _split=split,
            _batch_size=batch_size,
            _tokenizer=tokenizer,
            _processor=processor,
            _collate_fn=collate_fn,
        )
    except (InstantiationException, ImportError, ValueError, TypeError) as error:
        raise ValueError(f"Cannot construct custom {split} dataloader {config['_target_']!r}: {error}") from error

    methods = ["__iter__"]
    if split == "train" and any(
        getattr(loader, method, None) is not None for method in ("state_dict", "load_state_dict")
    ):
        methods += ["state_dict", "load_state_dict"]
    missing = [method for method in methods if not callable(getattr(loader, method, None))]
    if missing:
        close_dataloaders(loader)
        raise TypeError(f"Custom {split} dataloader must implement callable methods: {missing}")
    return loader


def validate_custom_batch(batch: Any, *, split: str, batch_size: int) -> None:
    """Check consumed batches, without probing or advancing a source iterator."""
    if not isinstance(batch, dict) or "raw_prompt" not in batch:
        raise TypeError(f"Custom {split} dataloader must yield batch dictionaries containing raw_prompt.")
    try:
        count = len(batch["raw_prompt"])
    except TypeError as error:
        raise TypeError(f"Custom {split} dataloader raw_prompt must be batched.") from error
    if count <= 0 or count > batch_size or (split == "train" and count != batch_size):
        raise ValueError(f"Custom {split} dataloader yielded {count} prompts; expected batch size {batch_size}.")
    for key, value in batch.items():
        if isinstance(value, (str | bytes | Mapping)):
            raise TypeError(f"Custom {split} dataloader field {key!r} must have a leading batch dimension.")
        try:
            size = len(value)
        except TypeError as error:
            raise TypeError(f"Custom {split} dataloader field {key!r} must be batched.") from error
        if size != count:
            raise ValueError(f"Custom {split} dataloader field {key!r} has {size} rows; expected {count}.")


def close_dataloaders(*loaders: PromptDataLoader | None) -> None:
    """Close each owned loader once, attempting all cleanups even if one fails."""
    has_active_exception = sys.exc_info()[0] is not None
    seen = set()
    first_error = None
    for loader in loaders:
        if loader is None or id(loader) in seen:
            continue
        seen.add(id(loader))
        close = getattr(loader, "close", None)
        if close is not None:
            try:
                close()
            except Exception as error:
                logger.exception("Custom dataloader cleanup failed")
                if first_error is None:
                    first_error = error
    if first_error is not None and not has_active_exception:
        raise first_error
