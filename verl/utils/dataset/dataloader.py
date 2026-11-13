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
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)

RUNTIME_ARGUMENTS = frozenset({"_data_config", "_split", "_batch_size", "_tokenizer", "_processor", "_collate_fn"})


class PromptDataLoader(Protocol):
    """Collated batches; the training loader owns repetition, validation needs fresh finite iterations."""

    def __iter__(self) -> Iterator[dict[str, Any]]: ...


class StatefulPromptDataLoader(PromptDataLoader, Protocol):
    """Optional resume contract; implementations need not inherit this class."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class DataLoaderExhausted(Exception):
    """A custom training source cannot supply another complete update."""


def resolve_gen_batch_size(config: DictConfig, trainer_mode: str) -> int:
    """Use single-prompt fetches when sampling may request exact replacement counts."""
    filter_groups = config.algorithm.get("filter_groups", None)
    dapo_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
    sync_refill_failed_groups = bool(config.trainer.v1.sampler.get("sync_refill_failed_groups", False))
    requires_exact_refill = trainer_mode != "sync" or dapo_enabled or sync_refill_failed_groups
    if requires_exact_refill:
        user_gen_batch_size = config.data.get("gen_batch_size", None)
        if user_gen_batch_size not in (None, 1):
            logger.warning(f"data.gen_batch_size={user_gen_batch_size} is overridden to 1.")
        elif user_gen_batch_size is None:
            logger.info("data.gen_batch_size defaulted to 1.")
        with open_dict(config):
            config.data.gen_batch_size = 1
    return config.data.get("gen_batch_size", None) or config.data.train_batch_size


def create_custom_dataloaders(
    config: DictConfig, *, trainer_mode: str, tokenizer: Any, processor: Any, collate_fn: Any
) -> tuple[PromptDataLoader, PromptDataLoader | None, int | None, int]:
    """Build train/validation loaders and return their epoch size and validation fetch size.

    Construction owns cleanup until both loaders are successfully returned to the trainer.
    """
    train_batch_size = config.data.train_batch_size
    requested_fetch_size = config.data.get("gen_batch_size")
    if requested_fetch_size is not None and (type(requested_fetch_size) is not int or requested_fetch_size <= 0):
        raise ValueError("data.gen_batch_size must be a positive integer or null.")
    gen_batch_size = resolve_gen_batch_size(config, trainer_mode)
    if type(train_batch_size) is not int or train_batch_size <= 0:
        raise ValueError("data.train_batch_size must be a positive integer.")
    if type(gen_batch_size) is not int or gen_batch_size <= 0 or train_batch_size % gen_batch_size:
        raise ValueError("Custom dataloader fetch size must be a positive divisor of data.train_batch_size.")
    total_steps = config.trainer.total_training_steps
    if total_steps is not None and (type(total_steps) is not int or total_steps <= 0):
        raise ValueError("trainer.total_training_steps must be a positive integer.")

    train = val = None
    try:
        train = create_custom_dataloader(
            config.data,
            split="train",
            batch_size=gen_batch_size,
            tokenizer=tokenizer,
            processor=processor,
            collate_fn=collate_fn,
        )
        if total_steps is not None:
            steps_per_epoch = None
            logger.info("Custom dataloader uses step-based training; trainer.total_epochs is not used.")
        else:
            try:
                num_batches = len(train)
            except TypeError as error:
                raise ValueError("Unsized custom dataloaders require trainer.total_training_steps.") from error
            steps_per_epoch = num_batches * gen_batch_size // train_batch_size
            if steps_per_epoch <= 0:
                raise ValueError("Custom training dataloader must provide at least one full training batch.")

        val_batch_size = config.data.get("val_batch_size")
        val_batch_size = train_batch_size if val_batch_size is None else val_batch_size
        if (
            config.trainer.get("val_before_train", True)
            or config.trainer.get("val_only", False)
            or config.trainer.get("test_freq", -1) > 0
        ):
            val = create_custom_dataloader(
                config.data,
                split="val",
                batch_size=val_batch_size,
                tokenizer=tokenizer,
                processor=processor,
                collate_fn=collate_fn,
            )
            if val is train:
                raise ValueError("Custom train and validation dataloaders must be separate instances.")
        return train, val, steps_per_epoch, val_batch_size
    except BaseException:
        close_dataloaders(train, val)
        raise


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
