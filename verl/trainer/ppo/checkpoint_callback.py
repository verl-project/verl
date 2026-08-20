# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Driver-side checkpoint callback hook for the PPO trainers.

Users plug in a custom callback by setting ``trainer.checkpoint_callback_class``
to the fully qualified class name of a :class:`CheckpointCallback` subclass.
The trainer instantiates it on the driver process and calls :meth:`CheckpointCallback.on_save`
after each checkpoint save, mirroring the ``on_save`` event of the HuggingFace
``transformers`` ``TrainerCallback``. The hook runs only on the driver, after the
worker-group RPCs; per-rank hooks inside the FSDP/Megatron checkpoint managers
are out of scope.

Exceptions raised by the hook propagate and abort training: checkpoint callbacks
typically perform durability-critical side effects (uploading shards, registering
a model version), and silently swallowing a failure could lose checkpoints without
any signal. Wrap the hook body in ``try/except`` for best-effort semantics.
"""

from verl.utils.import_utils import load_class_from_fqn


class CheckpointCallback:
    """No-op base class for trainer checkpoint callbacks.

    Subclass and override :meth:`on_save`. The hook is invoked with keyword
    arguments and must accept ``**kwargs`` so that additional context can be
    passed in future versions without breaking user subclasses.
    """

    def __init__(self, config=None):
        """Store the full trainer config for use by subclass hooks.

        Args:
            config: The full trainer ``DictConfig``.
        """
        self.config = config

    def on_save(self, trainer, global_step: int, checkpoint_dir: str, async_save: bool = False, **kwargs) -> None:
        """Event called after a checkpoint save.

        Not called when a save step raises. With ``async_save=True`` (Megatron
        asynchronous checkpointing), worker-side writes may still be in flight
        when this fires and ``latest_checkpointed_iteration.txt`` has not been
        written, so the checkpoint must not be assumed durable yet.

        Args:
            trainer: The trainer instance performing the save.
            global_step: The training step that was checkpointed.
            checkpoint_dir: The local ``global_step_{N}`` directory that was written.
            async_save: Whether the checkpoint was saved asynchronously.
            **kwargs: Reserved for future context.
        """


def build_checkpoint_callback(config) -> CheckpointCallback:
    """Instantiate the callback named by ``trainer.checkpoint_callback_class``.

    Returns a no-op :class:`CheckpointCallback` when the key is unset or null, so
    trainer call sites never need a None guard.

    Args:
        config: The full trainer ``DictConfig``.

    Returns:
        A :class:`CheckpointCallback` instance constructed with ``config``.
    """
    fqn = config.trainer.get("checkpoint_callback_class", None)
    if not fqn:
        return CheckpointCallback(config=config)
    callback_cls = load_class_from_fqn(fqn, "CheckpointCallback")
    return callback_cls(config=config)
