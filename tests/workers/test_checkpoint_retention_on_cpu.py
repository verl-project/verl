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

"""Worker retention deferral must reach engines before either save phase can delete files."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker


@pytest.mark.parametrize("defer,expected_limit", [(False, 1), (True, None)])
def test_worker_disables_retention_for_coordinated_save(defer, expected_limit):
    engine = SimpleNamespace(supports_deferred_checkpoint_retention=True, save_checkpoint=Mock())
    worker = SimpleNamespace(engine=engine)
    TrainingWorker.save_checkpoint(
        worker, "checkpoint", global_step=2, max_ckpt_to_keep=1, defer_retention=defer, update_tracker=False
    )
    engine.save_checkpoint.assert_called_once_with("checkpoint", None, 2, expected_limit, update_tracker=False)


def test_unsupported_engine_fails_before_saving():
    engine = SimpleNamespace(supports_deferred_checkpoint_retention=False, save_checkpoint=Mock())
    with pytest.raises(NotImplementedError, match="deferred checkpoint retention"):
        TrainingWorker.save_checkpoint(SimpleNamespace(engine=engine), "checkpoint", defer_retention=True)
    engine.save_checkpoint.assert_not_called()


def test_actor_wrapper_forwards_deferral_and_post_publication_pruning():
    actor = SimpleNamespace(save_checkpoint=Mock(), prune_checkpoints=Mock())
    outer = SimpleNamespace(actor=actor, role="actor")
    ActorRolloutRefWorker.save_checkpoint(
        outer, "checkpoint", global_step=2, max_ckpt_to_keep=1, defer_retention=True, update_tracker=False
    )
    actor.save_checkpoint.assert_called_once_with("checkpoint", None, 2, 1, defer_retention=True, update_tracker=False)
    ActorRolloutRefWorker.prune_checkpoints(outer, max_ckpt_to_keep=1)
    actor.prune_checkpoints.assert_called_once_with(1)
    engine = SimpleNamespace(prune_checkpoints=Mock())
    TrainingWorker.prune_checkpoints(SimpleNamespace(engine=engine), max_ckpt_to_keep=1)
    engine.prune_checkpoints.assert_called_once_with(1)
