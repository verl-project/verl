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

"""Exercise both production SFT loops without importing distributed backends."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _load_fit(mode, namespace):
    """Load the actual trainer loop while substituting its runtime dependencies."""
    filename = "sft_trainer_ray.py" if mode == "ray" else "sft_trainer.py"
    path = Path(__file__).resolve().parents[2] / "verl/trainer" / filename
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SFTTrainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["fit"]


@pytest.mark.parametrize("mode", ["spmd", "ray"])
@pytest.mark.parametrize("async_save", [False, True])
@pytest.mark.parametrize(
    "step_limit,resume_step,save_freq,expected_saves",
    [
        (10, 0, 3, [(3, False), (6, True)]),
        (10, 0, 4, [(4, False), (6, True)]),
        (4, 0, 3, [(3, False), (4, True)]),
        (6, 0, 3, [(3, False), (6, True)]),
        (10, 2, 3, [(3, False), (6, True)]),
        (10, 3, 3, [(6, True)]),
        (10, 0, -1, [(6, True)]),
    ],
)
def test_final_sft_save_publishes_before_exit(mode, async_save, step_limit, resume_step, save_freq, expected_saves):
    """The epoch or step limit must publish every pending save before returning."""
    saves, pending, published = [], [], []
    steps_per_epoch = 3

    class ResumedLoader:
        """Yield the remaining first epoch, then full epochs after resumption."""

        first_epoch = True

        def __iter__(self):
            """Apply the resumed batch offset only to the first iteration."""
            count = steps_per_epoch - resume_step % steps_per_epoch if self.first_epoch else steps_per_epoch
            self.first_epoch = False
            return iter([{} for _ in range(count)])

    def save_checkpoint(*, step, blocking):
        """Keep async saves pending until a blocking save publishes the queue."""
        saves.append((step, blocking))
        pending.append(step)
        if blocking or not async_save:
            published.extend(pending)
            pending.clear()

    engine = MagicMock()
    engine.is_mp_src_rank_with_outputs.return_value = False
    client = MagicMock()

    def train_batch(*args, **kwargs):
        """Return minimal metrics using the direct or Ray-style result interface."""
        metrics = {"loss": 1.0, "grad_norm": 1.0, "lr": 0.1, "mfu": 0.1}
        result = {"metrics": metrics}
        return SimpleNamespace(get=lambda: result) if mode == "ray" else result

    client.train_batch.side_effect = train_batch
    fake_torch = MagicMock()
    fake_torch.sum.return_value.item.return_value = 1
    namespace = {
        "Tracking": MagicMock(),
        "OmegaConf": SimpleNamespace(to_container=lambda *args, **kwargs: {}),
        "log_with_rank": MagicMock(),
        "logger": None,
        "aggressive_empty_cache": MagicMock(),
        "log_gpu_memory_usage": MagicMock(),
        "tqdm": lambda data, **kwargs: data,
        "NonTensorData": lambda data: data,
        "torch": fake_torch,
        "tu": SimpleNamespace(
            get_tensordict=lambda **kwargs: kwargs["tensor_dict"],
            assign_non_tensor=lambda *args, **kwargs: None,
            get=lambda data, key: data[key],
        ),
    }
    trainer = SimpleNamespace(
        engine=engine,
        training_client=client,
        resume_global_step=resume_step,
        total_training_steps=step_limit,
        steps_per_epoch=steps_per_epoch,
        config=SimpleNamespace(
            trainer=SimpleNamespace(
                total_epochs=2, balance_batch=False, project_name="test", experiment_name="test", logger=[]
            ),
            model=SimpleNamespace(use_remove_padding=False),
            data=SimpleNamespace(
                use_dynamic_bsz=False,
                max_token_len_per_gpu=16,
                micro_batch_size_per_gpu=1,
                pad_mode="no_padding",
            ),
        ),
        checkpoint_config=SimpleNamespace(async_save=async_save),
        model_config=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        global_batch_size=1,
        train_sampler=MagicMock(),
        train_dataloader=ResumedLoader(),
        val_dataloader=None,
        rank=0,
        device_name="cpu",
        start_profile_step=-1,
        end_profile_step=-1,
        test_freq=-1,
        save_freq=save_freq,
        ckpt_handler=SimpleNamespace(save_checkpoint=save_checkpoint),
        _get_batch_seqlens=lambda **kwargs: SimpleNamespace(tolist=lambda: [1]),
    )

    _load_fit(mode, namespace)(trainer)

    assert saves == expected_saves
    assert pending == []
    assert published == [step for step, _ in expected_saves]
    assert client.train_batch.call_count == min(step_limit, 6) - resume_step
    assert trainer.total_training_steps == step_limit
