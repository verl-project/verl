"""Exercise the actual driver method without optional cluster/TQ dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("setting", [None, True, False])
def test_old_log_probs_preserved_when_entropy_monitoring_is_disabled(setting):
    source = Path(__file__).resolve().parents[3] / "verl/trainer/ppo/v1/trainer_base.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPOTrainer")
    method = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_old_log_prob"
    )
    enabled = setting is not False
    log_probs = torch.tensor([[-.3, -.8]])
    entropy = torch.tensor([[.2, .6]])
    requests, stored = [], {}

    class Fields(dict):
        def select(self, *names):
            return Fields({name: self[name] for name in names})

        def to_padded_tensor(self):
            return self

    class Batch:
        keys = ["trajectory"]
        partition_id = "train"
        extra_info = {}

        def __len__(self):
            return 1

    batch = Batch()
    values = Fields(log_probs=log_probs, response_mask=torch.ones_like(log_probs),
                    responses=torch.tensor([[1, 2]]), rollout_log_probs=log_probs.clone())

    def compute(data):
        assert data.extra_info["calculate_entropy"] is enabled
        if enabled:
            values["entropy"] = entropy
        return data

    def get(**kwargs):
        requests.append(kwargs["select_fields"])
        return values.select(*kwargs["select_fields"])

    def put(**kwargs):
        stored.update(kwargs["fields"])
        return batch

    def aggregate(loss_mat, **kwargs):
        assert enabled, "Entropy reduction must not run when disabled"
        return loss_mat.mean()

    def debug(data):
        assert "old_log_probs" in data.batch and "rollout_log_probs" in data.batch
        return {"debug_metric": 1}

    trainer_options = {} if setting is None else {"log_old_policy_entropy": setting}
    controller = SimpleNamespace(
        config=SimpleNamespace(
            trainer=trainer_options, algorithm={},
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(temperature=1.0, calculate_log_probs=True),
                actor=SimpleNamespace(loss_agg_mode="seq-mean-token-mean", loss_scale_factor=None),
            ),
        ),
        actor_rollout_wg=SimpleNamespace(compute_log_prob=compute),
    )
    namespace = {
        "KVBatchMeta": Batch, "DataProto": SimpleNamespace,
        "tq": SimpleNamespace(kv_batch_get=get, kv_batch_put=put),
        "response_from_nested": lambda value, mask: value,
        "agg_loss": aggregate, "calculate_debug_metrics": debug,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    metrics = {}
    result = namespace["_compute_old_log_prob"](controller, batch, metrics)
    assert result is batch and torch.equal(stored["old_log_probs"], log_probs)
    assert ("entropy" in stored) is enabled
    assert all(("entropy" in request) is enabled for request in requests)
    assert ("actor/entropy" in metrics) is enabled
    assert metrics["debug_metric"] == 1
