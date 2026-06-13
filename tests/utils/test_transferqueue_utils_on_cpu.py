import torch
from tensordict import TensorDict

from verl.utils import transferqueue_utils as tq_utils
from verl.utils.transferqueue_utils import force_tq_sequence_fields_nested


def test_force_tq_sequence_fields_nested_converts_dense_sequence_tensors():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "response_mask": torch.tensor([[0, 1], [1, 1]]),
            "position_ids": torch.arange(2 * 4 * 3, dtype=torch.long).reshape(2, 4, 3),
            "num_turns": torch.tensor([1, 2]),
        },
        batch_size=[2],
    )

    normalized = force_tq_sequence_fields_nested(data)

    assert normalized["prompts"].is_nested
    assert normalized["attention_mask"].is_nested
    assert normalized["response_mask"].is_nested
    assert normalized["position_ids"].is_nested
    assert normalized["position_ids"][0].shape == (4, 3)
    assert not normalized["num_turns"].is_nested


def test_install_tq_nested_readback_wrappers_can_be_env_gated(monkeypatch):
    class FakeTQ:
        def __init__(self):
            self.kv_batch_get_calls = 0

        def kv_batch_get(self, *args, **kwargs):
            self.kv_batch_get_calls += 1
            return TensorDict(
                {
                    "prompts": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                    "num_turns": torch.tensor([1, 2]),
                },
                batch_size=[2],
            )

    fake_tq = FakeTQ()
    monkeypatch.setattr(tq_utils, "tq", fake_tq)
    monkeypatch.setenv("VERL_FORCE_TQ_NESTED_READBACK", "1")

    tq_utils._install_tq_nested_readback_wrappers()

    normalized = tq_utils.tq.kv_batch_get(keys=["a", "b"], partition_id="train")
    assert normalized["prompts"].is_nested
    assert normalized["attention_mask"].is_nested
    assert not normalized["num_turns"].is_nested
    assert fake_tq.kv_batch_get_calls == 1
