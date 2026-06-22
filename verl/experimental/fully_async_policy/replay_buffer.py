import random

from verl.protocol import DataProto, DataProtoItem


class ReplayBuffer:
    """Global experience replay buffer for storing positive trajectory samples.

    Stores positive turn-level samples in a single global pool. When a GRPO group
    has all-negative rewards, a positive replay sample can be injected to keep a
    useful gradient signal.
    """

    def __init__(self, buffer_size: int = 64, reward_threshold: float = 0.1):
        self.buffer_size = buffer_size
        self.reward_threshold = reward_threshold
        self.pos_samples: list[DataProtoItem] = []

    def update(self, batch: DataProto):
        """Store positive samples from the current batch."""
        scores = batch.batch["rm_scores"].sum(dim=-1)
        for i in range(len(scores)):
            if scores[i].item() > self.reward_threshold:
                sample = batch[i]
                self.pos_samples.append(sample)
                if len(self.pos_samples) > self.buffer_size:
                    self.pos_samples.pop(0)

    def get_positive(self, n: int = 1):
        """Retrieve random positive samples from the global pool."""
        if not self.pos_samples:
            return None
        return random.choices(self.pos_samples, k=n)

    def has_positive(self) -> bool:
        return len(self.pos_samples) > 0

    def total_size(self) -> int:
        return len(self.pos_samples)
