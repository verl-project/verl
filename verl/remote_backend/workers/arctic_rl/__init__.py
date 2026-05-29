"""Arctic-RL backend worker.

Used when `trainer.remote_backend=arctic`. Pairs with the adapter at
:mod:`verl.workers.remote_client.arctic_rl`.
"""

from verl.remote_backend.workers.arctic_rl.worker import ArcticRLActorRolloutRefWorker

__all__ = ["ArcticRLActorRolloutRefWorker"]
