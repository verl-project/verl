PPO Code Architecture
=====================

Last updated: 11/10/2026.

``verl.trainer.main_ppo.TaskRunnerV1`` initializes TransferQueue, selects the
trainer using ``trainer.v1.trainer_mode``, and creates ``AgentLoopManagerTQ``.
The trainer initializes workers and runs the training loop.

* ``verl/trainer/ppo/v1/trainer_base.py``: shared trainer lifecycle and PPO updates.
* ``verl/trainer/ppo/v1/trainer_sync.py``: synchronous training.
* ``verl/trainer/ppo/v1/trainer_colocate_async.py``: colocated asynchronous training.
* ``verl/trainer/ppo/v1/trainer_separate_async.py``: separate asynchronous training.
* ``verl/trainer/ppo/trainer_utils.py``: KL penalties, advantages, response masks, and speculative decoding metrics.

See :doc:`../advance/v1_async_trainer` for scheduling and configuration,
and :doc:`../api/trainer` for the trainer API.
