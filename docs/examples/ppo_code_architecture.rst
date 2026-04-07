PPO Example Architecture
========================

Last updated: 02/17/2025.

Let's start with the Proximal Policy Optimization algorithm, which is
most widely used algorithm in LLM post-training.

The main entry point of the PPO algorithm example is:
`main_ppo.py <https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py>`_.
In this tutorial, we will go through the code architecture in `main_ppo.py <https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py>`_.

Define the data
---------------

Users need to preprocess and store the dataset in parquet files.
And we implement `RLHFDataset` to load and tokenize the parquet files.

For ``RLHFDataset`` (Default), at least 1 fields are required:

- ``prompt``: Contains the string prompt

We already provide some examples of processing the datasets to parquet
files in `data_preprocess directory <https://github.com/volcengine/verl/blob/main/examples/data_preprocess>`_. Currently, we support
preprocess of GSM8k, MATH, Hellasage, Full_hh_rlhf datasets. See :doc:`../preparation/prepare_data` for
more information.

Define the reward functions for different datasets
--------------------------------------------------

In this main entry point, the users only need to define their own reward
function based on the datasets (or applications) utilized in PPO
training.

For example, we already provide reward functions for `GSM8k <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py>`_ 
and `MATH <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math.py>`_
datasets in the ``_select_rm_score_fn``. In the ``RewardManager``, we
will compute the reward score based on the data_source to select
corresponding reward functions. For some RLHF datasets (e.g.,
full_hh_rlhf), the reward model is utilized to assess the responses
without any reward functions. In this case, the ``RewardManager`` will
return the ``rm_score`` computed by the reward model directly.

See `reward functions <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score>`_ for detailed implementation.

Define worker classes
---------------------

.. code:: python

   from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
   from verl.single_controller.ray import RayWorkerGroup

   ray_worker_group_cls = RayWorkerGroup
   CriticWorker = TrainingWorker

   from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

   role_worker_mapping = {
       Role.ActorRollout: ActorRolloutRefWorker,
       Role.Critic: CriticWorker,
       Role.RefPolicy: ActorRolloutRefWorker
   }

   global_pool_id = 'global_pool'
   resource_pool_spec = {
       global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
   }
   mapping = {
       Role.ActorRollout: global_pool_id,
       Role.Critic: global_pool_id,
       Role.RefPolicy: global_pool_id,
   }

Step 1: Construct the mapping between roles and workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A role represents a group of workers in the same process. We have
pre-defined several roles in `ray_trainer.py <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py#L38>`_.

.. code:: python

   class Role(Enum):
       """
       To create more roles dynamically, you can subclass Role and add new members
       """
       Actor = 0  # This worker only has Actor
       Rollout = 1 # This worker only has Rollout
       ActorRollout = 2 # This worker has both actor and rollout, it's a HybridEngine
       Critic = 3 # This worker only has critic
       RefPolicy = 4 # This worker only has reference policy
       RewardModel = 5 # This worker only has reward model
       ActorRolloutRef = 6 # This worker contains actor, rollout and reference policy simultaneously 

Step 2: Define the worker class corresponding to this role
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- We have pre-implemented the ``ActorRolloutRefWorker``. Through
  different configs, it can be a standalone actor, a standalone rollout,
  an ActorRollout HybridEngine, or an ActorRolloutRef HybridEngine
- We also pre-implemented workers for ``Actor``, ``Rollout``,
  ``Critic``, ``Reward Model`` and ``Reference model`` supporting both FSDP
  and Megatron-LM backends.
  See `engine_workers.py <https://github.com/verl-project/verl/blob/main/verl/workers/engine_workers.py>`_
  for more information.

Step 3: Define resource pool id and resource pool spec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Resource pool is a division of global GPU resources,
  ``resource_pool_spec`` is a dict, mapping from id to # of GPUs

  - In the above example, we defined a global resource pool:
    global_pool_id, and then put all roles on this one resource pool
    with all the GPUs in this post-training task. This refers to
    *co-locate* placement where all the models share the same set of
    GPUs.

- See resource pool and placement for advance usage.

Defining reward model/function
------------------------------

.. code:: python

   # we should adopt a multi-source reward function here
   # - for rule-based rm, we directly call a reward score
   # - for model-based rm, we call a model
   # - for code related prompt, we send to a sandbox if there are test cases
   # - finally, we combine all the rewards together
   # - The reward type depends on the tag of the data
   if config.reward_model.enable:
       from verl.workers.engine_workers import TrainingWorker as RewardModelWorker
       role_worker_mapping[Role.RewardModel] = RewardModelWorker
       mapping[Role.RewardModel] = global_pool_id
    
   reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

   # Note that we always use function-based RM for validation
   val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

   resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

Since not all tasks use model-based RM, users need to define here
whether it's a model-based RM or a function-based RM

- If it's a model-based RM, directly add the ``RewardModel`` role in the
  resource mapping and add it to the resource pool mapping.

  - Note that the pre-defined ``RewardModelWorker`` only supports models
    with the structure of huggingface
    ``AutoModelForSequenceClassification``. If it's not this model, you
    need to define your own RewardModelWorker in
    `engine_workers.py <https://github.com/verl-project/verl/blob/main/verl/workers/engine_workers.py>`_.

- If it's a function-based RM, the users are required to classified the
  reward function for each datasets.

.. code:: python

   def _select_rm_score_fn(data_source):
       if data_source == 'openai/gsm8k':
           return gsm8k.compute_score
       elif data_source == 'lighteval/MATH':
           return math.compute_score
       else:
           raise NotImplementedError

See reward functions implemented in `directory <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/>`_ 
for more information.

Define, init and run the PPO Trainer
------------------------------------

.. code:: python

   trainer = RayPPOTrainer(config=config,
                           tokenizer=tokenizer,
                           role_worker_mapping=role_worker_mapping,
                           resource_pool_manager=resource_pool_manager,
                           ray_worker_group_cls=ray_worker_group_cls,
                           reward_fn=reward_fn,
                           val_reward_fn=val_reward_fn)
   trainer.init_workers()
   trainer.fit()

- We first initialize the ``RayPPOTrainer`` with user config, tokenizer
  and all the above worker mapping, resource pool, worker group and
  reward functions
- We first call the ``trainer.init_workers()`` to initialize the models
  on the allocated GPUs (in the resource pool)
- The actual PPO training will be executed in ``trainer.fit()``

verl can be easily extended to other RL algorithms by reusing the Ray
model workers, resource pool and reward functions. See :doc:`extension<../advance/dpo_extension>` for
more information.

Details of the ``RayPPOTrainer`` is discussed in :doc:`Ray Trainer<../workers/ray_trainer>`.
