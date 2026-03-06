PyTorch FSDP Backend
======================

Last updated: 12/01/2025.

We support PyTorch FSDP Backend by implementing various workers for
actor, critic, reference, rollout and reward models.

**Pros**

- Readily support various models.

  - Users only need to implement the corresponding
    ``dtensor_weight_loader`` for weight synchronization between FSDP
    and vLLM. While for ``hf_weight_loader``, users can directly apply
    any models supported both in HF and vLLM without any code change.

- Easy to organize the forward and backward computation for each model.

**Cons**

- Poor scalability when it comes to large-scale models (e.g. Llama 70B
  and 405B)
- The resharding overhead between actor and rollout could be larger than
  Megatron-LM backend.

Due to the simplicity, we recommend using FSDP backend for algorithm
research and prototyping.

FSDP Workers
--------------

ActorRolloutRefWorker
^^^^^^^^^^^^^^^^^^^^^

Actor/Rollout HybridEngine
''''''''''''''''''''''''''

1. HybridEngine, Actor and Rollout initialization API.

.. code:: python

   @register(dispatch_mode=Dispatch.ONE_TO_ALL)
   def init_model(self):

``ONE_TO_ALL``: when calling the ``init_model`` function from the driver
process, each worker (on a GPU) will execute the following model
initialization process.

The initialization details of HybridEngine, Actor and Rollout are
highlighted below:

1. ``DataParallelPPOActor`` implements the simple PPO computation logics
   when the model is built with FSDP, including compute log prob, model
   update.
2. ``vLLMRollout`` support generation with vLLM. We modify the vLLM
   Engine and make it executed under SPMD to fit into our
   ``WorkerGroup`` design.

See `source code <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py>`_. for more information.

1. Generate sequence and recompute log prob

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def generate_sequences(self, prompts: DataProto):

- ``Dispatch.DP_COMPUTE_PROTO``: The data will be dispatched and
  collected along the DP dimension

- In this function, the rollout model will perform auto-regressive
  generation and the actor model will recompute the old log prob for the
  generated response.

3. Update actor model

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def update_actor(self, data: DataProto):

- Update the actor model weight using PPO & entropy loss.

ReferenceModel
''''''''''''''

1. Reference model initialization

The reference model is initialized using the same function as the actor
model without initializing the HybridEngine and Optimizer. Then the
actor model is also wrapped by the ``DataParallelPPOActor``.

2. Compute reference log prob

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def compute_ref_log_prob(self, data: DataProto):

- In this function, the reference model will call the compute log prob
  function in ``DataParallelPPOActor`` to compute the reference log
  prob.

CriticWorker and RewardWorker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Model initialization

Quite similar to reference model. The CriticWorker will perform
additional initialization for the Optimizer.

2. Compute Values for CriticWorker

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def compute_values(self, data: DataProto):

3. Update Critic

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def update_critic(self, data: DataProto):

4. Compute Reward

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def compute_rm_score(self, data: DataProto):


HybridShard
------------

We didn't support FSDP `HybridShard`. To support this, we may need to
construct a 2D device mesh and test the corresponding
``dtensor_weight_loader`` and ``hf_weight_loader`` for each model.
