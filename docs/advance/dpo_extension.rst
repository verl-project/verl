Extend to other RL(HF) algorithms
=================================

Last updated: 02/25/2025.

We already implemented the complete training pipeline of the PPO
algorithms. To extend to other algorithms, we analyze the high-level
principle to use verl and provide a tutorial to implement the DPO
algorithm. Users can follow the similar paradigm to extend to other RL algorithms.

.. note:: **Key ideas**: Single process drives multi-process computation and data communication.

Overall Approach
----------------

Step 1: Consider what multi-machine multi-GPU computations are needed
for each model, such as ``generate_sequence`` , ``compute_log_prob`` and
``update_policy`` in the actor_rollout model. Implement distributed
single-process-multiple-data (SPMD) computation and encapsulate them
into APIs

Step 2: Based on different distributed scenarios, including FSDP and 3D
parallelism in Megatron-LM, implement single-process control of data
interaction among multi-process computations.

Step 3: Utilize the encapsulated APIs to implement the control flow

Example: Online DPO
-------------------

We use verl to implement a simple online DPO algorithm. The algorithm
flow of Online DPO is as follows:

1. There is a prompt (rollout) generator which has the same weight as
   the actor model. After a batch of prompts are fed into the generator,
   it generates N responses for each prompt.
2. Send all the prompts + responses to a verifier for scoring, which can
   be reward model or a rule-based function. Then sort them in pairs to
   form a training batch.
3. Use this training batch to train the actor model using DPO. During
   the process, a reference policy is needed.

Step 1: What are the multi-machine multi-GPU computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Sample Generator**

Implementation details:

.. code:: python

   from verl.single_controller.base import Worker
   from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs, RayResourcePool
   import ray

   @ray.remote
   class SampleGenerator(Worker):
       def __init__(self, config):
           super().__init__()
           self.config = config
           
       def generate_sequences(self, data):
           pass

Here, ``SampleGenerator`` can be viewed as a multi-process pulled up by
``torchrun``, with each process running the same code (SPMD).
``SampleGenerator`` needs to implement a ``generate_sequences`` API for
the control flow to call. The implementation details inside can use any
inference engine including vllm, sglang and huggingface. Users can
largely reuse the code in
verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py and we won't
go into details here.

**ReferencePolicy inference**

API: compute reference log probability

.. code:: python

   from verl.single_controller.base import Worker
   import ray

   @ray.remote
   class ReferencePolicy(Worker):
       def __init__(self):
           super().__init__()
           self.model = Model()
           
       def infer(self, data):
           return self.model(data)

**Actor update**

API: Update actor model parameters

.. code:: python

   from verl.single_controller.base import Worker
   import ray

   @ray.remote
   class DPOActor(Worker):
       def __init__(self):
           super().__init__()
           self.model = Model()
           self.model = FSDP(self.model)  # or other distributed strategy
           self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
           self.loss_fn = xxx
           
       def update(self, data):
           self.optimizer.zero_grad()
           logits = self.model(data)
           loss = self.loss_fn(logits)
           loss.backward()
           self.optimizer.step()

**Notes: How to distinguish between control processes and distributed computation processes**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Control processes are generally functions directly decorated with
  ``@ray.remote``
- Computation processes are all wrapped into a ``RayWorkerGroup``.

Users can reuse most of the distribtued computation logics implemented
in PPO algorithm, including FSDP and Megatron-LM backend in
verl/verl/trainer/ppo.

Step 2: Based on different distributed scenarios, implement single-process control of multi-process data interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The core problem to solve here is how a single process sends data to
multiple processes, drives multi-process computation, and how the
control process obtains the results of multi-process computation.**
First, we initialize the multi-process ``WorkerGroup`` in the control
process.

.. code:: python

   @ray.remote(num_cpus=1)
   def main_task(config):
       # construct SampleGenerator
       resource_pool = RayResourcePool(process_on_nodes=[8] * 2)  # 16 GPUs
       ray_cls = RayClassWithInitArgs(SampleGenerator, config=config)
       # put SampleGenerator onto resource pool
       worker_group = RayWorkerGroup(resource_pool, ray_cls)
       
       # construct reference policy

As we can see, in the control process, multiple processes are wrapped
into a ``RayWorkerGroup``. Inside this ``WorkerGroup``, there is a
``self._workers`` member, where each worker is a RayActor
(https://docs.ray.io/en/latest/ray-core/actors.html) of SampleGenerator.
ray_trainer.md also provide an implementation of
``MegatronRayWorkerGroup``.

Assuming the model is distributed using FSDP, and there is a batch of
data on the control process, for data parallelism, the underlying
calling process is:

.. code:: python

   data = xxx
   data_list = data.chunk(dp_size)

   output = []
   for d in data_list:
       # worker_group._workers[i] is a SampleGenerator
       output.append(worker_group._workers[i].generate_sequences.remote(d))

   output = ray.get(output)
   output = torch.cat(output)

Single process calling multiple processes involves the following 3
steps:

1. Split the data into DP parts on the control process.
2. Send the data to remote, call the remote computation through RPC, and
   utilize multi-process computation.
3. Obtain the computation results of each worker on the control process
   and merge them.

Frequently calling these 3 steps on the controller process greatly hurts
code readability. **In verl, we have abstracted and encapsulated these 3
steps, so that the worker's method + dispatch + collect can be
registered into the worker_group**

.. code:: python

   from verl.single_controller.base.decorator import register

   def dispatch_data(worker_group, data):
       return data.chunk(worker_group.world_size)
       
   def collect_data(worker_group, data):
       return torch.cat(data)

   dispatch_mode = {
       'dispatch_fn': dispatch_data,
       'collect_fn': collect_data
   }

   @register(dispatch_mode=dispatch_mode)
   def generate_sequences(self, data):
       pass

In this way, we can directly call the method inside the worker through
the ``worker_group`` on the control (driver) process (which is a single
process):

.. code:: python

   output = worker_group.generate_sequences(data)

This single line includes data splitting, data distribution and
computation, and data collection.

Furthermore, the model parallelism size of each model is usually fixed,
including dp, tp, pp. So for these common distributed scenarios, we have
pre-implemented specific dispatch and collect methods,in `decorator.py <https://github.com/verl-project/verl/blob/main/verl/single_controller/base/decorator.py>`_, which can be directly used to wrap the computations.

.. code:: python

   from verl.single_controller.base.decorator import register, Dispatch

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def generate_sequences(self, data: DataProto) -> DataProto:
       pass

Here it requires the data interface to be ``DataProto``. Definition of
``DataProto`` is in `protocol.py <https://github.com/verl-project/verl/blob/main/verl/protocol.py>`_.

Step 3: Main training loop
~~~~~~~~~~~~~~~~~~~~~~~~~~

With the above training flows, we can implement the algorithm's control
flow. It is recommended that ``main_task`` is also a ray remote process.

.. code:: python

   @ray.remote(num_cpus=1)
   def main_task(config):
       # construct SampleGenerator
       resource_pool = RayResourcePool(process_on_nodes=[8] * 2)  # 16 GPUs
       ray_cls = RayClassWithInitArgs(SampleGenerator, config=config) 
       # put SampleGenerator onto resource pool
       sample_gen = RayWorkerGroup(resource_pool, ray_cls)
       
       # construct reference policy
       ray_cls = RayClassWithInitArgs(ReferencePolicy)
       ref_policy = RayWorkerGroup(resource_pool, ray_cls)
       
       # construct actor
       ray_cls = RayClassWithInitArgs(DPOActor)  
       dpo_policy = RayWorkerGroup(resource_pool, ray_cls)
       
       dataloader = DataLoader()
       
       for data in dataloader:
           # generate data
           data = sample_gen.generate_sequences(data)
           # generate scores for each data 
           data = generate_scores(data)
           # generate pairwise data using scores
           data = generate_pairwise_data(data)
           # generate ref_log_prob
           data.batch['ref_log_prob'] = ref_policy.infer(data)
           # update using dpo
           dpo_policy.update(data)
           # logging

Here, different ``WorkerGroups`` can be placed in the same resource pool or
in different resource pools using ``create_colocated_worker_cls``
similar as in `ray_trainer.py <https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/ray_trainer.py>`_.
