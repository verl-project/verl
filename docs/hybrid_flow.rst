=========================================================
HybridFlow Programming Guide
=========================================================

Last updated: 06/02/2025.

.. _vermouth: https://github.com/vermouth1992

Author: `Chi Zhang <https://github.com/vermouth1992>`_

verl is an open source implementation of the paper `HybridFlow <https://arxiv.org/abs/2409.19256v2>`_ [1]_. In this section, we will introduce the basic concepts of HybridFlow, the motivation and how to program with verl APIs.

Motivation and Design
------------------------
We use dataflow to represent RL systems. [4]_.

DataFlow
~~~~~~~~~~~~~~~~~~~~

Dataflow is an abstraction of computations. Neural Network training is a typical dataflow. It can be represented by computational graph. 

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/dataflow.jpeg?raw=true
   :alt: The dataflow graph from CS231n 2024 lecture 4

This figure [2]_ represents the computation graph of a polynomial function followed by a sigmoid function. In the data flow of neural network computation, each node represents an operator, and each edge represents the direction of forward/backward propagation. The computation graph determines the architecture of the neural network.

RL as a dataflow problem
++++++++++++++++++++++++++++++++++++++++++++++

Reinforcement learning (RL) training can also be represented as a dataflow. Below is the dataflow graph that represents the PPO algorithm used in RLHF [3]_:

.. image:: https://picx.zhimg.com/70/v2-cb8ab5ee946a105aab6a563e92682ffa_1440w.avis?source=172ae18b&biz_tag=Post
  :alt: PPO dataflow graph, credit to Zhihu 低级炼丹师

However, the dataflow of RL has fundamental differences compared with dataflow of neural network training as follows:

+--------------------------+--------------------------------------------------+---------------------+
| Workload                 | Node                                             | Edge                |
+--------------------------+--------------------------------------------------+---------------------+
| Neural Network Training  | Operator (+/-/matmul/softmax)                    | Tensor movement     |
+--------------------------+--------------------------------------------------+---------------------+
| Reinforcement Learning   | High-level operators (rollout/model forward)     | Data Movement       |
+--------------------------+--------------------------------------------------+---------------------+

In the case of tabular reinforcement learning, each operator is a simple scalar math operation (e.g., bellman update). In deep reinforcement learning(DRL), each operator is a high-level neural network computation such as model inference/update. This makes RL a two-level dataflow problem:

- Control flow: defines how the high-level operators are executed (e.g., In PPO, we first perform rollout. Then, we perform advantage computation. Finally, we perform training). It expresses the **core logics of RL algorithms**.
- Computation flow: defines the dataflow of **neural network computation** (e.g., model forward/backward/optimizer).


Design Choices
~~~~~~~~~~~~~~~~~~~~
The model size used in DRL before the LLM era is typically small. Thus, the high-level neural network computation can be done in a single process. This enables embedding the computation flow inside the control flow as a single process.

However, in the LLM era, the computation flow (e.g., training neural network) becomes a multi-process program. This naturally leads to two design choices:

1. Convert the control flow into a multi-process program as well. Then colocate with computation flow (unified multi-controller)

- Advantages:

  - Achieves the **optimal performance** under fixed computation flow and control flow as the communication overhead in both training and data transfer is minimized.

- Disadvantages:

  - The computation and/or control flow is **hard to reuse** from software perspective as computation code is coupled with specific controller code. For example, the training loop of PPO is generic. Say we have an PPO training flow implemented with a specific computation flow such as FSDP. Neither the control flow or computation flow can be reused if we want to switch the computation flow from FSDP to Megatron, due to the coupling of control and computation flows.
  - Requires more efforts from the user under flexible and dynamic control flows, due to the multi-process nature of the program.

2. Separate the flows: single process for the control flow and multi-process for computation flow

- Advantages:

  - The computation flow defined elsewhere can be **easily reused** after the decoupling.
  - The controller runs on a single process. Implementing a new RL algorithm with a **different control flow is simple and easy**.

- Disadvantages:

  - Additional **data communication overhead** each time the controller process and computatation processes interact. The data has to be sent back and forth.

In verl, the latter strategy with separate control flow and computation flow is adopted. verl is designed to decouple the control flow of RL algorithms, and the implementation of computation engines.

Overall Execution Diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a simplified diagram denoting the execution of a reinforcement learning job. In the diagram, the controller runs on a single process, while the generator/actor workers, critic workers run on multiple processes, placed with specific resource groups. For rollout, the controller passes the data to the generator to perform sample generation. When the rollout is done, the data is passed back to controller for the next step of the algorithm. Similar execution is done for other workers. With the hybrid controller design, the data flow and computation is decoupled to provide both efficiency in computation and flexibility in defining algorithm training loops.

.. figure:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/driver_worker.png?raw=true
   :alt: The execution diagram

Codebase walkthrough (PPO)
------------------------------------------------

Entry function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Code: https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py

In this file, we define a remote function `main_task` that serves as the controller (driver) process as shown in the above figure. We also define a ``RewardManager``, where users can customize their reward function based on the data source in the dataset. Note that `RewardManager` should return the final token-level reward that is optimized by RL algorithms. Note that users can combine model-based rewards and rule-based rewards.
The ``main_task`` constructs a RayPPOTrainer instance and launch the fit. Note that ``main_task`` **runs as a single process**.

We highly recommend that the ``main_task`` is NOT scheduled on the head of the ray cluster because ``main_task`` will consume a lot of memory but the head usually contains very few resources.

Ray trainer
~~~~~~~~~~~~~~~~~~~~
Code: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py

The RayPPOTrainer manages 

- Worker and WorkerGroup construction
- Runs the main loop of PPO algorithm

Note that, the fit function of RayPPOTrainer **runs as a single process**.

Worker and WorkerGroup construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each workerGroup manages a list of workers that runs remotely. Note that the worker group runs in the process of its constructor.
Each worker inside the WorkerGroup runs on a GPU. The worker group serves as a proxy for the controller process to interact with a list of workers, in order to perform certain computations. **In order to do so, we have to bind the methods of the worker into the method of the WorkerGroup and define the data dispatch and data collection**. This is done via simple decoration that will be introduced in the Worker definition section.

For example, in PPO, we define 3 worker groups:

- ActorRolloutRef: manages actor, rollout and reference policy. ActorRolloutRefWorker can be instantiated as a single actor, a single rollout, a single reference policy, a combined actor/rollout or a combined actor/rollout/ref. This design is aimed for the maximum code reuse in various scenarios. The reason for colocating actor and rollout is for fast weight transfer using nccl. The reason for coloating actor and reference is to implement an efficient lora PPO as the reference policy is simply the base model of PPO in lora. The colocation is done via ``verl.single_controller.ray.base.create_colocated_worker_cls``, where it creates a single ray remote class exposing all class methods from these roles.
- Critic: manages the critic model
- Reward: manages the reward model

The worker group will be constructed on the resource pool it designates. The resource pool is a set of GPUs in the ray cluster.

Worker definition
~~~~~~~~~~~~~~~~~~~~

.. _ActorRolloutRefWorker: https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py

We take `ActorRolloutRefWorker <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py>`_ for an example.
The APIs it should expose to the controller process are:

- init_model: build the underlying model
- generate_sequences: given prompts, generate responses
- compute_log_prob: compute the log-probability of a generated sequence using actor
- compute_ref_log_prob: compute the log-probability of a generated sequence using reference policy
- save_checkpoint: save the checkpoint

Note that these methods are defined in the worker that can only be invoked via remote calls. For example, if the controller process wants to initialize the model, it has to call

.. code-block:: python

   for worker in actor_rollout_ref_wg:
       worker.init_model.remote()

If the controller process wants to generate sequences, it has to call

.. code-block:: python

   data = xxx
   # split the data into dp chunks
   data_dp_lst = data.split(dp_size)
   output_dp_lst = []
   for i, worker in enumerate(actor_rollout_ref_wg):
       output_future = worker.generate_sequences.remote(data_dp_lst[i])
       output_dp_lst.append(output_future)
   output = torch.cat(ray.get(output_dp_lst), dim=0)

We observe that controller process calling worker group methods in general can be divided into 3 parts:

- Split the data into data parallel sizes
- Dispatch the corresponding data into each worker
- Collect and concatenate the data when the computation finishes

In verl, we design a syntax sugar to encapsulate the 3 processes into a single call from the controller process.

.. code-block:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def generate_sequences(data):
       ...

   # on the driver
   output = actor_rollout_ref_wg.generate_sequences(data)

We decorate the method of the worker with a ``register`` that explicitly defines how the input data should be split and dispatched to each worker, and how the output data should be collected and concatenated by the controller. For example, ``Dispatch.DP_COMPUTE_PROTO`` splits the input data into dp chunks, dispatch each data to each worker, collect the output and concatenate the results. Note that this function requires the input and output to be a DataProto defined here (https://github.com/volcengine/verl/blob/main/verl/protocol.py).


PPO main loop
~~~~~~~~~~~~~~~~~~~~
With the aforementioned APIs, we can implement the main loop of PPO as if it is a single process program

.. code-block:: python

   for prompt in dataloader:
       output = actor_rollout_ref_wg.generate_sequences(prompt)
       old_log_prob = actor_rollout_ref_wg.compute_log_prob(output)
       ref_log_prob = actor_rollout_ref_wg.compute_ref_log_prob(output)
       values = critic_wg.compute_values(output)
       rewards = reward_wg.compute_scores(output)
       # compute_advantages is running directly on the control process
       advantages = compute_advantages(values, rewards)
       output = output.union(old_log_prob)
       output = output.union(ref_log_prob)
       output = output.union(values)
       output = output.union(rewards)
       output = output.union(advantages)
       # update actor
       actor_rollout_ref_wg.update_actor(output)
       critic.update_critic(output)

Takeaways
~~~~~~~~~~~~~~~~~~~~
- This programming paradigm enables users to use different computation backend without modification of the control process.
- This programming paradigm enables flexible placement (by changing the mapping of WorkerGroup and ResourcePool) without modification of the control process.

Repository organization
------------------------------------------------

Important code files in the repository are organized as below:

.. code-block:: bash

   verl # the verl package
     trainer
       main_ppo.py  # the entrypoint for RL training
       ppo
         ray_trainer.py  # the training loop for RL algorithms such as PPO
       sft_trainer.py  # the SFT trainer with FSDP backend
     config
       generation.yaml  # configuration template for rollout
       ppo_trainer.yaml  # configuration template for the RL trainer
     workers
       protocol.py  # the interface of DataProto
       fsdp_workers.py   # the FSDP worker interfaces: ActorRolloutRefWorker, CriticWorker, RewardModelWorker
       megatron_workers.py  # the Megatron worker interfaces: ActorRolloutRefWorker, CriticWorker, RewardModelWorker
       actor
         dp_actor.py  #  data parallel actor with FSDP backend
         megatron_actor.py  # nD parallel actor with Megatron backend
       critic
         dp_critic.py  # data parallel critic with FSDP backend
         megatron_critic.py  # nD parallel critic with FSDP backend
       reward_model
         megatron
           reward_model.py  # reward model with Megatron backend
       rollout
         vllm
           vllm_rollout.py  # rollout with vllm backend
         hf_rollout.py  # rollout with huggingface TGI backend
       sharding_manager
         fsdp_ulysses.py  # data and model resharding when using FSDP + ulysses
         fsdp_vllm.py  # data and model resharding when using FSDP + ulysses + vllm
         megatron_vllm.py  # data and model resharding when using Megatron + vllm
     utils
       dataset  # datasets for SFT/RM/RL
       reward_score  # function based reward
         gsm8k.py  # reward function for gsm8k dataset
         math.py  # reward function for math dataset
       seqlen_balancing.py  # the sequence balance optimization
     models
       llama  # Megatron implementation for llama, deepseek, mistral, etc
       transformers  # ulysses integration with transformer models such as llama, qwen, etc
       weight_loader_registery.py  # registry of weight loaders for loading hf ckpt into Megatron
     third_party
       vllm  # adaptor for vllm's usage in RL
         vllm_spmd  # vllm >= v0.7 adaptor
   examples  # example scripts
   tests  # integration and unit tests
   .github  # the configuration of continuous integration tests


.. [1] HybridFlow: A Flexible and Efficient RLHF Framework: https://arxiv.org/abs/2409.19256v2
.. [2] Data flow graph credit to CS231n 2024 lecture 4: https://cs231n.stanford.edu/slides/2024/lecture_4.pdf
.. [3] PPO dataflow graph credit to 低级炼丹师 from Zhihu​: https://zhuanlan.zhihu.com/p/635757674
.. [4] RLFlow
