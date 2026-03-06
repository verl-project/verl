Reward Loop
===========

.. _yyding: https://yyding1.github.io

Author: `Yuyang Ding <https://yyding1.github.io>`_

Last updated: 2/10/2026.

Introduction
------------

Reward Loop is the default reward computation implementation in verl.
It is designed to support efficient, flexible, and easy-to-use reward computation.

This document introduces the usage and architectural design.

Key features include:

1. **Distributed reward manager**, enabling scalable and efficient reward computation.
2. **Support for hybrid reward settings**, including both generative and discriminative reward models, as well as more complex reward scenarios.
3. **Simple and extensible interface**, for easily defining customized reward functions.

Distributed Reward manager
--------------------------

.. image:: https://github.com/yyDing1/verl-materials/blob/main/distributed_reward_manager.svg?raw=true

How distributed
~~~~~~~~~~~~~~~

Under the single_controller setup, actor rollout and reward computation can be abstracted as:

.. code:: python

   # initalize rollout manager and async reward loop manager
   async_rollout_manager = AgentLoopManager(config)
   async_reward_manager = RewardLoopManager(config)
   # actor rollout using `async_rollout_manager`
   gen_batch = async_rollout_manager.generate_sequences(batch)
   # compute reward using `async_reward_manager`
   reward_batch = async_reward_manager.compute_rm_score(gen_batch)

Within the ``RewardLoopManager``, multiple ``RewardWorker`` are launched across all nodes to enable distributed reward computation. 
The number of parallel workers can be configured via ``config.reward.num_workers``.

Upon receiving a batch reward request, the batch is partitioned into smaller chunks and distributed to each reward worker for parallel execution.
User only need to invoke ``compute_rm_score``.

.. code:: python

   class RewardLoopManager:
      """
      RewardLoopManager run in single controller.
      This class will create reward loop workers and manage them.
      """
      def _init_reward_loop_workers(self):
         self.reward_loop_workers = [...]

      def compute_rm_score(self, data):
         chunks = data.chunk(len(self.reward_loop_workers))
         outputs = ray.get(
            [
               worker.compute_score_batch.remote(chunk)
               for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
         )
         outputs_flat = [item for sublist in outputs for item in sublist]
         ...

This is how the reward manager is parallelized and distributed across all nodes.

Streaming Reward with Rollout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Furthermore, we check whether actor rollout and reward computation can be performed in a streaming manner,
where the reward is calculated as soon as each sample is rolled out.

.. code:: python

   # agent_reward_loop: streaming reward computation with actor rollout
   # two conditions satisfied: (1) rule-based reward, or (2) reward model with extra resource pool
   enable_agent_reward_loop = not use_rm or config.reward.reward_model.enable_resource_pool

   # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
   # to stream reward computation with actor rollout
   reward_loop_worker_handles = async_reward_manager.reward_loop_workers if enable_agent_reward_loop else None
   async_rollout_manager = AgentLoopManager(
      config=config,
      worker_group=actor_rollout_wg,
      rollout_resource_pool=actor_rollout_resource_pool,
      reward_loop_worker_handles=reward_loop_worker_handles,
   )

Hybrid Reward Scenarios Usage
-----------------------------

As described above, each ``reward_loop_worker`` is responsible for handling reward requests.
The rewards can be categorized as follows:

- **Rule-based Reward**: The reward is determined by predefined rules, e.g., checking whether the predicted answer matches the ground truth via string matching.
- **Discriminative Reward Model (DisRM)**: The reward is produced by a specified discriminative reward model, such as ``Skywork/Skywork-Reward-Llama-3.1-8B-v0.2``.
- **Generative Reward Model (GenRM)**: The reward is obtained using a generative reward model, for example ``dyyyyyyyy/FAPO-GenRM-4B``.
- **Hybrid Reward Scenarios**: A combination of the above reward types, e.g., rule + GenRM.

.. code:: python

   class RewardLoopWorker:

      async def compute_score_batch(self, data: DataProto) -> list[dict]:
         tasks = []
         for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
         outputs = await asyncio.gather(*tasks)
         return outputs

      async def compute_score(self, data: DataProto) -> dict:
         assert len(data) == 1, "RewardLoopWorker only support single data item"
         if self.config.reward.custom_reward_function.path is not None:
            # directly use user-customized reward function
            return await self.reward_manager.run_single(data)
         else:
            if self.config.reward.reward_model.enable:
               # we assume the rm is disrm
               # genrm must set custom_reward_function
               return await self.compute_score_disrm(data)
            else:
               return await self.reward_manager.run_single(data)

Each ``RewardLoopWorker`` will initalize one ``RewardManager``, splits the batch into individual data items and processes them in parallel using asynchronous tasks.

Reward Manager
~~~~~~~~~~~~~~

The ``RewardManager`` maintains a reward function and defines its computation logic, including:

- **naive**: The simplest implementation.
- **dapo**: DAPO implementation with an overlong reward penalty.
- **limit**: Restricts the concurrency of the reward function, useful when external API calls are rate-limited.
- **remote**: Runs in a separate process, effective for CPU-intensive tasks such as ``Math-Verify``.

Users can also customize their own ``RewardManager``, inheriting from ``RewardManagerBase``, and implementing the ``run_single`` function.

.. code:: python

   @register("user_costomized")
   class UserCostomizedRewardManager(RewardManagerBase):
      async def run_single(self, data: DataProto) -> dict:
         assert len(data) == 1, "Only support single data item"
         # your own reward manager
         ...

After defining it, users can specify their custom reward manager by setting ``reward.reward_manager.name=user_costomized``.

Rule-Based Reward
~~~~~~~~~~~~~~~~~

If ``reward.custom_reward_function`` is provided, the user-defined reward function will be used. Otherwise, it falls back to the default reward function.

Note that The custom function can be either synchronous or asynchronous; the system automatically detects its type and loads it accordingly.

We recommend **using asynchronous functions** when reward computation need to involve external model API calls or sandboxed execution, as they are significantly more efficient.

.. code:: python

   async def compute_score(data_source, solution_str, ground_truth, extra_info):
      """Compute a score by sending an async request to a remote service."""
      
      # prepare request payload
      payload = {"messages": [{"role": "user", "content": "check the correcness of the question and response ..."}], ...}

      # send async HTTP request
      async with aiohttp.ClientSession() as session:
         async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as resp:
               result = await resp.json()

      # parse and return score
      score = int(result["choices"][0]["message"]["content"].strip().split("\n")[-1])
      return {"score": score}

Model-Base Reward
~~~~~~~~~~~~~~~~~

**For discriminative reward model (DisRM)**, we provide a simple implementation:

.. code:: python

   class RewardLoopWorker:
      async def compute_score_disrm(self, data) -> dict:
         disrm_prompt = await self._preprocess_reward_inputs(data)
         payloads = {
            "model": model_name,
            "input": disrm_prompt,
            "activation": False,
         }
         output = await self._post_request(payloads, "classify")
         rm_score = output["data"][-1]["probs"][-1]
         return {"reward_score": rm_score}

pass the question and the model rollout as inputs to the reward model and obtain a reward score. This is also the standard practice for most DisRM.

Users should provide ``reward.reward_model.model_path`` to specify the reward model.

**For generative reward model (GenRM)**

For generative reward model scenarios, users need to specify both ``reward.reward_model.model_path`` and ``reward.custom_reward_function``.

The custom reward function should implement the following components:

- Convert the question and the model rollout into a GenRM input prompt using a custom prompt template.
- Invoke the GenRM to perform generation with custom sampling parameters. For this purpose, the Reward Loop provides an HTTP interface (i.e., ``reward_router_address``) for interacting with GenRM.
- Parse the GenRM output using a custom parser and extract the reward score.

As these steps are highly customizable and task-dependent, we offer this flexibility entirely to the user-defined reward function.

Below we provide an example of a custom reward function using GenRM.

.. code:: python

   async def compute_score_gsm8k(
      data_source: str,
      solution_str: str,
      ground_truth: str,
      extra_info: dict,
      reward_router_address: str,  # an HTTP router endpoint provided by Reward Loop
      reward_model_tokenizer: PreTrainedTokenizer,
   ):
      """Compute the reward score."""

      # Step 1: Prepare prompt and request payload
      grm_prompt = GRM_PROMPT_TEMPLATE.format(problem=extra_info["question"], solution=solution_str)
      messages = [{"role": "user", "content": grm_prompt}]
      sampling_params = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}
      chat_complete_request = {"messages": messages, **sampling_params}

      # Step 2: Send async request to the reward model
      # here, chat_complete sends async http request to the router address
      result = await chat_complete(
         router_address=reward_router_address,
         chat_complete_request=chat_complete_request,
      )

      # Step 3: Parse model response and extract score
      grm_response = result.choices[0].message.content.strip()
      try:
         score_str = grm_response.split("\n\n")[-1].strip()
         score = int(score_str)
      except Exception:
         score = 0

      return {"score": score}

**For hybrid reward scenarios**, such as combining rule-based rewards with GenRM similarly as above,

.. _recipe/fapo: https://github.com/verl-project/verl-recipe/tree/main/fapo

A runnable and reproducible example that demonstrates how to use a rule-based reward function together with a GenRM is provided in the `recipe/fapo`_ directory for reference. Welcome to use and cite.

Reward Model Arch Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We support multiple execution modes for reward models during:

- **Colocate Mode**: The reward model shares the same resource pool as the actor/rollout/reference models. In this setup, all rollouts must complete first, after which the reward model is awakened to perform inference.
- **Standalone Mode**: The reward model runs on a separate resource pool, independent from the actor/rollout/reference models. In this setup, each sample is evaluated by the reward model immediately after its rollout finishes.

The standalone mode can enable the streaming manner stated above.

By default, the system runs in colocate mode. Users can enable standalone mode by setting ``reward.reward_model.enable_resource_pool=True`` and allocating the corresponding resources via ``reward.reward_model.nnodes`` and ``reward.reward_model.n_gpus_per_node``.

.. image:: https://github.com/yyDing1/verl-materials/blob/main/reward_loop.svg?raw=true


To support flexible and scalable reward model computation, we implement a reward router that coordinates requests among multiple reward model servers.

Each reward model runs as an independent server and is registered with the router.
This router will forward the requests to the registered reward servers with load balancing and return the results.
This design allows us to expose a single unified router address to user-defined reward functions, enabling them to access various reward models seamlessly through the same interface.

.. image:: https://github.com/yyDing1/verl-materials/blob/main/reward_loop_full.svg?raw=true

.. code:: python

   class RewardModelManager:
      """Reward model manager."""

      def __init__(
         self,
         config: RewardModelConfig,
         resource_pool: RayResourcePool = None,
      ):
         """
         Initialize the reward model manager.

         Args:
            config (RewardModelConfig): Reward model configuration.
            resource_pool (RayResourcePool, optional): Resource pool. Defaults to None.
         """
         self.config = config
         self.resource_pool = resource_pool
         self._initialize_llm_servers()
         self._initialize_router()

