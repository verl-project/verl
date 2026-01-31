Reward Loop
===========

.. _yyding: https://yyding1.github.io

Author: `Yuyang Ding <https://yyding1.github.io>`_

Last updated: 12/20/2025.

.. warning::
   Reward Loop is ready for use, but the API may change in future releases.
   User can set ``reward_model.use_reward_loop=True`` or ``False`` to control whether to enable reward loop.

Reward Loop is designed to support flexible and user-friendly reward computation, with most implementation in ``verl/experimental/reward_loop``.

Compared with the previous reward mechanism, the Reward Loop offers the following key features:

1. provides a more flexible and user-friendly design for reward-model settings, enabling hybrid reward scenarios where multiple reward sources can be seamlessly integrated.
2. implements asynchronous reward computation instead of the previous batch-based computation, improving efficiency for both rule-based rewards and reward-model-based scenarios.

Hybrid Reward Scenarios
-----------------------

Reward Loop covers all typical reward-computation scenarios.

- **Rule-based Reward**: The reward is determined by predefined rules, e.g., checking whether the predicted answer matches the ground truth via simple string matching.
- **Discriminative Reward Model (DisRM)**: The reward is produced by a specified discriminative reward model, such as ``Skywork/Skywork-Reward-Llama-3.1-8B-v0.2``.
- **Generative Reward Model (GenRM)**: The reward is obtained using a generative reward model, for example ``dyyyyyyyy/FAPO-GenRM-4B``.
- **Hybrid Reward Scenarios**: Reward Loop provides interfaces for plugging in reward models, allowing users to define custom reward logic based on their needs (e.g., combining rule-based methods with GenRM).

Rule-based Reward
~~~~~~~~~~~~~~~~~

If ``custom_reward_function`` is not provided, the reward loop will fall back to the default rule-based reward function.
Otherwise, only the user-defined reward function will be used. The files under ``verl/utils/reward_score/`` provide some examples.

Reward Loop supports both synchronous and asynchronous user-defined reward functions. It automatically detects the function type and executes it accordingly, ensuring that reward computation remains non-blocking and efficient.

Discriminative Reward Model (DisRM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For scenarios involving a discriminative reward model, users should provide ``reward_model.model.path`` to specify the reward model.

The Reward Loop will pass the question and the model rollout as inputs to the reward model and obtain a reward score from its output.

Generative Reward Model (GenRM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For generative reward model scenarios, users need to specify both ``reward_model.model.path`` and ``custom_reward_function``.

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

Hybrid Reward Scenarios
~~~~~~~~~~~~~~~~~~~~~~~

For more complex application settings, such as combining rule-based rewards with GenRM, or mixing rule-based rewards with DisRM, users can also achieve this by specifying the ``reward_model.model.path`` together with the ``custom_reward_function``.
The implementation of the customized reward function follows the same pattern as illustrated above.

A runnable and reproducible example that demonstrates how to use a rule-based reward function together with a GenRM is provided in the ``recipe/fapo`` directory for reference. Welcome to use and cite.

Architecture Design
-------------------

Reward Loop supports multiple execution modes for reward training:

- **Colocate Mode**: The reward model shares the same resource pool as the actor/rollout/reference models. In this setup, all rollouts must complete first, after which the reward model is awakened to perform inference.
- **Standalone Mode**: The reward model runs on a separate resource pool, independent from the actor/rollout/reference models. In this setup, each sample is evaluated by the reward model immediately after its rollout finishes.

.. image:: https://github.com/yyDing1/verl-materials/blob/main/reward_loop.svg?raw=true

RewardLoopWorker
~~~~~~~~~~~~~~~~~

The ``RewardLoopWorker`` is responsible for handling batch-level reward computation, operating in an asynchronous manner.

.. image:: https://github.com/yyDing1/verl-materials/blob/main/reward_loop_worker.svg?raw=true

For each sample, the reward is computed according to the following logic:

- if ``custom_reward_function`` is provided, we directly use user-customized reward function
- if ``custom_reward_function`` is not provided:
   - **reward model is not enabled**: use default rule-based reward function
   - **reward model is discriminative**: compute reward score using disrm
   - **reward model is generative**: this is not permitted (user-customized reward func **must be** provided)

In most cases, we encourage users to define and use their own customized reward functions.

``RewardLoopWorker`` will initialize a ``RewardManager`` via ``_init_reward_fn()``.
Then the batch reward computation request of ``compute_score_batch`` will be processed asynchronously.

.. code:: python

   @ray.remote
   class RewardLoopWorker:
      def __init__(self, config: DictConfig, reward_router_address: str = None):
         self.config = config
         self.reward_router_address = reward_router_address
         self._init_reward_fn()

      def _init_reward_fn(self):
         input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
         self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
         self.reward_model_tokenizer = None
         if self.config.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)
         self.reward_fn = get_custom_reward_fn(self.config)
         reward_manager_cls = get_reward_manager_cls(self.config.reward_model.reward_manager)
         self.reward_loop = reward_manager_cls(
            self.config, self.input_tokenizer, self.reward_fn, self.reward_router_address, self.reward_model_tokenizer
         )

      async def compute_score_batch(self, data: DataProto) -> list[dict]:
         tasks = []
         for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
         outputs = await asyncio.gather(*tasks)
         return outputs

      async def compute_score(self, data: DataProto) -> dict:
         assert len(data) == 1, "RewardLoopWorker only support single data item"
         if self.config.custom_reward_function.path is not None:
            # directly use user-customized reward function
            return await self.reward_loop.run_single(data)
         else:
            if self.config.reward_model.enable:
               # we assume the rm is disrm
               # genrm must set custom_reward_function
               return await self.compute_score_disrm(data)
            else:
               return await self.reward_loop.run_single(data)

RewardManager
~~~~~~~~~~~~~

Reward Loop refactors the previous reward manager, which processed rewards sequentially on batched inputs.
Instead, the Reward Loop performs reward computation asynchronously and in parallel at the per-sample level.

In the ``RewardManager`` of Reward Loop, we implement a ``run_single`` function to compute the score for single sample. All the reward functions are executed by ``compute_score_fn``. The input should be a ``DataProto`` containing only one item.

.. code:: python

   @register("naive")
   class NaiveRewardManager(RewardManagerBase):
      async def run_single(self, data: DataProto) -> dict:
         assert len(data) == 1, "Only support single data item"
         ...

Commonly used reward managers, such as ``DAPORewardManager`` has been implemented in reward loop.
In addition, ``RateLimitRewardManager`` is also ready for use for external API-based reward computation scenarios like ChatGPT.

Users can also customize their own ``RewardManager``, by adding the ``@register`` decorator, inheriting from ``RewardManagerBase``, and implementing the ``run_single`` function.
See ``verl/experimental/reward_manager/*`` for reference.

.. code:: python

   @register("user_costomized")
   class UserCostomizedRewardManager(RewardManagerBase):
      async def run_single(self, data: DataProto) -> dict:
         assert len(data) == 1, "Only support single data item"
         # your own reward manager
         ...

After defining it, users can specify their custom reward manager by setting ``reward_model.reward_manager=user_costomized``.

RewardLoopManager
~~~~~~~~~~~~~~~~~

To enable parallel reward computation, the Reward Loop launches multiple reward workers that handle reward computation requests concurrently.

In **standalone mode**, we directly launch one ``RewardLoopWorker`` for each ``AgentLoopWorker`` to handle reward computation independently.

In **colocate mode**, we launch a ``RewardLoopManager`` to

1. launch reward model if enabled
2. manage multiple ``RewardLoopWorker`` instances to parallelize reward computation.

Users can specify the number of workers by setting ``reward_model.num_workers`` in colocate mode.

.. code:: python

   class RewardLoopManager:
      """
      RewardLoopManager run in single controller.
      This class will create reward loop workers and manage them.
      RewardLoopManager will deprecate fsdp/megatron RewardModelWorker in the future.
      """
   def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
      self.config = config
      if self.config.reward_model.enable:
         self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
         self.reward_router_address = self.reward_model_manager.get_router_address()
      else:
         self.reward_model_manager = None
         self.reward_router_address = None

      self._init_reward_loop_workers()

   def _init_reward_loop_workers(self):
      self.reward_loop_workers = []
      num_workers = self.config.reward_model.get("num_workers", 1)
      node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

      for i in range(num_workers):
         # Round-robin scheduling over the all nodes
         node_id = node_ids[i % len(node_ids)]
         self.reward_loop_workers.append(
            RewardLoopWorker.options(
               name=f"reward_loop_worker_{i}",
               scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                  node_id=node_id,
                  soft=True,
               ),
            ).remote(self.config, self.reward_router_address)
         )

   def compute_rm_score(self, data: DataProto) -> DataProto:
      """
      Compute reward score for the given data.
      """
      ...


RewardModelManager
~~~~~~~~~~~~~~~~~~

To support flexible and scalable reward model computation, Reward Loop implement a reward router that coordinates requests among multiple reward model servers.

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
         assert self.config.rollout.skip_tokenizer_init is False, "Reward model should not skip tokenizer init."
         if self.config.rollout.free_cache_engine:
               self.sleep()
