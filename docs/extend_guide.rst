How to Extend verl
===================

Last updated: 06/23/2026.

Author: `Xibin Wu <https://github.com/wuxibin89>`_

RL Researcher
-------------

How do I extend verl to support my own reward function?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

verl supports different types of reward functions:

- Rule-based reward: math, code, etc with ground truth
- Discriminative reward model (DisRM)
- Generative reward model (GenRM)
- Hybrid reward: rule-based + GenRM/DisRM

All types of reward functions are supported to be customized by user, for more details, see: :doc:`Reward Loop<advance/reward_loop>`.

How do I extend verl to support my own tool calls?
++++++++++++++++++++++++++++++++++++++++++++++++++

verl provides a built-in ReAct agent loop implementation: `ToolAgentLoop <https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py>`_.
ToolAgentLoop support two types of tool definitions:

- Stateless function-based tool: decorate a function with ``@function_tool``
- Stateful class-based tool: inherit from ``BaseTool`` and implement the ``execute`` method

After defining your tools, you can set the tool agent loop in config:

.. code:: bash

    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.multi_turn.format=hermes # hermes,gpt-oss,qwen3_coder,etc.
    actor_rollout_ref.rollout.multi_turn.function_tool_path=path/to/your_tools.py # function-based tool path
    actor_rollout_ref.rollout.multi_turn.tool_config_path=path/to/your_tools.yaml # class-based tool path

For more details, see:

- :doc:`Multi-turn Rollout Support <sglang_multiturn/multiturn>`
- :doc:`Agent Loop <advance/agent_loop>`
- `Train ReAct agent with code sandbox <https://github.com/verl-project/verl/blob/main/examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb>`_

ToolAgentLoop doesn't meet my requirements, how do I extend verl to support my own agent Loop?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If ToolAgentLoop doesn't meet your requirements, you can customize your own agent loop by inheriting from ``AgentLoopBase`` and implementing the ``run`` method.

.. warning:: It's user's responsibility to request LLM server in `TITO(token-in-token-out) <https://qgallouedec-tito.hf.space/>`_, be careful to adhere to a golden rule: **never re-encode tokens you’ve decoded**.

.. code:: python

   class MyAgentLoop(AgentLoopBase):
       async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
           """Run agent loop to interact with LLM server and environment.

           Args:
               sampling_params (Dict[str, Any]): LLM sampling params.
               **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

           Returns:
               AgentLoopOutput: Agent loop output.
           """
           ...

After defining MyAgentLoop, you can set the agent loop class in config:

.. code:: bash

    actor_rollout_ref.rollout.agent.agent_loop_config_path=path/to/your_agent.yaml

For more details, see: :doc:`Agent Loop <advance/agent_loop>`.

I'm doing async training, how do I customize my own replay buffer sampling strategy?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In async training, the agent framework streams generated trajectories into ``TransferQueue``, and the
trainer uses `ReplayBuffer <https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/v1/replay_buffer.py>`_ to sample a batch from TransferQueue for training.

While we provide a default sampling strategy, it's very common for users to want to customize it to meet their own needs.
To do so, inherit from ``ReplayBuffer`` and implement the ``sample`` method.

.. code:: python

    class UserCustomReplayBuffer(ReplayBuffer):
        def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
            """Sample a batch of data from the replay buffer.

            Args:
                global_steps (int): Global steps of the current training.
                partition_id (str): Partition of TransferQueue, e.g. "train" or "val".
                batch_size (int, optional): Batch size.

            Returns:
                KVBatchMeta: A batch of data.
                dict: Auxiliary metrics, e.g. off-policy staleness stats.
            """
            ...

After defining UserCustomReplayBuffer, you can set the custom sampler in config:

.. code:: bash

    trainer.v1.sampler.custom_sampler.path = "path/to/your/sampler.py"
    trainer.v1.sampler.custom_sampler.name = "UserCustomReplayBuffer"

How do I customize sync/async trainer behavior?
+++++++++++++++++++++++++++++++++++++++++++++++

User may want to change the trainer's default behavior, for example:

- over-sampling: sample more trajectories than the batch size
- dynamic filtering: filter out samples with group responses are all correct or incorrect

verl `v1 PPO trainer <https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/v1/trainer_base.py>`_ 
provides a set of hooks to customize trainer behavior:

- on_init_end
- on_train_begin
- on_train_end
- on_validate_begin
- on_validate_end
- on_step_begin
- on_step_end
- on_sample_begin
- on_sample_end

These hooks are also used by the ``sync``, ``colocate_async``, and ``separate_async`` trainers to change model engine, LLM server, and checkpoint engine behavior.

Agent Framework Developer
-------------------------

How do I replace verl's AgentLoopManager with my own agent framework?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

AgentLoopManager is a reference implementation of an agent framework and is designed to be fully replaceable by other agent frameworks. 
You can plug in your own agent framework, the only requirement is:

- implement a non-blocking ``generate_sequences`` method
- put trajectory fields(e.g. ``prompt_ids``, ``response_ids``, ``response_mask``, ...) into ``TransferQueue`` once rollout finished

.. code:: python

    class MyAgentLoopManager:
        @classmethod
        @auto_await
        async def create(
            cls,
            config: DictConfig,
            llm_client: LLMServerClient,
            teacher_client: dict[str, LLMServerClient] = None,
            reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
        ):
            """Create agent loop manager.

            Args:
                config (DictConfig): whole config for main entrypoint.
                llm_client (LLMServerClient): Client for the LLM server.
                teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
                reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
            """
            ...

        def generate_sequences(self, prompts: TensorDict) -> None:
            """Add batch of prompts to agent framework for rollout without blocking. Agent framework should put trajectory
            fields(e.g. prompt_ids, response_ids, response_mask, ...) into TransferQueue once rollout finished.

            Args:
                prompts (TensorDict): batch of prompts from train or validation dataset.
            """
            ...

After defining MyAgentLoopManager, you can set the agent loop manager class in config:

.. code:: bash

    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=my_package.module.MyAgentLoopManager

I want to train my model with Claude code/Codex/Trae etc, how do I integrate these agent frameworks in blackbox?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We have launched a sub-project: `verl-project/uni-agent <https://github.com/verl-project/uni-agent>`_, in which we provide an agent gateway:

- **Message API**: Provide OpenAI ``/v1/chat/completions`` and Anthropic ``/v1/messages`` compatible API
- **Token-in-token-out**: encode ``user,tool`` messages into token ids and request LLM server, decode response ids and parsing tools into ``assistant`` messages
- **Trajectory tracking**: messages prefix matching, spawn a new trajectory if prefix changed
- **Session management**: multiple active sessions management

For more details, see:

- `Agent Gateway RFC <https://github.com/verl-project/verl/issues/5790>`_
- `Agent Gateway Implementation <https://github.com/verl-project/uni-agent/tree/main/uni_agent/gateway>`_

Training/Inference Framework Developer
--------------------------------------

I'm an inference framework developer, how do I extend verl to support my own inference framework?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

verl provides an environment variable hook ``VERL_USE_EXTERNAL_MODULES`` to load external modules. You can define a register hook in your own module and set the environment variable to dynamically register your own modules.

- ``RolloutReplica``: custom rollout replica class to define how to launch your own inference server.
- ``ServerAdapter``: custom server adapter class to define how to update weights with your own inference server.

For example, this is how `verl-project/vexact <https://github.com/verl-project/vexact>`_ integrate with verl. vexact define a register hook in `register.py <https://github.com/verl-project/vexact/blob/main/vexact/integrations/verl/register.py>`_:

.. code:: python

    def _load_vexact_replica():
        """Lazy loader for VeXactReplica to avoid circular imports."""
        from vexact.integrations.verl.async_server import VeXactReplica

        return VeXactReplica


    # Register VeXact rollout replica (for server mode)
    RolloutReplicaRegistry.register("vexact", _load_vexact_replica)

    # Register VeXact rollout base (for hybrid mode with device mesh)
    _ROLLOUT_REGISTRY[("vexact", "async")] = "vexact.integrations.verl.rollout.ServerAdapter"


And user can set the environment variable to load vexact:

.. code:: bash

    export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register

I'm a training framework developer, how do I extend verl to support my own training framework?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

verl provides a unified training engine abstraction: `BaseEngine <https://github.com/verl-project/verl/blob/main/verl/workers/engine/base.py>`_.
With this abstraction, we provide native support for some popular training frameworks:

- FSDP: FSDP1/2+SP
- Megatron: DP+TP+CP+EP+PP
- VeOmni: FSDP2+SP+EP
- TorchTitan: FSDP2+TP+CP+EP+PP
- Automodel: FSDP2+TP+CP+EP+PP

For training framework developer who want to integrate with verl, you can inherit from ``BaseEngine`` and implement all the interfaces.
Then you can register your own training engine in verl with ``VERL_USE_EXTERNAL_MODULES`` same as inference framework.

For example, this is how FlagOS integrate with verl. FlagOS define a register hook in `__init__.py <https://github.com/verl-project/verl-hardware-plugin/blob/main/verl_hardware_plugin/__init__.py>`_:

.. code:: python

    from verl_hardware_plugin.engines import register_all_engines
    from verl_hardware_plugin.platforms import register_all_platforms

    register_all_platforms()
    register_all_engines()

And user can set the environment variable to load your own training framework:

.. code:: bash

    export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin

For more details, see: :doc:`Model Engine <workers/model_engine>`

I'm a hardware vendor, how do I extend verl to support my own chip?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

verl provides native support for NVIDIA GPU, Huawei Ascend NPU, AMD GPU in the main branch, and provides a unified plugin system to support other hardware platforms.

For more details, see:

- :doc:`Multi-chip Support <hardware/multi_chip_support>`
- `verl-project/verl-hardware-plugin <https://github.com/verl-project/verl-hardware-plugin>`_: external hardware plugin for MLU, XPU, MetaX, etc.
