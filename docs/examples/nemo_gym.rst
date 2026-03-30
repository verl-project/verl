NVIDIA NeMo Gym Integration
==================================

`NVIDIA NeMo Gym <https://github.com/NVIDIA-NeMo/Gym>`_ (`docs <https://docs.nvidia.com/nemo/gym/latest/index.html>`_)
is an RL environment framework for scalable, multi-environment, agentic RL. This integration enables
running NeMo Gym environments with verl using a custom agent loop manager.

Overview
--------

The integration adds three components to ``verl/experimental/nemo_gym/``:

- ``agent_loop.py`` — ``NemoGymAgentLoopManager``: drives multi-turn rollouts
  via NeMo Gym and formats results into verl's DataProto format.
- ``dataset.py`` — ``NemoGymJSONLDataset``: loads NeMo Gym JSONL datasets
  including messages, tools, agent refs, and metadata into verl format.
- ``server_patch.py`` — patches vLLM's ``OpenAIServingChat`` and
  ``OpenAIServingTokenization`` to fix retokenization across multi-turn calls,
  matching NeMo RL's approach.

Requirements
------------

- A NeMo Gym clone with the environment you want to train on.
- ``pip install -e /path/to/gym-ref`` installed into the container at job start.

Quick Start
-----------

1. **Install NeMo Gym** in your container startup script::

    pip install -e /path/to/gym-ref

2. **Prepare training datasets** in NeMo Gym JSONL format. Each line should be a
   JSON object with a ``responses_create_params`` field containing the initial
   messages and any tools, plus an ``agent_ref`` pointing at your environment's
   agent server.

3. **Add these overrides** to your verl training command::

    +data.custom_cls.path=verl/experimental/nemo_gym/dataset.py
    +data.custom_cls.name=NemoGymJSONLDataset
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.nemo_gym.agent_loop.NemoGymAgentLoopManager
    "+actor_rollout_ref.rollout.agent.nemo_gym.config_paths=[/path/to/env.yaml]"
    +actor_rollout_ref.rollout.agent.nemo_gym.nemo_gym_root=/path/to/gym-ref

See ``submit_workplace.sh`` and ``submit_math.sh`` for working examples.

Configuration
-------------

The ``nemo_gym`` block in ``AgentLoopConfig`` accepts:

.. code-block:: yaml

    actor_rollout_ref:
      rollout:
        agent:
          nemo_gym:
            nemo_gym_root: /path/to/gym-ref
            uses_reasoning_parser: false
            config_paths:
              - /path/to/env.yaml

For environments that use tool calling (e.g. workplace assistant), use a tool parser. For reasoning models, use a reasoning parser.