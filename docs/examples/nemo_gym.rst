NVIDIA NeMo Gym Integration
==================================

`NVIDIA NeMo Gym <https://github.com/NVIDIA-NeMo/Gym>`_ (`docs <https://docs.nvidia.com/nemo/gym/latest/index.html>`_)
is an RL environment framework for scalable, multi-environment, agentic RL. This integration enables
running NeMo Gym environments with verl using a custom agent loop manager.

Overview
--------

The integration adds two components to ``verl/experimental/nemo_gym/``:

- ``agent_loop.py`` — ``NemoGymAgentLoopManager``: offloads multi-turn rollouts
  to NeMo Gym, handles retokenization correction across turns, and formats output.
  The retokenization logic may change shortly to follow verl approach.
- ``dataset.py`` — ``NemoGymJSONLDataset``: loads NeMo Gym datasets
  including messages, tools, agent refs, and metadata into verl format.

Requirements
------------

- A NeMo Gym local clone (``gym-ref``) with the environment you want to train on. TODO finalize submodule decision
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
    "+actor_rollout_ref.rollout.agent.nemo_gym.initial_global_config_dict.config_paths=[/path/to/env.yaml]"
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
            initial_global_config_dict:
              config_paths:
                - /path/to/env.yaml

Tool Calling
------------

For environments that use tool calling (e.g. workplace assistant), use a tool parser, for example::

    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable-auto-tool-choice=true'
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.tool-call-parser=hermes'
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.max-model-len=32768'
