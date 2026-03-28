Multi-Turn Tool Use with NeMo Gym
==================================

`NeMo Gym <https://github.com/NVIDIA-NeMo/Gym>`_ is an environment framework for
multi-turn RL rollouts with tool-calling agents. The verl integration lets you
replace verl's standard single-turn rollout with NeMo Gym's ``simple_agent``,
which runs full multi-turn conversations through an OpenAI-compatible HTTP
endpoint and returns token IDs and log-probs for training.

Overview
--------

The integration adds two components to ``verl/experimental/nemo_gym/``:

- ``agent_loop.py`` — ``NemoGymAgentLoopManager``: drives multi-turn rollouts
  via NeMo Gym, handles token ID reconciliation across turns, and returns a
  ``DataProto`` compatible with verl's Megatron actor update.
- ``dataset.py`` — ``NemoGymJSONLDataset``: loads NeMo Gym JSONL files
  (including tool definitions, agent refs, and ground-truth answers) into
  verl's data pipeline.

Requirements
------------

- A NeMo Gym checkout (``gym-ref``) with the environment you want to train on.
- ``pip install -e /path/to/gym-ref`` installed into the container at job start.

Quick Start
-----------

1. **Install NeMo Gym** in your container startup script::

    pip install -e /path/to/gym-ref

2. **Prepare your dataset** in NeMo Gym JSONL format. Each line should be a
   JSON object with a ``responses_create_params`` field containing the initial
   messages and any tools, plus an ``agent_ref`` pointing at your environment's
   agent server.

3. **Add these overrides** to your verl training command::

    +data.custom_cls.path=verl/experimental/nemo_gym/dataset.py
    +data.custom_cls.name=NemoGymJSONLDataset
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.nemo_gym.agent_loop.NemoGymAgentLoopManager
    "+actor_rollout_ref.rollout.agent.nemo_gym.initial_global_config_dict.config_paths=[/path/to/env.yaml]"
    +actor_rollout_ref.rollout.agent.nemo_gym.nemo_gym_root=/path/to/gym-ref

See ``submit_workplace.sh`` and ``submit_math.sh`` for complete working examples.

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

For environments that use tool calling (e.g. workplace assistant), pass the
vLLM engine kwargs to enable the hermes tool parser::

    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable-auto-tool-choice=true'
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.tool-call-parser=hermes'
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.max-model-len=32768'
