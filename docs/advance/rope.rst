RoPE Scaling override
=======================================

Last updated: 05/14/2025.

Some models such as `Qwen/Qwen2.5-7B-Instruct <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts>`_ support RoPE Scaling but don't have it defined in their config.json file.
For example, this model supports this configuration:

.. code:: python

    {
        ...,
        "rope_scaling": {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn"
        }
    }



In order to support a longer context for such models, you must override the model configs when starting the trainer.

PPO example:

.. code:: bash

    +actor_rollout_ref.model.override_config.rope_scaling.type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=32768 \


And for the critic model

.. code:: bash

    +critic.model.override_config.rope_scaling.type=yarn \
    +critic.model.override_config.rope_scaling.factor=4.0 \
    +critic.model.override_config.rope_scaling.original_max_position_embeddings=32768 \
