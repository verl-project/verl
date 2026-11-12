Implement Reward Function for Dataset
======================================

Last updated: 11/12/2026.

For each dataset, we need to implement a reward function or utilize a reward model to compute the rewards for the generated responses.
We already pre-implemented some reward functions in `reward_score directory <https://github.com/verl-project/verl/blob/main/verl/utils/reward_score>`_.
You can also use customized reward functions.

Currently, we support reward functions for GSM8k and MATH datasets. For RLHF datasets (e.g.,
full_hh_rlhf) and Code Generation (e.g., APPS), we utilize reward model
and SandBox (will opensource soon) for evaluation respectively.

RewardManager
-------------

In the entrypoint of the PPO Post-Training script `main_ppo.py <https://github.com/verl-project/verl/blob/main/verl/trainer/main_ppo.py#L33>`_,
we implement a ``RewardManager`` that utilize pre-implemented reward functions to compute the scores for each response.

In the ``RewardManager``, we implemented a ``__call__`` function to
compute the score for each response. 
All the reward functions are executed by ``compute_score_fn``.
The input is a ``DataProto``, which includes:

- ``input_ids``, ``attention_mask``: ``input_ids`` and ``attention_mask`` after applying
  chat_template, including prompt and response
- ``responses``: response tokens
- ``ground_truth``: The ground truth string of the current prompt.
  Stored in ``non_tensor_batch`` in the ``DataProto``, which should be
  preprocessed in the parquet files.
- ``data_source``: The dataset name of the current prompt. Stored in
  ``non_tensor_batch`` in the ``DataProto``, which should be
  preprocessed in the parquet files.

After detokenize the responses, the responses string and the ground
truth string will be input to the ``compute_score_fn`` to compute the
score for each response.

Reward Functions
----------------

Pre-implemented
~~~~~~~~~~~~~~~

We already pre-implemented some reward functions in `reward_score directory <https://github.com/verl-project/verl/blob/main/verl/utils/reward_score>`_.

- In the `GSM8k example <https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/gsm8k.py>`_, we
  force the response to output the final answer after four ####, then
  use string matching to compare with the ground truth. If completely
  correct, score 1 point; if the format is correct, score 0.1 points; if
  the format is incorrect, score 0 points.
- In the `MATH example <https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_reward.py>`_, we follow
  the implementation in `lm-evaluation-harness repository <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py>`_.

Customized
~~~~~~~~~~

You can implement customized reward functions in a separate file and specify them using ``custom_reward_function.path`` and ``custom_reward_function.name``. For the set of them, please refer to :ref:`config-explain-page`.

The parameters of your reward function should be ``data_source``, ``solution_str``, ``ground_truth``, and ``extra_info``.
For example:

.. code:: python

  def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    return len(solution_str)/100

If you are testing only a single customized reward function, you can simply name it 'compute_score' and leave ``custom_reward_function.name`` unset.

To run multiple tests with different customized reward functions, you can modify both ``custom_reward_function.path`` and ``custom_reward_function.name`` for each trial. 
For instance, you might create a single `my_reward.py` file and implement multiple reward functions within it. This way, for different trials, you only need to adjust ``custom_reward_function.name``, making it more convenient to conduct multiple tests within scripts.

Rollout Metadata in ``extra_info``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When scoring through the reward loop, rollout ``extra_fields`` are forwarded into ``extra_info`` before calling ``compute_score``. For example, vLLM rollouts populate ``extra_info["ended_with_eos"]`` (``bool``) indicating whether the final generation terminated on an effective native EOS token ID (combining the renderer's primary EOS token ID and ``generation_config.json`` ``eos_token_id`` entries, and setting ``False`` if post-processing or tool-budget limits truncate the response). Because ``ended_with_eos`` is omitted when a rollout backend or engine does not expose native EOS token IDs, custom reward functions should check ``extra_info.get("ended_with_eos") is False`` before penalizing unfinished responses so unknown backends remain unaffected.
