Implement Reward Function for Dataset
======================================

Last updated: 11/12/2026.

For each dataset, we need to implement a reward function or utilize a reward model to compute the rewards for the generated responses.
We already pre-implemented some reward functions in `reward_score directory <https://github.com/verl-project/verl/blob/main/verl/utils/reward_score>`_.
You can also use customized reward functions.

Currently, we support reward functions for GSM8k, MATH, DAPO-Math, Math-Verify, PRIME math/code, Geometry3K, and Search-R1 QA datasets. For RLHF datasets (e.g.,
full_hh_rlhf) and Code Generation (e.g., APPS), we utilize reward model
and SandBox for evaluation respectively.

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

- In the `GSM8k example <https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/gsm8k.py>`_ (``gsm8k``), we
  force the response to output the final answer after four ``####``, then
  use string matching to compare with the ground truth.
- In the `MATH example <https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_reward.py>`_ (``math_reward``), we follow
  the implementation in `lm-evaluation-harness repository <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py>`_.
- Additional built-in presets include ``math_dapo``, ``math_verify``, ``prime_math``, ``prime_code``, ``geo3k``, ``search_r1_like_qa_em``, and ``search_r1_like_qa_subem``.

Selecting Presets via CLI without Modifying Parquet Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ``default_compute_score`` dispatches to a built-in preset based on each row's ``data_source`` column. When using a public Parquet file whose ``data_source`` is not in the built-in table (for example, ``"dapo_filter"`` or sub-source tags such as ``"MATH##Aops"``), or when switching a known dataset to a different built-in scorer (such as ``math_verify``), you can configure preset routing directly on the command line without editing the Parquet file:

- ``reward.preset=<preset_name>`` sets the fallback preset for any unmapped ``data_source`` while keeping built-in ``data_source`` mappings (such as ``openai/gsm8k`` or ``aime*``) intact for multi-dataset validation.
- ``reward.data_source_presets`` maps exact ``data_source`` strings or ``fnmatch`` globs (such as ``"MATH##*"`` or ``"*"``) to a built-in preset, taking precedence over the built-in ``data_source`` table. It accepts either a YAML dictionary (``+reward.data_source_presets.dapo_filter=math_dapo``) or a Hydra list of ``"pattern=preset"`` strings when keys contain ``/``, ``#``, or ``*``.

.. code:: bash

  # Use math_dapo as the fallback preset for unmapped data_source values (e.g. dapo_filter)
  python3 -m verl.trainer.main_ppo ... reward.preset=math_dapo

  # Map specific data_source names or globs while preserving other validation scorers
  python3 -m verl.trainer.main_ppo ... \
      'reward.data_source_presets=["dapo_filter=math_dapo","MATH##*=math_dapo","HuggingFaceH4/MATH-500=math_verify"]'

Customized
~~~~~~~~~~

You can implement customized reward functions in a separate file and specify them using ``reward.custom_reward_function.path`` and ``reward.custom_reward_function.name``. For the set of them, please refer to :ref:`config-explain-page`.

The parameters of your reward function should be ``data_source``, ``solution_str``, ``ground_truth``, and ``extra_info``.
For example:

.. code:: python

  def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    return len(solution_str)/100

If you are testing only a single customized reward function, you can simply name it 'compute_score' and leave ``reward.custom_reward_function.name`` unset.

To run multiple tests with different customized reward functions, you can modify both ``reward.custom_reward_function.path`` and ``reward.custom_reward_function.name`` for each trial. 
For instance, you might create a single `my_reward.py` file and implement multiple reward functions within it. This way, for different trials, you only need to adjust ``reward.custom_reward_function.name``, making it more convenient to conduct multiple tests within scripts.
