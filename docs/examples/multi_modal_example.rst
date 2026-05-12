Multi-Modal Example Architecture
=================================

Last updated: 04/28/2025.

Introduction
------------

Now, verl has supported multi-modal training. You can use fsdp and 
vllm/sglang to start a multi-modal RL task. Megatron supports is also 
on the way.

Follow the steps below to quickly start a multi-modal RL task.

Step 1: Prepare dataset
-----------------------

.. code:: python

    # it will be saved in the $HOME/data/geo3k folder
    python examples/data_preprocess/geo3k.py

Step 2: Download Model
----------------------

.. code:: bash

    # download the model from huggingface
    python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"

Step 3: Perform GRPO training with multi-modal model on Geo3K Dataset
---------------------------------------------------------------------

.. code:: bash

    # run the task
    bash examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh








