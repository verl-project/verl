Prepare Data for Post-Training
========================================

Last updated: 02/09/2025.

Before starting the post-training job, we need to prepare the data for
the policy training. The data should be stored in the parquet format.

We provide several data preprocess scripts for different datasets,
including GSM8K, MATH, HelloSwag, Full_hh_rlhf. To prepare other datasets, we need
to follow the following steps: The data preprocess script can be divided
into two parts:

1. The first part is the common part, which loads the dataset from
   huggingface's ``datasets`` package. Then preprocess the datasets with
   the ``make_map_fn`` and then store in the parquet format.

.. code:: python

   import re
   import os
   import datasets

   from verl.utils.hdfs_io import copy, makedirs
   import argparse

   # To extract the solution for each prompts in the dataset
   # def extract_solution(solution_str): 
   # ...


   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--local_dir', default='/opt/tiger/gsm8k')
       parser.add_argument('--hdfs_dir', default=None)

       args = parser.parse_args()

       num_few_shot = 5
       data_source = 'openai/gsm8k'

       dataset = datasets.load_dataset(data_source, 'main')

       train_dataset = dataset['train']
       test_dataset = dataset['test']

           # Construct a `def make_map_fn(split)` for the corresponding datasets.
       # ...
           
       train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
       test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

       local_dir = args.local_dir
       hdfs_dir = args.hdfs_dir

       train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
       test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

       makedirs(hdfs_dir)

       copy(src=local_dir, dst=hdfs_dir)

2. The users are required to implement the ``make_map_fn()`` function
   (as well as the ``extract_solution``) on their own to support
   different datasets or tasks.

We already implemented the data preprocess of GSM8k, MATH, Hellaswag and Full_hh_rlhf
datasets. And we take the GSM8k dataset as an example:

**GSM8K**

In the ``make_map_fn``, each data field should consist of the following
5 fields:

1. ``data_source``: The name of the dataset. To index the corresponding
   reward function in the ``RewardModel``
2. ``prompt``: This field should be constructed in the format of
   huggingface chat_template. The tokenizer in ``RLHFDataset`` will
   apply chat template and tokenize the prompt.
3. ``ability``: Define the task category.
4. ``reward_model``: Currently, we only utilize the ``ground_truth``
   field during evaluation. The ``ground_truth`` is computed by the
   ``extract_solution`` function. **NOTED** that the implementation of
   the corresponding reward function should align with this extracted
   ``ground_truth``.
5. ``extra_info``: Record some information of the current prompt. Not
   use for now.

.. code:: python

   def extract_solution(solution_str):
       solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
       assert solution is not None
       final_solution = solution.group(0)
       final_solution = final_solution.split('#### ')[1].replace(',', '')
       return final_solution

   instruction_following = "Let's think step by step and output the final answer after \"####\"."

   # add a row to each data item that represents a unique id
   def make_map_fn(split):

       def process_fn(example, idx):
           question = example.pop('question')

           question = question + ' ' + instruction_following

           answer = example.pop('answer')
           solution = extract_solution(answer)
           data = {
               "data_source": data_source,
               "prompt": [{
                   "role": "user",
                   "content": question
               }],
               "ability": "math",
               "reward_model": {
                   "style": "rule",
                   "ground_truth": solution
               },
               "extra_info": {
                   'split': split,
                   'index': idx
               }
           }
           return data

       return process_fn
