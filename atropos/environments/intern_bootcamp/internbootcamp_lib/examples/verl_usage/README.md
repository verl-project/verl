# VERL USAGE

This directory contains examples and data processing scripts for training with bootcamp datasets in the VERL framework. Here is a brief overview of the file and its functions:

---

## Directory structure

```
/InternBootcamp/examples/verl_usage
├── run_bootcamp.sh
├── bootcamp_reward_for_verl.py
└── verl_data_preprocess.py
```

---

### 1. `run_bootcamp.sh`

This is a Shell script for setting up and running VERL training experiments. The main functions include:

- Set basic parameters such as experiment name and data path.
- Install the necessary dependencies and configure the environment.
- Start the VERL trainer and configure the experiment parameters (such as model path, batch size, learning rate, etc.).



#### How to use

- Before starting, ensure that important parameters are correctly configured, such as internbootcamp_path, model.path, trainer.default_local_dir, etc

```bash
./run_bootcamp.sh
```

---

### 2. `bootcamp_reward_for_verl.py`

This Python script is used to dynamically calculate the reward score of the bootcamp during training. Need to be embedded in the verl framework for use.

The script provides a function _default_compute_score to calculate a score based on 'data_source', 'model_output', and 'groud_truth' (here, a dictionary of parameters needed to validate model_output). Support for multiple data sources, including:
- **Non-bootcamped data sets** : Scores are calculated using modules such as gsm8k, math, prime_math, prime_code, etc.
- **bootcamp dataset** : Dynamically import external bootcamp modules to calculate scores.
- Raises NotImplementedError if the data source is not supported.


### 3. `verl_data_preprocess.py`



This Python script is used to convert raw data into a VerL-compatible format. The main functions include:

- **Recursively process input directories or files** : Converts the '.jsonl 'raw data file to the'.parquet 'format and preserves the directory structure.
- **Data partition** : According to the file path, the data belongs to the 'train' or 'test' partition.
- **Random shuffle and merge** : Combine multiple '.parquet 'files into one file and randomly shuffle the data.
- **Metadata Add** : Add 'split' and other necessary information for each piece of data.



#### Directory Structure of Outputs

```merged
examples/bootcamp_generator_outputs/<time_stamp>_for_verl_merged/
├── train/
│   └── bootcamps.parquet
└── test/
    └── bootcamps.parquet
```
```not merged
examples/bootcamp_generator_outputs/<time_stamp>_for_verl/
├── train/
│   ├── bootcamp1.parquet
│   ├── bootcamp2.parquet
│   └── ...
└── test/
    ├── bootcamp1.parquet
    ├── bootcamp2.parquet
    └── ...    
```


#### Example Command

```bash
python examples/verl_usage/verl_preprocess.py --src examples/bootcamp_generator_outputs/2025-03-07-16:48:28
```



This command converts all.jsonl files in examples/bootcamp_generator_outputs/2025-03-07-16:48:28 to VerL-compatible.parquet files and outputs to the default directory:

```
examples/bootcamp_generator_outputs/2025-03-07-16:48:28_for_verl_merged/
examples/bootcamp_generator_outputs/2025-03-07-16:48:28_for_verl/
```



---



### 4. Sumarizing the Workflow 


1. **Data Preprocessing** :

- Convert raw.jsonL data to VerL-compatible.parquet format using verl_data_preprocess.py.
- merged output directory is' <src>_for_verl_merged ', containing 'train/bootcamps.parquet' and 'test/bootcamps.parquet' files.

merged output directory <src>_for_verl ', containing multiple.parquet files, each corresponding to a.jsonl file.



2. **Embed bootcamp_reward_for_verl.py into the verl framework**

This is the following code snippet

```python
    elif data_source.startswith("bootcamp/"):
        try:
            import importlib
            import json
            bootcamp_name = data_source.split('/')[1]
            class_name = bootcamp_name[0].upper() + bootcamp_name[1:] + "bootcamp"
            module = importlib.import_module(f"internbootcamp")
            ground_truth = json.loads(ground_truth)
            return getattr(module, class_name).verify_score(solution_str, ground_truth, format_score=0)
        except Exception as e:
            print("Something woring with bootcamp reward because of ",e)
            return 0
```

Embed the relevant code in the VeRL framework as' /verl/utils/reward_score/__init__.py '
```
/verl
└── utils
    └── reward_score
        └── __init__.py
```

3. **Start the training experiment** :

- Complete the Settings in 'run_bootcamp.sh' `experiment_name`,`internbootcamp_path`,`train_files`,`test_files`,`actor_rollout_ref.model.path`,`trainer.default_local _dir 'and other important experimental parameters.

- Run 'run_bootcamp.sh' to start VERL training.



---



With the above tools and processes, you can efficiently prepare data and run VERL experiments.
