# Pipeline Usage

## Configuration files 

**puzzle_configs**: you can configure the parameters for `__init__` a bootcamp. Different parameters lead to different distribution of the generated samples.

**data_configs**: configuration files to run the final generation pipeline.
- You can manually add the tasks you want to generate data for in the file.
- You can use  `examples/pipelines/puzzle_configs/` to run `examples/pipelines/data_config_gen.py`. This will automatically generate **data_config_train.jsonl** and **data_config_test.jsonl** under `data_configs`.

For example, an example to include `futoshiki` is as follows.
```json
{"bootcamp_name": "futoshiki", "sample_number": 100, "config_file": "futoshiki", "bootcamp_cls_name": "Futoshikibootcamp"}
```
Here, `sample_number` means the number of data samples to generate, `config_file` the name of the task configuration file, and `bootcamp_cls_name` represent the class name of the bootcamp used to generate data.


## Running the Data Generation Pipeline

**run_pipeline.sh** contains the unified pipeline to generate data for all tasks based on the configurations. 

## Quick Start
1. Run the following command to gather all the bootcamp into a configuration file to specify options for data generation..
```bash
python examples/pipelines/quickgen_data_configs.py
```
You can adjust the `train_sample_number` and `test_sample_number` to control the number to samples to generate for the two sets.

2. Run `bash examples/pipelines/run_pipline.sh` to generate data with the output under `examples/bootcamp_generator_outputs`.