# Fast Optimizer Benchmark

Fast and cheap Benchmark for HPO and Optimizer.

Master Project at Machine Learning Lab Freiburg,
Simon Blauth, Tobias Bürger, Zacharias Häringer

This benchmark aims to be fast while maintaining a wide selection of different tasks. It also tries to be independent of the hardware used, however it requires a minimum of 4 gpus ideally capable of bfloat16 mixed precision.
One run of all tasks in this suite on a single optimizer configuration should not take more than a day.
A benchmark should state the following for each task: time taken per optimization step compared to baseline, best model performance, final model performance.

## Tasks

We try to cover a large range of deep learning tasks in this benchmark.

Instructions on how to write your own task can be found [here](pytorch_fob/tasks/README.md)

### Available Tasks

| Name | Dataset | Model | Task | Target Metric | Baseline Score | Baseline Runtime | Hardware |
| ------- | ---- | ----- | ---- | ------------- | -------------- | ---------------- | -------- |
| [mnist](pytorch_fob/tasks/mnist) | MNIST | MLP | Image Classification | Top-1 Accuracy | 0.97 | 1 min | 1 gpu |
| [classification](pytorch_fob/tasks/classification) | [Imagenet-64x64](https://patrykchrabaszcz.github.io/Imagenet32/) | [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf) | Image Classification | Top-1 Accuracy | 0.69 | 4h | 4 gpu |
| [classification_small](pytorch_fob/tasks/classification_small) | [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) | [Resnet18](https://arxiv.org/pdf/1512.03385.pdf) | Image Classification | Top-1 Accuracy | 0.77 | 10 min | 1 gpu |
| [segmentation](pytorch_fob/tasks/segmentation) | [MIT Scene Parse](http://sceneparsing.csail.mit.edu/) | [SegFormer](https://arxiv.org/abs/2105.15203) | Semantic Segmentation | Intersection over Union (IoU) | 35.6 | 5h | 4 gpu |
| [graph](pytorch_fob/tasks/graph) | [ogbg-molhiv](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) | [Graph Isomorphism Network (GIN)](https://arxiv.org/pdf/1810.00826.pdf) | Graph Property Prediction | ROC-AUC | 0.77 | 20min | 1 gpu |
| [graph_tiny](pytorch_fob/tasks/graph_tiny) | [Cora](https://paperswithcode.com/sota/node-classification-on-cora) | [GCN](https://arxiv.org/abs/1609.02907) | Node Classification | Accuracy | 0.82 | 1min | 1 gpu |
| [tabular](pytorch_fob/tasks/tabular) | [California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) | [FT Transformer](https://arxiv.org/pdf/2106.11959.pdf) | Tabular Regression | Test RMSE | 0.40 | 2 min | 1 gpu |
| [translation](pytorch_fob/tasks/translation) | [WMT17(en-de)](https://machinetranslate.org/wmt17) | [T5 small](https://jmlr.org/papers/volume21/20-074/20-074.pdf) | Machine Translation | BLEU (sacrebleu) | 26.3 | 6h | 4 gpus |

## Optimizer and Scheduler

An optimizer (together with the learning rate scheduler) contains the deep learning training algorithm to benchmark. Each optimizer has its own subfolder in the `optimizers` folder.
We currently have the following optimizers:

| Name | Optimizer | LR Scheduler |
| ---- | --------- | ------------ |
| adamw_baseline | [AdamW](https://arxiv.org/abs/1711.05101) | [Cosine Annealing](https://arxiv.org/abs/1608.03983) with linear warmup |
| adamcpr | [AdamCPR](https://arxiv.org/abs/2311.09058v2) | [Cosine Annealing](https://arxiv.org/abs/1608.03983) with linear warmup |
| sgd_baseline | Stochastic Gradient Descent | [Cosine Annealing](https://arxiv.org/abs/1608.03983) |

Instructions on how to add your own optimizer can be found [here](pytorch_fob/optimizers/README.md)

## Usage Instructions

### Installation

This repo was tested with Python 3.10, Python 3.11 works as well.
Some libraries are not updates so currently Python 3.12 breaks.
Create conda environment:
```bash
conda env create --file environment.yml
```

or alternatively:
```bash
conda create -n fob python=3.10 -y
```
Activate and install requirements
```bash
conda activate fob
pip install -r requirements.txt
pip install -e .
```

#### Troubleshooting

Sometimes pip fails to install the correct version of `mmcv`. If you encounter errors, try to install the correct version of `mmcv` as instructed on their [website](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

### How to run an experiment

Make sure you have the conda environment set up and activated.
Then you write an `experiment.yaml` (can be named differently) where you specify which optimizer and task you want to use. Every value can also be a list of values if you want to perform a gridsearch over them (more details below).

As an example we use this `experiment.yaml`:
```yaml
task:
  name:
    - mnist
    - classification_small
optimizer:
  - name: adamw_baseline
    beta2: 0.98
  - name: sgd_baseline
    momentum: 0.5
engine:
  seed: [42, 47]
```
This will produce 2x2x2=8 runs in total.
Each undefined parameter will be set using either `engine/default.yaml`, `optimizers/<optimizer>/default.yaml` or `tasks/<task>/default.yaml`.

Before you run the experiment make sure the datasets are prepared:
```bash
python -m pytorch_fob.dataset_setup experiment.yaml
```

Then you run the experiment:
```bash
python -m pytorch_fob.run_experiment experiment.yaml
```
This runs all tasks with all optimizers and hyperparameter specified inside `experiment.yaml` using grid-search.
You can either supply one value or a list of values for each entry. Grid-search combines each possible combination.
For example: you specified 3 task, 2 optimizer, 2 different learning rates and 4 seeds then you need a total 3 x 2 x 2 x 4 = 48 runs

You can additionally set values trough the command line (this overrides existing values). For example you can set the `data_dir` where datasets are stored using either:
```bash
python -m script experiment.yaml "engine.data_dir=<path>"
```
or you can specify it inside the `experiment.yaml`:
```yaml
engine:
  data_dir: <path>
```
## Usage Examples

In the following you can find example use cases for experiments. Here we will focus on running the training and testing pipeline. For instructions on how to plot the results, refer to the [evaluation/README.md](pytorch_fob/evaluation/README.md).

In these examples we will perform 'dry-runs' by setting the following parameters in the `experiment.yaml`:

```yaml
engine:
  train: false
  test: false
  plot: false
```

(Note: it might be a good idea to perform a dry run locally before wasting compute ressources)

### Example 1: Running a single task

This is an (quite) minimal example of how to run a single task. The model and training are customized. All other values will be taken from their respective `default.yaml`.

```yaml
task:
  name: mnist
  max_epochs: 1
  model:
    num_hidden: 42
```

Full experiment file: [examples/usage/1_single_task.yaml](examples/usage/1_single_task.yaml)

```bash
python -m pytorch_fob.run_experiment examples/usage/1_single_task.yaml
```

Take a look at the [output directory](examples/usage/outputs/experiment-1/) to see the results.

Note on the *folder name* of the runs:
Any hyperparameter that differs from the default will be included in the directory name. This is helpful for example when observing runs with Tensorboard.

Note on the *directory structure* of the outputs:
The individual runs will be placed

```
examples/usage/outputs/experiment-1  # (customize via: engine.output_dir)
└── taskname                         # (customize via: task.output_dir_name)
    └── optimizer name               # (customize via: optimizer.output_dir_name)
        ├── run_1                    # (name includes non-default parameters)
        ├── ...
        └── run_n
```

### Example 2: Comparing optimizers

To quickly run multiple optimizers on multiple hyperparameters, you can declare a list of values. This will perform a grid search over the values.

```yaml
optimizer:
  - name: adamw_baseline
    learning_rate: [1.0e-2, 1.0e-3]
    weight_decay: [0.1, 0.01]
  - name: adamcpr
    learning_rate: [1.0e-2, 1.0e-3]
    kappa_init_param: [0.5, 1, 2, 4, 8, 16, 32]
```

AdamW is used 4 (= 2 x 2) times, AdamCPR is used 14 (= 2 x 7) times, for a total of 18 runs.

Full experiment file: [examples/usage/2_comparing_optimizers.yaml](examples/usage/2_comparing_optimizers.yaml)

```bash
python -m pytorch_fob.run_experiment examples/usage/2_comparing_optimizers.yaml
```

Take a look at the [output directory](examples/usage/outputs/experiment-2/) to see the 18 run folders.

### Example 3: Running multiple tasks

If you want to use this repository for benchmarking an optimizer you most likely want to run multiple tasks, on multiple seeds.

```yaml
task:
  - classification
  - classification_small
  - graph
  - graph_tiny
  - mnist
  - segmentation
  - tabular
  - translation
engine:
  seed: [1, 2, 3]
```

You can use any subset of the full task list, if some tasks are not relevant for you.
Every task will be run on every seed. By default, the benchmark uses deterministic algorithms wherever possible and logs a warning otherwise.

Full experiment file: [examples/usage/3_benchmark_optimizers.yaml](examples/usage/3_benchmark_optimizers.yaml)

```bash
python -m pytorch_fob.run_experiment examples/usage/3_benchmark_optimizers.yaml
```

Take a look at the [output directory](examples/usage/outputs/experiment-3/) to see the results.

### Example 4: Running different versions of the same task

You can also run different versions of the same task (or optimizer).
This might be useful when you do not want a full grid search, but only want to combine certain groups.

The full grid search would be 2x2x2x2, we only want 8
🟦: group1  normalizer=quantile and noise=1.e-3 (+optimizer)
🟧: group2  normalizer=standard and noise=0
⬜: unwanted parameter combinations

🟦⬜⬜🟧
⬜🟦🟧⬜
⬜🟧🟦⬜
🟧⬜⬜🟦

```yaml
task:
  - name: tabular
    output_dir_name: tabular_quantile
    train_transforms:
      normalizer: quantile
      noise: 1.e-3
  - name: tabular
    output_dir_name: tabular_standard
    train_transforms:
      normalizer: standard
      noise: 0
optimizer:
  name: adamw_baseline
  learning_rate: [1.e-2, 1.e-3]
  weight_decay: [1.e-2, 1.e-3]
```

Full experiment file: [examples/usage/4_multiple_task_versions.yaml](examples/usage/4_multiple_task_versions.yaml)

```bash
python -m pytorch_fob.run_experiment examples/usage/4_multiple_task_versions.yaml
```

Take a look at the [output directory](examples/usage/outputs/experiment-4/) to see the results.

### Example 5: Running experiments with SLURM (convenience)

You can run experiments with SLURM. This is a convenience feature that allows you to run experiments on remote clusters. It splits each run of the experiment into a seperate job.

```yaml
engine:
  run_scheduler: slurm_array
  sbatch_args:
    partition: my_gpu_partition  # adapt to your cluster
  sbatch_script_template: path/to/template.sh
```

- The `slurm_array` scheduler will put the runs into an array job. Therefore all slurm relevant parameters (e.g. devices, time, workers, ...) need to be equal across all runs. Using this scheduler is only recommended when running a single task.
The `slurm_jobs` scheduler on the other hand will put each run into a seperate job.
- arguments put in `sbatch_args` will be passed to sbatch.
  e.g. `partition: my_gpu_partition` is parsed to `--partition=my_gpu_partition`

  - per default gpus equal to `engine.devices` and a number of cpus according to `engine.workers` are requested.
  - The requested time is set according to the defaults per task. It is recommended to use the `engine.sbatch_time_factor` to scale the default time per task for slower / faster machines.
- Wrap the FOB execution in your pre- and post commands (e.g. conda activation) with an `sbatch_script_template` the placeholder `__FOB_COMMAND__` in [examples/usage/sbatch_template.sh](examples/usage/sbatch_template.sh) will be replaced.


Full experiment file: [examples/usage/5_slurm.yaml](examples/usage/5_slurm.yaml)

Running this command without slurm will crash, but save the individual slurm scripts into [`path/to/sbatch_scripts`](examples/usage/outputs/experiment-5/sbatch_scripts) for us to look at.

```bash
python -m pytorch_fob.run_experiment examples/usage/5_slurm.yaml
```

Take a look at the [output directory](examples/usage/outputs/experiment-5/) to see the results.

## License
This repository is licensed under the Apache License 2.0.

However, please be aware that the repository includes various models and datasets, each of which may have its own licensing terms. It is the responsibility of the users to ensure that they comply with the specific licenses of these models and datasets.

By using this repository, you agree to respect and comply with all relevant licenses associated with the models and datasets. The Apache License 2.0 applies only to the original content and code provided in this repository.
