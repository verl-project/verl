# Evaluation

During training you can monitor your experiments with [Tensorboard](https://www.tensorflow.org/tensorboard).
We also try to provide some useful functionality to quickly evaluate and compare the results of your experiments.

One can use the ```evaluate_experiment.py``` to get a quick first impression of a finished experiment run.

## Plotting vs. raw data

You can use the plotting pipeline with your customized setting (as shown in the usage examples).
Alternatively you can use the script to export your data to a .csv and process the data to your own needs.

In this scenario, set ```evaluation.output_types: [csv]  # no plotting, just the data``` in your experiment yaml.

## Usage Examples

In the following you can find 4 example use cases for experiments and how to visualize the results as heatmaps.

1. testing an optimizer on a task
2. comparing two optimizers on the same task
3. comparing multiple optimizers on different tasks
4. comparing the influence of a single hyperparameter

Here we want to focus on the plotting. For instructions on how to run experiments, refer to the main [README](../../README.md). To get started right away, we provide the data for this example. If you want to reproduce it, refer to [this section](#reproducing-the-data).

### Plotting the experiment

By default, calling the `run_experiment.py` will plot the experiment after training and testing. To disable, set `engine.plot=false`.
To plot your experiment afterwards, call the `evaluate_experiment.py` with the same experiment yaml. To adjust how to plot, change the values under the `evaluation` key of the experiment. Take a look at the [evaluation/default.yaml](default.yaml) to see which settings are available. Some of these keys are explained in the examples below to give the reader a first impression. Note that some default parameters are set in the respective tasks (e.g. in [tasks/mnist/default.yaml](../tasks/mnist/default.yaml)).

### Example use cases

Here are some example scenarios to give you an understanding of how our plotting works. Run the commands from the root of the repository. Take a look at the yaml files used in the command to see what is going on.

#### Example 1

This example is a good starting point; it shows the performance of a single default optimizer on one of the tasks.
Experiment file: [examples/plotting/1_mnist-adamw.yaml](../../examples/plotting/1_mnist-adamw.yaml)

```python -m pytorch_fob.evaluate_experiment examples/plotting/1_mnist-adamw.yaml```

![your plot is not finished yet](../../examples/plotting/1_mnist-adamw-last-heatmap.png)

This example uses only the final model performance and only creates the plot as png.

Helpful settings:

- ```checkpoints: [last]```  # you could use [last, best] to additionaly plot the model with the best validation
- ```output_types: [png]```  # you could use [pdf, png] to also create a pdf


#### Example 2

You can compare two different optimizers.
Experiment file: [examples/plotting/2_adamw-vs-sgd.yaml](../../examples/plotting/2_adamw-vs-sgd.yaml)

```python -m pytorch_fob.evaluate_experiment examples/plotting/2_adamw-vs-sgd.yaml```

![your plot is not finished yet](../../examples/plotting/2_adamw-vs-sgd-last-heatmap.png)

Helpful settings:

- ```plot.x_axis: [optimizer.weight_decay, optimizer.kappa_init_param]```  # the values given here are used as the value for the axis. The order in the list is used from left to right for the plot columns
- `column_split_key: optimizer.name` This creates a column for each different optimizer (default behavior). You can set this to null to disable columns or choose a different key.


#### Example 3

There are multiple tasks in the benchmark, this example shows how to get a quick overview over multiple at the same time.
Experiment file: [examples/plotting/3_mnist-and-tabular_adamw-vs-sgd.yaml](../../examples/plotting/3_mnist-and-tabular_adamw-vs-sgd.yaml)

```python -m pytorch_fob.evaluate_experiment examples/plotting/3_mnist-and-tabular_adamw-vs-sgd.yaml```

![your plot is not finished yet](../../examples/plotting/3_mnist-and-tabular_adamw-vs-sgd-last-heatmap.png)

Helpful settings:

 - ```split_groups: ["task.name"]```

Every non unique value for each parameter name in `split_groups` will create its own subplot.
Instead of a list you can set to `false` to disable splitting or `true` to split on every parameter that is different between runs (except those already in `column_split_key` or `aggregate_groups`).
This list is useful if there are just a few parameters you want to split.

#### Example 4

Any parameter that is neither on the x-axis nor y-axis will either be aggregated over or split into subplots.
Any individual square of a heatmap shows the *mean* and *std* over multiple runs (as seen in the previous plots). Here we show how to choose the runs to aggregate.
Experiment file: [examples/plotting/4_adamw-vs-sgd_seeds.yaml](../../examples/plotting/4_adamw-vs-sgd_seeds.yaml)

```python -m pytorch_fob.evaluate_experiment examples/plotting/4_adamw-vs-sgd_seeds.yaml```

![your plot is not finished yet](../../examples/plotting/4_adamw-vs-sgd_seeds-last-heatmap.png)

Helpful settings:

- Control the std with
    - ```plot.std```  # toggle off with ```False```
    - ```plot.aggfunc: std```  # also try ```var```
- control the rows with
    - ```split_groups: ["engine.seed"]```
    - ```aggregate_groups: []```

Per default the plot will display the *mean* and *std* calculated over the seeds.
We need to remove the seed from the ```aggregate_groups``` list (by giving an empty list instead). This list is useful if there are additional parameters you want to aggregate over.


-------------------------------------------------------------------------------

### Reproducing the Data

Lets create some data that we can plot; from the root directory call:

#### Data Download

first we make sure the data is already downloaded beforehand:

```python -m pytorch_fob.dataset_setup examples/plotting/3_mnist-and-tabular_adamw-vs-sgd.yaml```

This will download the mnist data (required for 1-4) and tabular (required for 3) into the [examples/data](../../examples/data) directory - path can be changed in the corresponding yaml you want to use (e.g. [examples/plotting/1_mnist-adamw.yaml](../../examples/plotting/1_mnist-adamw.yaml) if you have already set up your benchmark).

Estimated disk usage for the data: ~65M

#### Training

The 2 tasks will be run on 2x2 hyperparameter on 2 different seeds per optimizer for a total of 32 runs.

```python -m pytorch_fob.run_experiment examples/plotting/3_mnist-and-tabular_adamw-vs-sgd.yaml```

After training finished you should find 32 run directories in [examples/plotting/outputs](../../examples/plotting/outputs)

All parameters that differ from the default value are noted in the directory name.
