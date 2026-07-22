# RE-ARC: Reverse-Engineering the Abstraction and Reasoning Corpus
## [Addressing ARC via Procedural Example Generation](https://arxiv.org/abs/2404.07353)

This repository presents code to procedurally generate examples for the [ARC](https://github.com/fchollet/ARC) training tasks. For each of the 400 tasks, an example generator is provided. See the [demo notebook](demo.ipynb) for example usage of the code and a visualization of the data. The primary entry point is the `generate_dataset` function defined in [main.py](main.py). The file [re_arc.zip](re_arc.zip) contains 1000 verified generated examples for each of the 400 training tasks (re_arc/tasks contains a json file for each ARC task containing an array of json objects each with keys "input" and "output") alongside two difficulty metrics for each example and some task-level metadata about runtime and sample-efficiency, the result of running the notebook, which calls `generate_dataset` with the default parameter values. The only major dependency is the [ARC-DSL](https://github.com/michaelhodel/arc-dsl), which is however included as a single file in [dsl.py](dsl.py), as it is not provided as a Python package. Other relevant files are [generators.py](generators.py), which contains the task-specific example generators, and [verifiers.py](verifiers.py), which contains the corresponding task solver programs used for keeping only generated examples that are valid.

For a more in-depth description of the work, see the [notes on arxiv](https://arxiv.org/abs/2404.07353).


### Example usage:

```python 
from main import demo_generator
demo_generator('00d62c1b')
```

#### 00d62c1b (original)

![00d62c1b (original)](00d62c1b_original.png "00d62c1b (original)")


#### 00d62c1b (generated)

![00d62c1b (generated)](00d62c1b_generated.png "00d62c1b (generated)")

