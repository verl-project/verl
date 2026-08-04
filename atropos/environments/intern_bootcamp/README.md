# InternBootcamp RL Training Environment

## Overview

The InternBootcamp RL Training Environment is a flexible and extensible framework for training large reasoning models using reinforcement learning on verifiable reasoning tasks. Based on the [InternBootcamp](https://github.com/InternLM/InternBootcamp) library, this environment provides a seamless integration between InternBootcamp's comprehensive collection of reasoning tasks and the Atropos RL training infrastructure.

## How InternBootcamp Works

InternBootcamp is a library that provides:

1. **Standardized Task Interface**: Each task (called a "bootcamp") implements three core methods:
   - `case_generator()`: Generates problem instances with controllable difficulty
   - `prompt_func()`: Converts problem instances into natural language prompts
   - `verify_score()`: Verifies and scores model responses

2. **Diverse Task Coverage**: Over 1,000 verifiable reasoning tasks including:
   - Logic puzzles (e.g., Game24, Sudoku, N-Queens)
   - Mathematical problems (algebra, geometry, calculus)
   - Algorithm challenges (sorting, searching, optimization)
   - Game-based reasoning (chess, Go, strategic games)
   - Pattern recognition and sequence problems

3. **Automatic Task Generation**: Tasks can generate unlimited problem instances with:
   - Controllable difficulty parameters
   - Consistent verification methods
   - Scalable complexity

## Architecture

```
InternBootcamp RL Environment
├── Task Selection Layer
│   ├── Single Task Mode (train on one specific bootcamp)
│   ├── Multi-Task Mode (train on multiple bootcamps - TBD)
│   └── Curriculum Mode (progressive difficulty - TBD)
│
├── InternBootcamp Integration
│   ├── Bootcamp Registry (dynamic task discovery)
│   ├── Bootcamp Instance Management
│   ├── Problem Generation Pipeline
│   └── Response Verification System
│
├── RL Training Loop
│   ├── Trajectory Collection
│   ├── Reward Calculation
│   └── Policy Updates
│
└── Atropos Base Environment
    ├── Server Management
    ├── Batch Processing
    └── Wandb Logging
```

## Key Features

### 1. Dynamic Task Discovery
The environment automatically discovers all available bootcamp tasks (1000+) without manual imports:

```python
from environments.intern_bootcamp.bootcamp_registry import get_available_bootcamps

# List all available tasks
tasks = get_available_bootcamps()
print(f"Found {len(tasks)} bootcamp tasks")
# Output: Found 1069 bootcamp tasks
```

### 2. Simple Task Selection
Train on any available bootcamp task by name:

```python
# Train on Game24
env = InternBootcampEnv(task_name="Game24bootcamp", task_params={"num_numbers": 4})

# Train on Sudoku
env = InternBootcampEnv(task_name="Sudokubootcamp")

# Train on Maze solving
env = InternBootcampEnv(task_name="Mazebootcamp")
```

### 3. Automatic Problem Generation
Each training step:
1. Instantiates the selected bootcamp with specified parameters
2. Generates a new problem instance using `case_generator()`
3. Converts it to a natural language prompt via `prompt_func()`
4. Collects model responses
5. Verifies correctness using `verify_score()`

### 4. Flexible Reward System
- **Base rewards**: Correct/incorrect responses (configurable)
- **Format bonuses**: Proper answer formatting (e.g., `\boxed{}` for math)
- **Reasoning bonuses**: Quality of step-by-step explanations
- **Task-specific scoring**: Each bootcamp can define its own scoring logic

## Installation

1. Clone the repository and navigate to the environment:
```bash
cd environments/intern_bootcamp
```

2. Install InternBootcamp (already included as a submodule):
```bash
cd internbootcamp_lib && uv pip install -e .
```

## Usage Examples

### 1. Single Task Training
Train on Game24 puzzles with specific difficulty:

```bash
python -m environments.intern_bootcamp serve \
    --env--task_name "Game24bootcamp" \
    --env--task_params '{"num_numbers": 4, "range_max": 100}' \
    --env--group_size 8 \
    --env--total_steps 10000
```

### 2. Exploring Available Tasks
List all available bootcamp tasks:

```python
from environments.intern_bootcamp.bootcamp_registry import get_available_bootcamps

tasks = get_available_bootcamps()
for task in tasks[:20]:  # Show first 20
    print(task)
```

### 3. Custom Configuration File
Use a YAML configuration for training:

```yaml
# config/intern_bootcamp_game24.yaml
env:
  task_name: "Game24bootcamp"
  task_params:
    num_numbers: 4
    range_max: 50
    target_max: 50

  correct_reward: 1.0
  incorrect_reward: -0.5
  format_bonus: 0.2

  group_size: 8
  total_steps: 10000
  steps_per_eval: 100

openai:
  model_name: "gpt-4"
  temperature: 0.7
  max_tokens: 2048
```

Run with config:
```bash
python -m environments.intern_bootcamp serve --config config/intern_bootcamp_game24.yaml
```

## Available Bootcamp Tasks

The environment supports over 1000 bootcamp tasks. Some examples include:

- **Math & Logic**: Game24bootcamp, Sudokubootcamp, Kakurobootcamp
- **Algorithms**: Mazebootcamp, Slitherlinkbootcamp, Bridgesbootcamp
- **Games**: InternGObootcamp, Chessbootcamp
- **Pattern Recognition**: Arcbootcamp, Nonogramsbootcamp
- **Code Generation**: CodeIObootcamp, BigCodeBenchbootcamp
- **Language Tasks**: Cipherbootcamp, WordSortingbootcamp

Use `get_available_bootcamps()` to see the full list.

## Implementation Details

### Environment Configuration

```python
class InternBootcampEnvConfig(BaseEnvConfig):
    # Task selection
    task_name: str = "Game24bootcamp"  # Bootcamp task name
    task_params: Dict[str, Any] = {}   # Task-specific parameters

    # Reward configuration
    correct_reward: float = 1.0
    incorrect_reward: float = -0.5
    format_bonus: float = 0.2

    # Training parameters
    require_reasoning: bool = True
    min_reasoning_length: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
```

### Bootcamp Registry

The environment uses a dynamic registry system to discover and manage bootcamp tasks:

```python
from environments.intern_bootcamp.bootcamp_registry import (
    create_bootcamp,
    get_available_bootcamps,
    bootcamp_registry
)

# Create a bootcamp instance
bootcamp = create_bootcamp("Game24bootcamp", num_numbers=4, range_max=50)

# Get information about a bootcamp
info = bootcamp_registry.get_bootcamp_info("Game24bootcamp")
print(info["parameters"])  # Shows accepted parameters
```

## Evaluation and Metrics

The environment tracks comprehensive metrics:

### Performance Metrics
- **Task accuracy**: Success rate on the specific bootcamp task
- **Format compliance**: Rate of properly formatted responses
- **Reasoning quality**: Length and coherence of explanations

### Training Metrics
- **Reward statistics**: Mean, std, min, max rewards
- **Problem diversity**: Variety of generated problems
- **Learning progress**: Improvement over time

## Troubleshooting

### Common Issues

1. **Task Not Found**
   ```
   ValueError: Unknown bootcamp: XYZBootcamp
   ```
   Solution: Check available tasks with `get_available_bootcamps()`

2. **Import Errors**
   ```
   ImportError: No module named 'internbootcamp'
   ```
   Solution: Install InternBootcamp: `cd internbootcamp_lib && pip install -e .`

3. **Parameter Errors**
   ```
   TypeError: __init__() got an unexpected keyword argument
   ```
   Solution: Check accepted parameters with `bootcamp_registry.get_bootcamp_info(task_name)`

## Future Enhancements

1. **Multi-Task Training**: Train on multiple bootcamps simultaneously
2. **Curriculum Learning**: Progressive difficulty advancement
3. **Task Composition**: Combine multiple bootcamps into complex reasoning chains
4. **Custom Bootcamps**: Easy integration of new reasoning tasks

## Contributing

To add new features or improvements:

1. Fork the repository
2. Create a feature branch
3. Implement your changes following the existing patterns
4. Add tests for new functionality
5. Submit a pull request with a clear description

## License

This environment follows the same license as the Atropos framework and InternBootcamp library.
