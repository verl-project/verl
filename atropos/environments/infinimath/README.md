# InfiniteMath Environment

## Environment Overview

This environment provides procedurally generated math problems with curriculum-based advancement. It allows an agent to solve increasingly difficult math problems, with the difficulty level adapting based on performance.

**Demonstrates:**
- Procedural content generation (math problems).
- Curriculum learning: The environment automatically adjusts the difficulty (levels 1-7) based on the LLM's success rate.
- Step-by-step reasoning evaluation: Rewards correctness, the presence of reasoning steps (within `<think>` tags), and the final answer format (`\boxed{}`).
- Handling LaTeX formatting for problems and answers.

**Training Goal:**
- To train LLMs to solve mathematical problems accurately.
- To encourage explicit step-by-step reasoning before providing an answer.
- To improve the LLM's ability to follow specific formatting instructions (using `<think>` tags and `\boxed{}`).
- To teach the model to handle progressively more complex problems through the curriculum.

## Features

- Progressive difficulty scaling across 7 levels of math problems
- Built-in curriculum system that adapts to agent performance
- Automatic problem generation with solutions
- Reward functions for accuracy, formatting, and boxed answer checking

## Usage

Before running the environment, ensure you have installed the necessary dependencies. Navigate to the `environments/infinimath/` directory and run:

```bash
pip install -r requirements.txt
```

To run the InfiniteMath environment for local testing of the curriculum advancement:

```bash
python environments/infinimath/infinimath_local_server.py
```

The `infinimath_local_server.py` script contains the primary configuration for the environment when run in this standalone mode. You can modify this script directly to change parameters such as the model used, API keys (via environment variables), and various curriculum or reward settings.
The script is designed for local debugging and demonstration of the environment's capabilities.
