# Environments

This directory contains various environments for training and evaluating language models on different tasks. Each environment implements a specific task with its own input format, reward function, and evaluation metrics.

## Available Environments

---

###  MCQA Thinking Environment (`mcqa_thinking_env.py`)

Multiple Choice Question Answering environment that requires models to think through problems systematically.

**Input Format:**
- Questions from the MMLU (Massive Multitask Language Understanding) dataset
- Each item contains:
  - `prompt`: The question text
  - `answer`: Index of correct answer
  - `ground_truth`: Letter (A, B, C, D) of correct answer
  - `options`: List of possible answers

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if the model's answer matches the ground truth letter
- Score of 0.0 if incorrect or invalid response (multiple think tags, malformed thinking sections)
- Length penalty applied if all responses are correct:
  - No penalty for responses under 50% of max token length
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 50% and 100% of max length
  - Returns None if all scores are identical (no learning signal)

---

### GSM8K Environment (`gsm8k_server.py`)

Mathematical reasoning environment using the GSM8K dataset.

**Input Format:**
- Questions from GSM8K dataset
- Each item contains:
  - `question`: The math problem
  - `answer`: The numerical answer

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

You are allocated a maximum of 2048 tokens, please strive to use less.

You will then provide your answer like this: \boxed{your answer here}
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \boxed{your answer here}
```

**Reward Function:**
- Score of 1.0 if the model's answer matches the ground truth (using LaTeX verification)
- Score of 0.0 if incorrect or if ground truth is not parseable
- Length penalty applied if all responses are correct:
  - No penalty for responses under 50% of max token length
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 50% and 100% of max length
  - Returns None if all scores are identical (no learning signal)

---

### Tool Calling Environment (`tool_calling_server.py`)

Environment for training models to make function calls in a structured format.

**Input Format:**
- Conversations from ShareGPT-Hermes function call dataset
- Each item contains:
  - `conversations`: List of messages with roles (system, human, gpt)
  - Expected tool calls in JSON format

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if all expected tool calls are present and match exactly (including nested JSON fields)
- Score of 0.0 if any tool calls are missing, incorrect, or malformed
- Length penalty applied if all responses are correct:
  - No penalty for responses under 50% of max token length
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 50% and 100% of max length
  - Returns None if all scores are identical (no learning signal)

---

### RLAIF Server Environment (`rlaif_server.py`)

Environment for Reinforcement Learning from AI Feedback (RLAIF). Used for aligning models to specific personalities or styles based on AI-generated preferences or reward signals.

**Input Format:**
- Typically involves prompts for which responses are generated and then evaluated by a reward model or preference model to guide the LLM's behavior. Specifics depend on the RLAIF setup.

**System Prompt:**
- Varies based on the desired personality/style (e.g., "Egregore," "Ascension Maze").

**Reward Function:**
- Based on the output of an AI judge/reward model, designed to score responses according to the target alignment criteria.

---

### Financial Fundamentals Prediction Environment (`fundamental_prediction_environment.py`)

Environment for training models to predict financial fundamentals using the "NousResearch/company-fundamentals-prediction-lite" dataset.

**Input Format:**
- Items include `context` (company fundamentals, news, macroeconomic data), `fundamental_metric` (e.g., revenue, EPS), and ground truth `answer` ("maintained", "raised", or "reduced") and `magnitude` (percentage change). The model analyzes the `context` to predict the `answer` and `magnitude` for the given `fundamental_metric`.

**Task:**
- Predict directional changes and magnitude for company financial fundamentals.

**Reward Function:**
- Based on the accuracy of predictions for both direction and magnitude.

---

### Math Server Environment (`math_server.py`)

A versatile math problem-solving environment supporting multiple datasets and operational modes.

**Datasets:**
- Integrates `gsm8k` (various subsets), `competition_math`, `math_qa`, and `MetaMathQA`.

**Operational Modes:**
- Supports standard problem solving, RLAIF (Reinforcement Learning from AI Feedback) for preference learning between solutions, a "judge" mode for evaluating solution correctness, and a "retry/self-correct" mode utilizing feedback on previous attempts.

**Input Format:**
- Mathematical problems, varying slightly by operational mode (e.g., including solutions for judging/RLAIF).

**System Prompt:**
- Dynamically constructed based on the operational mode. For standard problem solving, the prompt focuses on the problem itself. Other modes include specific instructions for judging, preference selection, or self-correction.

**Reward Function:**
- Based on the correctness of the mathematical solution, with variations depending on the mode (e.g., preference scores in RLAIF).

---

### Math Server Zero Environment (`math_server_zero.py`)

A math problem-solving environment using the "zwhe99/DeepMath-103K" dataset, with a structured prompt format inspired by the Open-Reasoner-Zero project.

**Input Format:**
- Mathematical problems from the "zwhe99/DeepMath-103K" dataset.

**System Prompt Structure:**
- Utilizes a specific conversational format where the AI is instructed to first think (using `<think> </think>` tags) and then provide the answer (using `<answer> </answer>` tags, with the final numerical answer in `\boxed{}`). The overall prompt guides the model through this structured reasoning and response process.
  - `prompt_format = "A conversation between User and Assistant... User: {prompt}\nAssistant: <think>"`
  - `problem_format = "You must put your answer inside <answer> </answer> tags... This is the problem:\n{problem}"`

**Reward Function:**
- Based on the correctness of the mathematical solution within the `<answer>` tag, verified using LaTeX parsing.

---

### Coding Server Environment (`code_execution_server/coding_server.py`)

Environment for training models to generate and potentially execute code.

**Input Format:**
- Coding problems or prompts (e.g., from datasets like MBPP, HumanEval).

**System Prompt:**
- Instructs the model to generate code for a given problem.

**Reward Function:**
- Based on correctness of the generated code, often involving execution and unit test passing.
- The `code_execution_server/` directory also contains a `Dockerfile` for containerized execution.

---

### Dataset Environment (`dataset_environment/dataset_env.py`)

A highly configurable environment for working with Hugging Face datasets. For more details, see the [Dataset Environment README](dataset_environment/README.md).

**Purpose:**
- Allows users to easily define RL environments using existing datasets from Hugging Face Hub.

**Input Format:**
- Defined by the chosen Hugging Face dataset (user specifies prompt and answer fields).

**System Prompt:**
- Customizable by the user.

**Reward Function:**
- Highly flexible, supports a registry of predefined reward functions (e.g., `accuracy`, `format`, `cosine_scaled`) and allows users to create and register custom reward functions. Multiple reward functions can be combined with weights.

**Configuration:**
- Primarily through YAML files specifying dataset details, generation parameters, and reward functions.

---

### Multimodal DPO Environments (`multimodal_dpo/`)

A collection of environments for Direct Preference Optimization (DPO) with multimodal inputs. These environments are designed for tasks that involve processing both text and images.

**Files:**
- `ocr_vqa.py`
- `pixmo_clocks.py`
- `pixmo_count.py`
- `pixmo_point_explanations.py`
- `clevr_cogen_a_train.py`
- `clevr_complex.py`

**Purpose:**
- Training models on tasks such as Optical Character Recognition VQA, visual counting, and interpreting complex visual scenes (e.g., Clevr).

**Input Format:**
- Typically pairs of (image, text prompt) and corresponding preferred/dispreferred responses.

**Reward Function:**
- Based on the DPO mechanism, implicitly learned from preference data.

---

### Game Environments (`game_environments/`)

This section covers environments based on interactive games.

#### Gymnasium Taxi (`game_environments/gymnasium/gym_taxi.py`)

- **Game:** Based on the classic Gymnasium Taxi-v3 environment.
- **Task:** The agent controls a taxi to pick up a passenger and drop them off at the correct location.
- **Objective:** Optimize for efficient navigation and task completion.

#### Gymnasium Blackjack (`game_environments/gymnasium/blackjack/`)

Two Blackjack environment implementations are provided. For more details, see the [Blackjack README](game_environments/gymnasium/blackjack/README.md).

- **`blackjack_env_no_thinking.py` (Standard Blackjack):**
    - **Gameplay:** A standard version of Blackjack.
    - **Objective:** Achieve a hand total closer to 21 than the dealer without exceeding 21.
    - **Interaction:** Designed for shorter episodes without complex intermediate "thinking" steps. Aiming to teach the LLM to be a better policy model in uncertain environments.

- **`blackjack_env_thinking.py` (Blackjack with Windowed Decision Making & Counterfactuals):**
    - **Gameplay:** A more complex version designed for agents that produce long interaction sequences, including "thinking" steps.
    - **Features:** Windowed decision making, local alternative generation, value-based pruning, and counterfactual data for training (GRPO).
    - **Use Case:** Ideal for training LLMs that engage in explicit multi-step reasoning before action. Teaches the model to be more "confident" about selecting optimal moves & taking informed risks in uncertain environments, even with the knowledge that it might still lose with optimal play.

### Instruction Following Environment (`instruction_following_algorithm_environment.py`)

**Dependencies:**
- `datasets` (Hugging Face)
- `langdetect`

This environment was inspired by AllenAI's RLVR-IFEVAL environment and uses AllenAI's dataset from their Tulu3 paper and project:
- Dataset: https://huggingface.co/datasets/allenai/RLVR-IFeval
- Paper: https://arxiv.org/abs/2411.15124

Environment for training models to follow natural language instructions and constraints, based on the `allenai/RLVR-IFeval` dataset and environment.

**Input Format:**
- Each item from the processed `allenai/RLVR-IFeval` dataset contains:
  - `prompt`: The user's instruction string.
  - `func_name`: The string name of the verifier function (from a predefined map) used to check if the instruction is followed.
  - `args`: A dictionary of arguments for the specified verifier function.

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if the model's response correctly follows the instruction, as determined by the specific verifier function associated with the input prompt.
- Score of 0.0 if the response fails the verifier function.
- Length penalty applied if all responses in a batch are correct (receive a score of 1.0 before penalty):
  - No penalty for responses under a certain percentage (e.g., 75%) of max token length.
  - Linear penalty scaling from 1.0 down to 0.0 for responses between the threshold and 100% of max length.
  - Returns None if all scores are identical after potential penalties (no learning signal).

**Unique Configuration and Features:**
- **Dataset Configuration (`IFConfig`):
  - `dataset_name`: Specifies the primary dataset to use (defaults to `allenai/RLVR-IFeval`).
  - `dataset_config_name`: Optional name for a specific configuration or subset of the dataset.
  - `test_set_ratio`: Defines the proportion of the dataset reserved for testing (defaults to 5%).

- **Verifier-Based Scoring:** Utilizes a comprehensive map of verifier functions (`IF_FUNCTIONS_MAP`) to evaluate whether the model's
output adheres to diverse and specific constraints defined in the input instructions (e.g., keyword presence, response length, JSON format, etc.).

- **Specialized Dataset Processing:** The `setup` method is specifically designed to parse the `allenai/RLVR-IFeval` dataset, extracting user instructions, the corresponding verifier function name, and its arguments.

- **Fallback Mechanism:** Includes a fallback to a small, predefined dummy dataset if the primary dataset (`allenai/RLVR-IFeval`) cannot be loaded, ensuring operational continuity for testing or development.

---

### SWE-RL Environment (`swe_rl_env.py`)

Software Engineering Reinforcement Learning environment for training models to fix bugs based on issue descriptions and code context.

**Dependencies:**
- `datasets` (Hugging Face)
- `difflib`
- `wandb`
- `pydantic`

**Dataset:**
- Default: `princeton-nlp/SWE-bench_Lite_oracle`
- Configurable via `SWERLEnvConfig` (e.g., `dataset_name`, `dataset_split_train`, `dataset_split_eval`).

**Input Format (for the model via prompts):**
- `problem_statement`: The issue text.
- `content`: Relevant code segments from one or more files.

**System Prompts:**
1.  **Thinking System Prompt:**
    ```
    You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
    ```
2.  **Task System Prompt:**
    ```
    A user will ask you to solve a task. You should generate the solution. Your response format must follow the template below:
    ```
    (Followed by instructions on the SEARCH/REPLACE format)

**User Prompt Template:**
```
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
--- BEGIN FILE ---
``` {content} ```
--- END FILE ---
Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.
Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE
Here is an example:
```python
### mathweb/flask/app.py
import math
from flask import Flask
```
Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line ’ print(x)’, you must fully write that out, with all those spaces before the code!
Wrap each *SEARCH/REPLACE* edit in a code block as shown in the example above. If you have multiple *SEARCH/REPLACE* edits, use a separate code block for each one.
```

**Reward Function:**
- Primary reward is based on the `SequenceMatcher` ratio between the model's reconstructed generated patch and the oracle patch.
- A score of -1.0 is given initially.
- If the model's response has a `finish_reason` of "length", or if `<think>` tags are present but malformed, the reward remains -1.0 and advantage is set to zero for "length".
- If the SEARCH/REPLACE patch format is correctly parsed from the model's output (after potentially extracting content from `<think> </think>` tags):
    - The `SequenceMatcher.ratio()` between the reconstructed predicted patch and the `oracle_patch_str` is used as the reward.
- Buffers track:
    - `percent_format_correct_buffer`: Percentage of responses with correctly formatted patches.
    - `similarity_score_buffer`: List of similarity scores for correctly formatted patches.
    - `think_tags_present_buffer`: Percentage of responses where `<think>` tags were present.
    - `think_tags_well_formed_buffer`: Percentage of responses where `<think>` tags were present AND well-formed.

**Evaluation Metrics:**
- `eval/avg_similarity_score_correct_patch_format`: Average similarity score for responses that had a correctly formatted patch.
- `eval/patch_format_accuracy`: Proportion of evaluation items where the patch was correctly formatted.
- `eval/pass_at_1`: Proportion of evaluation items where the patch was correct and achieved a similarity score of 1.0.
- `eval/avg_think_tags_present`: Average presence of think tags in evaluation responses.
- `eval/avg_think_tags_well_formed`: Average well-formedness of think tags in evaluation responses.

**Unique Configuration and Features:**
- **Dataset Handling:** Loads training and test data from Hugging Face datasets, specifically tailored for SWE-bench like formats.
- **Patch Parsing:** Implements robust parsing for a specific SEARCH/REPLACE patch format.
- **Thinking Tag Processing:** Extracts content after `<think> </think>` tags for patch generation, and scores based on presence and well-formedness of these tags.
- **Wandb Logging:** Logs detailed training and evaluation metrics, including rollout tables with problem statements, full interaction text, oracle patches, and scores.

## Common Features

All environments share these common features:

1. **Training/Test Split:**
   - 98% training, 2% test split
   - Random shuffling with fixed seed (42)

2. **Metrics Tracking:**
   - Percent correct buffer
   - Completion lengths
   - Wandb integration for visualization
   - Rollout tracking

3. **Token Management:**
   - Maximum token length limits
   - Token length statistics tracking
   - Length penalty for excessive responses

4. **Evaluation:**
   - Separate evaluation on test set
   - Comprehensive metrics logging
   - Support for multiple model completions per prompt

5. **Detailed Documentation:**
   - Many environments, especially those with more complexity, include detailed `README.md` files within their respective subdirectories to provide specific context and usage instructions.

6. **Additional Libraries:**
   - If an environment requires specific libraries not covered by the main project dependencies, its subdirectory may include a `requirements.txt` file for easy installation via `pip`, or provide installation instructions in its `README.md`.

## Usage

Each environment can be initialized with:
- `config`: BaseEnvConfig object
- `server_configs`: List of OpenAI API configurations
- `slurm`: Boolean for distributed training
- `testing`: Boolean for testing mode

The environments follow a common interface with methods for:
- `setup()`: Loading and preparing datasets
- `get_next_item()`: Retrieving next training item
- `collect_trajectories()`: Generating model responses
- `score()`: Computing rewards
- `evaluate()`: Running evaluation on test set
- `wandb_log()`: Logging metrics to Weights & Biases

## 31. Cybersecurity Sigma Rule Generation Environment

**Location:** `environments/community/cybersecurity_sigma/`
**Contributor:** [Subrahmanyam2305](https://github.com/Subrahmanyam2305)
**PR:** [#74](https://github.com/NousResearch/atropos/pull/74)

### Core Features
- **Dual Reward Systems**: Jaccard similarity scoring and LLM-as-a-judge evaluation
- **Structured Output Generation**: Enforces YAML format with LaTeX `\boxed{}` wrapper
- **Cybersecurity Domain**: Trains models to generate Sigma detection rules from threat prompts
- **Dataset Integration**: Uses `mmaisel1/nous-rl-hackathon-sigma` from Hugging Face

### Technical Implementation
- **Environment Names**: `sigmarule` (Jaccard) and `llm_judge_sigmarule` (LLM judge)
- **Output Format**: `<think>...</think>` reasoning tags + YAML in `\boxed{}`
- **Reward Mechanisms**: Token-based Jaccard similarity vs. semantic LLM evaluation
- **Model Configuration**: DeepHermes-3-Llama-3-3B-Preview with 2048 token limit

### Research Applications
- **Cybersecurity Training**: Automated threat detection rule generation
- **Structured Generation**: Constrained output format research with YAML validation
- **Evaluation Methodology**: Comparison of token-based vs. semantic reward functions
- **Domain Expertise**: Training models on specialized cybersecurity knowledge

### Setup and Usage
```bash
# Environment variables
export OPENAI_API_KEY="your-key"  # For LLM judge (optional)
export NOUS_API_KEY="your-key"    # For model inference

# Run environments
python environments/community/cybersecurity_sigma/jaccard_reward_env.py
python environments/community/cybersecurity_sigma/llm_judge_env.py
```

### Performance Characteristics
- **Jaccard Rewards**: 0.1-0.3 range, fast but structurally sensitive
- **LLM Judge Rewards**: Binary 0.0/1.0, semantic understanding but API latency
- **W&B Integration**: Comprehensive experiment tracking and visualization
- **Length Penalties**: Applied for overly verbose rule generation

## 32. Wikipedia Article Research Environment

**Location:** `environments/community/wikipedia_research/`
**Contributor:** [aniemerg](https://github.com/aniemerg)
**PR:** [#72](https://github.com/NousResearch/atropos/pull/72)

### Core Features
- **Multi-Step Research Process**: Web search and content extraction with Tavily API integration
- **Factual Accuracy Evaluation**: OpenAI-powered line-by-line fact-checking against reference articles
- **Wikipedia Blocking**: Prevents direct Wikipedia access to encourage diverse source usage
- **Quality Assessment Framework**: Structure, comprehensiveness, and fact usage scoring

### Technical Implementation
- **Environment Name**: `WikipediaArticleCreator`
- **Research Tools**: `web_search` and `visit_page` with error handling and filtering
- **Evaluation System**: Dual scoring combining structural quality with factual accuracy
- **Episode Management**: Tracks complete research sessions with conversation history

### Research Applications
- **Information Synthesis**: Training models to combine multiple sources into coherent articles
- **Research Methodology**: Multi-step information gathering and fact verification
- **Quality Assessment**: Comprehensive article evaluation across multiple dimensions
- **Tool Usage Training**: Effective utilization of search and extraction capabilities

### Setup and Usage
```bash
# Environment variables
export TAVILY_API_KEY="your-tavily-key"    # Required for web research
export OPENAI_API_KEY="your-openai-key"    # Required for LLM and evaluation

# Direct usage
cd environments/community/wikipedia_research
python run_with_openai.py --topic "Climate change in Antarctica" --model "gpt-4o"

# Training mode
python -m atroposlib.cli.dpo \
    --env-module "environments.community.wikipedia_research.wikipedia_article_creator"
```

### Performance Characteristics
- **Research Efficiency**: 10-50 tool calls per article depending on complexity
- **Quality Metrics**: Structure (0-1), comprehensiveness (0-1), fact usage (0-1)
- **Accuracy Evaluation**: CORRECT/INCORRECT/UNKNOWN statement categorization
- **Combined Scoring**: Overall article score in [-1, 1] range balancing quality and accuracy
- **W&B Integration**: Complete research session tracking with tool usage analytics

## 33. Goofy Math Environment

**Location:** `environments/community/goofy_math/`
**Contributor:** [chinguun101](https://github.com/chinguun101)
**PR:** [#91](https://github.com/NousResearch/atropos/pull/91)

### Core Features
- **Dual Reward System**: Mathematical correctness verification + goofiness scoring
- **RLAIF-Based Judging**: AI feedback system for ranking entertaining vs. standard solutions
- **GSM8K Integration**: Uses standard math dataset with humor enhancement overlay
- **Position Bias Elimination**: Forward/reverse judgment pairs to ensure fair evaluation

### Technical Implementation
- **Environment Name**: `goofy_math`
- **Correctness Verification**: Uses `math_verify` and `latex2sympy2_extended` for objective scoring
- **Goofiness Assessment**: LLM judge evaluates entertainment value of mathematically correct solutions
- **Reward Formula**: `score = correctness_score + (goofiness_bonus * 0.5)`
- **Output Format**: `<think>...</think>` reasoning + `\boxed{answer}` format

### Research Applications
- **Educational AI**: Training math tutors that are both accurate and engaging
- **Personality Injection**: Adding entertainment value while maintaining technical correctness
- **Multi-Objective Optimization**: Balancing objective accuracy with subjective entertainment
- **Humor in AI**: Systematic approach to training models for appropriate comedic timing

### Setup and Usage
```bash
# Install requirements
pip install -r environments/community/goofy_math/requirements.txt

# Environment variables
export OPENAI_API_KEY="your-key"

# Process mode for examples
python environments/community/goofy_math/goofy_math_server.py process \
  --env.data_path_to_save_groups goofy_math_demo.jsonl \
  --env.total_steps 3

# Training mode
python -m atroposlib.cli.dpo \
    --env-module "environments.community.goofy_math.goofy_math_server"
```

### Performance Characteristics
- **Correctness Requirement**: Solutions must pass mathematical verification to receive any reward
- **Goofiness Scoring**: 0-1 range based on humor, sound effects, and creative explanations
- **Reward Distribution**: Base 1.0 for correctness + up to 0.5 bonus for entertainment value
- **Anti-Reward Hacking**: Goofiness only evaluated after correctness verification
- **W&B Integration**: Tracks goofiness histograms, judgment tables, and accuracy metrics

### Demo and Results
- **Video Demo**: [1-minute demonstration](https://www.loom.com/share/8704f63e2d2e4b4db23eab673d7990a2)
- **WandB Run**: [Experiment tracking](https://wandb.ai/goofymath/goofy_math/runs/z92gd2j4)
- **Unique Metrics**: `train/avg_goofiness_score`, `train/goofiness_histogram`, `train/judgement_table`

## 34. Options Implied Volatility Prediction Environment

**Location:** `environments/community/options_iv_prediction/`
**Contributor:** [michaelwaves](https://github.com/michaelwaves)
**PR:** [#78](https://github.com/NousResearch/atropos/pull/78)

### Core Features
- **Real Market Data Integration**: Live options data fetching via Yahoo Finance API (`yahooquery`)
- **Financial Analysis Training**: Teaches models options pricing relationships and implied volatility prediction
- **Thinking Process Framework**: Encourages step-by-step reasoning with `<think>` tags for complex financial analysis
- **Dual Scoring System**: Magnitude accuracy and binary correctness evaluation

### Technical Implementation
- **Environment Name**: `OptionsIVPrediction`
- **Data Source**: Real-time UNH (UnitedHealth Group) options chain data
- **Input Parameters**: Option price, stock price, strike price, time to expiry, risk-free rate
- **Output Format**: Structured prediction with exact format requirement: "The implied volatility will be: {percentage}%"

### Research Applications
- **Financial AI Development**: Training models to understand complex options pricing mechanisms
- **Quantitative Analysis**: Automated volatility prediction for trading and risk management
- **Educational Applications**: Teaching AI systems fundamental financial concepts
- **Real-World Integration**: Direct application to live market data and trading scenarios

### Setup and Usage
```bash
# Dependencies
pip install pandas wandb datasets tqdm yahooquery atroposlib

# Training mode
python environments/community/options_iv_prediction/options_iv_prediction.py serve \
    --env.total_steps 2000 --env.batch_size 1024

# Process mode (data generation)
python environments/community/options_iv_prediction/options_iv_prediction.py process \
    --env.data_path_to_save_groups ./outputs/options_rollouts.jsonl \
    --openai.api_key YOUR_KEY
```

### Performance Characteristics
- **Memory Usage**: ~2-4 GB RAM for typical configurations with live data processing
- **Data Processing**: Automatic filtering of invalid options (negative prices, expired contracts)
- **Scoring Metrics**: Magnitude accuracy (0-1 scale) and binary correctness (within 10% threshold)
- **Combined Reward**: Weighted combination (70% magnitude + 30% binary) for balanced learning
- **Market Integration**: Real-time data fetching with robust error handling for market anomalies

---
