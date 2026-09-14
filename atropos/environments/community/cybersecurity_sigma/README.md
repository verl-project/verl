# Cybersecurity Sigma Rule Generation Environment

This environment trains LLMs to generate semantically correct Sigma detection rules from threat-hunting prompts. It provides two different reward mechanisms for evaluating generated rules.

## Overview

The environment focuses on structured generation tasks where outputs must be valid YAML conforming to Sigma detection rule schemas. It includes two implementations with different reward functions:

1. **Jaccard Similarity Reward** (`jaccard_reward_env.py`) - Uses token-based similarity scoring
2. **LLM Judge Reward** (`llm_judge_env.py`) - Uses LLM-based semantic evaluation

## Core Features

### Dataset Integration
- Uses the `mmaisel1/nous-rl-hackathon-sigma` dataset from Hugging Face
- Contains threat-hunting prompts paired with corresponding Sigma rules
- Automatic train/test split with shuffling for reproducibility

### Structured Output Format
- Enforces specific output format with `<think>...</think>` reasoning tags
- Requires YAML output wrapped in LaTeX `\boxed{...}` environment
- Validates YAML syntax and Sigma rule structure

### Dual Reward Mechanisms

#### Jaccard Similarity Scoring
- Compares flattened key paths of gold and generated YAML under `detection:` section
- Uses scikit-learn's Jaccard similarity for token-based matching
- Tends to produce low and sparse rewards due to structural mismatches

#### LLM-as-a-Judge Scoring
- Uses binary LLM evaluation for semantic equivalence assessment
- Returns 1.0 if generated rule is functionally equivalent to gold standard
- Provides higher-fidelity supervision even when structure varies

### Advanced Features
- Length penalty system for overly verbose outputs
- Comprehensive evaluation metrics tracking
- W&B integration for experiment monitoring
- Configurable token limits and batch sizes

## Technical Implementation

### Environment Configuration
- **Model**: NousResearch/DeepHermes-3-Llama-3-3B-Preview
- **Max Token Length**: 2048 tokens
- **Group Size**: 8 completions per prompt
- **Batch Size**: 12 items per batch
- **Evaluation Frequency**: Every 100 steps

### System Prompt Structure
The environment uses a detailed system prompt that:
- Enforces structured reasoning with `<think>` tags
- Requires YAML output in `\boxed{}` environment
- Provides Sigma rule best practices and examples
- Specifies exact formatting requirements for parser compatibility

### Scoring Pipeline
1. **Extraction**: Parse YAML from `\boxed{}` wrapper using regex
2. **Validation**: Attempt YAML parsing and structure validation
3. **Evaluation**: Apply either Jaccard similarity or LLM judge scoring
4. **Aggregation**: Collect scores for batch-level reward computation

## Setup and Usage

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"  # For LLM judge (optional)
export NOUS_API_KEY="your-nous-api-key"     # For model inference
```

### Command Line Usage
```bash
# Jaccard similarity reward
python environments/community/cybersecurity_sigma/jaccard_reward_env.py

# LLM judge reward
python environments/community/cybersecurity_sigma/llm_judge_env.py
```

### Dependencies
- `datasets` - Hugging Face dataset loading
- `scikit-learn` - Jaccard similarity computation (jaccard_reward_env only)
- `latex2sympy2_extended` - LaTeX parsing utilities
- `math_verify` - YAML extraction from LaTeX boxes
- `openai` - LLM judge API calls (llm_judge_env only)

## Research Applications

### Cybersecurity Training
- Train models to understand threat detection patterns
- Generate rules for various attack vectors and techniques
- Develop automated threat hunting capabilities

### Structured Generation Research
- Study LLM performance on constrained output formats
- Compare token-based vs. semantic evaluation methods
- Investigate reasoning quality in cybersecurity domains

### Evaluation Methodology Development
- Benchmark different reward function approaches
- Analyze correlation between structural and semantic correctness
- Develop better automated evaluation metrics for domain-specific tasks

## Performance Characteristics

### Jaccard Similarity Results
- **Typical Rewards**: 0.1-0.3 range due to structural sensitivity
- **Strengths**: Fast computation, deterministic scoring
- **Limitations**: Sensitive to formatting differences, low reward density

### LLM Judge Results
- **Typical Rewards**: Binary 0.0/1.0 with higher success rates
- **Strengths**: Semantic understanding, format flexibility
- **Limitations**: API latency, potential inconsistency, cost considerations

## Example Outputs

### Input Prompt
```
DotNET Assembly DLL Loaded Via Office Application: Detects any assembly DLL being loaded by an Office Product
```

### Expected Sigma Rule Format
```yaml
detection:
  condition: selection
  selection:
    process_name:
      - excel.exe
      - word.exe
      - powerpnt.exe
    dll_loaded: "*.dll"
logsource:
  category: process
  product: windows
```

The environment provides a robust framework for training LLMs on cybersecurity detection rule generation with flexible evaluation mechanisms suited for different research objectives.
