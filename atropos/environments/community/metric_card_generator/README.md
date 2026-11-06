# Metric Card Generator Environment

## Design and Motivation

This environment generates structured JSON configurations for Metric Card UI components for AI model evaluation dashboards. It demonstrates a closed-loop generation, evaluation, and visualization pipeline using Atropos.

The environment challenges language models to produce well-structured, valid JSON metric card configurations that can be directly used in front-end applications. This tests the model's ability to:
- Follow specific schema requirements
- Generate complex nested structures
- Maintain consistent JSON formatting
- Create semantically meaningful metric descriptions

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the environment with process command to generate rollouts
python metric_card_generator.py process --env.data_path_to_save_groups artifacts/metric_rollouts.jsonl

# View the generated HTML visualization
# Open artifacts/metric_rollouts.html in a browser
```

## Environment Components

- **metric_card_generator.py**: Main environment implementation with prompting and evaluation logic
- **extract_metric_training.py**: Utility to extract high-quality examples for training
- **trainingDataScript.py**: Creates training datasets from collected examples
- **show_score_distribution.py**: Visualization tool for analyzing model performance

## Artifacts

The artifacts folder includes:
- **metric_rollouts.jsonl**: Raw model outputs with scores
- **metric_rollouts.html**: Visualization of model outputs and scores
- **metric_training.jsonl**: Processed examples suitable for fine-tuning
- **metric_training_high_quality.jsonl**: Filtered high-quality examples

## Evaluation Metrics

The environment evaluates model outputs on several dimensions:

- **JSON Validity**: Whether the output is valid, parseable JSON
- **Schema Compliance**: Whether the output follows the required structure
- **Semantic Quality**: Whether the metrics described make sense for the given context
- **Formatting**: Proper nesting, field types, and attribute consistency

## WandB Integration

Performance metrics are logged to Weights & Biases, including:
- Percent of valid JSON responses
- Average scores across evaluation criteria
- Token usage efficiency
- Examples of best and worst performing generations

## Use with Training

This environment can be integrated into the Atropos training loop to improve a model's ability to generate structured JSON output:

```bash
# Example training command
python example_trainer/trainer.py --environment metric_card_generator --model your_model --iterations 1000
```
