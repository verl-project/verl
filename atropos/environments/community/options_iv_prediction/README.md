# Options Implied Volatility Prediction Environment

**Location:** `environments/community/options_iv_prediction/`
**Contributor:** [michaelwaves](https://github.com/michaelwaves)
**PR:** [#78](https://github.com/NousResearch/atropos/pull/78)

## Overview

This environment trains language models to predict implied volatility (IV) for stock options using real market data. The model analyzes option pricing parameters including option price, stock price, strike price, time to expiry, and risk-free rate to predict the implied volatility.

## Core Features

- **Real Market Data**: Uses Yahoo Finance API via `yahooquery` to fetch live options data
- **Financial Analysis**: Trains models to understand options pricing relationships
- **Thinking Process**: Encourages step-by-step reasoning with `<think>` tags
- **Accuracy Scoring**: Evaluates predictions based on magnitude accuracy and percentage correctness
- **WandB Integration**: Comprehensive logging and visualization of training metrics

## Environment Details

### Task Description
Given option market data (option price, stock price, strike price, time to expiry, risk-free rate), predict the implied volatility as a percentage.

### Input Format
```
Option Price: $X.XX
Stock Price: $X.XX
Strike Price: $X.XX
Time to Expiry: X.XXXXXX years
Risk-Free Rate: X.XX

Your final answer MUST use the exact format: "The implied volatility will be: {answer}"
Where {answer} is the implied volatility as a string in percent (e.g. 70%)
```

### Scoring Methodology
- **Magnitude Accuracy**: Measures how close the predicted IV is to the actual IV
- **Binary Correctness**: Whether the prediction is within an acceptable threshold
- **Combined Score**: Weighted combination of magnitude and binary accuracy

## Setup Instructions

### Dependencies
Install required packages:
```bash
pip install pandas wandb datasets tqdm yahooquery atroposlib
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

### Environment Variables
- **API Keys**: Configure OpenAI or other LLM provider API keys
- **WandB**: Set up Weights & Biases for experiment tracking

### Data Source
The environment automatically fetches real-time options data for UNH (UnitedHealth Group) using the Yahoo Finance API. No manual data preparation is required.

## Usage Examples

### Training Mode
```bash
python options_iv_prediction.py serve --env.total_steps 2000 --env.batch_size 1024
```

### Process Mode (Data Generation)
```bash
python options_iv_prediction.py process --env.data_path_to_save_groups ./outputs/options_rollouts.jsonl --openai.api_key YOUR_KEY
```

### Configuration Options
- `group_size`: Number of predictions per training group (default: 16)
- `max_token_length`: Maximum tokens for model responses (default: 16384)
- `steps_per_eval`: Evaluation frequency (default: 20)
- `wandb_name`: Custom name for WandB runs

## Performance Characteristics

- **Memory Usage**: ~2-4 GB RAM for typical configurations
- **API Calls**: Fetches live market data on startup, then uses cached data
- **Processing Time**: 1-3 minutes per batch depending on model size
- **Accuracy Metrics**: Tracks both percentage correctness and magnitude accuracy

## Technical Implementation

### Data Processing
1. Fetches real-time options chain data for UNH stock
2. Calculates time to expiry from current date to option expiration
3. Filters out invalid options (negative prices, expired options)
4. Creates train/test split (95%/5%)

### Scoring Algorithm
```python
def _calculate_iv_score(self, predicted_iv, expected_iv):
    # Magnitude accuracy (0-1 scale)
    magnitude_accuracy = max(0, 1 - abs(predicted_iv - expected_iv) / 100)

    # Binary correctness (within 10% threshold)
    is_correct = abs(predicted_iv - expected_iv) <= 10

    # Combined score
    return magnitude_accuracy * 0.7 + (1.0 if is_correct else 0.0) * 0.3
```

### Model Integration
- Compatible with any OpenAI-compatible API
- Supports both local and cloud-based language models
- Automatic tokenization and conversation management

## Output Format

### WandB Metrics
- `train/percent_correct`: Percentage of predictions within threshold
- `train/magnitude_accuracy`: Average magnitude accuracy score
- `eval/percent_correct`: Evaluation accuracy
- `train/rollouts`: Sample predictions with scores

### Data Files
- Generated rollouts saved to specified JSONL file
- HTML visualization of training conversations
- Detailed prediction analysis and scoring breakdown

## Research Applications

This environment is valuable for:
- **Financial AI Research**: Training models to understand options pricing
- **Quantitative Analysis**: Developing AI-powered trading strategies
- **Risk Management**: Automated volatility prediction for portfolio management
- **Educational Purposes**: Teaching AI systems financial concepts

## Example Output

```
<think>
Let me analyze this option:
- Option Price: $5.50
- Stock Price: $100.00
- Strike Price: $105.00
- Time to Expiry: 0.25 years
- Risk-Free Rate: 5%

This is an out-of-the-money call option. Given the option price and other parameters, I need to work backwards to find the implied volatility that would justify this price using the Black-Scholes model...
</think>

The implied volatility will be: 25.3%
```

## Contributing

To contribute improvements:
1. Test changes with the provided example data
2. Ensure all scoring metrics work correctly
3. Verify WandB integration functions properly
4. Update documentation for any new features

## License

This environment follows the same license as the Atropos project.
