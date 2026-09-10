# Solitaire Winning Probability Environment

This environment is designed to analyze and predict winning probabilities in various solitaire-style games using both theoretical mathematical analysis and empirical simulation.

## Overview

The system combines two approaches to determine game winning probabilities:
1. **Theoretical Analysis**: Uses AI to derive mathematical formulas for exact probability calculations
2. **Empirical Simulation**: Runs Monte Carlo simulations to verify theoretical predictions

## Key Components

### GamePredictor Class
The core component that handles:
- AI-powered probability analysis
- Mathematical formula evaluation
- Game simulation
- Probability comparison between theoretical and empirical results

### Features

- **AI Analysis**: Uses LLM to analyze game mechanics and derive mathematical formulas
- **Formula Evaluation**: Supports complex mathematical expressions including:
  - Factorials
  - Combinations (C(n,r))
  - Permutations (P(n,r))
  - Standard mathematical operations
- **Simulation Engine**: Runs multiple game simulations to verify theoretical predictions
- **QA Dataset Generation**: Creates training data for AI models by generating question-answer pairs

### Reward Function

The environment implements a sophisticated reward function that evaluates the quality of probability predictions:

1. **Base Reward Calculation**:
   - Compares the predicted probability with the ground truth probability
   - Calculates the relative error: `1 - min(abs(gt - predicted) / gt, 2)`
   - Adds a small bonus of 0.2 for valid predictions
   - Clips the final reward between -1 and 1

2. **Length Penalty**:
   - Applies a length-based penalty for responses that exceed 50% of the maximum token length
   - No penalty for responses under the threshold
   - Linear scaling of penalty based on response length
   - Helps encourage concise and efficient solutions

3. **Validation Checks**:
   - Verifies proper formula formatting and syntax
   - Ensures responses contain valid mathematical expressions
   - Handles edge cases and invalid responses gracefully

4. **Quality Metrics**:
   - Tracks percentage of correct predictions
   - Monitors response lengths and quality
   - Provides feedback for model improvement

## Usage

```python
# Initialize the predictor
predictor = GamePredictor(openai_api_key, openai_api_base)

# Define games to analyze
games = {
    'game_name': game_function,
    # ... more games
}

# Get predictions for all games
results = await predictor.predict_games(games)

# Generate QA dataset
await predictor.generate_qa_csv(games, n_simulations, "output.csv")
```

## Output Format

The system provides comprehensive analysis for each game:
- AI's mathematical reasoning
- Derived probability formula
- Calculated theoretical probability
- Simulated empirical probability
- Comparison assessment between theory and simulation

## Supported Games

The environment includes several example games:
- Easy games (1-4)
- Card matching games (2-4 cards)
- Odd card game

## Requirements

- Python 3.x
- OpenAI API access
- Required packages:
  - openai
  - asteval
  - asyncio

## Purpose

This environment serves multiple purposes:
1. Educational: Demonstrates probability theory in practical game scenarios
2. Research: Provides a framework for analyzing game mechanics
3. AI Training: Generates datasets for training AI models in probability analysis
4. Verification: Validates theoretical probability calculations through simulation

## Contributing

New games can be added by implementing game functions that return a boolean indicating win/loss. The system will automatically analyze and provide probability predictions for any valid game implementation.
