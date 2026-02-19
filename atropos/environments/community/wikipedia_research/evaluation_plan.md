# Wikipedia Article Evaluation System Plan

## Overview

This document outlines the plan to implement an evaluation system that uses OpenAI models to assess AI-generated Wikipedia articles against reference articles from the existing JSON data. This system will be integrated directly into the `score()` function of the `WikipediaArticleCreatorEnv` class.

## Core Components

### 1. Data Access Module

**Purpose**: Access reference Wikipedia articles from the existing JSON data.

**Implementation Details**:
- Utilize the existing JSON loading functionality in `wikipedia_article_creator.py`
- Access reference article content via the "plain_text" key already available in the JSON
- Match generated articles to reference articles by title

```python
def get_reference_article(self, topic: str) -> str:
    """
    Retrieves the reference article text for a given topic from the loaded JSON data.
    """
    # Access the article JSON that's already loaded by _load_topics()
    # Return the "plain_text" content for the matching article
    pass
```

### 2. Content Preparation Module

**Purpose**: Prepare AI-generated articles for evaluation against reference content.

**Implementation Details**:
- Split AI-generated article into numbered lines for granular assessment
- No need to normalize reference text - the OpenAI model can work with raw text

```python
def prepare_article_for_evaluation(self, article_content: str) -> Tuple[str, List[str]]:
    """
    Prepares an AI-generated article for evaluation.
    Returns both the numbered version (for the prompt) and the original lines (for scoring).
    """
    # Split article into lines
    # Add line numbers
    # Return both formatted text and original lines
    pass
```

### 3. Evaluation Engine

**Purpose**: Compare AI-generated article against the reference using OpenAI models.

**Implementation Details**:
- Create a focused prompt for the OpenAI model
- Generate YAML-formatted assessment of each line
- Categorize statements as CORRECT, INCORRECT, or UNKNOWN
- Include brief justification for each classification

```python
async def evaluate_article_accuracy(
    self,
    reference_content: str,
    generated_article: str
) -> Dict:
    """
    Evaluates the factual accuracy of a generated article against a reference.
    Returns structured accuracy data.
    """
    # Format the prompt with reference and generated content
    # Call the OpenAI API
    # Parse YAML response
    # Return structured accuracy data
    pass
```

### 4. Scoring Integration

**Purpose**: Calculate accuracy score and integrate with existing scoring mechanism.

**Implementation Details**:
- Convert evaluation results into a normalized score
- Integrate with existing article quality metrics
- Add accuracy metrics to wandb logging

```python
def calculate_accuracy_score(self, evaluation_data: Dict) -> float:
    """
    Calculates a normalized accuracy score from evaluation data.
    Returns a score between -1 and 1 for compatibility with existing scoring.
    """
    # Calculate percentage of CORRECT, INCORRECT, and UNKNOWN statements
    # Convert to a normalized score in the range [-1, 1]
    # More CORRECT = higher score, more INCORRECT = lower score
    pass
```

## Integration with Existing Environment

### Updating the `score()` Function

```python
async def score(self, rollout_group_data: List[ScoredDataGroup]) -> List[ScoredDataGroup]:
    """
    Enhanced scoring function that incorporates factual accuracy evaluation.
    """
    # For each terminal step with a final article:
    #   1. Get the corresponding topic
    #   2. Retrieve reference article from JSON data
    #   3. Evaluate article accuracy
    #   4. Calculate accuracy score
    #   5. Combine with existing quality metrics
    #   6. Update the score in the ScoredDataGroup

    # Add accuracy metrics to article_quality_metrics for wandb logging

    return rollout_group_data
```

## OpenAI Prompt Design

```
You are an expert fact-checker comparing an AI-generated article with a reference Wikipedia article.

# Classification Criteria
- CORRECT: The statement is accurate and verifiable in the reference article
- INCORRECT: The statement contradicts information in the reference article
- UNKNOWN: The reference doesn't mention this information or provides insufficient details to verify

# Output Format
You must produce valid YAML with this exact structure for each numbered line:
1:
  analysis: "Brief analysis of line 1"
  accuracy: "CORRECT|INCORRECT|UNKNOWN"
2:
  analysis: "Brief analysis of line 2"
  accuracy: "CORRECT|INCORRECT|UNKNOWN"
...

# REFERENCE ARTICLE:
{wiki_content}

# AI-GENERATED ARTICLE (NUMBERED LINES):
{numbered_ai_content}
```

## Implementation Steps

1. Implement the `get_reference_article()` function to extract reference text from JSON
2. Create the `prepare_article_for_evaluation()` function to number article lines
3. Develop the `evaluate_article_accuracy()` function with OpenAI integration
4. Implement the `calculate_accuracy_score()` function
5. Update the `score()` method to incorporate accuracy evaluation
6. Extend `_assess_article_quality()` to include the new accuracy metrics
7. Update wandb logging to include accuracy statistics

## Accuracy Scoring Formula

The accuracy score will be calculated as follows:

```python
# Example scoring formula
def calculate_accuracy_score(evaluation_data):
    total_lines = len(evaluation_data)
    correct_count = sum(1 for item in evaluation_data.values() if item['accuracy'] == 'CORRECT')
    incorrect_count = sum(1 for item in evaluation_data.values() if item['accuracy'] == 'INCORRECT')

    # Calculate percentages
    pct_correct = correct_count / total_lines if total_lines > 0 else 0
    pct_incorrect = incorrect_count / total_lines if total_lines > 0 else 0

    # Convert to score between -1 and 1
    # Formula: correct% * 2 - 1 with adjustment for incorrect%
    score = pct_correct * 2 - 1 - (pct_incorrect * 0.5)

    # Ensure score is within [-1, 1] range
    return max(-1, min(1, score))
```

## Updates to wandb Logging

The existing wandb logging will be extended to include:

```python
# Add to article_quality_metrics
accuracy_metrics = {
    "pct_correct": percentage_correct,
    "pct_incorrect": percentage_incorrect,
    "pct_unknown": percentage_unknown,
    "accuracy_score": accuracy_score
}
self.article_quality_metrics[-1].update(accuracy_metrics)

# Add to wandb metrics table
wandb_metrics["train/article_quality"] = table.add_column("factual_accuracy", [
    m["accuracy_score"] for m in self.article_quality_metrics
])
```

## Expected Benefits

1. More comprehensive evaluation of generated articles
2. Better feedback for the model on factual accuracy
3. Improved ability to detect hallucinations or fabricated information
4. Enhanced scoring mechanism that values factual correctness
