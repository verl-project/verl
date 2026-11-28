# Humor Generation Environment

## Overview
A reinforcement learning environment for training language models to generate humor in the style of specific comedians and formats. The environment uses a multi-dimensional scoring rubric to evaluate joke quality across relevance, style consistency, creativity, humor effectiveness, virality, and cognitive coherence.

## Features
- **Multi-Comedian Training**: Supports various comedian styles (Norm Macdonald, John Mulaney, Hasan Minhaj, Dave Chappelle, Ali Wong, Chris Rock)
- **Format Diversity**: Trains on different humor formats (haiku, one-liner, q/a over SMS)
- **Comprehensive Scoring**: 6-dimensional evaluation rubric for joke quality assessment
- **Dataset Generation**: Automated dataset creation using GPT-4o-mini
- **WandB Integration**: Comprehensive experiment tracking and visualization

## Environment Structure
- `humor_env.py`: Main environment implementation with scoring logic
- `generate_humor_dataset.py`: Script for creating training datasets
- `humor_dataset.jsonl`: Pre-generated dataset with comedian/format combinations

## Scoring Rubric
The environment evaluates generated jokes across six dimensions (0-3 points each):
1. **Relevance to Format** (0-2): How well the joke fits the specified format
2. **Style Consistency** (0-2): Adherence to the target comedian's style
3. **Creativity** (0-3): Originality and inventiveness of the humor
4. **Humor Effectiveness** (0-3): How funny and engaging the joke is
5. **Virality** (0-3): Potential for widespread appeal and sharing
6. **Cognitive Coherence** (0-3): Logical structure and comprehensibility

## Usage

### Running the Environment
```bash
python environments/community/humor_generation/humor_env.py serve
```

### Generating New Datasets
```bash
cd environments/community/humor_generation/
python generate_humor_dataset.py
```

## Configuration
- **Model**: GPT-4o-mini for both generation and evaluation
- **Group Size**: 2 completions per prompt
- **Max Tokens**: 2048 for joke generation, 512 for scoring
- **Evaluation**: LLM-based scoring using detailed rubric prompts

## Requirements
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Standard Atropos dependencies
- WandB account for experiment tracking

## Dataset Format
Each record contains:
- `comedian`: Target comedian style
- `format`: Humor format (haiku, one-liner, q/a over SMS)
- `question`: Prompt asking for model recommendations and example jokes
- `response`: GPT-4o-mini generated response with explanations and examples

## Training Applications
- **Style Transfer**: Learning to mimic specific comedian voices
- **Format Adaptation**: Generating humor in constrained formats
- **Quality Assessment**: Training models to evaluate humor effectiveness
- **Creative Writing**: Developing AI systems for entertainment content creation
