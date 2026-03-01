# 🎓 ExamCraft: Adaptive LLM Teacher Training Environment

**Hackathon Submission**: Train language models to become adaptive teachers through reinforcement learning.

## 🌟 Overview

ExamCraft trains LLMs to be better teachers by generating adaptive questions, providing explanations, and creating personalized lesson plans. The environment rewards effective teaching strategies and penalizes poor ones.

### Key Features
- **Adaptive Question Generation**: Targets student weak areas automatically
- **Real-time Difficulty Adjustment**: Matches challenge level to student ability
- **Comprehensive Teaching Actions**: Questions, explanations, and lesson plans
- **Sophisticated Reward System**: Multi-factor scoring for teaching effectiveness
- **Student Learning Simulation**: Realistic proficiency progression

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Environment

1. **Start Atropos trajectory API**:
```bash
run-api
```

2. **Run environment in serve mode**:
```bash
python examcraft_server.py serve --slurm false
```

3. **Generate inference-only rollouts**:
```bash
python examcraft_server.py process --env.data_path_to_save_groups demo_output.jsonl
```

### Training with Atropos
```bash
# Generate SFT data
atropos-sft-gen examcraft_sft.jsonl --tokenizer NousResearch/Hermes-3-Llama-3.1-8B

# Generate DPO data
atropos-dpo-gen examcraft_dpo.jsonl --tokenizer NousResearch/Hermes-3-Llama-3.1-8B
```

## 🎯 Environment Design

### Teaching Actions
- **QUESTION**: Generate adaptive multiple-choice questions
- **EXPLANATION**: Provide detailed concept explanations
- **LESSON_PLAN**: Create personalized study plans

### Reward Components
1. **Correctness Reward**: Base reward for student getting questions right
2. **Targeting Bonus**: Extra points for focusing on weak topics
3. **Difficulty Appropriateness**: Rewards for matching difficulty to ability
4. **Quality Bonus**: Higher scores for detailed, well-structured content
5. **Learning Impact**: Bonuses for explanations that boost understanding

### Student Simulation
- Probabilistic responses based on topic proficiency
- Dynamic learning from good teaching
- Realistic difficulty sensitivity
- Session momentum effects

## 📊 Example Metrics

The environment tracks:
- Student accuracy improvement across topics
- Teaching effectiveness scores
- Adaptive difficulty selection
- Content quality metrics
- Learning progression over time

## 🏆 Why This Matters

Adaptive AI tutoring can revolutionize education by:
- Personalizing learning experiences at scale
- Identifying knowledge gaps automatically
- Providing instant, detailed feedback
- Making quality education globally accessible

## 🔧 Configuration

### Student Profile Format
```json
{
  "student_id": "student001",
  "target_grade": "11th grade",
  "learning_goal": "Master linear algebra basics",
  "current_avg_score": 73,
  "topics": [
    {"name": "vectors", "proficiency": 0.65},
    {"name": "matrices", "proficiency": 0.50}
  ],
  "preferred_learning_style": "visual"
}
```

### Environment Parameters
- `max_questions_per_episode`: 8
- `student_learning_rate`: 0.03
- `enable_lesson_plans`: true

## 📈 Results Preview

After training, teachers learn to:
- Prioritize topics where students struggle most
- Adapt question difficulty based on recent performance
- Generate detailed explanations that boost understanding
- Create comprehensive lesson plans targeting weak areas

Built for the **Nous Research RL Environments Hackathon** 🚀
