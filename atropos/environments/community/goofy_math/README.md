# GoofyMath 😂➗

A reinforcement learning environment that trains math models to be both *accurate* and *entertaining*.

## Demo Video

🎬 [Watch the 1-minute demo on YouTube]
( https://www.loom.com/share/8704f63e2d2e4b4db23eab673d7990a2?sid=3b78d63d-7cb0-44b2-a279-281c1be702b9 )

## Motivation & Design

Can a math tutor be both correct AND entertaining? We believe humor can dramatically improve learning outcomes.

The **GoofyMath** environment:
1. Takes standard GSM8K math problems
2. Uses a two-stage judging system:
   - First filters for mathematically correct solutions
   - Then ranks solutions by "goofiness" to reward entertaining explanations
3. Combines RLAIF (AI feedback) with objective correctness verification

The reward function: `score = correctness_score + (goofiness_bonus * 0.5)`
- Solutions MUST be correct (pass verification)
- Extra points (up to +0.5) for humor, sound effects, and creative explanations

## Quickstart

```bash
# Install requirements
pip install -r requirements.txt

# Run process mode to generate examples
export OPENAI_API_KEY=your_key_here
cd atropos
python environments/hack0/goofy_math_server.py process \
  --env.data_path_to_save_groups goofy_math_demo.jsonl \
  --env.total_steps 3
```

## WandB Run

📊 [View our WandB run](https://wandb.ai/goofymath/goofy_math/runs/z92gd2j4)

### Added Metrics
- **train/avg_goofiness_score**: Average goofiness score across solutions (0-1)
- **train/goofiness_histogram**: Distribution of goofiness scores
- **train/judgement_table**: Comparison table showing goofy vs standard solutions
- **train/percent_correct**: Accuracy rate (must maintain high performance)

## Technical Details

### Reward Hacking Prevention
- Goofiness is only rewarded AFTER correctness is verified
- Position bias eliminated by swapping solutions A/B in judgments
- Goofiness bonus capped at 50% of base reward

### Implementation Notes
- Uses RLAIF pattern with a novel twist: combining objective verification with subjective personality scoring
- Differentiator: most math tutoring systems optimize ONLY for correctness
- High-quality goofiness prompting designed to make explanations entertaining without sacrificing clarity

### Future Work
- Context-aware humor (different tones for different math concepts)
- Age-appropriate adjustments for younger vs. older students
- Personalized humor adaptation based on student feedback
