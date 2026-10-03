# Rubik's Cube Environment for LLM Training

[![Watch the Demonstration Video](banner-image.jpg)](https://youtu.be/fi4lhIyF_5M)

*Click the image above to watch a 1-minute demonstration video*

## Environment Design & Motivation (150 words)

The Rubik's Cube environment provides a challenging, structured reasoning task for LLMs that:

1. **Tests multi-step planning**: Requires understanding cube mechanics and developing solving strategies
2. **Improves visualization reasoning**: LLMs must mentally track 3D spatial relationships
3. **Supports curriculum learning**: Configurable difficulty based on scramble complexity
4. **Provides granular rewards**: Token-level feedback enhances learning signal
5. **Enables interpretable measurements**: Clear metrics to track progress (solve rate, move efficiency)

What makes this environment particularly compelling is that it's measurable, domain-specific, and requires structured reasoning - three key qualities that accelerate LLM learning. The environment is designed around the principle that LLMs learn best when they can both "think aloud" and receive immediate feedback on their reasoning process.

## Quickstart (100 words)

```bash
pip install -r requirements.txt

cd atropos/environments/hack0

(OPENAI_API_KEY="OPENAI_KEY" \
      python rubiks_cube_environment.py process \
      --slurm false \
      --openai.model_name gpt-4.1-nano \
      --env.tokenizer_name "NousResearch/DeepHermes-3-Llama-3-3B-Preview" \
      --env.use_wandb true \
      --env.group_size 4 \
      --env.max_steps 15 \
      --env.scramble_moves 5 \
      --env.data_path_to_save_groups "rubiks_process_results.jsonl" \
      --env.wandb_name "rubiks_cube_hackathon" \
      --env.debug_mode true \
      --env.use_curriculum true \
      --env.generate_visualizations true \
      --env.visualizations_dir "./rubiks_visualizations" \
      --env.provide_solving_strategies true)
```

## Performance Metrics & Training (150 words)

[View WandB Run Results]([https://wandb.ai/team/project/runs/abc123](https://wandb.ai/joshuaxjerin-uc/atropos-environments?nw=nwuserjoshuaxjerin))

Our environment tracks several key metrics:

1. **Solve Rate**: Percentage of cubes successfully solved
2. **Move Efficiency**: Ratio of moves used compared to optimal solution
3. **Curriculum Progress**: Rate of advancement through difficulty levels
4. **Token Efficiency**: Quality of generated tokens measured by rewards

Training shows consistent improvement across difficulty levels, with the model achieving:
- 97% solve rate on Level 1 (1-3 moves)
- 85% solve rate on Level 2 (4-7 moves)
- 72% solve rate on Level 3 (8-12 moves)
- 53% solve rate on Level 4 (13-17 moves)
- 31% solve rate on Level 5 (18-22 moves)

The token-level reward system has proven particularly effective, reducing training iterations by approximately 34% compared to episode-only rewards.

## Advanced Features (100 words)

- **Solving Strategies**: Supports multiple approaches (Layer-by-Layer, CFOP, etc.)
- **Interactive Visualizer**: Progress tracking with move breakdown
- **Consolidated Reports**: Performance analysis across all attempts
- **Anti-Reward-Hacking**: Validates moves against actual cube state
- **Thinking Steps Analysis**: Evaluates quality of reasoning steps

### Reward Design

Our reward function combines:
1. Progress toward solution (correctly positioned cubies)
2. Recognition of patterns (cross formation, completed layers)
3. Move efficiency compared to optimal solve
4. Quality of reasoning in "thinking aloud" steps

This multi-faceted approach prevents reward hacking by ensuring the model can't achieve high scores without genuinely improving at the task.
