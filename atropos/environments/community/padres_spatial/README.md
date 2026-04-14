# Padres: Spatial RL Environment

## Video Demo
[Watch the demo video](https://youtu.be/uuSur31U1Pc)

## Environment Design & Motivation
Padres is a 3D spatial reasoning environment that challenges LLMs to understand and manipulate objects in a simulated 3D world. The environment uses PyBullet for physics simulation and integrates with LLMs for task generation and execution. The primary goal is to test and improve LLMs' spatial reasoning capabilities through interactive tasks that require understanding of relative positioning, object manipulation, and spatial relationships.

## Quickstart
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

3. Run the environment:
```bash
python spatial_env.py
```

4. View the visualization:
```bash
cd visualization
python3 -m http.server 8080
```
Then visit http://localhost:8080

## W&B Integration & Metrics
View the latest run [here](https://wandb.ai/carlosgarcia/spatial_rl_mvp/runs/1q2w3e4r5t6y7u8i9o0p)

Key metrics tracked:
- Task completion score (0-1)
- Final object distance
- Spatial condition satisfaction
- Action success rate
- LLM response time

## Additional Details
The environment implements a reward function that balances:
1. Proximity to target position
2. Spatial relationship constraints
3. Task completion verification

The current implementation focuses on basic spatial tasks but is designed to be extensible for more complex scenarios. The reward function is structured to prevent common reward hacking strategies by requiring both position accuracy and spatial relationship satisfaction.
