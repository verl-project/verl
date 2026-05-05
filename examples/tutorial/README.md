# verl Tutorials & Launchers

Learning-oriented content that is not tied to a specific algorithm.

## Contents

| Subdir                     | What it is                                                                 |
|----------------------------|----------------------------------------------------------------------------|
| `agent_loop_get_started/`  | Notebook walkthrough of the agent-loop API (`agent_loop_tutorial.ipynb`). |
| `ray/`                     | Ray API crash-course notebook (`tutorial.ipynb`).                          |
| `slurm/`                   | SLURM job template for running Ray-on-SLURM (`ray_on_slurm.slurm`).        |
| `skypilot/`                | SkyPilot task specs for cloud / Kubernetes (`verl-ppo.yaml`, `verl-grpo.yaml`, `verl-multiturn-tools.yaml`). |

Use these as starting points; the runnable training scripts live under the
trainer directories (`grpo_trainer/`, `ppo_trainer/`, `sft/`, ...).
