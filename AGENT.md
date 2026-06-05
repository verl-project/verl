# Required Test

Only run this custom test (no other verl tests):

python -m pytest tests_custom/test_wandb_rank_runs_steps_since_best.py

Notes:
- Requires `WANDB_API_KEY` (loaded from `.env` via `load_wandb_api_key`).
- Targets W&B runs created on or after 2026-02-03T18:59:00Z (post early-stopping change).
