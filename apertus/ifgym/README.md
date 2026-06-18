# IFGym multi-turn instruction-following RL

Port of the [swiss-ai/if-gym](https://github.com/swiss-ai/if-gym/tree/final)
multi-turn training setup (`train/mt/run_mt.sh`) onto this fork of verl. It
trains a model on multi-turn conversations where each user turn carries a set of
instruction-following constraints, and each assistant turn is scored against its
turn's constraints.

## Why this differs from upstream IFGym

Upstream IFGym drives multi-turn rollout through verl's `BaseInteraction` /
`interaction_config_path` mechanism. **This fork does not have that mechanism**
(its multi-turn rollout is tool-driven). So the scripted user simulator is
implemented here as a dedicated agent loop, `ifgym_agent`
(`ifgym_agent_loop.py`), selected with `rollout.agent.default_agent_loop`
instead of `interaction_config_path`. The original `run_mt.sh` also monkey-patched
the installed verl package; those patches are integrated into the repo instead:

- Per-turn credit assignment (each turn's score at that turn's last token in
  `rm_scores`) lives in `verl/experimental/agent_loop/agent_loop.py`
  (`_postprocess`), gated on a non-empty `turn_scores` so other agent loops are
  unaffected.
- The custom advantage estimators and agent loop are registered by
  `apertus.ifgym.main_ifgym` (import side effects) and `ifgym_agent.yaml`.

## Contents

| File | Purpose |
|------|---------|
| `ifgym_instructions/` | Vendored constraint checker from if-gym (`instructions_registry`, `instructions`, `instructions_util`). |
| `ifgym_agent_loop.py` | `ifgym_agent` agent loop: scripts user turns, scores each assistant turn, sets `turn_scores` + `reward_score`. |
| `ifgym_advantage.py` | `ifgym_per_turn_grpo` advantage estimator. |
| `ifgym_per_turn_rloo.py` | `ifgym_per_turn_rloo` advantage estimator. |
| `ifgym_mt_reward.py` | `compute_score` reward fn for the reward-manager / offline-eval path (unused during agent-loop rollout). |
| `ifgym_agent.yaml` | Agent loop registry (`agent_loop_config_path`). |
| `ifgym_multiturn.yaml` | Hydra config (extends `ppo_trainer`). |
| `prepare_ifgym_mt_data.py` | Build `train.parquet` / `test.parquet` from the if-gym multi-turn dataset. |
| `main_ifgym.py` | Entry point; registers estimators + agent loop, then runs PPO. |
| `run_mt.sh` | Launch script. |
| `requirements.txt` | Extra pip deps for the constraint checker. |

## Usage

```bash
pip install -r apertus/ifgym/requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 1. Build data from the if-gym v2-multiturn-if dataset (<src>/<split>/data.jsonl):
python -m apertus.ifgym.prepare_ifgym_mt_data \
    --src /path/to/v2-multiturn-if --out /path/to/ifgym_mt_data

# 2. Train (run from the repo root):
MODEL_PATH=/path/to/model \
EXP_NAME=ifgym_mt_perturn_grpo \
DATA_DIR=/path/to/ifgym_mt_data \
ALGO=perturn_grpo \
bash apertus/ifgym/run_mt.sh
```

`ALGO` selects the advantage estimator / loss: `perturn_grpo` (default),
`trajectory_grpo`, `trajectory_rloo`, `perturn_rloo`, `trajectory_gspo`,
`perturn_gspo`.

## Data format

Each parquet row has `prompt` (first user turn) and
`extra_info.interaction_kwargs.turns_json` — a JSON list of turns, each
`{"prompt", "active_constraints": [{"constraint_id", "kwargs"}, ...]}`.
`constraint_id` must be a key in
`ifgym_instructions.instructions_registry.INSTRUCTION_DICT`; history-aware
constraints receive the previous assistant response automatically.
