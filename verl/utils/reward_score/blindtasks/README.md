# blindtasks

Procedural visual-reasoning gym with 22 task variants across 10 families:
- each task is a deterministic `(seed) -> (image, question, ground_truth)` sample
- tasks are programmatic, meaning the same seed always produces the same instance

---

## Tasks

| Task | Description |
|---|---|
| `blindtasks.nested_squares` | Count concentric squares (2–6) |
| `blindtasks.nested_squares.hard` | Count random-offset nested squares (2–5) |
| `blindtasks.lines` | Count parallel lines with optional distractor (2-15) |
| `blindtasks.lines.hard` | Count intersections of 2 coloured polylines (0/1/2) |
| `blindtasks.circles` | Count concentric circles (2–8) |
| `blindtasks.circles.hard` | Two circles: do they overlap? (yes/no) |
| `blindtasks.circled_letter` | Letter inside one of 4 labelled circles: which label? (A/B/C/D) |
| `blindtasks.circled_letter.hard` | Long word with a red-ellipse'd char: which one? |
| `blindtasks.olympic` | Count non-overlapping circles (3–6) |
| `blindtasks.olympic.hard` | Count non-overlapping circles (5–9) |
| `blindtasks.olympic.interlocking` | Count overlapping circles (5–9, canonical 2-row Olympic-rings layout) |
| `blindtasks.grid` | Ask rows OR cols separately (2–12) |
| `blindtasks.grid.hard` | Joint rows × cols answer |
| `blindtasks.grid.word` | Each cell contains a different word; ask joint rows × cols |
| `blindtasks.subway` | 4 stations on edges + 1–2 coloured paths; count paths |
| `blindtasks.subway.hard` | 4 stations + 1–3 paths, larger image |
| `blindtasks.orientation` | Arrow at one of 4 cardinal directions +-10° jitter (N/E/S/W) |
| `blindtasks.orientation.hard` | Arrow at one of 8 directions +-5° jitter (N/NE/E/SE/S/SW/W/NW) |
| `blindtasks.spatial_relation` | 2 distinguishable shapes; 4-way relative position (above/below/left/right) |
| `blindtasks.spatial_relation.hard` | 3 shapes (target + reference + distractor); same 4-way answer |
| `blindtasks.size_compare` | 2 circles A/B; pick the larger (ratio in {1.5, 2.0, 3.0}) |
| `blindtasks.size_compare.hard` | 3 circles A/B/C; pick the largest (ratio in {1.3, 1.5, 2.0}) |

---

## Per-Instance Diversity

Every `task.sample(seed)` call also samples a `RenderConfig` (see [`base.py`](base.py))

The same seed produces the same image, but adjacent seeds vary along:

- Image size: model can't memorize one resolution (ex. 224 / 256 / 336 px)
- Line width: thin vs thick strokes
- Fore/Back-ground color swap: dark-on-light and light-on-dark
- Font: drawn from a per-instance font pool
- Layout jitter: position, rotation, spacing within task constraints

---

## Reward and Scoring

Single source of truth: [`evaluation/blindtasks_scorer.py::score`](../evaluation/blindtasks_scorer.py).

Both Apertus and Qwen reward functions ([`rewards/blindtasks_reward.py`](../rewards/blindtasks_reward.py)) parse the model's answer string, then call `score(task_name, ground_truth, prediction, *, shaped=False)`

**Two reward modes:**

| Mode | Range | When to use |
|---|---|---|
| Strict (default) | {0.0, 1.0} | Standard RL signal |
| Shaped (`extra_info.shaped=True`) | [0.0, 1.0] | Partial credit by distance from answer if 0/1 signal is too sparse |

**Ground truth schema**:

- Count / yes-no / label / char / relation tasks: `{"answer": <value>}`
- Grid joint tasks: `{"answer": ..., "rows": int, "cols": int}`
- Orientation tasks: `{"answer": ..., "n_directions": 4 or 8}`

---

## Seed Convention

Splits (train/val/test) are kept disjoint by seed range instead of file enumeration:

| Split | Seed range |
|---|---|
| train | `[0, 1_000_000)` |
| val | `[1_000_000, 2_000_000)` |
| test | `[2_000_000, inf)` |

The parser ([`data_prep/prepare_blindtasks_rl_parse.py`](../data_prep/prepare_blindtasks_rl_parse.py)) enforces this with `_check_seed_bounds` and ([`data_prep/audit_blindtasks_rl.py`](../data_prep/audit_blindtasks_rl.py)) re-verifies

---

## Balanced Sampling

Some tasks have natural class labels (`count`, `direction`, `label`). `BALANCED_CLASSES` in [`registry.py`](registry.py) lists the expected classes for each. `sample_balanced(task, n_per_class=N, expected_classes=...)` in [`utils.py`](utils.py) iterates seeds until it has N samples per class

Combinatorial-answer tasks (`grid.hard`, `grid.word`, `circled_letter.hard`) are not in `BALANCED_CLASSES` and fall back to seed enumeration

---

## Multi-Task Training

[`data_prep/prepare_blindtasks_rl_parse.py`](../data_prep/prepare_blindtasks_rl_parse.py) supports three `--mix` values:

| Mix | Behavior |
|---|---|
| `uniform` | Equal rows per task |
| `cognitive` | Equal budget per cognitive primitive (counting / yes-no / letter-id / direction / metric-compare / spatial) |
| `inverse_baseline` | Per-task weight inversely proportional to baseline accuracy |

---

## Adding a Task

1. Create `blindtasks/<name>.py` exporting `TASK_EASY` / `TASK_HARD` instances of a class implementing `.sample(seed)` and `.verify(inst, pred)`
2. Register in [`registry.py`](registry.py): add the module to `_TASK_MODULES`
3. Add a scorer dispatch entry in [`evaluation/blindtasks_scorer.py`](../evaluation/blindtasks_scorer.py)
4. If the task has natural classes, add to `BALANCED_CLASSES` in [`registry.py`](registry.py)
5. Add the task to `_COGNITIVE_PRIMITIVES` in the parser
6. Add tests

---

## RL Entrypoints

The BlindTasks SLURM scripts assume the container starts with this repository
mounted as the container working directory. They intentionally use paths
relative to `BASE="$(pwd)"`, captured at script start; configure dataset/result
locations with `OUT_DIR` or `OUTPUT_DIRECTORY` instead of changing the project
path in the scripts.

Generate parquet -> train:

```bash
# Qwen
sbatch slurm/prepare_blindtasks_rl_qwen.slurm --tasks blindtasks.olympic --n-per-task 5000
sbatch slurm/blindtasks_rl_qwen.slurm

# Apertus
sbatch slurm/prepare_blindtasks_rl.slurm --tasks blindtasks.olympic --n-per-task 5000
sbatch slurm/blindtasks_rl.slurm
```
