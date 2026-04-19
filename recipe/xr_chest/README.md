# XR Chest GRPO — Quickstart

Qwen3-VL-4B GRPO on the XR-chest dataset, starting from an SFT
checkpoint. Full design doc:
[`verl/docs/xr_grpo.md`](../../docs/xr_grpo.md).

## Files

- `prepare_xr_data.py` — converts llama-factory JSONL splits to verl
  parquet (paths-in-images).
- `rave_decode.py` — `load_rave_image(path)`; byte-identical to
  llama-factory's `Qwen3VLRavePlugin._load_rave_image`.
- `xr_dataset.py` — `XRChestRLHFDataset(RLHFDataset)`; decodes RAVE
  paths -> PIL on the fly and wraps with `max_pixels=131072`.
- `reward_xr_report.py` — placeholder ROUGE-L reward. Swap body for
  RATE-backed reward in phase 2.
- `run_qwen3vl_4b_grpo.sh` — launcher for `b200-9` (8xB200).
- `tests/test_pixel_parity.py` — asserts byte-equal decode vs lf.

## One-time setup

Generate parquets on a node that has the RAVE cache locally (e.g. `b200-9`):

```bash
ssh b200-9
cd /mnt/nfs/home/michael/verl
/mnt/nfs/home/michael/llama-factory-voio/.venv/bin/python \
    recipe/xr_chest/prepare_xr_data.py --verify-paths
```

Output: `~/data/xr_chest_grpo/{train,valid_mini,test,train_mini}.parquet`.

## Verify pixel parity

```bash
ssh b200-9
cd /mnt/nfs/home/michael/verl
/mnt/nfs/home/michael/llama-factory-voio/.venv/bin/python \
    -m pytest recipe/xr_chest/tests/test_pixel_parity.py -v
```

Must show 10 passed before the first training run.

## Dry-run the config (no training)

```bash
ssh b200-9
cd /mnt/nfs/home/michael/verl
DRY_RUN=1 SMOKE=1 bash recipe/xr_chest/run_qwen3vl_4b_grpo.sh
```

Loads the config, builds the dataset and the model, skips training
(`total_epochs=0`). Good for catching schema errors fast.

## Smoke run

32-row `train_mini` × 1 epoch:

```bash
ssh b200-9
cd /mnt/nfs/home/michael/verl
SMOKE=1 bash recipe/xr_chest/run_qwen3vl_4b_grpo.sh
```

## Full run

```bash
ssh b200-9
cd /mnt/nfs/home/michael/verl
bash recipe/xr_chest/run_qwen3vl_4b_grpo.sh
```

wandb: project `pillar-rl`, experiment `xr-chest-qwen3vl-4b-grpo-<tag>`.

## Environment

All scripts assume
`/mnt/nfs/home/michael/llama-factory-voio/.venv/bin/python` — it has
verl, vLLM, transformers, qwen-vl-utils, peft, and the extra
`rouge-score` dep. If a fresh env is needed:

```bash
uv pip install rouge-score
```
