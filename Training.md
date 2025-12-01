# 训练脚本

`/home/4dtc/rio/verl/examples/grpo_trainer/run_qwen2.5-14_pairwise.sh`

## 需要修改的参数

```bash
data.train_files=/home/4dtc/rio/verl/data/ui_design_parquet/train.parquet \
data.val_files=/home/4dtc/rio/verl/data/ui_design_parquet/val.parquet \
```

还需要修改/home/4dtc/rio/verl/verl/workers/reward_manager/pairwise_comparison.py里面的reward model的api url