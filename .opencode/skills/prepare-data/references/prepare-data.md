# 数据集参考

## 官方文档

Prepare Data for Post-Training: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html

## 支持的数据集

| 数据集 | 脚本路径 | 数据源 | 说明 |
| ---- | ---- | ---- | ---- |
| GSM8K | examples/data_preprocess/gsm8k.py | openai/gsm8k | 数学推理（小学应用题） |
| GSM8K multiturn sft | examples/data_preprocess/gsm8k_multiturn_sft.py | openai/gsm8k | 数学推理（小学应用题） |
| MATH | examples/data_preprocess/math_dataset.py | DigitalLearningGmbH/MATH-lighteval | 数学竞赛题 |
| HelloSwag | examples/data_preprocess/hellaswag.py | Rowan/hellaswag | 常识推理 |
| Full_hh_rlhf | examples/data_preprocess/full_hh_rlhf.py | Dahoas/full-hh-rlhf | 人类反馈强化学习 |
| Geo3K | examples/data_preprocess/geo3k.py | hiyouga/geometry3k | 几何问答 |
| Pokemon | examples/data_preprocess/pokemon.py | llamafactory/pokemon-gpt4o-captions | 图像描述 |
| DAPO-Math-17k | examples/data_preprocess/dapo_multiturn_w_tool.py | BytedTsinghua-SIA/DAPO-Math-17k | 多轮数学推理，支持code_interpreter |
| AIME-2024 | examples/data_preprocess/aime2024_multiturn_w_tool.py | BytedTsinghua-SIA/AIME-2024 | AIME 2024数学竞赛题（重复32次） |

## 脚本调用

```bash
python3 /verl/examples/data_preprocess/<dataset>.py --local_save_dir <output_dir>
```

full_hh_rlhf 需要额外参数：
```bash
python3 ... full_hh_rlhf.py --local_save_dir <dir> --split <sft|rm|rl>
```