import pandas as pd
import json

def format_aqua_prompt(row):
    """将问题和选项格式化为对话形式"""
    question = row['question']
    options = row['options']  # 列表: ["A)21", "B)21.5", ...]
    
    # 格式化选项为字符串
    options_text = "\n".join(options)
    
    prompt_content = f"""Solve the following multiple choice question step by step.
Your final answer should be one of the options A, B, C, D, or E.

Question: {question}

Options:
{options_text}

Please explain your reasoning, then clearly state your final answer (A, B, C, D, or E)."""

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_content}
    ]

# 读取AQuA-RAT数据
df = pd.read_parquet('/data/open_datasets/aqua_rat/raw/test-00000-of-00001.parquet')

# 关键：创建包含 'ground_truth' 键的 reward_model 列
# main_eval 会执行 reward_data["ground_truth"] 提取 correct 值
df['reward_model'] = df['correct'].apply(lambda x: {
    'ground_truth': str(x).strip().upper(),  # 确保是 "A", "B", "C", "D", 或 "E"
})

# 创建prompt
df['prompt'] = df.apply(format_aqua_prompt, axis=1)

df['data_source'] = 'aqua_rat'

# 保存（只保留需要的列）
df[['prompt', 'question', 'options', 'rationale', 'correct', 'data_source', 'reward_model']].to_parquet(
    '/data/open_datasets/aqua_rat/processed/test-processed.parquet',
    index=False
)

print(f"处理了 {len(df)} 条数据")
print("示例 reward_model:", df.iloc[0]['reward_model'])