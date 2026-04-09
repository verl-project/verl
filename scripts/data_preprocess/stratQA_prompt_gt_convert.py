import pandas as pd
import numpy as np

def format_stratQA_prompt(row):
    
    prompt_content = f"""Answer the following true of false question step by step, using the given facts.
Your should answer the question only in \"true\" or \"false\".

Term & Description: {row['term']} / {row['description']}
Question: {row['question']}

Facts: {row['facts']}

Please explain your reasoning, then clearly state your final answer (true or false)."""

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_content}
    ]


# Read Original Data
input_path = '/data/open_datasets/StrategyQA/data/test-00000-of-00001-bae602f3ee37f4ca.parquet'
df = pd.read_parquet(input_path)

# 关键：创建包含 'ground_truth' 键的 reward_model 列
# answerKey 是正确选项的标签，如 "A", "B", "C", "D"
df['reward_model'] = df['answer'].apply(lambda x: {
    'ground_truth': x
})

# 创建 prompt 列
df['prompt'] = df.apply(format_stratQA_prompt, axis=1)

# 创建 data_source 列用于标识数据集
df['data_source'] = 'strategyQA'

# 保存处理后的数据（保留原始列便于调试）
output_columns = ['prompt', 'qid', 'term', 'description', 'question', 'answer', 'facts', 'data_source', 'reward_model']
df[output_columns].to_parquet(
    '/data/open_datasets/StrategyQA/data/test-processed.parquet',
    index=False
)

print(f"成功处理了 {len(df)} 条 StrategyQA 数据")
print("\n示例数据:")
print(f"Prompt: {df.iloc[0]['prompt']}")
print(f"Reward Model: {df.iloc[0]['reward_model']}")
print(f"Answer: {df.iloc[0]['answer']}")