import pandas as pd
import numpy as np

"""
This preprocessing script can be used for ARC-Challenge and ARC-Easy.
Change the paths and the data source to use it for different datasets.
"""

def format_arc_prompt(row):
    """将ARC-Challenge问题和选项格式化为对话形式"""
    question = row['question']
    choices = row['choices']  # 字典格式: {'text': array([...]), 'label': array([...])}
    
    # 提取选项文本和标签（处理numpy array或普通列表）
    if isinstance(choices, dict):
        choice_texts = choices['text']
        choice_labels = choices['label']
        # 如果是numpy array，转换为list
        if isinstance(choice_texts, np.ndarray):
            choice_texts = choice_texts.tolist()
        if isinstance(choice_labels, np.ndarray):
            choice_labels = choice_labels.tolist()
    else:
        raise TypeError("Column \"choices\" is not of type \"dict\". Please check the parquet file.")
    
    # 格式化选项为字符串 (例如: "A) xxxxxx")
    options_lines = []
    for label, text in zip(choice_labels, choice_texts):
        options_lines.append(f"{label}) {text}")
    options_text = "\n".join(options_lines)
    
    # 构建有效的选项标签列表（用于提示模型）
    valid_labels = ", ".join(choice_labels)
    
    prompt_content = f"""Answer the following multiple choice question step by step.
Your final answer should be exactly one of the option labels: {valid_labels}.

Question: {question}

Options:
{options_text}

Please explain your reasoning, then clearly state your final answer using the option label ({valid_labels})."""

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_content}
    ]

# Read Original Data
input_path = '/data/open_datasets/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet'
df = pd.read_parquet(input_path)

# 关键：创建包含 'ground_truth' 键的 reward_model 列
# answerKey 是正确选项的标签，如 "A", "B", "C", "D"
df['reward_model'] = df['answerKey'].apply(lambda x: {
    'ground_truth': str(x).strip().upper(),  # 确保是 "A", "B", "C", 或 "D"
})

# 创建 prompt 列
df['prompt'] = df.apply(format_arc_prompt, axis=1)

# 创建 data_source 列用于标识数据集
df['data_source'] = 'ai2_arc'

# 保存处理后的数据（保留原始列便于调试）
output_columns = ['prompt', 'question', 'choices', 'answerKey', 'id', 'data_source', 'reward_model']
df[output_columns].to_parquet(
    '/data/open_datasets/ai2_arc/ARC-Challenge/test-processed.parquet',
    index=False
)

print(f"成功处理了 {len(df)} 条 ARC-Challenge 数据")
print("\n示例数据:")
print(f"Prompt: {df.iloc[0]['prompt']}")
print(f"Reward Model: {df.iloc[0]['reward_model']}")
print(f"Answer Key: {df.iloc[0]['answerKey']}")