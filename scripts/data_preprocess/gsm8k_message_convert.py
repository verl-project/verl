import pandas as pd

# 读取原数据
train_df = pd.read_parquet('/home/hjw/CoT-Data-verl/data/gsm8k/train.parquet')
test_df = pd.read_parquet('/home/hjw/CoT-Data-verl/data/gsm8k/test.parquet')

def convert_to_messages(row):
    # 提取 prompt 内容（处理 numpy 数组的情况）
    prompt = row['prompt']
    if isinstance(prompt, (list, tuple)) or (hasattr(prompt, '__len__') and not isinstance(prompt, str)):
        prompt_content = prompt[0]['content']
    else:
        prompt_content = prompt
    
    # 直接返回 Python 列表，不要 json.dumps！
    return [
        {"role": "user", "content": prompt_content},
        {"role": "assistant", "content": row['extra_info']['answer']}
    ]

# 应用到 dataframe
train_df['messages'] = train_df.apply(convert_to_messages, axis=1)
test_df['messages'] = test_df.apply(convert_to_messages, axis=1)

# 可选：只保留需要的列
# train_df = train_df[['messages']]
# test_df = test_df[['messages']]

# 保存（Parquet 支持嵌套结构，无需转为 JSON 字符串）
train_df.to_parquet('/home/hjw/CoT-Data-verl/data/gsm8k/train_messages.parquet')
test_df.to_parquet('/home/hjw/CoT-Data-verl/data/gsm8k/test_messages.parquet')

print("转换完成！数据类型:", type(train_df['messages'].iloc[0]))