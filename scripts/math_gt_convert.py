import pandas as pd
import re

def extract_boxed_answer(solution_text):
    """从 MATH solution 中提取 \boxed{answer}"""
    if pd.isna(solution_text):
        return ""
    # 匹配 \boxed{content}，支持嵌套括号
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, str(solution_text))
    if matches:
        return matches[-1].strip()  # 取最后一个 \boxed{} 的内容
    return ""

# 读取 MATH 数据
df = pd.read_parquet('~/CoT-Data-verl/data/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet')

# 添加 prompt 列（如果需要在 problem 前加指令）
df['prompt'] = df['problem'].apply(lambda x: [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\nProblem: {x}"}
])

# 添加 data_source 列
df['data_source'] = 'math'

# 添加 reward_model 列（包含提取的 ground_truth）
df['reward_model'] = df['solution'].apply(lambda x: {
    'ground_truth': extract_boxed_answer(x),
    'raw_solution': x  # 保留原始解答用于调试
})

# 保存处理后的数据
df[['prompt', 'problem', 'solution', 'data_source', 'reward_model', 'level', 'type']].to_parquet(
    '/home/hjw/CoT-Data-verl/data/MATH/train_processed.parquet',
    index=False
)