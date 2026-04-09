import pandas as pd

def remove_asy_problems(input_file, output_file=None):
    """
    读取 parquet 文件，移除包含 [asy] 绘图指令的行
    
    参数:
        input_file: 输入的 parquet 文件路径
        output_file: 可选，保存过滤后数据的 parquet 文件路径
                    如果不提供，则只返回过滤后的 DataFrame
    
    返回:
        过滤后的 DataFrame
    """
    # 读取数据
    df = pd.read_parquet(input_file)
    
    # 显示原始数据统计
    print(f"原始数据行数: {len(df)}")
    
    # 检查 problem 列中是否包含 [asy] 或 [/asy] 标签
    # 使用正则表达式匹配 [asy] 或 [/asy]
    has_asy = df['problem'].str.contains(r'\[asy\]|\[/asy\]', 
                                          regex=True, 
                                          na=False)
    
    # 过滤掉包含 asy 标签的行（取反）
    df_filtered = df[~has_asy].copy()
    
    print(f"包含绘图指令的行数: {has_asy.sum()}")
    print(f"过滤后数据行数: {len(df_filtered)}")
    
    # 可选：保存结果
    if output_file:
        df_filtered.to_parquet(output_file)
        print(f"已保存到: {output_file}")
    
    return df_filtered


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    input_path = "/data/open_datasets/MATH-500/test-processed.parquet"  # 或 train.parquet, test.parquet 等
    output_path = "/data/open_datasets/MATH-500/test-processed-noasy.parquet"
    
    # 执行过滤
    df_clean = remove_asy_problems(input_path, output_path)
    
    # 验证：查看前几行
    print("\n过滤后的样本:")
    print(df_clean['problem'].iloc[0])