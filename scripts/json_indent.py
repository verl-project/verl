import json
import pandas as pd


def format_json(input_file, output_file=None, indent=2):
    """
    将未分行的JSON文件格式化为可读形式
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径，若为None则覆盖原文件
        indent: 缩进空格数，默认4
    """
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame.from_dict(data, orient='columns')

        print(f"原始数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"前3行预览:\n{df.head(3)}")
        
        # 如果未指定输出文件，默认覆盖原文件
        if output_file is None:
            output_file = input_file
        
        # 写入格式化后的JSON
        df.to_json(
            output_file,
            orient='records',
            force_ascii=False,  # 保留中文不转义
            indent=indent       # 美化格式，2空格缩进
        )
            
        print(f"\n✅ 转换完成！已保存至: {output_file}")
        print(f"总行数: {len(df)}")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
    except FileNotFoundError:
        print(f"❌ 文件未找到: {input_file}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 直接修改下面的文件路径
    input_path = "/home/hjw/CoT-Data-verl/evals/Qwen2.5-Coder-7B--math-500--eval--0325-141549/generated/responses_labeled.json"  # 你的输入文件
    output_path = "/home/hjw/CoT-Data-verl/evals/Qwen2.5-Coder-7B--math-500--eval--0325-141549/generated/responses_labeled_formatted.json"  # 输出文件，若为None则覆盖原文件
    
    format_json(input_path, output_path)