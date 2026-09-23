import random
from libs.wordladder.solver_main import solver_main2
from unittest.mock import patch
from libs.wordladder.file_io import *

"""
原代码地址:https://github.com/marrow16/pyWordLadder/tree/master
solver_main.py solver_main()是原程序main.py main()
现在使用main.py() solver_main2() solve2()
"""

def main():
    """
    弃用,现在用solver,此方法为无参数测试方法,所有需要参数全部来自config.json,打印结果
    原程序需要sys.args输入
    str     start_word  修改为传参
    str     end_word    修改为传参
    int     ladder      修改为自动递增直到答案足够/上限
    y/n     打印所有     修改为打印+后处理
    y/n     重新运行     预输入"n"截断
    """
    param_data = {
        "start_word" : "",
        "end_word" : "",
        "solutions" : 0,
        "max_ladder" : 0,
        "output_filepath" : "",
        "is_nlp": True
        }
    
    if is_param_random():
        if is_word_random():
            param_data["start_word"] , param_data["end_word"] = get_random_puzzle()
        else:
            words = get_random_words(get_random_mode_word_length())
            param_data["start_word"] = words[0]
            param_data["end_word"] = words[1]
    else:
        param_data["start_word"] = get_specify_mode_start_word()
        param_data["end_word"] = get_specify_mode_end_word()
    param_data["solutions"] = get_solutions_count()
    param_data["max_ladder"] = get_ladders_count()
    param_data["output_filepath"] = get_output_filepath()
    param_data["is_nlp"] = is_nl_describe()
    print(param_data)

    # 预输入数据 目的截断所有后续需要y/n的sys.args输入
    inputs = iter(['n'] * 100)

    # 模拟 input()
    with patch("builtins.input", lambda _: next(inputs)):
        solver_main2(param_data)

def solver(start_word:str,end_word:str):
    """
    :param  start_word:str      第一个单词
    :param  end_word:str        最后一个单词
    :return solutions:list[str] n条可能的解 (config.json控制)
    """
    param_data = {
        "start_word" : start_word,
        "end_word" : end_word,
        "solutions" : 0,
        "max_ladder" : 0,
        "output_filepath" : "",
        "is_nlp": True
        }
    
    # if is_param_random():
    #     if is_word_random():
    #         param_data["start_word"] , param_data["end_word"] = get_random_puzzle()
    #     else:
    #         words = get_random_words(get_random_mode_word_length())
    #         param_data["start_word"] = words[0]
    #         param_data["end_word"] = words[1]
    # else:
    #     param_data["start_word"] = get_specify_mode_start_word()
    #     param_data["end_word"] = get_specify_mode_end_word()
    param_data["solutions"] = get_solutions_count()
    param_data["max_ladder"] = get_ladders_count()
    param_data["output_filepath"] = get_output_filepath()
    param_data["is_nlp"] = is_nl_describe()
    print(param_data)

    # 预输入数据 目的截断所有后续需要y/n的sys.args输入
    inputs = iter(['n'] * 100)

    # 模拟 input()
    with patch("builtins.input", lambda _: next(inputs)):
        return solver_main2(param_data)

def get_random_puzzle():
    random_number = random.randint(2, 15)
    words = get_random_words(random_number)
    start_word = words[0]
    end_word = words[1]
    
    return start_word,end_word

def get_random_words(size):
    #:param n:int 单词长度
    #:return :["word1","word2"]
    if not (2 <= size <= 15):
        raise ValueError("参数n必须在2到15之间")
    file_path = f'libs/wordladder/words/resources/dictionary-{size}-letter-words.txt'

    try:
        # 读取文件中的所有单词
        with open(file_path, 'r', encoding='utf-8') as file:
            words = [line.strip() for line in file.readlines()]
        
        if len(words) < 2:
            raise ValueError("文件中的单词数量不足两个")
        
        # 随机选择两个不同的单词
        selected_words = random.sample(words, 2)
        
        return selected_words
    
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return []
    except Exception as e:
        print(f"发生错误: {e}")
        return []

if __name__ == '__main__':
    main()