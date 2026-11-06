import sys

from libs.wordladder.solving.puzzle import Puzzle
from libs.wordladder.solving.solver import Solver
from libs.wordladder.words.dictionary import Dictionary
from libs.wordladder.words.word import Word
from libs.wordladder.interactive import Interactive
from time import perf_counter
import json


def solver_main():
    if len(sys.argv) > 2:
        solve(sys.argv[1:])
    else:
        Interactive(sys.argv[1:]).run()

def solver_main2(param_data):
    if len(sys.argv) > 0:
        return solve2(sys.argv[1:],param_data)
    else:
        Interactive(sys.argv[1:]).run()

def solve(args):
    first = args[0]
    second = args[1]
    if len(second) != len(first):
        print('Start word \'%s\' and end word \'%s\' not the same length!' % (first, second))
        sys.exit(-1)
    start = perf_counter()
    dictionary = Dictionary(len(first))
    took = (perf_counter() - start) * 1000
    print('Took %.2fms to load dictionary' % took)
    start_word: Word = dictionary[first]
    if start_word is None:
        print('Start word \'%s\' not in dictionary' % first)
        sys.exit(-1)
    elif start_word.is_island:
        print('Start word \'%s\' is an island word' % first)
        sys.exit(-1)
    end_word: Word = dictionary[second]
    if end_word is None:
        print('End word \'%s\' not in dictionary' % second)
        sys.exit(-1)
    elif end_word.is_island:
        print('End word \'%s\' is an island word' % second)
        sys.exit(-1)

    puzzle: Puzzle = Puzzle(start_word, end_word)
    max_ladder_length: int = -1
    if len(args) > 2:
        try:
            max_ladder_length = int(args[2])
            if max_ladder_length < 1:
                raise ValueError
        except ValueError:
            print('Max ladder length arg must be an integer (greater than zero)')
            sys.exit(-1)
    else:
        start = perf_counter()
        min_ladder = puzzle.calculate_minimum_ladder_length()
        took = (perf_counter() - start) * 1000
        if min_ladder is None:
            print('Cannot solve \'%s\' to \'%s\'' % (first, second))
            sys.exit(-1)
        max_ladder_length = min_ladder
        print('Took %.2fms to determine minimum ladder length of %d' % (took, max_ladder_length))

    solver = Solver(puzzle)
    start = perf_counter()
    solutions = solver.solve(max_ladder_length)
    took = (perf_counter() - start) * 1000
    if len(solutions) == 0:
        print('Cannot solve \'%s\' to \'%s\' in ladder length %d (took %.2fms)' % (first, second, max_ladder_length, took))
        sys.exit(-1)
    slen = len(solutions)
    print('Took %.2fms to find %d solutions (explored %d solutions)' % (took, slen, solver.explored_count))
    solutions.sort()
    for i in range(slen):
        print('%d/%d %s' % (i + 1, slen, solutions[i]))

def solve2(args,param_data):
    solutions_cnt = 0
    current_ladder = 1
    solutions_rounds = 0

    first = param_data["start_word"]
    second = param_data["end_word"]

    if len(second) != len(first):
        print('Start word \'%s\' and end word \'%s\' not the same length!' % (first, second))
        sys.exit(-1)
    start = perf_counter()
    dictionary = Dictionary(len(first))
    took = (perf_counter() - start) * 1000
    print('Took %.2fms to load dictionary' % took)
    start_word: Word = dictionary[first]
    if start_word is None:
        print('Start word \'%s\' not in dictionary' % first)
        return save_to_file(param_data,None,code = 2)
        sys.exit(-1)
    elif start_word.is_island:
        print('Start word \'%s\' is an island word' % first)
        return save_to_file(param_data,None,code = 3)
        sys.exit(-1)
    end_word: Word = dictionary[second]
    if end_word is None:
        print('End word \'%s\' not in dictionary' % second)
        return save_to_file(param_data,None,code = 4)
        sys.exit(-1)
    elif end_word.is_island:
        print('End word \'%s\' is an island word' % second)
        return save_to_file(param_data,None,code = 5)
        sys.exit(-1)

    puzzle: Puzzle = Puzzle(start_word, end_word)

    max_ladder_length: int = -1
    while(solutions_cnt<param_data["solutions"] ):
        
        if len(args) > 2:
            try:
                max_ladder_length = current_ladder
                if max_ladder_length < 1:
                    raise ValueError
            except ValueError:
                print('Max ladder length arg must be an integer (greater than zero)')
                sys.exit(-1)
        else:
            start = perf_counter()
            min_ladder = puzzle.calculate_minimum_ladder_length()
            took = (perf_counter() - start) * 1000
            if min_ladder is None:
                print('Cannot solve \'%s\' to \'%s\'' % (first, second))
                return save_to_file(param_data,solutions="", code = 6)
                sys.exit(-1)
            max_ladder_length = min_ladder + solutions_rounds
            # print('Took %.2fms to determine minimum ladder length of %d' % (took, max_ladder_length))

        if( max_ladder_length > param_data["max_ladder"] or max_ladder_length > 20):
            print("max_ladder 超出限额")
            break

        solver = Solver(puzzle)
        start = perf_counter()
        # print("-- max_ladder_length (start solve):" + str(max_ladder_length))
        solutions = solver.solve(max_ladder_length)
        took = (perf_counter() - start) * 1000
        if len(solutions) == 0:
            print('Cannot solve \'%s\' to \'%s\' in ladder length %d (took %.2fms)' % (first, second, max_ladder_length, took))
            sys.exit(-1)
        slen = len(solutions)
        solutions_rounds += 1
        if slen < param_data["solutions"]:
            print('Took %.2fms to find %d solutions (explored %d solutions)' % (took, slen, solver.explored_count))
            print("not enough,pass to next ladder")
            current_ladder += 1
            max_ladder_length += 1
            continue
        else:
            solutions_cnt += slen
            print('Took %.2fms to find %d solutions (explored %d solutions)' % (took, slen, solver.explored_count))
            solutions.sort()
            for i in range(slen):
                print('%d/%d %s' % (i + 1, slen, solutions[i]))
            return save_to_file(param_data,solutions, code = 0)
            sys.exit(-1)
            
    return save_to_file(param_data,None,code = 1)


def save_to_file2(param_data, solutions , code):
    # 获取前n个元素
    # txt输出，已弃用
    is_nlp = param_data["is_nlp"]

    filepath = param_data["output_filepath"]
    n = param_data["solutions"]

    with open(filepath, 'a', encoding='utf-8') as file:
        if code == 0:
                if is_nlp :
                    title = "在一次Word Ladder中,当start word是" + param_data["start_word"] + "，end word是" + param_data["end_word"] + "时，至少可以得出以下" + str (n) + "条ladder路径："
                else:
                    title = param_data["start_word"] + " " + param_data["end_word"]
                file.write(title + '\n')
                for i in range(n):
                    file.write(str(solutions[i]) + '\n')
        elif code == 1:
                if is_nlp :
                    title = "在一次Word Ladder中,当start word是" + param_data["start_word"] + "，end word是" + param_data["end_word"] + "时，没能解出" + str (n) + "条ladder路径"
                else:
                    title = param_data["start_word"] + " " + param_data["end_word"]
                file.write(title + '\n')
        elif code == 2:
                if is_nlp :
                    title = "在一次Word Ladder中,当start word是" + param_data["start_word"] + "，end word是" + param_data["end_word"] + "时，无法求解，因为" + param_data["start_word"] + "不在字典中"
                else:
                    title = param_data["start_word"] + " " + param_data["end_word"]
                file.write(title + '\n')
        elif code == 3:
                if is_nlp :
                    title = "在一次Word Ladder中,当start word是" + param_data["start_word"] + "，end word是" + param_data["end_word"] + "时，无法求解，因为" + param_data["start_word"] + "是一个island word"
                else:
                    title = param_data["start_word"] + " " + param_data["end_word"]
                file.write(title + '\n')
        elif code == 4:
                if is_nlp :
                    title = "在一次Word Ladder中,当start word是" + param_data["start_word"] + "，end word是" + param_data["end_word"] + "时，无法求解，因为" + param_data["end_word"] + "不在字典中"
                else:
                    title = param_data["start_word"] + " " + param_data["end_word"]
                file.write(title + '\n')
        elif code == 5:
                if is_nlp :
                    title = "在一次Word Ladder中,当start word是" + param_data["start_word"] + "，end word是" + param_data["end_word"] + "时，无法求解，因为" + param_data["end_word"] + "是一个island word"
                else:
                    title = param_data["start_word"] + " " + param_data["end_word"]
                file.write(title + '\n')
        elif code == 6:
                if is_nlp :
                    title = "在一次Word Ladder中,当start word是" + param_data["start_word"] + "，end word是" + param_data["end_word"] + "时，无法求解，因为没有路径可以从start word到end word"
                else:
                    title = param_data["start_word"] + " " + param_data["end_word"]
                file.write(title + '\n')
        else:
            if is_nlp :
                title = "【出现未知异常，跳过此条训练数据1】"
            else:
                title = "error"
            file.write(title + '\n')

def save_to_file(param_data, solutions, code):
    is_nlp = param_data["is_nlp"]
    filepath = param_data["output_filepath"]
    start_word = param_data["start_word"]
    end_word = param_data["end_word"]
    n = param_data["solutions"]

    # 定义基础描述
    base_title = f"{start_word} {end_word}" if not is_nlp else f"在一次Word Ladder中,当start word是{start_word}，end word是{end_word}时，"

    # 处理不同的code
    if code == 0:
        describe = f"{base_title}至少可以得出以下{n}条ladder路径：" if is_nlp else base_title
        solutions_data = [str(solutions[i]) for i in range(n)] if solutions else []
        describe = parse_word_ladder(solutions_data,start_word=start_word,end_word=end_word,n=n)
    elif code == 1:
        describe = f"{base_title}没能解出{n}条ladder路径" if is_nlp else base_title
        solutions_data = "此题目无解"
    elif code == 2:
        describe = f"{base_title}无法求解，因为{start_word}不在字典中" if is_nlp else base_title
        solutions_data = "此题目无解"
    elif code == 3:
        describe = f"{base_title}无法求解，因为{start_word}是一个island word" if is_nlp else base_title
        solutions_data = "此题目无解"
    elif code == 4:
        describe = f"{base_title}无法求解，因为{end_word}不在字典中" if is_nlp else base_title
        solutions_data = "此题目无解"
    elif code == 5:
        describe = f"{base_title}无法求解，因为{end_word}是一个island word" if is_nlp else base_title
        solutions_data = "此题目无解"
    elif code == 6:
        describe = f"{base_title}无法求解，因为没有路径可以从start word到end word" if is_nlp else base_title
        solutions_data = "此题目无解"
    else:
        describe = "【出现未知异常，跳过此条训练数据2】" if is_nlp else "error"

        solutions_data = "此题目无解"



    return solutions_data,describe

    # 准备要写入的数据
    record = {
        "start_word": start_word,
        "end_word": end_word,
        "describe": describe,
        "solutions": solutions_data
    }

    

    # 写入文件
    with open(filepath, 'a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')

def parse_word_ladder(data, start_word, end_word, n):
    """
    解析 Word Ladder 数据为自然语言格式。

    参数：
        data (list of str): 包含每条路径的字符串列表。
        start_word (str): 初始单词。
        end_word (str): 结束单词。
        n (int): 数据中路径的数量。

    返回：
        str: 格式化后的自然语言文本。
    """
    nl_output = []
    nl_output.append(f"在一次Word Ladder中，从{start_word}到{end_word}至少可以得出以下{n}条ladder路径：")

    for i, path in enumerate(data[:n], start=1):
        words = path.strip('[]').split(',')
        words = [word.strip() for word in words]
        changes = []

        # 解析路径中的每步变化
        for j in range(1, len(words)):
            prev_word, curr_word = words[j - 1], words[j]
            for k, (prev_char, curr_char) in enumerate(zip(prev_word, curr_word)):
                if prev_char != curr_char:
                    changes.append(f"将第{k + 1}个字母由{prev_char}更换成{curr_char}")

        # 构建路径描述
        path_description = f"第{i}条：{words[0]}" + ''.join([f"，{change}，得到{words[j]}" for j, change in enumerate(changes, start=1)])
        nl_output.append(path_description + '；')

    return '\n'.join(nl_output)

# 完整的原solver
if __name__ == '__main__':
    solver_main()