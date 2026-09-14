import random
import string

def generate_crypto_math(num_letters, num_puzzles, num_add=2):
    """
    生成 Crypto-math puzzle 对的函数，可指定加号数量 (num_add)。

    :param num_letters: int
        表示题目中所有字母的数量（即所用到的独立字母总数）。
    :param num_puzzles: int
        需要生成的 puzzle 对（question-answer 对）的数量。
    :param num_add: int
        题目里的加号数量，即加数个数。例如:
         - num_add=2 -> X + Y = Z
         - num_add=3 -> X + Y + W = Z
        ...
    :return: list
        返回一个列表，列表元素为 {"question": question_str, "answer": answer_str} 形式的字典。
    """
    # 若所需字母数超过可用数字(0-9)数量，直接返回空列表
    if num_letters > 10:
        return []

    results = []
    seen_questions = set()  # 用于去重

    while len(results) < num_puzzles:
        # 1. 随机生成 num_add 个加数
        addends = [random.randint(1, 999) for _ in range(num_add)]
        Z = sum(addends)  # 结果

        # 2. 提取所有数字
        union_digits = set()
        for val in addends:
            union_digits |= set(str(val))
        union_digits |= set(str(Z))

        # 如果数字种类不等于所需字母数，则跳过
        if len(union_digits) != num_letters:
            continue

        # 3. 将数字随机映射到字母
        #    - 随机选出 num_letters 个字母
        letters_pool = random.sample(string.ascii_uppercase, num_letters)
        union_digits_list = list(union_digits)
        random.shuffle(union_digits_list)

        digit_to_letter = {}
        for i, d in enumerate(union_digits_list):
            digit_to_letter[d] = letters_pool[i]

        # 4. 构造 question
        def num_to_letters(num):
            """将整数 num 转为对应的字母串"""
            return ''.join(digit_to_letter[d] for d in str(num))

        # 拼接加数部分 (e.g. "AB + CD + EF")
        addends_str = "+".join(num_to_letters(val) for val in addends)
        question_str = f"{addends_str}={num_to_letters(Z)}"

        # 去重检测
        if question_str in seen_questions:
            continue

        # 5. 构造 answer（字母->数字）
        letter_to_digit = {v: k for k, v in digit_to_letter.items()}
        sorted_letters = sorted(letter_to_digit.keys())  # 按字母排序
        answer_str = "[[" + ",".join(f"{letter}={letter_to_digit[letter]}" 
                                     for letter in sorted_letters) + "]]"

        puzzle_dict = {
            "puzzle": question_str,
            "target": answer_str,
            "num_add": num_add,
            "num_letters": num_letters
        }

        # 记录并保存
        seen_questions.add(question_str)
        results.append(puzzle_dict)

    return results


# # ============= 测试示例 =============
# if __name__ == "__main__":
#     # 生成 3 道 puzzle，每道 puzzle 中有 3 个加数(num_add=3)，并且题目中恰好有 4 个独立字母(num_letters=4)
#     puzzles = generate_crypto_math(num_letters=10, num_puzzles=10, num_add=20)
#     for idx, p in enumerate(puzzles, start=1):
#         print(f"Puzzle {idx}:")
#         question = p["puzzle"]
#         answer = p["target"]
#         print("  Question:", question)
#         print("  Answer:  ", answer)
#         answer = {item.split('=')[0]: int(item.split('=')[1]) for item in answer[2:-2].split(",")}
#         for letter, digit in answer.items():
#             question = question.replace(letter, str(digit))
#         print("  Replaced question:", question)
#         print(p)
#         try:
#             eval_res = eval(question.replace('=', '=='))
#         except:
#             print("  Eval failed!")
#             eval_res = False
#         if not eval_res:
#             print("  Check failed!")
#             break
#         print()




import itertools


def solve_crypto_math(question, max_iterations=2_000_000):
    """
    对给定的 Crypto-math puzzle (如 "A+BC=DB") 进行求解。
    如果有解，返回形如 "[[A=7,B=2,C=5,D=3]]" 的字符串；无解则返回 False。

    :param question: str, 形如 "A+BC=DB" 或 "ONE+ONE+TWO=FOUR"
    :param max_iterations: int, 限制最大尝试次数，防止极端情况下计算过长
    :return: str 或 bool
    """
    print(f"[Step 0] 准备开始求解题目: {question}")
    # 1. 解析题目
    if '=' not in question:
        print("[Step 1] 解析失败：题目中没有 '='，返回 False")
        return False

    left_side, right_side = question.split('=')
    left_side = left_side.strip()
    right_side = right_side.strip()

    print(f"[Step 1] 成功用 '=' 将题目分为：left_side='{left_side}'，right_side='{right_side}'")

    left_terms = left_side.split('+')  # 左侧有多个加数
    # 这里假设右侧只有一个整体表达式，如 "DB" 或 "FOUR"
    # 如果右侧也可能有多个加数，可以类似地再 split('+')
    right_term = right_side

    print(f"[Step 2] 左侧加数列表 = {left_terms}")
    print(f"[Step 2] 右侧表达式 = {right_term}")

    # 2. 收集所有出现的字母 & 首字母
    #    （用于确保多位数首字母不为 0）
    letters_set = set()
    leading_letters = set()

    def analyze_term(term):
        """
        term 为字符串（如 'ABC'），
        收集其中的所有字母，并找出首字母。
        """
        term = term.strip()
        for ch in term:
            if ch.isalpha():
                letters_set.add(ch)
        # 若是多位数或至少 2 位，则把首字母记录为 leading letter
        if len(term) > 1 and term[0].isalpha():
            leading_letters.add(term[0])

    # 分析左侧
    for t in left_terms:
        analyze_term(t)

    # 分析右侧
    analyze_term(right_term)

    letters = sorted(letters_set)  # 收集到的全部字母并排序
    print(f"[Step 3] 收集到的所有字母 = {letters}")
    print(f"[Step 3] 需要确保以下首字母 != 0: {sorted(list(leading_letters))}")

    # 如果字母数量大于10，无法映射到 0-9 的唯一数字
    if len(letters) > 10:
        print(f"[Step 3] 字母数量 = {len(letters)}，超过 10 个，不可能一一映射到数字 0-9。返回 False")
        return False

    # 3. 准备求解
    #    定义一个小函数，将某个映射应用到字符串上，返回对应的数字值
    def term_value(term, mapping):
        """
        将类似 'ABC' 根据 mapping 替换成数字，然后转换为 int。
        比如 'ABC' -> '123' -> int(123)。
        """
        return int(''.join(str(mapping[ch]) for ch in term if ch.isalpha()))

    # 4. 穷举：letters 的所有数字排列
    #    注意 leading_letters 不能为 0
    #    leading_letters 可能是集合，如 {'A', 'B', 'D'} => 对应的映射必须 > 0
    attempt_count = 0
    print(f"[Step 4] 准备开始尝试所有可能的字母->数字排列…(最多 {max_iterations} 次)")

    for perm in itertools.permutations(range(10), len(letters)):
        attempt_count += 1
        if attempt_count > max_iterations:
            # 超过给定的迭代上限，直接返回 False 以防无限拖延
            print(f"[Step 4] 已尝试超过 {max_iterations} 种排列，仍未找到解，返回 False")
            return False

        # 构造 letter -> digit 映射
        mapping = {}
        for i, letter in enumerate(letters):
            mapping[letter] = perm[i]

        # 如果有任何 leading letter 被映射成了 0，跳过
        if any(mapping[ld] == 0 for ld in leading_letters):
            continue

        # 为了演示，可以在每隔一定数量的 permutation 时打印提示
        # if attempt_count % 10000 == 0:
        print(f"[Progress] 已尝试 {attempt_count} 次排列，当前映射示例 = {mapping}")

        # 计算左侧总和
        try:
            left_sum = sum(term_value(t, mapping) for t in left_terms)
            right_val = term_value(right_term, mapping)
        except ValueError:
            # 若转换出错，跳过（一般不会发生，防御性写法）
            print(f"[Warning] 第 {attempt_count} 次尝试转换数值时出错，跳过。映射 = {mapping}")
            continue

        # 打印一下当前计算值（可以注释掉或更精简）
        # print(f"尝试第 {attempt_count} 次映射: {mapping}, 左边和= {left_sum}, 右边= {right_val}")

        if left_sum == right_val:
            # 找到可行解，返回
            print(f"[Success] 在第 {attempt_count} 次尝试中找到解: 左侧 {left_sum} = 右侧 {right_val}")
            letter_to_digit_strs = []
            for letter in sorted(mapping.keys()):
                letter_to_digit_strs.append(f"{letter}={mapping[letter]}")
            answer_str = "[[" + ",".join(letter_to_digit_strs) + "]]"
            print(f"[Step 5] 返回解: {answer_str}")
            return answer_str

    # 所有排列尝试完都没找到可行解，返回 False
    print(f"[Step 6] 穷举完所有排列也没能找到可行解，返回 False")
    return False


# # ==================== 测试示例 ====================
# if __name__ == "__main__":
#     # SYL+JAT+FOO+TJJ+KJF+OA+OLL+TSS+JK+FYR+JAS+TTT+OFY+AOO+JFR+FJS+YJT+JSL+LSA+OTF=LLKTT -> [[A=5,F=6,J=4,K=2,L=1,O=9,R=0,S=3,T=7,Y=8]]
#     # 381+457+699+744+246+95+911+733+42+680+453+777+968+599+460+643+847+431+135+976=11277
#     for puzzle in ["A+BC=DB"]: # , "A+BB=A", "ONE+ONE+TWO=FOUR", "SYL+JAT+FOO+TJJ+KJF+OA+OLL+TSS+JK+FYR+JAS+TTT+OFY+AOO+JFR+FJS+YJT+JSL+LSA+OTF=LLKTT"
#         sol = solve_crypto_math(puzzle, max_iterations=2_000_000)
#         print(puzzle, "->", sol)
#         if sol != False:
#             sol = {item.split('=')[0]: int(item.split('=')[1]) for item in sol[2:-2].split(",")}
#             for letter, digit in sol.items():
#                 puzzle = puzzle.replace(letter, str(digit))
#             try:
#                 eval_res = eval(puzzle.replace('=', '=='))
#             except:
#                 print("  Eval failed!")
#                 eval_res = False
#             if not eval_res:
#                 print("  Check failed!")
#                 break
#             print(puzzle)
#         print()