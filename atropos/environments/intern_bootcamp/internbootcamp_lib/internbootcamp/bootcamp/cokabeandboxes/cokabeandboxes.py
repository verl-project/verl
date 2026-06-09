"""# 

### 谜题描述
Okabe and Super Hacker Daru are stacking and removing boxes. There are n boxes numbered from 1 to n. Initially there are no boxes on the stack.

Okabe, being a control freak, gives Daru 2n commands: n of which are to add a box to the top of the stack, and n of which are to remove a box from the top of the stack and throw it in the trash. Okabe wants Daru to throw away the boxes in the order from 1 to n. Of course, this means that it might be impossible for Daru to perform some of Okabe's remove commands, because the required box is not on the top of the stack.

That's why Daru can decide to wait until Okabe looks away and then reorder the boxes in the stack in any way he wants. He can do it at any point of time between Okabe's commands, but he can't add or remove boxes while he does it.

Tell Daru the minimum number of times he needs to reorder the boxes so that he can successfully complete all of Okabe's commands. It is guaranteed that every box is added before it is required to be removed.

Input

The first line of input contains the integer n (1 ≤ n ≤ 3·105) — the number of boxes.

Each of the next 2n lines of input starts with a string \"add\" or \"remove\". If the line starts with the \"add\", an integer x (1 ≤ x ≤ n) follows, indicating that Daru should add the box with number x to the top of the stack. 

It is guaranteed that exactly n lines contain \"add\" operations, all the boxes added are distinct, and n lines contain \"remove\" operations. It is also guaranteed that a box is always added before it is required to be removed.

Output

Print the minimum number of times Daru needs to reorder the boxes to successfully complete all of Okabe's commands.

Examples

Input

3
add 1
remove
add 2
add 3
remove
remove


Output

1


Input

7
add 3
add 2
add 1
remove
add 4
remove
remove
remove
add 6
add 7
add 5
remove
remove
remove


Output

2

Note

In the first sample, Daru should reorder the boxes after adding box 3 to the stack.

In the second sample, Daru should reorder the boxes after adding box 4 and box 7 to the stack.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
def main():
    n = int(stdin.readline())
    st = []
    ans = p = l = 0
    i = 1
    pu = st.append
    po = st.pop
    for line in stdin:
        if line[0] == 'r':
            if l > p and i != st[-1]:
                ans += 1
                p = l
            po()
            l -= 1
            if p > l:
                p = l
            i += 1
        else:
            t = int(line.split()[1], 10)
            pu(t)
            l += 1
    print ans
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cokabeandboxesbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10):
        """初始化参数，允许配置生成的盒子数量范围"""
        self.min_n = min_n
        self.max_n = max_n

    def case_generator(self):
        """生成一个合法的命令序列，包含交替的add和remove操作"""
        n = random.randint(self.min_n, self.max_n)
        add_order = list(range(1, n+1))
        random.shuffle(add_order)  # 随机生成添加顺序
        
        stack = []
        commands = []
        add_count = 0
        remove_count = 0
        expected_i = 1  # 当前期望移除的盒子编号
        
        while add_count < n or remove_count < n:
            # 优先添加未添加的盒子，但在适当条件下允许remove
            can_add = add_count < n
            can_remove = remove_count < add_count and expected_i <= n
            
            if can_add and (not can_remove or random.random() < 0.5):
                x = add_order[add_count]
                commands.append(f'add {x}')
                stack.append(x)
                add_count += 1
            else:
                commands.append('remove')
                # 即使栈顶不是期望的i也强制移除（模拟问题中的不可行操作）
                if stack and stack[-1] == expected_i:
                    stack.pop()
                expected_i += 1
                remove_count += 1
        
        # 计算正确答案
        expected_ans = self._compute_ans(n, commands)
        return {
            'n': n,
            'commands': commands,
            'expected_ans': expected_ans
        }

    @staticmethod
    def _compute_ans(n, commands):
        """根据输入命令计算最小重排次数（参考原题解逻辑）"""
        stack = []
        ans = p = stack_length = 0
        current_expected = 1
        for cmd in commands:
            if cmd.startswith('remove'):
                if stack_length > p and (stack and stack[-1] != current_expected):
                    ans += 1
                    p = stack_length
                if stack:
                    stack.pop()
                stack_length -= 1
                if p > stack_length:
                    p = stack_length
                current_expected += 1
            else:
                x = int(cmd.split()[1])
                stack.append(x)
                stack_length += 1
        return ans

    @staticmethod
    def prompt_func(question_case) -> str:
        """构造包含详细规则描述和示例的问题文本"""
        n = question_case['n']
        commands = '\n'.join(question_case['commands'])
        prompt = (
            "你是Daru，需要处理Okabe的指令来添加和移除盒子。当无法直接移除时，你可以重新排序栈。\n\n"
            "规则：\n"
            "1. 初始栈为空，共有n（n={}）个盒子，编号1~{}。\n"
            "2. 你会收到2n条命令：'add x'将x添加到栈顶，'remove'移除栈顶盒子。\n"
            "3. 需要确保最终移除顺序为1,2,...,n。若无法直接移除，可任选时机重排栈。\n"
            "4. 求最少需要重排的次数。\n\n"
            "输入格式：\n"
            "首行为n，随后2n行每行为命令。\n\n"
            "输入示例：\n{}\n\n"
            "请输出答案，并用[answer]和[/answer]标签包裹。例如：[answer]2[/answer]。"
        ).format(n, n, f"{n}\n{commands}")
        return prompt

    @staticmethod
    def extract_output(output):
        """从模型输出中提取最后一个[answer]标签内的答案"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """验证答案是否正确"""
        try:
            return int(solution) == identity['expected_ans']
        except (ValueError, TypeError):
            return False
