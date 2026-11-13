"""# 

### 谜题描述
Each month Blake gets the report containing main economic indicators of the company \"Blake Technologies\". There are n commodities produced by the company. For each of them there is exactly one integer in the final report, that denotes corresponding revenue. Before the report gets to Blake, it passes through the hands of m managers. Each of them may reorder the elements in some order. Namely, the i-th manager either sorts first ri numbers in non-descending or non-ascending order and then passes the report to the manager i + 1, or directly to Blake (if this manager has number i = m).

Employees of the \"Blake Technologies\" are preparing the report right now. You know the initial sequence ai of length n and the description of each manager, that is value ri and his favourite order. You are asked to speed up the process and determine how the final report will look like.

Input

The first line of the input contains two integers n and m (1 ≤ n, m ≤ 200 000) — the number of commodities in the report and the number of managers, respectively.

The second line contains n integers ai (|ai| ≤ 109) — the initial report before it gets to the first manager.

Then follow m lines with the descriptions of the operations managers are going to perform. The i-th of these lines contains two integers ti and ri (<image>, 1 ≤ ri ≤ n), meaning that the i-th manager sorts the first ri numbers either in the non-descending (if ti = 1) or non-ascending (if ti = 2) order.

Output

Print n integers — the final report, which will be passed to Blake by manager number m.

Examples

Input

3 1
1 2 3
2 2


Output

2 1 3 

Input

4 2
1 2 4 3
2 3
1 2


Output

2 4 1 3 

Note

In the first sample, the initial report looked like: 1 2 3. After the first manager the first two numbers were transposed: 2 1 3. The report got to Blake in this form.

In the second sample the original report was like this: 1 2 4 3. After the first manager the report changed to: 4 2 1 3. After the second manager the report changed to: 2 4 1 3. This report was handed over to Blake.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
if __name__ == \"__main__\":
    n, m = [int(x) for x in raw_input().split()]
    a = [int(x) for x in raw_input().split()]

    stack = []
    for i in range(m):
        t, r = [int(x) for x in raw_input().split()]
        while stack and stack[-1][1] <= r:
            stack.pop()
        stack.append((t, r))

    i, j, k = 0, stack[0][1] - 1, stack[0][1]
    a[:k] = sorted(a[:k])
    res = []
    for index, (t, r) in enumerate(stack):
        count = r - stack[index+1][1] if index+1 < len(stack) else r
        for _ in range(count):
            if t == 1:
                res.append(a[j])
                j -= 1
            else:
                res.append(a[i])
                i += 1

    res.reverse()
    for ele in a[k:]:
        res.append(ele)

    print ' '.join([str(x) for x in res])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def compute_solution(n, m, a, operations):
    """完整实现排序逻辑的解决方案"""
    stack = []
    for t, r in operations:
        # 维护单调递减栈，保留有效操作
        while stack and stack[-1][1] <= r:
            stack.pop()
        stack.append((t, r))

    if not stack:
        return a.copy()

    # 预处理初始排序
    k = stack[0][1]
    sorted_part = sorted(a[:k])  # 初始升序排序
    current_a = sorted_part + a[k:]

    i, j = 0, k-1  # 双指针初始化
    res = []
    
    # 逆序处理有效操作
    for idx in range(len(stack)):
        t, r = stack[idx]
        next_r = stack[idx+1][1] if idx+1 < len(stack) else 0
        cnt = r - next_r

        # 根据操作类型选择取数方向
        for _ in range(cnt):
            if t == 1:
                res.append(current_a[j])
                j -= 1
            else:
                res.append(current_a[i])
                i += 1

    # 组合最终结果
    res.reverse()
    return res + current_a[k:]

class Creportbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.max_n = params.get('max_n', 100)   # 扩大默认规模
        self.max_m = params.get('max_m', 50)    # 扩大默认规模
        self.min_val = params.get('min_val', -1e9)
        self.max_val = params.get('max_val', 1e9)

    def case_generator(self):
        """生成包含边界情况的测试用例"""
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        
        # 生成可包含极大值的测试数据
        a = [
            random.choice([self.min_val, self.max_val, random.randint(-100, 100)])
            for _ in range(n)
        ]
        
        operations = []
        prev_r = 0
        for _ in range(m):
            t = random.choice([1, 2])
            # 生成包含覆盖性操作的测试序列
            r = random.choice([
                n,                        # 全排列
                random.randint(1, n//2),  # 前部操作
                random.randint(n//2, n)   # 后部操作
            ])
            operations.append((t, r))
        
        return {
            'n': n,
            'm': m,
            'a': a.copy(),
            'operations': operations
        }

    @staticmethod
    def prompt_func(question_case):
        """增强问题描述的严谨性"""
        a_str = ' '.join(map(str, question_case['a']))
        operations_desc = []
        for idx, (t, r) in enumerate(question_case['operations'], 1):
            order_type = "非递减（从小到大）" if t == 1 else "非递增（从大到小）"
            operations_desc.append(f"{idx}. 对前 {r} 个元素进行{order_type}排序")
        
        return f"""## 经济报告排序系统 ##

您需要处理包含{question_case['n']}个商品的经济报告，初始序列为：
[{a_str}]

经过以下{len(question_case['operations'])}个排序操作：
{"".join(operations_desc)}

操作规则：
1. 每个操作都应用于前一个操作处理后的结果
2. 排序范围始终是当前序列的前r个元素
3. 最终输出应包含所有元素

请输出经所有操作处理后的完整序列，并将结果按以下格式包裹：
[answer]
（用空格分隔的数字序列）
[/answer]"""

    @staticmethod
    def extract_output(output):
        """强化抽取容错机制"""
        patterns = [
            r'\[answer\]\s*((?:-?\d+[\s\n]*)+)\[\/answer\]',  # 标准格式
            r'final[:：]?\s*((?:-?\d+[\s\n]*)+)',             # 兼容描述性前缀
            r'答案[:：]?\s*((?:-?\d+[\s\n]*)+)'               # 中文前缀
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    last_match = matches[-1].replace('\n', ' ').strip()
                    return list(map(int, re.split(r'\s+', last_match)))
                except (ValueError, AttributeError):
                    continue
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """增强验证鲁棒性"""
        if not solution or len(solution) != identity['n']:
            return False
        
        try:
            # 深拷贝避免污染原始数据
            test_case = {
                'n': identity['n'],
                'm': identity['m'],
                'a': list(identity['a']),
                'operations': list(identity['operations'])
            }
            correct = compute_solution(**test_case)
            return solution == correct
        except Exception as e:
            return False
