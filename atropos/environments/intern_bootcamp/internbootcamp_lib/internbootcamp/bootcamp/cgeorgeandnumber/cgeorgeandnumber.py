"""# 

### 谜题描述
George is a cat, so he really likes to play. Most of all he likes to play with his array of positive integers b. During the game, George modifies the array by using special changes. Let's mark George's current array as b1, b2, ..., b|b| (record |b| denotes the current length of the array). Then one change is a sequence of actions: 

  * Choose two distinct indexes i and j (1 ≤ i, j ≤ |b|; i ≠ j), such that bi ≥ bj. 
  * Get number v = concat(bi, bj), where concat(x, y) is a number obtained by adding number y to the end of the decimal record of number x. For example, concat(500, 10) = 50010, concat(2, 2) = 22. 
  * Add number v to the end of the array. The length of the array will increase by one. 
  * Remove from the array numbers with indexes i and j. The length of the array will decrease by two, and elements of the array will become re-numbered from 1 to current length of the array. 



George played for a long time with his array b and received from array b an array consisting of exactly one number p. Now George wants to know: what is the maximum number of elements array b could contain originally? Help him find this number. Note that originally the array could contain only positive integers.

Input

The first line of the input contains a single integer p (1 ≤ p < 10100000). It is guaranteed that number p doesn't contain any leading zeroes.

Output

Print an integer — the maximum number of elements array b could contain originally.

Examples

Input

9555


Output

4

Input

10000000005


Output

2

Input

800101


Output

3

Input

45


Output

1

Input

1000000000000001223300003342220044555


Output

17

Input

19992000


Output

1

Input

310200


Output

2

Note

Let's consider the test examples: 

  * Originally array b can be equal to {5, 9, 5, 5}. The sequence of George's changes could have been: {5, 9, 5, 5} → {5, 5, 95} → {95, 55} → {9555}. 
  * Originally array b could be equal to {1000000000, 5}. Please note that the array b cannot contain zeros. 
  * Originally array b could be equal to {800, 10, 1}. 
  * Originally array b could be equal to {45}. It cannot be equal to {4, 5}, because George can get only array {54} from this array in one operation. 



Note that the numbers can be very large.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin

s, ans, tem = stdin.readline().strip(), 0, []
for i in range(len(s) - 1, -1, -1):
    tem.append(s[i])
    if s[i] != '0':
        ans += 1
        cur, tem = ''.join(tem[::-1]), []
        if (len(cur) > i and i) or (len(cur) == i and s[:i] < cur):
            break

print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cgeorgeandnumberbootcamp(Basebootcamp):
    def __init__(self, max_p_length=20):
        """
        初始化参数，限制生成p的最大长度以避免计算过载
        :param max_p_length: p的最大长度，默认20位
        """
        self.max_p_length = max_p_length
    
    def case_generator(self):
        """
        生成有效案例的核心逻辑：先构造分解路径，再反向生成p
        """
        # 随机生成初始元素数量（至少1个）
        original_n = random.randint(1, 10)  # 控制原始数组大小避免指数爆炸
        
        # 生成模拟合并路径
        elements = [str(random.randint(1, 999)) for _ in range(original_n)]
        while len(elements) > 1:
            # 随机选择两个不同元素
            i, j = random.sample(range(len(elements)), 2)
            a, b = elements[i], elements[j]
            if int(a) < int(b):
                a, b = b, a  # 保证a >= b
            merged = a + b
            # 移除原元素并添加新元素
            elements = [e for idx, e in enumerate(elements) if idx not in {i,j}] + [merged]
        p = elements[0]
        
        # 根据参考代码逆向计算正确答案n
        ans = 0
        tem = []
        for i in range(len(p)-1, -1, -1):
            tem.append(p[i])
            if p[i] != '0':
                ans += 1
                cur = ''.join(tem[::-1])
                tem = []
                if (len(cur) > i and i) or (len(cur) == i and p[:i] < cur):
                    break
        
        return {'p': p, 'n': ans}
    
    @staticmethod
    def prompt_func(question_case):
        p = question_case['p']
        return f"""# 谜题描述

George的数组游戏规则：

1. 初始数组包含若干正整数
2. 每次操作：
   - 选择两个不同元素bi ≥ bj
   - 拼接生成新数v = concat(bi, bj)（如concat(500, 10)=50010）
   - 将v加入数组并移除bi和bj
3. 最终数组只剩一个数{p}

请计算初始数组可能的最大元素数量，将答案用[answer]标签包裹。例如：答案为4则写[answer]4[/answer]"""

    @staticmethod
    def extract_output(output):
        # 匹配最后一个有效答案
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['n']
