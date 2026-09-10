"""# 

### 谜题描述
The Greatest Secret Ever consists of n words, indexed by positive integers from 1 to n. The secret needs dividing between k Keepers (let's index them by positive integers from 1 to k), the i-th Keeper gets a non-empty set of words with numbers from the set Ui = (ui, 1, ui, 2, ..., ui, |Ui|). Here and below we'll presuppose that the set elements are written in the increasing order.

We'll say that the secret is safe if the following conditions are hold:

  * for any two indexes i, j (1 ≤ i < j ≤ k) the intersection of sets Ui and Uj is an empty set; 
  * the union of sets U1, U2, ..., Uk is set (1, 2, ..., n); 
  * in each set Ui, its elements ui, 1, ui, 2, ..., ui, |Ui| do not form an arithmetic progression (in particular, |Ui| ≥ 3 should hold). 



Let us remind you that the elements of set (u1, u2, ..., us) form an arithmetic progression if there is such number d, that for all i (1 ≤ i < s) fulfills ui + d = ui + 1. For example, the elements of sets (5), (1, 10) and (1, 5, 9) form arithmetic progressions and the elements of sets (1, 2, 4) and (3, 6, 8) don't.

Your task is to find any partition of the set of words into subsets U1, U2, ..., Uk so that the secret is safe. Otherwise indicate that there's no such partition.

Input

The input consists of a single line which contains two integers n and k (2 ≤ k ≤ n ≤ 106) — the number of words in the secret and the number of the Keepers. The numbers are separated by a single space.

Output

If there is no way to keep the secret safe, print a single integer \"-1\" (without the quotes). Otherwise, print n integers, the i-th of them representing the number of the Keeper who's got the i-th word of the secret.

If there are multiple solutions, print any of them.

Examples

Input

11 3


Output

3 1 2 1 1 2 3 2 2 3 1


Input

5 2


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, k = map(int, raw_input().split())
def solve(n, k):
    if k * 3 > n: return None
    r = n % k
    top = [None] * k
    belong = [None] * n
    j = 0
    for i in xrange(k):
        h = n / k
        if r > 0:
            h += 1
            r -= 1
        while h:
            h -= 1
            belong[j] = i
            j += 1
        top[i] = j - 1
    # print_belong(n, k, belong)
    if k == 2:
        belong[0] = 1
        belong[n-1] = 0
    else:
        for i in xrange(k):
            belong[top[i]] = (i + 2) % k
    # print_belong(n, k, belong)
    return belong

def print_belong(n, k, belong):
    stk = [[] for i in xrange(k)]
    for i in xrange(n):
        stk[belong[i]].append(i)
    print stk


ans = solve(n, k)
if ans:
    print ' '.join(str(x+1) for x in ans)
else:
    print -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Csecretbootcamp(Basebootcamp):
    def __init__(self, max_k=20, max_n=200, unsolvable_ratio=0.3):
        """
        参数说明:
            max_k: 最大守护者数 (默认20)
            max_n: 最大单词数 (默认200)
            unsolvable_ratio: 强制生成不可解案例的概率 (0.3)
        """
        self.max_k = max_k
        self.max_n = max_n
        self.unsolvable_ratio = unsolvable_ratio

    def case_generator(self):
        # 控制不可解案例生成
        if random.random() < self.unsolvable_ratio:
            k = random.randint(2, self.max_k//2)
            n = random.randint(k, 3*k -1)  # 确保3k >n
        else:
            k = random.randint(2, self.max_k)
            min_n = 3*k
            n = random.randint(min_n, min(min_n + 50, self.max_n))  # 生成有效解附近的值
        
        return {'n': n, 'k': k}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        return f"""你需要将{n}个单词分配给{k}位守护者，满足：
1. 每个守护者获得≥3个单词
2. 所有守护者的单词集合互不相交
3. 所有单词必须分配
4. 每个守护者的单词编号不能形成等差数列

输入：n={n} k={k}

若存在解决方案，输出{n}个1~{k}的整数（空格分隔）；否则输出-1。答案置于[answer][/answer]中。"""

    @staticmethod
    def extract_output(output):
        # 增强格式容错性
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.DOTALL|re.IGNORECASE)
        if not matches:
            return None
        solution = matches[-1].strip().replace(',', ' ').replace('\n', ' ')
        solution = ' '.join(solution.split())
        
        if solution == '-1':
            return '-1'
        
        if re.fullmatch(r'(-1)|((\d+ )*\d+)', solution) and len(solution.split()) == solution.count(' ') + 1:
            return solution
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n, k = identity['n'], identity['k']
        minimal_condition = 3*k <= n  # 存在解的必要条件
        
        # 类型判断
        if solution == '-1':
            return not minimal_condition  # 当且仅当无解时返回-1正确
        
        if not minimal_condition:
            return False  # 当必须无解时却返回解
        
        # 格式验证
        try:
            parts = list(map(int, solution.split()))
            if len(parts) != n or any(p < 1 or p > k for p in parts):
                return False
        except:
            return False
        
        # 构建分配字典
        allocation = [[] for _ in range(k)]
        for idx, keeper in enumerate(parts, 1):  # 单词编号从1开始
            allocation[keeper-1].append(idx)
        
        # 完整性检查
        all_words = {w for group in allocation for w in group}
        if all_words != set(range(1, n+1)):
            return False
        
        # 集合属性验证
        for group in allocation:
            if len(group) < 3:
                return False
            
            sorted_group = sorted(group)
            d = sorted_group[1] - sorted_group[0]
            # 快速检测等差数列
            for i in range(2, len(sorted_group)):
                if sorted_group[i] - sorted_group[i-1] != d:
                    break
            else:
                # 全部差值相同则失败
                return False
        
        return True
