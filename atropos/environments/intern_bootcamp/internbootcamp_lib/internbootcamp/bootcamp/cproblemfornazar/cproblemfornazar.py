"""# 

### 谜题描述
Nazar, a student of the scientific lyceum of the Kingdom of Kremland, is known for his outstanding mathematical abilities. Today a math teacher gave him a very difficult task.

Consider two infinite sets of numbers. The first set consists of odd positive numbers (1, 3, 5, 7, …), and the second set consists of even positive numbers (2, 4, 6, 8, …). At the first stage, the teacher writes the first number on the endless blackboard from the first set, in the second stage — the first two numbers from the second set, on the third stage — the next four numbers from the first set, on the fourth — the next eight numbers from the second set and so on. In other words, at each stage, starting from the second, he writes out two times more numbers than at the previous one, and also changes the set from which these numbers are written out to another. 

The ten first written numbers: 1, 2, 4, 3, 5, 7, 9, 6, 8, 10. Let's number the numbers written, starting with one.

The task is to find the sum of numbers with numbers from l to r for given integers l and r. The answer may be big, so you need to find the remainder of the division by 1000000007 (10^9+7).

Nazar thought about this problem for a long time, but didn't come up with a solution. Help him solve this problem.

Input

The first line contains two integers l and r (1 ≤ l ≤ r ≤ 10^{18}) — the range in which you need to find the sum.

Output

Print a single integer — the answer modulo 1000000007 (10^9+7).

Examples

Input


1 3


Output


7

Input


5 14


Output


105

Input


88005553535 99999999999


Output


761141116

Note

In the first example, the answer is the sum of the first three numbers written out (1 + 2 + 4 = 7).

In the second example, the numbers with numbers from 5 to 14: 5, 7, 9, 6, 8, 10, 12, 14, 16, 18. Their sum is 105.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin, stdout
from collections import Counter, defaultdict
from itertools import permutations, combinations
raw_input = stdin.readline
pr = stdout.write


def in_num():
    return int(raw_input())


def in_arr():
    return map(int,raw_input().split())


def pr_num(n):
    stdout.write(str(n)+'\n')


def pr_arr(arr):
    pr(' '.join(map(str,arr))+'\n')

# fast read function for total integer input

def inp():
    # this function returns whole input of
    # space/line seperated integers
    # Use Ctrl+D to flush stdin.
    return map(int,stdin.read().split())

range = xrange # not for python 3.0+

# main code
mod=10**9+7
def fun(x):
    sm=0
    c1,c2=1,2
    p=1
    while x:
        temp=min(x,p)
        sm+=(c1+temp-1)*temp
        sm%=mod
        c1+=2*temp
        c1,c2=c2,c1
        x-=temp
        p*=2
    return sm
l,r=in_arr()
pr_num((fun(r)-fun(l-1))%mod)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Cproblemfornazarbootcamp(Basebootcamp):
    def __init__(self, max_lr=10**18):
        self.max_lr = max_lr
    
    def case_generator(self):
        # 生成边界案例的比例提升到50%
        if random.random() < 0.5:
            max_stage = self._find_max_stage()
            if max_stage == 0:
                return {'l': 1, 'r': 1}
            
            # 生成4种边界类型：阶段开始/中间/结束/跨阶段
            boundary_type = random.choice([1,2,3,4])
            
            if boundary_type == 1:  # 阶段边界
                stage = random.randint(1, max_stage)
                pos = (1 << stage) - 1  # 阶段结束位置
                delta = random.choice([-1, 0, 1])
                candidate = max(1, min(pos + delta, self.max_lr))
                return self._build_case_around(candidate)
            
            elif boundary_type == 2:  # 奇偶切换点
                stage = random.randint(1, max_stage-1)
                pos = (1 << stage) - 1
                return self._build_case_around(pos)
            
            elif boundary_type == 3:  # 大数边界
                return {'l': self.max_lr-10, 'r': self.max_lr}
            
            else:  # 跨多阶段案例
                stage1 = random.randint(1, max_stage-3)
                stage2 = stage1 + 3
                start = (1 << stage1) - 100
                end = (1 << stage2) + 100
                end = min(end, self.max_lr)
                start = max(1, start)
                r = random.randint(start, end)
                l = random.randint(start, r)
                return {'l': l, 'r': r}
        
        # 生成普通案例（覆盖各种数值范围）
        return self._generate_normal_case()
    
    def _find_max_stage(self):
        """动态计算最大可能的阶段数"""
        total, stage = 0, 0
        while True:
            add = 1 << stage
            if total + add > self.max_lr:
                return stage
            total += add
            stage += 1
    
    def _build_case_around(self, pos):
        """生成围绕特定位置的测试案例"""
        if random.choice([True, False]):
            l = max(1, pos - random.randint(0, 100))
            r = min(self.max_lr, pos + random.randint(0, 100))
        else:
            r = min(self.max_lr, pos + random.randint(0, 1000))
            l = max(1, r - random.randint(0, 1000))
        return {'l': l, 'r': r}
    
    def _generate_normal_case(self):
        """生成覆盖不同范围的普通案例"""
        range_type = random.choice([
            'tiny', 'small', 'medium', 'large', 'huge'
        ])
        
        ranges = {
            'tiny': (1, 100),
            'small': (100, 10**6),
            'medium': (10**6, 10**12),
            'large': (10**12, 10**15),
            'huge': (10**15, self.max_lr)
        }
        min_r, max_r = ranges[range_type]
        r = self._get_random_in_range(min_r, max_r)
        l = random.randint(1, r)
        return {'l': l, 'r': r}
    
    def _get_random_in_range(self, min_val, max_val):
        """高效生成指定范围的随机数"""
        span = max_val - min_val
        if span < 0:
            return min_val
        return min_val + random.randint(0, span)
    
    @staticmethod
    def prompt_func(question_case):
        l = question_case['l']
        r = question_case['r']
        return f"""作为数学天才，你需要解决以下数列求和问题：

数列生成规则：
1. 生成阶段按奇偶交替，第1阶段（奇数阶段）生成1个奇数，第2阶段（偶数阶段）生成2个偶数，第3阶段生成4个奇数，依此类推，每个阶段的数目是前一个阶段的两倍
2. 数列起始值：
   - 奇数阶段：从当前最小的未使用奇数开始
   - 偶数阶段：从当前最小的未使用偶数开始
3. 示例数列开始部分：1, 2,4, 3,5,7,9, 6,8,10,12,14,16,18,20,...

现在需要计算第{l}个到第{r}个数字的和（包含两端），结果对10^9+7取模。

请按以下步骤解答：
1. 确定每个数字所属的阶段
2. 计算各阶段对应数字的和
3. 对总和取模

将最终答案放在[answer]标签内，例如：[answer]123456789[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(
            r'\[answer\s*\]\s*(\d+)\s*\[\s*/answer\s*\]',  # 允许含空白符
            output, 
            re.IGNORECASE
        )
        if not matches:
            return None
        try:
            last_match = matches[-1]
            # 处理包含分隔符的情况（如1,234,567）
            cleaned = last_match.replace(',', '').replace(' ', '')
            return int(cleaned) % MOD
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 增加调试信息捕获
        try:
            l = identity['l']
            r = identity['r']
            sum_r = cls._calculate_sum(r)
            sum_l_1 = cls._calculate_sum(l-1)
            correct = (sum_r - sum_l_1) % MOD
            return solution % MOD == correct
        except Exception as e:
            print(f"Verification Error: {str(e)}")
            return False
    
    @staticmethod
    def _calculate_sum(x):
        sum_total = 0
        stage_size = 1  # 当前阶段元素个数
        is_odd = True    # 当前阶段奇偶性
        next_odd = 1     # 下一个奇数起始值
        next_even = 2    # 下一个偶数起始值
        remaining = x
        
        while remaining > 0:
            take = min(stage_size, remaining)
            
            if is_odd:
                start = next_odd
                end = start + 2*(take-1)
                segment_sum = take * (start + end) // 2
                next_odd = end + 2
            else:
                start = next_even
                end = start + 2*(take-1)
                segment_sum = take * (start + end) // 2
                next_even = end + 2
            
            sum_total = (sum_total + segment_sum) % MOD
            remaining -= take
            stage_size *= 2
            is_odd = not is_odd
        
        return sum_total
