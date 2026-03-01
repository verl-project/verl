"""# 

### 谜题描述
Vova, the Ultimate Thule new shaman, wants to build a pipeline. As there are exactly n houses in Ultimate Thule, Vova wants the city to have exactly n pipes, each such pipe should be connected to the water supply. A pipe can be connected to the water supply if there's water flowing out of it. Initially Vova has only one pipe with flowing water. Besides, Vova has several splitters.

A splitter is a construction that consists of one input (it can be connected to a water pipe) and x output pipes. When a splitter is connected to a water pipe, water flows from each output pipe. You can assume that the output pipes are ordinary pipes. For example, you can connect water supply to such pipe if there's water flowing out from it. At most one splitter can be connected to any water pipe.

<image> The figure shows a 4-output splitter

Vova has one splitter of each kind: with 2, 3, 4, ..., k outputs. Help Vova use the minimum number of splitters to build the required pipeline or otherwise state that it's impossible.

Vova needs the pipeline to have exactly n pipes with flowing out water. Note that some of those pipes can be the output pipes of the splitters.

Input

The first line contains two space-separated integers n and k (1 ≤ n ≤ 1018, 2 ≤ k ≤ 109).

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

Print a single integer — the minimum number of splitters needed to build the pipeline. If it is impossible to build a pipeline with the given splitters, print -1.

Examples

Input

4 3


Output

2


Input

5 5


Output

1


Input

8 4


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, k = map(int, raw_input().split())
n -= 1
k -= 1

if n == 0:
    print 0 ###!!!
    exit()

def get_sum(l, r):
    return (l+r)*(r-l+1)/2

l = 1
r = k
ret = -1
while l<=r:
    mid = (l+r) >> 1
    sm = get_sum(mid, k)
    if n > sm:
        r = mid - 1
    elif n <= sm:
        l = mid + 1
        ret = k-mid+1

print ret
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Bpipelinebootcamp(Basebootcamp):
    def __init__(self, seed=None, solvable_prob=0.5):
        super().__init__()
        self.seed = seed
        self.solvable_prob = solvable_prob
        random.seed(seed)
    
    def case_generator(self):
        # 参数有效性验证
        if random.random() < self.solvable_prob:
            # 生成有解案例（扩展参数范围）
            k = random.randint(2, 10**6)  # 扩展k范围到百万级
            max_m = k - 1
            m = random.randint(1, max_m)
            
            # 更严谨的等差数列和计算
            sum_val = (m + (k-1)) * (k - m) // 2
            n = sum_val + 1
            
            # 强制有效性验证
            assert sum_val >= 0, f"Invalid sum_val: m={m}, k={k}"
            assert 1 <= n <= 10**18, "Generated n exceeds constraints"
        else:
            # 严格的无解案例生成
            k = random.randint(2, 10**9)
            sum_max = k * (k-1) // 2
            # 生成两种无解类型：n-1 > sum_max 或 无法找到合适的m
            n = random.choice([
                sum_max + random.randint(1, 1000),  # 类型1：明显超出范围
                random.randint(2, sum_max)           # 类型2：潜在不可达数值
            ]) + 1
            
        return {'n': n, 'k': k}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        return f"""
Vova需要建造一个有正好{n}个流水管的管道系统。初始时有1个流水管。现有分水器类型从2输出到{k}输出各一个。每个流水管最多连接一个分水器。

规则详解：
1. 每次操作会消耗一个分水器（例如使用3输出的分水器后，就不能再使用其他分水器）
2. 分水器连接后，原水管停止流水，同时产生x个新流水管（x为分水器输出数）
3. 最终需要总流水管数恰好等于{n}

请计算需要的最少分水器数量（如果不可能输出-1）。答案必须用[answer]标签包裹，如：[answer]-1[/answer]
"""

    @staticmethod
    def extract_output(output):
        # 增强模式匹配避免误提取
        matches = re.findall(
            r'\[\s*answer\s*\]\s*(-?\d+)\s*\[\s*/answer\s*\]', 
            output, 
            flags=re.IGNORECASE
        )
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        k = identity['k']
        
        # 精确的数学验证逻辑
        def is_valid(splitters_used):
            total = 1
            available = sorted(range(2, k+1), reverse=True)
            for s in sorted(splitters_used, reverse=True):
                if s not in available:
                    return False
                total = total - 1 + s
                available.remove(s)
            return total == n

        # 转换为二分验证逻辑
        required = n - 1
        max_possible = k*(k-1)//2
        
        if required < 0:
            return solution == -1
        if required == 0:
            return solution == 0
        if required > max_possible:
            return solution == -1
        
        # 二分查找验证
        low = 1
        high = k-1
        answer = -1
        while low <= high:
            mid = (low + high) // 2
            current_sum = (mid + k-1) * (k-1 - mid + 1) // 2
            if required <= current_sum:
                answer = k-1 - mid + 1
                low = mid + 1
            else:
                high = mid - 1
        
        valid_answer = answer if answer != -1 and (mid + k-1)*(k-1 - mid +1)//2 >= required else -1
        return solution == valid_answer
