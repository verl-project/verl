"""# 

### 谜题描述
Vasya has written some permutation p_1, p_2, …, p_n of integers from 1 to n, so for all 1 ≤ i ≤ n it is true that 1 ≤ p_i ≤ n and all p_1, p_2, …, p_n are different. After that he wrote n numbers next_1, next_2, …, next_n. The number next_i is equal to the minimal index i < j ≤ n, such that p_j > p_i. If there is no such j let's let's define as next_i = n + 1.

In the evening Vasya went home from school and due to rain, his notebook got wet. Now it is impossible to read some written numbers. Permutation and some values next_i are completely lost! If for some i the value next_i is lost, let's say that next_i = -1.

You are given numbers next_1, next_2, …, next_n (maybe some of them are equal to -1). Help Vasya to find such permutation p_1, p_2, …, p_n of integers from 1 to n, that he can write it to the notebook and all numbers next_i, which are not equal to -1, will be correct. 

Input

The first line contains one integer t — the number of test cases (1 ≤ t ≤ 100 000).

Next 2 ⋅ t lines contains the description of test cases,two lines for each. The first line contains one integer n — the length of the permutation, written by Vasya (1 ≤ n ≤ 500 000). The second line contains n integers next_1, next_2, …, next_n, separated by spaces (next_i = -1 or i < next_i ≤ n + 1).

It is guaranteed, that the sum of n in all test cases doesn't exceed 500 000.

In hacks you can only use one test case, so T = 1.

Output

Print T lines, in i-th of them answer to the i-th test case.

If there is no such permutations p_1, p_2, …, p_n of integers from 1 to n, that Vasya could write, print the only number -1.

In the other case print n different integers p_1, p_2, …, p_n, separated by spaces (1 ≤ p_i ≤ n). All defined values of next_i which are not equal to -1 should be computed correctly p_1, p_2, …, p_n using defenition given in the statement of the problem. If there exists more than one solution you can find any of them.

Example

Input


6
3
2 3 4
2
3 3
3
-1 -1 -1
3
3 4 -1
1
2
4
4 -1 4 5


Output


1 2 3
2 1
2 1 3
-1
1
3 2 1 4

Note

In the first test case for permutation p = [1, 2, 3] Vasya should write next = [2, 3, 4], because each number in permutation is less than next. It's easy to see, that it is the only satisfying permutation.

In the third test case, any permutation can be the answer because all numbers next_i are lost.

In the fourth test case, there is no satisfying permutation, so the answer is -1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
from atexit import register
from io import BytesIO

sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))

input = lambda: sys.stdin.readline().rstrip('\r\n')
t=int(input())

for u in xrange(t):
    n=int(input())
    a=list(map(int,input().split()))
    long=[]
    lens=0
    fucked=False
    for i in xrange(n):
        if a[i]==-1:
            if lens==0:
                a[i]=i+2
            else:
                a[i]=long[-1]
        else:
            if lens==0 or a[i]<long[-1]:
                long.append(a[i])
                lens+=1
            elif lens>0 and a[i]>long[-1]:
                fucked=True
        if lens>0:
            if i>=long[-1]-2:
                long.pop()
                lens-=1
    if fucked:
        print(-1)
    else:
        back={}
        for i in xrange(n+1):
            back[i+1]=[]
        for i in xrange(n):
            back[a[i]].append(i+1)
        perm=[0]*n
        q=[n+1]
        big=n
        while q!=[]:
            newq=[]
            for guy in q:
                for boi in back[guy]:
                    perm[boi-1]=big
                    big-=1
                newq+=back[guy]
            q=newq
        perm=[str(guy) for guy in perm]
        print(\" \".join(perm))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cpermutationrecoverybootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_n = params.get('min_n', 3)
        self.max_n = params.get('max_n', 6)
        self.prob_negative = params.get('prob_negative', 0.3)
        self.prob_invalid = params.get('prob_invalid', 0.3)
        
    @staticmethod
    def compute_next(p):
        n = len(p)
        next_list = [n + 1] * n
        stack = []
        for i in range(n):
            while stack and p[i] > p[stack[-1]]:
                j = stack.pop()
                next_list[j] = i + 1
            stack.append(i)
        return next_list
    
    @classmethod
    def _is_case_invalid(cls, n, next_list):
        long = []
        lens = 0
        fucked = False
        a = next_list.copy()
        for i in range(n):
            if a[i] == -1:
                a[i] = i+2 if lens == 0 else long[-1]
            current = a[i]
            
            if current != -1:
                if lens > 0 and current > long[-1]:
                    fucked = True
                elif lens == 0 or current < long[-1]:
                    long.append(current)
                    lens += 1
            
            if lens > 0 and i >= (long[-1]-1):
                long.pop()
                lens -= 1
        return fucked
    
    def case_generator(self):
        generate_invalid = random.random() < self.prob_invalid
        
        for _ in range(10):  # 多次尝试生成
            n = random.randint(self.min_n, self.max_n)
            p = list(range(1, n+1))
            random.shuffle(p)
            original_next = self.compute_next(p)
            case_next = [
                val if random.random() > self.prob_negative else -1
                for val in original_next
            ]
            
            if not generate_invalid:
                return {'n': n, 'next': case_next}
            else:
                modified_next = case_next.copy()
                valid_modifications = [i for i in range(n) if modified_next[i] != -1]
                random.shuffle(valid_modifications)
                
                for idx in valid_modifications[:5]:  # 最多尝试修改5个有效位置
                    i_1based = idx + 1
                    current_val = modified_next[idx]
                    
                    if random.random() < 0.5:
                        # 设置为比i小的值或无效大值
                        if current_val <= i_1based:
                            continue
                        new_val = random.randint(1, i_1based)
                    else:
                        new_val = random.randint(n+2, n+5)
                    
                    modified_next[idx] = new_val
                    if self._is_case_invalid(n, modified_next):
                        return {'n': n, 'next': modified_next}
                
                return {'n': n, 'next': case_next}
        
        return {'n': 3, 'next': [3,4,-1]}  # fallback

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        next_list = question_case['next']
        problem = f"""给定一个长度{n}的next数组，其中-1表示信息丢失。请恢复原始排列，使得非-1的next值满足：
- next_i是i之后第一个更大元素的最小下标(1-based)
- 若无则设为{n+1}

如果解不存在请输出-1。答案请放在[answer]标签内。

输入：
n={n}
next={next_list}

示例：当n=3且next=[2,3,4]时，排列应为[answer]1 2 3[/answer]"""
        return problem
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        if content == '-1':
            return -1
        try:
            return list(map(int, content.split()))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        next_list = identity['next']
        
        if solution == -1:
            return cls._is_case_invalid(n, next_list)
        
        if len(solution) != n or set(solution) != set(range(1, n+1)):
            return False
            
        computed_next = cls.compute_next(solution)
        for i in range(n):
            expected = next_list[i]
            if expected != -1 and computed_next[i] != expected:
                return False
        return True
