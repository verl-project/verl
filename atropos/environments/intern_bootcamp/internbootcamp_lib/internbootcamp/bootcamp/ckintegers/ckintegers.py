"""# 

### 谜题描述
You are given a permutation p_1, p_2, …, p_n.

In one move you can swap two adjacent values.

You want to perform a minimum number of moves, such that in the end there will exist a subsegment 1,2,…, k, in other words in the end there should be an integer i, 1 ≤ i ≤ n-k+1 such that p_i = 1, p_{i+1} = 2, …, p_{i+k-1}=k.

Let f(k) be the minimum number of moves that you need to make a subsegment with values 1,2,…,k appear in the permutation.

You need to find f(1), f(2), …, f(n).

Input

The first line of input contains one integer n (1 ≤ n ≤ 200 000): the number of elements in the permutation.

The next line of input contains n integers p_1, p_2, …, p_n: given permutation (1 ≤ p_i ≤ n).

Output

Print n integers, the minimum number of moves that you need to make a subsegment with values 1,2,…,k appear in the permutation, for k=1, 2, …, n.

Examples

Input


5
5 4 3 2 1


Output


0 1 3 6 10 


Input


3
1 2 3


Output


0 0 0 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
arr = map(int,raw_input().split(\" \"))
trees = [0]*(1+n)
dic = [0]*(n+1)
ans = [0]*n

def update(t,i,v):
        while i < len(t):
                t[i] += v
                i += lowbit(i)
def lowbit(x):
        return x&-x
def sum(t,i):
        ans = 0
        while i>0:
                ans += t[i]
                i -= lowbit(i)
        return ans

def getmid(arr,l1,flag):
        low,high = 1,n
        if l1%2 == 0 and  flag:
                midv = l1/2
        else:
                midv = l1/2+1
        while low <= high:
                mid = (low+high)/2
                ret = sum(arr,mid)
                if ret >= midv:
                        high = mid-1
                else:
                        low = mid+1
        return low

for i in range(n):
        dic[arr[i]]=i+1

for i in range(1,n+1):
        ans[i-1] += sum(trees,n)-sum(trees,dic[i])
        if i>=2:
                ans[i-1] += ans[i-2]
        update(trees,dic[i],1)
visited = [0]*(1+n)
mid = 0
last = 0
for i in range(1,n+1):
        update(visited,dic[i],1)
        mid = getmid(visited,i,dic[i]>mid)
        tt = sum(visited,dic[i])
        minus = min(tt-1,i-tt)
        tmp = abs(dic[i]-mid-(tt-sum(visited,mid)))- minus
        ans[i-1] += tmp+last
        last = tmp+last
print \" \".join(map(str,ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 使用1-based索引

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

class Ckintegersbootcamp(Basebootcamp):
    def __init__(self, n=5):
        self.n = n

    def case_generator(self):
        arr = list(range(1, self.n + 1))
        random.shuffle(arr)
        fk = self.compute_fk(arr)
        return {
            'permutation': arr,
            'answers': fk
        }

    @staticmethod
    def prompt_func(question_case):
        permutation = question_case['permutation']
        n = len(permutation)
        prompt = (
            f"给定一个长度为{n}的排列p = {permutation}，计算f(1)到f({n})的值。\n"
            f"f(k)表示将1到k排列成连续子序列所需的最少交换次数。例如，f(1)=0，因为单个元素无需交换。\n"
            f"请将答案以空格分隔的形式放在[answer]和[/answer]之间，例如：[answer]0 1 3 6 10[/answer]\n"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output)
        if not matches:
            return None
        answer_str = matches[-1].strip()
        try:
            answers = list(map(int, answer_str.split()))
            return answers
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_answers = identity['answers']
        if len(solution) != len(correct_answers):
            return False
        return all(a == b for a, b in zip(solution, correct_answers))

    def compute_fk(self, arr):
        n = len(arr)
        pos = [0] * (n + 1)  # pos[i]表示元素i的位置（1-based）
        for idx, num in enumerate(arr):
            pos[num] = idx + 1

        ft = FenwickTree(n)
        ans = [0] * n

        for i in range(1, n + 1):
            inv = ft.query(n) - ft.query(pos[i])
            ans[i - 1] = inv
            if i >= 2:
                ans[i - 1] += ans[i - 2]
            ft.update(pos[i], 1)

        visited = FenwickTree(n)
        last = 0

        for i in range(1, n + 1):
            visited.update(pos[i], 1)
            l, r = 1, n
            mid = 0
            target = (i + 1) // 2 if i % 2 == 1 else i // 2
            while l <= r:
                m = (l + r) // 2
                cnt = visited.query(m)
                if cnt >= target:
                    r = m - 1
                    mid = m
                else:
                    l = m + 1
            if l > r:
                mid = l
            tt = visited.query(mid)
            minus = min(tt - 1, i - tt)
            tmp = abs(pos[i] - mid - (tt - visited.query(mid))) - minus
            ans[i - 1] += tmp + last
            last = tmp + last

        return ans
