"""# 

### 谜题描述
Young Teodor enjoys drawing. His favourite hobby is drawing segments with integer borders inside his huge [1;m] segment. One day Teodor noticed that picture he just drawn has one interesting feature: there doesn't exist an integer point, that belongs each of segments in the picture. Having discovered this fact, Teodor decided to share it with Sasha.

Sasha knows that Teodor likes to show off so he never trusts him. Teodor wants to prove that he can be trusted sometimes, so he decided to convince Sasha that there is no such integer point in his picture, which belongs to each segment. However Teodor is lazy person and neither wills to tell Sasha all coordinates of segments' ends nor wills to tell him their amount, so he suggested Sasha to ask him series of questions 'Given the integer point xi, how many segments in Fedya's picture contain that point?', promising to tell correct answers for this questions.

Both boys are very busy studying and don't have much time, so they ask you to find out how many questions can Sasha ask Teodor, that having only answers on his questions, Sasha can't be sure that Teodor isn't lying to him. Note that Sasha doesn't know amount of segments in Teodor's picture. Sure, Sasha is smart person and never asks about same point twice.

Input

First line of input contains two integer numbers: n and m (1 ≤ n, m ≤ 100 000) — amount of segments of Teodor's picture and maximal coordinate of point that Sasha can ask about.

ith of next n lines contains two integer numbers li and ri (1 ≤ li ≤ ri ≤ m) — left and right ends of ith segment in the picture. Note that that left and right ends of segment can be the same point.

It is guaranteed that there is no integer point, that belongs to all segments.

Output

Single line of output should contain one integer number k – size of largest set (xi, cnt(xi)) where all xi are different, 1 ≤ xi ≤ m, and cnt(xi) is amount of segments, containing point with coordinate xi, such that one can't be sure that there doesn't exist point, belonging to all of segments in initial picture, if he knows only this set(and doesn't know n).

Examples

Input

2 4
1 2
3 4


Output

4


Input

4 6
1 3
2 3
4 6
5 6


Output

5

Note

First example shows situation where Sasha can never be sure that Teodor isn't lying to him, because even if one knows cnt(xi) for each point in segment [1;4], he can't distinguish this case from situation Teodor has drawn whole [1;4] segment.

In second example Sasha can ask about 5 points e.g. 1, 2, 3, 5, 6, still not being sure if Teodor haven't lied to him. But once he knows information about all points in [1;6] segment, Sasha can be sure that Teodor haven't lied to him.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from itertools import repeat
from bisect import bisect
def main():
    n, m = map(int, stdin.readline().split())
    dat = map(int, stdin.read().split(), repeat(10, 2 * n))
    s = [0] * (m + 1)
    for i, x in enumerate(dat):
        if i & 1:
            s[x] -= 1
        else:
            s[x-1] += 1
    for i in xrange(m):
        s[i+1] += s[i]
    s.pop()
    l = 0
    b = [-1] * (m + 1)
    L = [0] * m
    R = [0] * m
    for i in xrange(m):
        p = bisect(b, s[i], 0, l)
        b[p] = s[i]
        if p == l:
            l += 1
        L[i] = p
    s.reverse()
    l = 0
    b = [-1] * (m + 1)
    for i in xrange(m):
        p = bisect(b, s[i], 0, l)
        b[p] = s[i]
        if p == l:
            l += 1
        R[i] = p
    R.reverse()
    for i in xrange(m):
        L[i] += R[i] + 1
    print max(L)
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import bisect
import random
from bootcamp import Basebootcamp

class Cteodorisnotaliarbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=5, min_m=5, max_m=20):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
    
    def case_generator(self):
        for _ in range(100):  # 增加生成尝试次数
            # 生成随机参数
            n = random.randint(self.min_n, self.max_n)
            m = random.randint(max(n, self.min_m), self.max_m)
            
            # 确保存在分割点
            if m < 2:
                continue
                
            # 随机选择分割点
            x = random.randint(1, m-1)
            k = random.randint(1, n)
            
            # 生成左组和右组线段
            left_segments = []
            for _ in range(k):
                l = random.randint(1, x)
                r = random.randint(l, x)
                left_segments.append((l, r))
                
            right_segments = []
            for _ in range(n-k):
                l = random.randint(x+1, m)
                r = random.randint(l, m)
                right_segments.append((l, r))
            
            # 合并并验证有效性
            segments = left_segments + right_segments
            if self._validate_segments(segments, m):
                return {'n': n, 'm': m, 'segments': segments}
        raise ValueError("Failed to generate valid case")

    def _validate_segments(self, segments, m):
        # 验证所有线段边界
        for l, r in segments:
            if not (1 <= l <= r <= m):
                return False
        
        # 验证线段交集为空
        max_li = max(l for l, _ in segments)
        min_ri = min(r for _, r in segments)
        return max_li > min_ri

    @staticmethod
    def prompt_func(question_case) -> str:
        return (
            f"Cteodorisnotaliar has {question_case['n']} segments in [1,{question_case['m']}]. "
            "Find the maximum number of distinct queries Sasha can ask while remaining uncertain "
            "about Cteodorisnotaliar's honesty.\n\nSegments:\n" +
            "\n".join(f"{l} {r}" for l, r in question_case['segments']) +
            "\n\nReturn the answer within [answer]...[/answer]."
        )

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](\d+)\[\/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 精确复现参考代码逻辑
        n, m = identity['n'], identity['m']
        segments = identity['segments']
        
        # 计算覆盖次数（差分数组）
        s = [0]*(m+1)
        for l, r in segments:
            s[l-1] += 1
            if r <= m:
                s[r] -= 1
        
        # 前缀和计算
        for i in range(m):
            s[i+1] += s[i]
        s = s[:m]  # 截取有效部分
        
        # 计算LIS和LDS
        def compute_sequences(arr):
            dp = []
            result = []
            for num in arr:
                idx = bisect.bisect_right(dp, num)
                if idx == len(dp):
                    dp.append(num)
                else:
                    dp[idx] = num
                result.append(idx + 1)  # 长度索引调整
            return result
        
        left = compute_sequences(s)
        right = compute_sequences(s[::-1])[::-1]
        
        # 计算最大可能值
        max_total = max(l + r - 1 for l, r in zip(left, right))
        return solution == max_total
