"""# 

### 谜题描述
Vladik often travels by trains. He remembered some of his trips especially well and I would like to tell you about one of these trips:

Vladik is at initial train station, and now n people (including Vladik) want to get on the train. They are already lined up in some order, and for each of them the city code ai is known (the code of the city in which they are going to).

Train chief selects some number of disjoint segments of the original sequence of people (covering entire sequence by segments is not necessary). People who are in the same segment will be in the same train carriage. The segments are selected in such way that if at least one person travels to the city x, then all people who are going to city x should be in the same railway carriage. This means that they can’t belong to different segments. Note, that all people who travel to the city x, either go to it and in the same railway carriage, or do not go anywhere at all.

Comfort of a train trip with people on segment from position l to position r is equal to XOR of all distinct codes of cities for people on the segment from position l to position r. XOR operation also known as exclusive OR.

Total comfort of a train trip is equal to sum of comfort for each segment.

Help Vladik to know maximal possible total comfort.

Input

First line contains single integer n (1 ≤ n ≤ 5000) — number of people.

Second line contains n space-separated integers a1, a2, ..., an (0 ≤ ai ≤ 5000), where ai denotes code of the city to which i-th person is going.

Output

The output should contain a single integer — maximal possible total comfort.

Examples

Input

6
4 4 2 5 2 3


Output

14


Input

9
5 1 3 1 5 2 4 2 5


Output

9

Note

In the first test case best partition into segments is: [4, 4] [2, 5, 2] [3], answer is calculated as follows: 4 + (2 xor 5) + 3 = 4 + 7 + 3 = 14

In the second test case best partition into segments is: 5 1 [3] 1 5 [2, 4, 2] 5, answer calculated as follows: 3 + (2 xor 4) = 3 + 6 = 9.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
lst=map(int,raw_input().split())
lmost=[-1]*(5002)
rmost=[-1]*(5002)
dp2=[]


dp=[0]*(5003)
for i in range(0,n):
    if lmost[lst[i]]==-1:
        lmost[lst[i]]=i
for i in range(n-1,-1,-1):
    if rmost[lst[i]]==-1:
        rmost[lst[i]]=i

for i in range(0,n):
    val=0
    s=set()
    dp[i]=dp[i-1]
    m=lmost[lst[i]]
    for j in range(i,-1,-1):
        if rmost[lst[j]]>i:
            break
        if rmost[lst[j]]==j:
            val=val^lst[j]
        #sg=lst[j:i]
        m = min(m, lmost[lst[j]])
       # print j,\"j\",i,\"i\"
        if j== m:


            dp[i] = max(dp[i], dp[j-1] + val)
     #   print s,j,val

print max(dp)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

def compute_max_comfort(n, a):
    # 预处理每个城市的最左和最右出现位置
    lmost = {}
    rmost = {}
    for i in range(n):
        city = a[i]
        if city not in lmost:
            lmost[city] = i
        rmost[city] = i
    
    dp = [0] * (n + 1)
    
    for i in range(n):
        dp[i+1] = dp[i]  # 默认不选当前段
        
        segment_cities = set()
        current_xor = 0
        min_l = n  # 当前段最小左边界
        valid = True
        
        # 从i往左扫描
        for j in range(i, -1, -1):
            city = a[j]
            
            # 检查该城市是否违反右边界约束
            if rmost.get(city, -1) > i:
                valid = False
                break
            
            # 更新当前段最小左边界
            min_l = min(min_l, lmost[city])
            
            # 仅当j到达当前段理论最小左边界时进行状态转移
            if j == min_l and valid:
                # 计算当前段的XOR
                if city not in segment_cities:
                    segment_cities.add(city)
                    current_xor ^= city
                
                # 状态转移
                dp[i+1] = max(dp[i+1], dp[j] + current_xor)
    
    return dp[n]

class Cvladikandmemorabletripbootcamp(Basebootcamp):
    def __init__(self, min_n=4, max_n=9, max_city=6):
        self.min_n = min_n
        self.max_n = max_n
        self.max_city = max_city
    
    def case_generator(self):
        for _ in range(100):  # 防止无限循环
            n = random.randint(self.min_n, self.max_n)
            a = [random.randint(0, self.max_city) for _ in range(n)]
            
            # 确保每个城市的出现位置连续
            city_pos = {}
            for i in range(n):
                city = a[i]
                if city in city_pos:
                    last_pos = city_pos[city][-1]
                    if last_pos != i-1:
                        # 强制让相同城市连续出现
                        a[last_pos+1], a[i] = a[i], a[last_pos+1]
                city_pos.setdefault(city, []).append(i)
            
            try:
                answer = compute_max_comfort(n, a)
                return {"n": n, "a": a, "answer": answer}
            except Exception as e:
                continue
        return {"n":4, "a":[1,1,2,2], "answer":3}  # 保底案例
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        return f"""## 题目描述
{Cvladikandmemorabletripbootcamp._rule_description()}

## 当前实例
人数n: {n}
城市代码序列: {' '.join(map(str, a))}

## 要求
请给出最大总舒适度，答案置于[answer][/answer]标签内。示例：[answer]42[/answer]"""

    @staticmethod
    def _rule_description():
        return """## 规则详解
1. **分段规则**：选择的各分段必须满足：若某分段包含城市x的乘客，则该城市所有乘客必须在同一分段
2. **舒适度计算**：每个分段的舒适度是该段内不同城市代码的异或(XOR)值
3. **目标**：选择若干不相交分段，使总舒适度最大"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
