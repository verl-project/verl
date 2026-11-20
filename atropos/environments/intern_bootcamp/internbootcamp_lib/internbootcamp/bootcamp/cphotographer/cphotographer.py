"""# 

### 谜题描述
Valera's lifelong ambition was to be a photographer, so he bought a new camera. Every day he got more and more clients asking for photos, and one day Valera needed a program that would determine the maximum number of people he can serve.

The camera's memory is d megabytes. Valera's camera can take photos of high and low quality. One low quality photo takes a megabytes of memory, one high quality photo take b megabytes of memory. For unknown reasons, each client asks him to make several low quality photos and several high quality photos. More formally, the i-th client asks to make xi low quality photos and yi high quality photos.

Valera wants to serve as many clients per day as possible, provided that they will be pleased with his work. To please the i-th client, Valera needs to give him everything he wants, that is, to make xi low quality photos and yi high quality photos. To make one low quality photo, the camera must have at least a megabytes of free memory space. Similarly, to make one high quality photo, the camera must have at least b megabytes of free memory space. Initially the camera's memory is empty. Valera also does not delete photos from the camera so that the camera's memory gradually fills up.

Calculate the maximum number of clients Valera can successfully serve and print the numbers of these clients.

Input

The first line contains two integers n and d (1 ≤ n ≤ 105, 1 ≤ d ≤ 109) — the number of clients and the camera memory size, correspondingly. The second line contains two integers a and b (1 ≤ a ≤ b ≤ 104) — the size of one low quality photo and of one high quality photo, correspondingly. 

Next n lines describe the clients. The i-th line contains two integers xi and yi (0 ≤ xi, yi ≤ 105) — the number of low quality photos and high quality photos the i-th client wants, correspondingly. 

All numbers on all lines are separated by single spaces. 

Output

On the first line print the answer to the problem — the maximum number of clients that Valera can successfully serve. Print on the second line the numbers of the client in any order. All numbers must be distinct. If there are multiple answers, print any of them. The clients are numbered starting with 1 in the order in which they are defined in the input data.

Examples

Input

3 10
2 3
1 4
2 1
1 0


Output

2
3 2 

Input

3 6
6 6
1 1
1 0
1 0


Output

1
2 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split(\" \"));
a, b = map(int, raw_input().split(\" \"))
v = [];
for i in range(0, n):
    x, y = map(int, raw_input().split(\" \"));
    v.append((a * x + b * y, i + 1));
v.sort();
#print v;
ans = [];
res = 0;
for w, n in v:
    if w <= m:
        ans.append(n);
        res += 1;
        m -= w;
    else:
        break;
print res;
print \" \".join(str(x) for x in ans);
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cphotographerbootcamp(Basebootcamp):
    def __init__(self, n_clients=5, d=100, a=2, b=3):
        if a > b:
            raise ValueError("a must be <= b")
        self.n_clients = n_clients
        self.d = d
        self.a = a
        self.b = b
        
    def case_generator(self):
        a = self.a
        b = self.b
        d_val = self.d
        
        max_xi = min(10**5, d_val // a) if a != 0 else 0
        max_yi = min(10**5, d_val // b) if b != 0 else 0
        
        clients = []
        for _ in range(self.n_clients):
            xi = random.randint(0, max_xi)
            yi = random.randint(0, max_yi)
            clients.append((xi, yi))
        
        case = {
            'n': self.n_clients,
            'd': d_val,
            'a': a,
            'b': b,
            'clients': clients.copy()
        }
        return case
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        d = question_case['d']
        a = question_case['a']
        b = question_case['b']
        clients = question_case['clients']
        
        input_lines = [f"{n} {d}", f"{a} {b}"]
        input_lines.extend(f"{x} {y}" for x, y in clients)
        input_str = '\n'.join(input_lines)
        
        prompt = f"""Valera是一位摄影师，他需要确定一天中最多可以服务多少客户。每个客户请求一定数量的低质量照片和高质量照片。每张低质量照片占用{a} MB内存，每张高质量照片占用{b} MB内存。相机的总内存是{d} MB。每个客户的请求必须被完全处理才能被服务。初始时内存为空，照片不可删除。

任务规则：
1. 输入包含{n}个客户，每个客户需要xi张低质量照片和yi张高质量照片。
2. 处理客户的顺序可以任意选择，但必须一次性处理完一个客户的所有照片需求。
3. 对于每个客户，处理该客户需要足够的空闲内存：xi*{a} + yi*{b} MB。
4. 目标是尽可能多地处理客户，找出最大可服务的客户数目，并输出这些客户的编号（输入中的顺序，从1开始）。

输入数据如下：
{input_str}

请按照以下格式输出答案：
第一行为一个整数，表示可服务的客户数量。
第二行为这些客户的编号，顺序不限，用空格分隔。

请将答案放在[answer]和[/answer]标签内。例如：
[answer]
2
3 1
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = [line.strip() for line in last_match.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            k = int(lines[0])
            clients = list(map(int, lines[1].split()))
            if len(clients) != k or len(set(clients)) != k:
                return None
            return clients
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list) or not all(isinstance(x, int) for x in solution):
            return False
        selected = solution
        n = identity['n']
        d = identity['d']
        a = identity['a']
        b = identity['b']
        clients = identity['clients']
        
        if len(selected) == 0:
            return True  # 允许0个客户的情况
        
        seen = set()
        for num in selected:
            if not (1 <= num <= n) or num in seen:
                return False
            seen.add(num)
        
        total = 0
        for num in selected:
            idx = num - 1
            x, y = clients[idx]
            cost = a * x + b * y
            if cost > (d - total):
                return False
            total += cost
        
        c_list = [(a * x + b * y, i + 1) for i, (x, y) in enumerate(clients)]
        sorted_clients = sorted(c_list, key=lambda x: x[0])
        prefix_sum = 0
        max_k = 0
        optimal_selection = []
        for c, num in sorted_clients:
            if prefix_sum + c > d:
                break
            prefix_sum += c
            optimal_selection.append(num)
            max_k += 1
        
        # 允许任何满足以下条件的解：数目等于max_k，且总消耗不超d
        return len(selected) == max_k and sum(a * clients[num-1][0] + b * clients[num-1][1] for num in selected) <= d
