"""# 

### 谜题描述
One unknown hacker wants to get the admin's password of AtForces testing system, to get problems from the next contest. To achieve that, he sneaked into the administrator's office and stole a piece of paper with a list of n passwords — strings, consists of small Latin letters.

Hacker went home and started preparing to hack AtForces. He found that the system contains only passwords from the stolen list and that the system determines the equivalence of the passwords a and b as follows:

  * two passwords a and b are equivalent if there is a letter, that exists in both a and b; 
  * two passwords a and b are equivalent if there is a password c from the list, which is equivalent to both a and b. 



If a password is set in the system and an equivalent one is applied to access the system, then the user is accessed into the system.

For example, if the list contain passwords \"a\", \"b\", \"ab\", \"d\", then passwords \"a\", \"b\", \"ab\" are equivalent to each other, but the password \"d\" is not equivalent to any other password from list. In other words, if:

  * admin's password is \"b\", then you can access to system by using any of this passwords: \"a\", \"b\", \"ab\"; 
  * admin's password is \"d\", then you can access to system by using only \"d\". 



Only one password from the list is the admin's password from the testing system. Help hacker to calculate the minimal number of passwords, required to guaranteed access to the system. Keep in mind that the hacker does not know which password is set in the system.

Input

The first line contain integer n (1 ≤ n ≤ 2 ⋅ 10^5) — number of passwords in the list. Next n lines contains passwords from the list – non-empty strings s_i, with length at most 50 letters. Some of the passwords may be equal.

It is guaranteed that the total length of all passwords does not exceed 10^6 letters. All of them consist only of lowercase Latin letters.

Output

In a single line print the minimal number of passwords, the use of which will allow guaranteed to access the system.

Examples

Input


4
a
b
ab
d


Output


2

Input


3
ab
bc
abc


Output


1

Input


1
codeforces


Output


1

Note

In the second example hacker need to use any of the passwords to access the system.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import deque
from sys import stdin,stdout
read,write = stdin.readline,stdout.write
g = [[] for i in xrange(300000)]
visited = [False]*300000

def dfs(i):
	pilha = deque()
	pilha.append(i)
	visited[i] = 1
	while(pilha):
		i = pilha.pop()
		for v in g[i]:
			if not(visited[v]):
				visited[v] = True
				pilha.append(v)
n = int(read())
for i in xrange(n):
	word = read().strip()
	for j in word:
		g[i].append(n+ord(j))
		g[n+ord(j)].append(i)

resp = 0
for i in xrange(n):
	if not(visited[i]):
		dfs(i)
		resp += 1

print(resp)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict
import random
import re
from bootcamp import Basebootcamp

class Dsecretpasswordsbootcamp(Basebootcamp):
    def __init__(self, max_groups=3, min_passwords=1, max_passwords=5):
        self.max_groups = max_groups
        self.min_passwords = min_passwords
        self.max_passwords = max_passwords
    
    def case_generator(self):
        # 生成不交叠的字符分组
        group_chars = []
        all_chars = list('abcdefghijklmnopqrstuvwxyz')
        random.shuffle(all_chars)
        
        # 限制最大分组数不超过可用字符数
        k = min(random.randint(1, self.max_groups), len(all_chars))
        group_size = len(all_chars) // k
        
        # 分割字符到不同分组
        groups = []
        for i in range(k):
            start = i * group_size
            end = start + group_size if i < k-1 else len(all_chars)
            groups.append(set(all_chars[start:end]))
        
        passwords = []
        for group_id in range(k):
            chars = list(groups[group_id])
            if not chars:  # 确保每个分组至少有一个字符
                continue
            
            # 生成分组内的密码
            num = random.randint(self.min_passwords, self.max_passwords)
            for _ in range(num):
                length = random.randint(1, 50)
                password = ''.join(random.choices(chars, k=length))
                passwords.append(password)
        
        # 保证至少有一个密码
        if not passwords:
            passwords.append('default')
        
        return {'n': len(passwords), 'passwords': passwords}

    @staticmethod
    def prompt_func(case):
        input_lines = [str(case['n'])] + case['passwords']
        input_str = '\n'.join(input_lines)  # 提前处理换行符
        return f"""As a security expert, determine the minimal number of passwords needed for guaranteed system access. The equivalence rules are:
- Two passwords are equivalent if they share any common letter
- Equivalence is transitive through intermediate passwords

Given these passwords:
{input_str}

Calculate the minimal required passwords. Put your final answer between [answer] and [/answer], e.g.: [answer]2[/answer]"""

    @staticmethod 
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """改进的Union-Find实现"""
        n = identity['n']
        passwords = identity['passwords']
        
        parent = list(range(n))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # 路径压缩
                u = parent[u]
            return u
            
        # 构建字符到密码索引的映射
        char_map = defaultdict(list)
        for idx, pw in enumerate(passwords):
            for c in set(pw):
                char_map[c].append(idx)
        
        # 合并同一字符对应的所有密码
        for c in char_map.values():
            if len(c) < 2:
                continue
            root = find(c[0])
            for other in c[1:]:
                parent[find(other)] = root
        
        # 计算连通分量数量
        roots = set()
        for i in range(n):
            roots.add(find(i))
        
        return solution == len(roots)
