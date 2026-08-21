"""# 

### 谜题描述
Fox Ciel is going to publish a paper on FOCS (Foxes Operated Computer Systems, pronounce: \"Fox\"). She heard a rumor: the authors list on the paper is always sorted in the lexicographical order. 

After checking some examples, she found out that sometimes it wasn't true. On some papers authors' names weren't sorted in lexicographical order in normal sense. But it was always true that after some modification of the order of letters in alphabet, the order of authors becomes lexicographical!

She wants to know, if there exists an order of letters in Latin alphabet such that the names on the paper she is submitting are following in the lexicographical order. If so, you should find out any such order.

Lexicographical order is defined in following way. When we compare s and t, first we find the leftmost position with differing characters: si ≠ ti. If there is no such position (i. e. s is a prefix of t or vice versa) the shortest string is less. Otherwise, we compare characters si and ti according to their order in alphabet.

Input

The first line contains an integer n (1 ≤ n ≤ 100): number of names.

Each of the following n lines contain one string namei (1 ≤ |namei| ≤ 100), the i-th name. Each name contains only lowercase Latin letters. All names are different.

Output

If there exists such order of letters that the given names are sorted lexicographically, output any such order as a permutation of characters 'a'–'z' (i. e. first output the first letter of the modified alphabet, then the second, and so on).

Otherwise output a single word \"Impossible\" (without quotes).

Examples

Input

3
rivest
shamir
adleman


Output

bcdefghijklmnopqrsatuvwxyz


Input

10
tourist
petr
wjmzbmr
yeputons
vepifanov
scottwu
oooooooooooooooo
subscriber
rowdark
tankengineer


Output

Impossible


Input

10
petr
egor
endagorion
feferivan
ilovetanyaromanova
kostka
dmitriyh
maratsnowbear
bredorjaguarturnik
cgyforever


Output

aghjlnopefikdmbcqrstuvwxyz


Input

7
car
care
careful
carefully
becarefuldontforgetsomething
otherwiseyouwillbehacked
goodluck


Output

acbdefhijklmnogpqrstuvwxyz

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict
from itertools import takewhile, count

def sort_topologically(graph):
    levels_by_name = {}
    names_by_level = defaultdict(set)

    def walk_depth_first(name):
        if name in levels_by_name:
            return levels_by_name[name]
        children = graph.get(name, None)
        level = 0 if not children else (1 + max(walk_depth_first(lname) for lname in children))
        levels_by_name[name] = level
        names_by_level[level].add(name)
        return level

    for name in graph:
        walk_depth_first(name)

    return list(takewhile(lambda x: x is not None, (names_by_level.get(i, None) for i in count())))

N = input()
alpha = 'abcdefghijklmnopqrstuvwxyz'
graph = {k:[] for k in alpha}

# print(sort_topologically(graph))
names = [raw_input() for x in range(N)]
ans = True
for name_a, name_b in zip(names, names[1:]):
  temp_a = str(name_a)
  temp_b = str(name_b)
  while temp_a and temp_b and temp_a[0] == temp_b[0]:
    temp_a = temp_a[1:]
    temp_b = temp_b[1:]
  if temp_a and not temp_b:
    ans = False
    break
  if not temp_a or not temp_b:
    continue
  graph[temp_b[0]].append(temp_a[0])
# for key in graph:
#   print key, graph[key]

for key in graph:
  if not graph[key]:
    continue
  visited = set([key])
  root = graph[key]
  while root:
    if root[0] in visited:
      ans = False
      break
    visited.add(root[0])
    root = graph[root[0]]
  if not ans:
    break

# abc = 0
# for key in graph:
#   if not graph[key]:
#     abc += 1
# if abc == 26:
#   ans = False

if ans:
  print ''.join(sum([list(x) for x in sort_topologically(graph)], []))
  # print ''
else:
  print 'Impossible'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from collections import defaultdict, deque
from bootcamp import Basebootcamp

def solve_puzzle(names):
    graph = defaultdict(list)
    for c in string.ascii_lowercase:  # 初始化所有字母节点
        graph[c] = []
    
    # 构建字母约束关系图
    for i in range(len(names)-1):
        a, b = names[i], names[i+1]
        min_len = min(len(a), len(b))
        j = 0
        while j < min_len and a[j] == b[j]:
            j += 1
        
        if j == min_len:  # 处理前缀情况
            if len(a) > len(b):
                return "Impossible"
            continue
        
        # 添加字符顺序约束：a[j]必须出现在b[j]之前
        x, y = a[j], b[j]
        graph[y].append(x)  # 修正方向：y依赖x → x必须出现在y前面
    
    # 拓扑排序
    in_degree = {c:0 for c in string.ascii_lowercase}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    
    queue = deque([c for c in string.ascii_lowercase if in_degree[c] == 0])
    top_order = []
    
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    return "Impossible" if len(top_order)!=26 else "".join(reversed(top_order))

class Cfoxandnamesbootcamp(Basebootcamp):
    def __init__(self, min_names=1, max_names=10, min_length=1, max_length=10, valid_ratio=0.5):
        self.min_names = min_names
        self.max_names = max_names
        self.min_length = min_length
        self.max_length = max_length
        self.valid_ratio = valid_ratio

    def case_generator(self):
        for _ in range(1000):  # 增加尝试次数
            n = random.randint(self.min_names, self.max_names)
            names = self._generate_names(n)
            solution = solve_puzzle(names)
            
            # 动态调整有效案例生成概率
            target_valid = random.random() < self.valid_ratio
            if (solution != "Impossible") == target_valid:
                return {'names': names}
        
        # 回退案例：生成保证有效的案例
        return {'names': sorted(["a"*i for i in range(1,4)], key=lambda x: (-len(x), x))}

    def _generate_names(self, n):
        names = set()
        char_pool = random.sample(string.ascii_lowercase, random.randint(3,5))  # 限制字符集增加冲突
        
        while len(names) < n:
            length = random.randint(self.min_length, self.max_length)
            name = "".join(random.choices(char_pool, k=length))
            names.add(name)
        return list(names)

    @staticmethod
    def prompt_func(question_case):
        names = question_case['names']
        problem = (
            "Determine if a custom alphabet exists to make the names lex ordered.\n"
            f"Input:\n{len(names)}\n" + "\n".join(names) + "\n\n"
            "Output format: [answer]<LETTER_ORDER|Impossible>[/answer]"
        )
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL|re.IGNORECASE)
        if not matches:
            return None
        ans = matches[-1].strip().lower()
        return ans if ans == "impossible" or len(ans)==26 else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        names = identity['names']
        
        if solution == "impossible":
            return solve_puzzle(names) == "Impossible"
        
        if len(solution)!=26 or len(set(solution))!=26:
            return False
        
        order = {c:i for i,c in enumerate(solution)}
        for i in range(len(names)-1):
            a, b = names[i], names[i+1]
            found = False
            for j in range(min(len(a), len(b))):
                if a[j] != b[j]:
                    if order[a[j]] > order[b[j]]:
                        return False
                    found = True
                    break
            if not found and len(a) > len(b):
                return False
        return True
