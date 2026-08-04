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
l=int(raw_input())
a=list()
for j in range(l):
    a.append(raw_input())

def topologicalSortUtil(v,visited,stack,imp,exp): 
    
    exp[ord(v)-ord('a')] = True
    visited[v] = True
    if v in graph.keys():
        for i in graph[v]:
            if exp[ord(i)-ord('a')] == True and visited[i] == False:
                pass
                
            elif visited[i] == False:
                topologicalSortUtil(i,visited,stack,imp,exp)
                if imp[0] == True:
                    break
            else:
                imp[0] = True
                return
    
    visited[v] = False
    stack.insert(0,v)

def TOPSORT(graph): 
    alpha_bets = list(map(chr, range(97, 123)))
    visited  = { i : False for i in alpha_bets }
    all_explored  = [False for i in range(0,26)]
    stack =[]
    impossible = list()
    impossible.append(False)
    for i in reversed(alpha_bets):
        if all_explored[ord(i)-ord('a')] == False :
            topologicalSortUtil(i,visited,stack,impossible,all_explored)
            if impossible[0] == True:
                return -1
        
    return stack 


graph = dict()
np = False
for i in range(0,l-1):
    first = a[i]
    second = a[i+1]
    k=0
    while k<min(len(first),len(second)) and first[k] == second[k]:
        k=k+1
    
    if k == len(first):
        continue
    elif k == len(second):
        np = True
        break
    
    if first[k] in graph.keys() and second[k] not in graph[first[k]] :
        graph[first[k]].append(second[k])
        
    else:
        if first[k] not in graph.keys():
            graph[first[k]] = list()
            
        graph[first[k]].append(second[k])

order_req = TOPSORT(graph)

if order_req == -1 or np:
    print \"Impossible\"
    
else:
    print ''.join(str(i) for i in order_req)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import json

class Afoxandnamesbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 3)  # Number of names
        self.names = params.get('names', self.generate_random_names(self.n))
    
    def generate_random_names(self, n=3):
        """Generate random names for testing"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        names = []
        for _ in range(n):
            length = random.randint(3, 6)
            name = ''.join(random.choice(letters) for _ in range(length))
            names.append(name)
        return names
    
    def case_generator(self):
        """Generate a puzzle instance with random names"""
        # Generate random names and ensure they can form a valid order
        names = []
        while True:
            names = self.generate_random_names(self.n)
            # Check if these names can form a valid order
            valid = True
            for i in range(len(names) - 1):
                if len(names[i]) == len(names[i+1]):
                    if names[i] > names[i+1]:
                        valid = False
                        break
                else:
                    if not (names[i] < names[i+1]):
                        valid = False
                        break
            if valid:
                break
        
        # Shuffle the names to create a puzzle instance
        shuffled_names = names.copy()
        random.shuffle(shuffled_names)
        
        return {
            'n': self.n,
            'names': shuffled_names
        }
    
    @staticmethod
    def prompt_func(question_case):
        names = question_case['names']
        names_str = ', '.join(names)
        prompt = (
            "你是一名科学家，Fox Ciel，正在准备提交一篇论文。你需要确保作者列表按某种字母顺序排列。给你一组名字："
            f"{names_str}，判断是否存在一种字母顺序，使得这些名字按字典序排列。如果存在，输出该顺序；否则，输出'Impossible'。"
            "注意：名字的比较规则是逐字符比较，遇到第一个不同的字符按字母顺序决定大小。如果一个名字是另一个的前缀，则较短的名字排在前面。"
            "请将答案放在[answer]和[/answer]之间。"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        parts = output.split('[answer]')
        if len(parts) < 2:
            return None
        last_answer = parts[-1].split('[/answer]')[0].strip()
        return last_answer
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == 'Impossible':
            names = identity['names']
            graph = {}
            for i in range(len(names) - 1):
                first = names[i]
                second = names[i+1]
                k = 0
                while k < min(len(first), len(second)) and first[k] == second[k]:
                    k += 1
                if k == len(first):
                    continue
                elif k == len(second):
                    return False  # Cannot be ordered
                if first[k] not in graph:
                    graph[first[k]] = []
                if second[k] not in graph[first[k]]:
                    graph[first[k]].append(second[k])
            try:
                order = cls.topological_sort(graph)
                if order == -1:
                    return True  # Impossible is correct
                else:
                    return False  # Solution is not Impossible
            except:
                return False
        else:
            order = solution
            if len(order) != 26 or len(set(order)) != 26:
                return False
            order_dict = {char: idx for idx, char in enumerate(order)}
            for i in range(len(identity['names']) - 1):
                first = identity['names'][i]
                second = identity['names'][i+1]
                if not cls.is_ordered(first, second, order_dict):
                    return False
            return True
    
    @staticmethod
    def topological_sort(graph):
        visited = set()
        stack = []
        has_cycle = [False]
        
        def dfs(node):
            visited.add(node)
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor in visited:
                        has_cycle[0] = True
                        return
                    if neighbor not in visited:
                        dfs(neighbor)
            stack.append(node)
        
        for char in 'abcdefghijklmnopqrstuvwxyz':
            if char not in visited:
                dfs(char)
                if has_cycle[0]:
                    return -1
        return ''.join(stack[::-1])
    
    @staticmethod
    def is_ordered(a, b, order_dict):
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                return order_dict[a[i]] < order_dict[b[i]]
        return len(a) <= len(b)
