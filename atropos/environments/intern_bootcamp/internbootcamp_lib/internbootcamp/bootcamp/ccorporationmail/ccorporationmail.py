"""# 

### 谜题描述
The Beroil corporation structure is hierarchical, that is it can be represented as a tree. Let's examine the presentation of this structure as follows:

  * employee ::= name. | name:employee1,employee2, ... ,employeek.
  * name ::= name of an employee 



That is, the description of each employee consists of his name, a colon (:), the descriptions of all his subordinates separated by commas, and, finally, a dot. If an employee has no subordinates, then the colon is not present in his description.

For example, line MIKE:MAX.,ARTEM:MIKE..,DMITRY:DMITRY.,DMITRY... is the correct way of recording the structure of a corporation where the director MIKE has subordinates MAX, ARTEM and DMITRY. ARTEM has a subordinate whose name is MIKE, just as the name of his boss and two subordinates of DMITRY are called DMITRY, just like himself.

In the Beroil corporation every employee can only correspond with his subordinates, at that the subordinates are not necessarily direct. Let's call an uncomfortable situation the situation when a person whose name is s writes a letter to another person whose name is also s. In the example given above are two such pairs: a pair involving MIKE, and two pairs for DMITRY (a pair for each of his subordinates).

Your task is by the given structure of the corporation to find the number of uncomfortable pairs in it.

<image>

Input

The first and single line contains the corporation structure which is a string of length from 1 to 1000 characters. It is guaranteed that the description is correct. Every name is a string consisting of capital Latin letters from 1 to 10 symbols in length.

Output

Print a single number — the number of uncomfortable situations in the company.

Examples

Input

MIKE:MAX.,ARTEM:MIKE..,DMITRY:DMITRY.,DMITRY...


Output

3


Input

A:A..


Output

1


Input

A:C:C:C:C.....


Output

6

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import re

def dfs(u):
    while cin.next() != '.':
        v = cin.next()
        global ans
        ans += u.setdefault(v, 0)
        u[v] += 1
        dfs(u)
        u[v] -= 1

cin = iter(filter(lambda x: x, re.split(r'([.,:])', raw_input())))
ans = 0
dfs({cin.next(): 1})
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import string
from bootcamp import Basebootcamp

def compute_uncomfortable_pairs(structure_str):
    tokens = re.split(r'([.,:])', structure_str)
    tokens = [t.strip() for t in tokens if t.strip()]
    it = iter(tokens)
    ans = 0

    def dfs(u):
        nonlocal ans
        while True:
            try:
                token = next(it)
            except StopIteration:
                break
            if token == '.':
                break
            if token in (':', ','):
                continue
            v = token
            ans += u.get(v, 0)
            u[v] = u.get(v, 0) + 1
            dfs(u)
            u[v] -= 1
            if u[v] == 0:
                del u[v]

    try:
        root = next(it)
    except StopIteration:
        return 0
    initial_u = {root: 1}
    dfs(initial_u)
    return ans

class Ccorporationmailbootcamp(Basebootcamp):
    def __init__(self, max_depth=4, name_reuse_prob=0.3, max_children=3):
        self.max_depth = max_depth
        self.name_reuse_prob = name_reuse_prob
        self.max_children = max_children
    
    def case_generator(self):
        # 参数动态调整机制
        original_params = (self.max_depth, self.max_children)
        
        for attempt in range(100):
            try:
                root = self.build_tree()
                structure_str = self.node_to_string(root)
                if 1 <= len(structure_str) <= 1000:
                    return {"structure": structure_str}
            except:
                pass
            
            # 动态调整参数
            if attempt % 10 == 9:
                self.max_depth = max(2, self.max_depth - 1)
                self.max_children = max(1, self.max_children - 1)
        
        # 恢复原始参数并生成简单案例
        self.max_depth, self.max_children = original_params
        return {"structure": f"{self.generate_name()}."}  # 生成单节点作为保底
    
    def build_tree(self, current_depth=0, parent_name=None):
        # 名称生成策略
        if parent_name and random.random() < self.name_reuse_prob:
            name = parent_name
        else:
            name = self.generate_name()
        
        node = {'name': name, 'children': []}
        
        if current_depth < self.max_depth:
            num_children = random.randint(0, self.max_children)
            for _ in range(num_children):
                child = self.build_tree(current_depth+1, name)
                node['children'].append(child)
        
        return node
    
    @staticmethod
    def node_to_string(node):
        if not node['children']:
            return f"{node['name']}."
        children_str = ','.join(Ccorporationmailbootcamp.node_to_string(child) for child in node['children'])
        return f"{node['name']}:{children_str}."
    
    @staticmethod
    def generate_name():
        return ''.join(random.choices(
            string.ascii_uppercase, 
            k=random.randint(1, 10)  # 严格符合题目要求
        ))
    
    @staticmethod
    def prompt_func(question_case):
        structure_str = question_case["structure"]
        return f"""Given the hierarchical structure of Ccorporationmail Corporation represented as a string, calculate the number of uncomfortable pairs. 

Structure rules:
1. Each employee is represented as "name." if no subordinates
2. Employees with subordinates are "name:sub1,sub2,..subN."
3. Subordinates can be at any level below in the hierarchy

An uncomfortable pair occurs when an employee (at any level) has the same name as one of their subordinates (direct or indirect).

Structure string: {structure_str}

Format your answer as [answer]number[/answer]. Example: [answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        try:
            return int(matches[-1]) if matches else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            ground_truth = compute_uncomfortable_pairs(identity["structure"])
            return int(solution) == ground_truth
        except:
            return False
