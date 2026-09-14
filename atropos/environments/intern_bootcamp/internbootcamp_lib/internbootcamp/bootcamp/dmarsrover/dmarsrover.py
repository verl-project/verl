"""# 

### 谜题描述
Natasha travels around Mars in the Mars rover. But suddenly it broke down, namely — the logical scheme inside it. The scheme is an undirected tree (connected acyclic graph) with a root in the vertex 1, in which every leaf (excluding root) is an input, and all other vertices are logical elements, including the root, which is output. One bit is fed to each input. One bit is returned at the output.

There are four types of logical elements: [AND](https://en.wikipedia.org/wiki/Logical_conjunction) (2 inputs), [OR](https://en.wikipedia.org/wiki/Logical_disjunction) (2 inputs), [XOR](https://en.wikipedia.org/wiki/Exclusive_or) (2 inputs), [NOT](https://en.wikipedia.org/wiki/Negation) (1 input). Logical elements take values from their direct descendants (inputs) and return the result of the function they perform. Natasha knows the logical scheme of the Mars rover, as well as the fact that only one input is broken. In order to fix the Mars rover, she needs to change the value on this input.

For each input, determine what the output will be if Natasha changes this input.

Input

The first line contains a single integer n (2 ≤ n ≤ 10^6) — the number of vertices in the graph (both inputs and elements).

The i-th of the next n lines contains a description of i-th vertex: the first word \"AND\", \"OR\", \"XOR\", \"NOT\" or \"IN\" (means the input of the scheme) is the vertex type. If this vertex is \"IN\", then the value of this input follows (0 or 1), otherwise follow the indices of input vertices of this element: \"AND\", \"OR\", \"XOR\" have 2 inputs, whereas \"NOT\" has 1 input. The vertices are numbered from one.

It is guaranteed that input data contains a correct logical scheme with an output produced by the vertex 1.

Output

Print a string of characters '0' and '1' (without quotes) — answers to the problem for each input in the ascending order of their vertex indices.

Example

Input

10
AND 9 4
IN 1
IN 1
XOR 6 5
AND 3 7
IN 0
NOT 10
IN 1
IN 1
AND 2 8


Output

10110

Note

The original scheme from the example (before the input is changed):

<image>

Green indicates bits '1', yellow indicates bits '0'.

If Natasha changes the input bit 2 to 0, then the output will be 1.

If Natasha changes the input bit 3 to 0, then the output will be 0.

If Natasha changes the input bit 6 to 1, then the output will be 1.

If Natasha changes the input bit 8 to 0, then the output will be 1.

If Natasha changes the input bit 9 to 0, then the output will be 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
#pragma comment(linker, \"/stack:200000000\")
#pragma GCC optimize(\"Ofast\")
#pragma GCC target(\"sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native\")
vector<int> op;
vector<vector<int>> v;
vector<bool> val;
int mp[1000001][2];
vector<int> saved;
vector<int> cur;
vector<pair<int, int>> curc;
vector<int> parent;
int fans = -1;
static string s;
int solve(int k) {
  cur.push_back(k);
  if (v[k].size() == 0) return saved[k] = val[k];
  if (v[k].size() == 1) return saved[k] = !solve(v[k][0]);
  if (op[k] == 1) return saved[k] = (solve(v[k][0]) & solve(v[k][1]));
  if (op[k] == 2) return saved[k] = (solve(v[k][0]) | solve(v[k][1]));
  if (op[k] == 3) return saved[k] = (solve(v[k][0]) ^ solve(v[k][1]));
  return 0;
}
void solve2(int k, int num) {
  if (k == 0) {
    fans = num;
    return;
  }
  if (mp[k][num] != -1) {
    fans = mp[k][num];
    return;
  }
  int p = parent[k];
  if (v[p].size() == 1) {
    curc.push_back({p, 1 - num});
    solve2(p, 1 - num);
    return;
  }
  int num1 = v[p][0], num2 = v[p][1];
  bool ar1, ar2;
  if (num1 == k)
    ar1 = num, ar2 = saved[v[p][1]];
  else
    ar2 = num, ar1 = saved[v[p][0]];
  if (op[p] == 1) {
    curc.push_back({p, ar1 & ar2});
    solve2(p, ar1 & ar2);
  }
  if (op[p] == 2) {
    curc.push_back({p, ar1 | ar2});
    solve2(p, ar1 | ar2);
  }
  if (op[p] == 3) {
    curc.push_back({p, ar1 ^ ar2});
    solve2(p, ar1 ^ ar2);
  }
}
signed main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  int n;
  cin >> n;
  memset(mp, -1, sizeof(mp));
  op.resize(n);
  val.resize(n);
  parent.resize(n);
  v.assign(n, {});
  saved.resize(n, 0);
  for (int i = 0; i < n; i++) {
    cin >> s;
    if (s == \"IN\") {
      op[i] = 0;
      int num;
      cin >> num;
      val[i] = num;
    } else if (s == \"NOT\") {
      int num;
      cin >> num;
      op[i] = 0;
      num--;
      v[i].push_back(num);
      parent[num] = i;
    } else {
      if (s == \"AND\") op[i] = 1;
      if (s == \"OR\") op[i] = 2;
      if (s == \"XOR\") op[i] = 3;
      int num1, num2;
      cin >> num1 >> num2;
      num1--;
      num2--;
      v[i].push_back(num1);
      v[i].push_back(num2);
      parent[num1] = i;
      parent[num2] = i;
    }
  }
  bool ans = solve(0);
  for (int i = 0; i < n; i++) {
    mp[i][saved[i]] = ans;
  }
  parent[0] = -1;
  for (int i = 0; i < n; i++) {
    if (v[i].size() == 0) {
      solve2(i, 1 - val[i]);
      if ((int)curc.size()) {
        for (auto xx : curc) mp[xx.first][xx.second] = fans;
      }
      curc.clear();
      cout << fans;
    }
  }
  cout << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import copy
from collections import deque
from bootcamp import Basebootcamp

class Dmarsroverbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.default_params = {'max_depth': 5, 'min_inputs': 3}
        self.default_params.update(params)
    
    def case_generator(self):
        # 生成根节点（顶点1）
        root_type = random.choice(['AND', 'OR', 'XOR', 'NOT'])
        nodes = {1: {'type': root_type, 'children': []}}
        queue = deque([1])
        next_id = 2
        
        # 生成合法树结构
        while queue:
            current_id = queue.popleft()
            current_type = nodes[current_id]['type']
            
            # 确定必需子节点数
            required_children = {'AND':2, 'OR':2, 'XOR':2, 'NOT':1}.get(current_type,0)
            children = []
            
            # 生成子节点
            for _ in range(required_children):
                # 动态决定子节点类型
                if current_id == 1 or random.random() < 0.5:  # 增加逻辑门生成概率
                    child_type = random.choice(['AND', 'OR', 'XOR', 'NOT'])
                else:
                    child_type = 'IN'
                
                # 创建节点
                nodes[next_id] = {'type': child_type}
                if child_type == 'IN':
                    nodes[next_id]['value'] = random.choice([0,1])
                else:
                    nodes[next_id]['children'] = []
                    queue.append(next_id)
                children.append(next_id)
                next_id +=1
            
            nodes[current_id]['children'] = children
        
        # 确保至少min_inputs个输入节点
        def count_inputs():
            return sum(1 for n in nodes.values() if n['type'] == 'IN')
        
        while count_inputs() < self.default_params['min_inputs']:
            # 寻找可添加输入的节点
            candidates = [id for id in nodes 
                         if nodes[id]['type'] in ['AND','OR','XOR','NOT']
                         and len(nodes[id]['children']) < 
                             (2 if nodes[id]['type'] != 'NOT' else 1)]
            if not candidates:
                break
            parent = random.choice(candidates)
            new_id = next_id
            next_id +=1
            nodes[new_id] = {'type': 'IN', 'value': random.choice([0,1])}
            nodes[parent]['children'].append(new_id)
        
        # 收集所有输入节点（正确方法）
        def find_inputs():
            inputs = []
            stack = [1]
            while stack:
                node_id = stack.pop()
                node = nodes.get(node_id)
                if not node: continue
                if node['type'] == 'IN':
                    inputs.append(node_id)
                else:
                    stack.extend(node.get('children',[]))
            return sorted(inputs)
        
        input_nodes = find_inputs()
        
        # 构建节点列表（索引对齐）
        max_id = max(nodes.keys())
        nodes_list = [None]*(max_id+1)
        for id in nodes:
            nodes_list[id] = nodes[id]
        
        # 计算原始结果
        original_output = self.compute_output(nodes_list)
        
        # 计算每个输入翻转后的结果
        answer = []
        for input_id in input_nodes:
            # 深拷贝并修改
            new_nodes = copy.deepcopy(nodes_list)
            new_nodes[input_id]['value'] = 1 - new_nodes[input_id]['value']
            
            # 计算新输出
            new_output = self.compute_output(new_nodes)
            answer.append(str(new_output))
        
        return {
            'nodes': nodes_list,
            'input_order': input_nodes,
            'answer': ''.join(answer)
        }
    
    def compute_output(self, nodes_list):
        computed = {}
        
        def evaluate(node_id):
            if node_id in computed:
                return computed[node_id]
            
            node = nodes_list[node_id]
            if node['type'] == 'IN':
                res = node['value']
            elif node['type'] == 'NOT':
                res = 1 - evaluate(node['children'][0])
            elif node['type'] == 'AND':
                res = evaluate(node['children'][0]) & evaluate(node['children'][1])
            elif node['type'] == 'OR':
                res = evaluate(node['children'][0]) | evaluate(node['children'][1])
            elif node['type'] == 'XOR':
                res = evaluate(node['children'][0]) ^ evaluate(node['children'][1])
            else:
                raise ValueError(f"Invalid node type {node['type']}")
            
            computed[node_id] = res
            return res
        
        return evaluate(1)
    
    @staticmethod 
    def prompt_func(case):
        desc = []
        for node_id in range(1, len(case['nodes'])):
            node = case['nodes'][node_id]
            if not node: continue
            if node['type'] == 'IN':
                desc.append(f"顶点 {node_id}: 输入端口，初始值 {node['value']}")
            else:
                children = ', '.join(map(str, node['children']))
                desc.append(f"顶点 {node_id}: {node['type']}门，连接[{children}]")
        
        input_order = ', '.join(map(str, case['input_order']))
        return f"""火星车逻辑电路维修问题：
电路结构：
{chr(10).join(desc)}

需要分析以下输入端口（按编号排序）：{input_order}。
对于每个端口，计算其值翻转后的根节点输出。按顺序组合答案如[answer]10101[/answer]。答案只能是0和1组成的字符串。"""
    
    @staticmethod
    def extract_output(text):
        matches = re.findall(r'\[answer\]([01]+)\[/answer\]', text)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, sol, case):
        return sol == case['answer']
