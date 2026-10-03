"""# 

### 谜题描述
Owl Pacino has always been into trees — unweighted rooted trees in particular. He loves determining the diameter of every tree he sees — that is, the maximum length of any simple path in the tree.

Owl Pacino's owl friends decided to present him the Tree Generator™ — a powerful machine creating rooted trees from their descriptions. An n-vertex rooted tree can be described by a bracket sequence of length 2(n - 1) in the following way: find any walk starting and finishing in the root that traverses each edge exactly twice — once down the tree, and later up the tree. Then follow the path and write down \"(\" (an opening parenthesis) if an edge is followed down the tree, and \")\" (a closing parenthesis) otherwise.

The following figure shows sample rooted trees and their descriptions:

<image>

Owl wrote down the description of an n-vertex rooted tree. Then, he rewrote the description q times. However, each time he wrote a new description, he picked two different characters in the description he wrote the last time, swapped them and wrote down the resulting string. He always made sure that each written string was the description of a rooted tree.

Pacino then used Tree Generator™ for each description he wrote down. What is the diameter of each constructed tree?

Input

The first line of the input contains two integers n, q (3 ≤ n ≤ 100 000, 1 ≤ q ≤ 100 000) — the number of vertices in the tree and the number of changes to the tree description. The following line contains a description of the initial tree — a string of length 2(n-1) consisting of opening and closing parentheses.

Each of the following q lines describes a single change to the description and contains two space-separated integers a_i, b_i (2 ≤ a_i, b_i ≤ 2n-3) which identify the indices of two brackets to be swapped. You can assume that the description will change after each query, and that after each change a tree can be constructed from the description.

Output

Output q + 1 integers — the diameter of each constructed tree, in the order their descriptions have been written down.

Examples

Input


5 5
(((())))
4 5
3 4
5 6
3 6
2 5


Output


4
3
3
2
4
4


Input


6 4
(((())()))
6 7
5 4
6 4
7 4


Output


4
4
4
5
3

Note

The following figure shows each constructed tree and its description in the first example test: 

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
constexpr int MAXN = 112345;
constexpr int MAXSEQ = 2 * (MAXN - 1);
constexpr int MAXNODES = 4 * MAXSEQ;
struct Node {
  int maxc, minc, p, s, v, lazy;
  void show() {
    cerr << \"{\" << endl;
    cerr << \"maxc\"
         << \" == \" << (maxc) << endl;
    ;
    cerr << \"minc\"
         << \" == \" << (minc) << endl;
    ;
    cerr << \"p\"
         << \" == \" << (p) << endl;
    ;
    cerr << \"s\"
         << \" == \" << (s) << endl;
    ;
    cerr << \"v\"
         << \" == \" << (v) << endl;
    ;
    cerr << \"lazy\"
         << \" == \" << (lazy) << endl;
    ;
    cerr << \"}\" << endl;
  }
};
Node tree[MAXNODES];
void fix(int L, int R, int ID);
void showTree(int L, int R, int ID) {
  cerr << \"[\" << L << \", \" << R << \"] : \" << ID << endl;
  tree[ID].show();
  cerr << endl;
  if (L < R) {
    int mid = (L + R) / 2;
    fix(L, R, ID);
    showTree(L, mid, 2 * ID + 1);
    showTree(mid + 1, R, 2 * ID + 2);
  }
}
Node join(const Node& L, const Node& R) {
  Node ans;
  ans.lazy = 0;
  ans.maxc = max(L.maxc, R.maxc);
  ans.minc = min(L.minc, R.minc);
  ans.p = max(L.p, R.p);
  ans.p = max(ans.p, L.maxc - 2 * R.minc);
  ans.s = max(L.s, R.s);
  ans.s = max(ans.s, R.maxc - 2 * L.minc);
  ans.v = max(L.v, R.v);
  ans.v = max(ans.v, L.maxc + R.s);
  ans.v = max(ans.v, L.p + R.maxc);
  return ans;
}
void calc(int L, int R, int ID) {
  if (L >= R) return;
  const int IDL = 2 * ID + 1, IDR = 2 * ID + 2;
  tree[ID] = join(tree[IDL], tree[IDR]);
}
void build(int L, int R, int ID) {
  if (L >= R) {
    tree[ID].p = tree[ID].s = tree[ID].v = -MAXSEQ;
    return;
  }
  int mid = (L + R) / 2;
  build(mid + 1, R, 2 * ID + 2);
  calc(L, R, ID);
}
void update(int a, int b, int x, int L, int R, int ID) {
  if (a > R or b < L) return;
  if (a <= L and b >= R) {
    tree[ID].maxc += x;
    tree[ID].minc += x;
    tree[ID].p -= x;
    tree[ID].s -= x;
    tree[ID].lazy += x;
    return;
  }
  fix(L, R, ID);
  int mid = (L + R) / 2;
  update(a, b, x, L, mid, 2 * ID + 1);
  update(a, b, x, mid + 1, R, 2 * ID + 2);
  calc(L, R, ID);
}
Node query(int a, int b, int L, int R, int ID) {
  if (a > R or b < L) {
    Node bad;
    bad.maxc = -MAXSEQ;
    bad.minc = MAXSEQ;
    bad.v = bad.p = bad.s = -MAXSEQ;
    return bad;
  }
  if (a <= L and b >= R) {
    return tree[ID];
  }
  fix(L, R, ID);
  int mid = (L + R) / 2;
  const int IDL = 2 * ID + 1, IDR = 2 * ID + 2;
  Node AL = query(a, b, L, mid, IDL);
  Node AR = query(a, b, mid + 1, R, IDR);
  return join(AL, AR);
}
void fix(int L, int R, int ID) {
  if (L >= R or not tree[ID].lazy) return;
  int mid = (L + R) / 2;
  update(L, mid, tree[ID].lazy, L, mid, 2 * ID + 1);
  update(mid + 1, R, tree[ID].lazy, mid + 1, R, 2 * ID + 2);
  tree[ID].lazy = 0;
}
int n, S;
string seq;
void add(int i) {
  if (seq[i] == '(') {
    update(i + 1, S, 1, 0, S, 0);
  } else {
    update(i + 1, S, -1, 0, S, 0);
  }
}
void remove(int i) {
  if (seq[i] == '(') {
    update(i + 1, S, -1, 0, S, 0);
  } else {
    update(i + 1, S, 1, 0, S, 0);
  }
}
int main() {
  ios::sync_with_stdio(false);
  int q;
  cin >> n >> q;
  cin >> seq;
  S = 2 * (n - 1);
  build(0, S, 0);
  for (int i = 0; i < S; i++) {
    add(i);
  }
  cout << query(0, S, 0, S, 0).v << '\n';
  for (int qi = 0; qi < q; qi++) {
    int a, b;
    cin >> a >> b;
    a--;
    b--;
    remove(a);
    remove(b);
    swap(seq[a], seq[b]);
    add(a);
    add(b);
    cout << query(0, S, 0, S, 0).v << '\n';
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from collections import deque

class TreeNode:
    def __init__(self):
        self.children = []
        self.parent = None

def parse_bracket_sequence(s):
    root = TreeNode()
    stack = [root]
    for c in s:
        if c == '(':
            node = TreeNode()
            current = stack[-1]
            current.children.append(node)
            node.parent = current
            stack.append(node)
        else:
            if len(stack) > 1:
                stack.pop()
    return root

def compute_diameter(root):
    def bfs(start):
        visited = set()
        queue = deque([(start, 0)])
        max_dist = 0
        farthest_node = start
        while queue:
            node, dist = queue.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest_node = node
            neighbors = []
            if node.parent:
                neighbors.append(node.parent)
            neighbors.extend(node.children)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return farthest_node, max_dist

    if not root or not root.children:
        return 0
    u, _ = bfs(root)
    v, diameter = bfs(u)
    return diameter

def generate_valid_sequence(n):
    """Generate a valid bracket sequence using a random walk approach."""
    s = []
    balance = 0
    max_open = n - 1
    open_count = 0
    for _ in range(2 * (n - 1)):
        if open_count < max_open and (balance == 0 or (random.random() < 0.5)):
            s.append('(')
            open_count += 1
            balance += 1
        else:
            s.append(')')
            balance -= 1
            if balance < 0:  # Should theoretically not happen
                return generate_valid_sequence(n)
    return ''.join(s)

def find_valid_swaps(s):
    """Find all possible swaps of same-type brackets ensuring validity."""
    opens = [i + 1 for i, c in enumerate(s) if c == '(']  # 1-based
    closes = [i + 1 for i, c in enumerate(s) if c == ')']
    valid_swaps = []
    # Generate all possible same-type swaps
    for i in range(len(opens)):
        for j in range(i + 1, len(opens)):
            valid_swaps.append((opens[i], opens[j]))
    for i in range(len(closes)):
        for j in range(i + 1, len(closes)):
            valid_swaps.append((closes[i], closes[j]))
    return valid_swaps

class Ctreegeneratorbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_q=5):
        self.max_n = max_n
        self.max_q = max_q

    def case_generator(self):
        n = random.randint(3, self.max_n)
        initial_seq = generate_valid_sequence(n)
        s_len = 2 * (n - 1)
        
        swaps = []
        current_seq = list(initial_seq)
        for _ in range(self.max_q):
            valid_swaps = find_valid_swaps(current_seq)
            if not valid_swaps:
                break
            a, b = random.choice(valid_swaps)
            swaps.append((a, b))
            # Apply the swap (0-based)
            a_idx = a - 1
            b_idx = b - 1
            current_seq[a_idx], current_seq[b_idx] = current_seq[b_idx], current_seq[a_idx]
        
        q = len(swaps)
        if q == 0:
            return self.case_generator()
        
        # Precompute answers
        current_seq = list(initial_seq)
        answers = []
        # Initial state
        root = parse_bracket_sequence(''.join(current_seq))
        answers.append(compute_diameter(root))
        # Apply swaps and compute answers
        for a, b in swaps:
            a_idx = a - 1
            b_idx = b - 1
            current_seq[a_idx], current_seq[b_idx] = current_seq[b_idx], current_seq[a_idx]
            root = parse_bracket_sequence(''.join(current_seq))
            answers.append(compute_diameter(root))
        
        return {
            'n': n,
            'q': q,
            'initial_seq': initial_seq,
            'swaps': swaps,
            'answers': answers
        }

    @staticmethod
    def prompt_func(question_case):
        prompt = (
            "Owl Pacino is analyzing tree diameters based on bracket sequence swaps. Given the initial sequence and swap operations, output the diameter after each change.\n\n"
            "Input format:\n"
            f"n q\n<s>\n<swap1>\n...\n<swapq>\n\n"
            "Output q+1 integers separated by spaces.\n\n"
            "Example:\n"
            "Input:\n5 5\n(((())))\n4 5\n3 4\n5 6\n3 6\n2 5\n"
            "Output:\n4 3 3 2 4 4\n\n"
            "Problem:\n"
            f"{question_case['n']} {question_case['q']}\n"
            f"{question_case['initial_seq']}\n"
        )
        for a, b in question_case['swaps']:
            prompt += f"{a} {b}\n"
        prompt += "\nPlace your answer within [answer][/answer] as space-separated integers."
        return prompt

    @staticmethod
    def extract_output(output):
        start_tag = '[answer]'
        end_tag = '[/answer]'
        start_idx = output.rfind(start_tag)
        if start_idx == -1:
            return None
        end_idx = output.find(end_tag, start_idx)
        if end_idx == -1:
            return None
        content = output[start_idx + len(start_tag):end_idx].strip()
        try:
            return list(map(int, content.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answers']
