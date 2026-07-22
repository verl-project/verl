"""# 

### 谜题描述
Malek is a rich man. He also is very generous. That's why he decided to split his money between poor people. A charity institute knows n poor people numbered from 1 to n. The institute gave Malek q recommendations. A recommendation is a segment of people like [l, r] which means the institute recommended that Malek gives one dollar to every person whose number is in this segment.

However this charity has very odd rules about the recommendations. Because of those rules the recommendations are given in such a way that for every two recommendation [a, b] and [c, d] one of the following conditions holds: 

  * The two segments are completely disjoint. More formally either a ≤ b < c ≤ d or c ≤ d < a ≤ b
  * One of the two segments are inside another. More formally either a ≤ c ≤ d ≤ b or c ≤ a ≤ b ≤ d. 



The goodness of a charity is the value of maximum money a person has after Malek finishes giving his money. The institute knows for each recommendation what is the probability that Malek will accept it. They want to know the expected value of goodness of this charity. So they asked you for help.

You have been given the list of recommendations and for each recommendation the probability of it being accepted by Malek. You have also been given how much money each person initially has. You must find the expected value of goodness.

Input

In the first line two space-separated integers n, q (1 ≤ n ≤ 105, 1 ≤ q ≤ 5000) are given.

In the second line n space-separated integers a1, a2, ..., an (0 ≤ ai ≤ 109) are given meaning that person number i initially has ai dollars. 

Each of the next q lines contains three space-separated numbers li, ri, pi (1 ≤ li ≤ ri ≤ n, 0 ≤ p ≤ 1) where li and ri are two integers describing the segment of recommendation and pi is a real number given with exactly three digits after decimal point which is equal to probability of Malek accepting this recommendation.

Note that a segment may appear several times in recommendations.

Output

Output the sought value. Your answer will be considered correct if its absolute or relative error is less than 10 - 6.

Examples

Input

5 2
1 7 2 4 3
1 3 0.500
2 2 0.500


Output

8.000000000


Input

5 2
281 280 279 278 282
1 4 1.000
1 4 0.000


Output

282.000000000


Input

3 5
1 2 3
1 3 0.500
2 2 0.250
1 2 0.800
1 1 0.120
2 2 0.900


Output

4.465000000

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
template <class T1, class T2, class Pred = std::less<T2> >
struct sort_pair_second {
  bool operator()(const std::pair<T1, T2> &left,
                  const std::pair<T1, T2> &right) {
    Pred p;
    return p(left.second, right.second);
  }
};
using namespace std;
struct sol_t {
  int start, end;
  vector<double> p;
  vector<double> psum;
  sol_t() {}
  sol_t(int v) {
    start = end = v;
    p.push_back(1.0);
    psum.push_back(1.0);
  }
  sol_t(sol_t &s1, sol_t &s2) {
    start = max(s1.start, s2.start);
    end = max(s1.end, s2.end);
    p = vector<double>(end - start + 1);
    psum = vector<double>(end - start + 1);
    for (int v = start; v <= end; v++)
      p[v - start] = s1.fetchp(v) * s2.fetchpsum(v - 1) +
                     s1.fetchpsum(v - 1) * s2.fetchp(v) +
                     s1.fetchp(v) * s2.fetchp(v);
    update_psum();
  }
  double fetchp(int v) {
    if (!(start <= v && v <= end)) return 0.0;
    return p[v - start];
  }
  double fetchpsum(int v) {
    if (v < start) return 0.0;
    if (v > end) return 1.0;
    return psum[v - start];
  }
  void update(double prob) {
    end++;
    p.push_back(0.0);
    psum.push_back(-1);
    for (int v = end - 1; v >= start; v--) {
      p[v + 1 - start] += p[v - start] * prob;
      p[v - start] *= (1 - prob);
    }
    update_psum();
  }
  void update_psum() {
    for (int v = start; v <= end; v++) {
      psum[v - start] = p[v - start];
      if (v - 1 >= start) psum[v - start] += psum[v - 1 - start];
    }
  }
  double expectation() {
    double result = 0.0;
    for (int v = start; v <= end; v++) {
      result += v * p[v - start];
    }
    return result;
  }
};
int values[100000];
struct range_t {
  int a, b;
  double prob;
  int maxv;
  vector<int> child;
  sol_t solution;
  bool operator<(const range_t &other) const {
    if (a != other.a) return a < other.a;
    return (b - a) > (other.b - other.a);
  }
  void populate(int &pos, int nr);
  void solve();
} range[5000 + 1];
void range_t::populate(int &pos, int nr) {
  while (pos < nr && range[pos].b <= this->b) {
    this->child.push_back(pos);
    pos++;
    range[pos - 1].populate(pos, nr);
  }
}
void range_t::solve() {
  maxv = -1000000000;
  if (child.empty()) {
    for (int i = a; i <= b; i++) maxv = max(maxv, values[i]);
  } else {
    for (int i = a; i <= range[child.front()].a - 1; i++)
      maxv = max(maxv, values[i]);
    for (int k = 0; k + 1 < (int)child.size(); k++)
      for (int i = range[child[k]].b + 1; i <= range[child[k + 1]].a - 1; i++)
        maxv = max(maxv, values[i]);
    for (int i = range[child.back()].b + 1; i <= b; i++)
      maxv = max(maxv, values[i]);
  }
  solution = sol_t(maxv);
  for (int c : child) {
    range[c].solve();
    solution = sol_t(solution, range[c].solution);
  }
  solution.update(prob);
}
int main() {
  int n, nr;
  scanf(\"%d %d\", &n, &nr);
  for (int i = 0; i < n; i++) scanf(\"%d\", &values[i]);
  for (int r = 0; r < nr; r++)
    scanf(\"%d %d %lf\", &range[r].a, &range[r].b, &range[r].prob), range[r].a--,
        range[r].b--;
  range[nr].a = 0, range[nr].b = n - 1, range[nr].prob = 0.0;
  nr++;
  sort(range, range + nr);
  int pos = 1;
  range[0].populate(pos, nr);
  range[0].solve();
  printf(\"%.8f\n\", range[0].solution.expectation());
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ehelpingpeoplebootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_q=3, max_initial=100):
        self.max_n = max_n
        self.max_q = max_q
        self.max_initial = max_initial
    
    def case_generator(self):
        class RangeNode:
            def __init__(self, l, r, prob):
                self.l = l
                self.r = r
                self.prob = round(prob, 3)
                self.children = []
        
        def build_hierarchy(n, q):
            root = RangeNode(1, n, 0.0)
            nodes = [root]
            
            def add_children(parent, depth):
                if len(nodes) >= q or depth > 3:
                    return
                
                # Generate contained intervals
                if random.random() < 0.7:
                    cl = random.randint(parent.l, parent.r)
                    cr = random.randint(cl, parent.r)
                    if cr - cl >= 1:
                        child = RangeNode(cl, cr, random.uniform(0, 1))
                        parent.children.append(child)
                        nodes.append(child)
                        add_children(child, depth+1)
                
                # Generate adjacent intervals
                if random.random() < 0.3 and parent.r < n:
                    split = random.randint(parent.l, parent.r)
                    if split < parent.r:
                        left = RangeNode(parent.l, split, random.uniform(0, 1))
                        right = RangeNode(split+1, parent.r, random.uniform(0, 1))
                        parent.children.extend([left, right])
                        nodes.extend([left, right])
                        add_children(left, depth+1)
                        add_children(right, depth+1)
            
            add_children(root, 0)
            return nodes[:q]
        
        while True:
            try:
                n = random.randint(3, self.max_n)
                q = random.randint(2, self.max_q)
                a = [random.randint(0, self.max_initial) for _ in range(n)]
                nodes = build_hierarchy(n, q)
                
                # Validate hierarchy structure
                stack = [(nodes[0].l, nodes[0].r)]
                for node in nodes[1:]:
                    while stack and not (stack[-1][0] <= node.l and node.r <= stack[-1][1]):
                        stack.pop()
                    if not stack:
                        raise ValueError("Invalid interval structure")
                    stack.append((node.l, node.r))
                
                recommendations = [(node.l, node.r, node.prob) for node in nodes[1:]]  # Skip root
                expected = self._calculate(n, a, recommendations)
                return {
                    "n": n,
                    "q": len(recommendations),
                    "a": a,
                    "recommendations": recommendations,
                    "correct_output": expected
                }
            except:
                continue

    def _calculate(self, n, a, recommendations):
        class Segment:
            def __init__(self, a, b, p):
                self.a = a-1  # 0-based
                self.b = b-1
                self.p = p
                self.children = []
                self.maxv = 0
                self.dist = {}

            def solve(self, values):
                # Calculate base maximum
                self.maxv = max(values[self.a:self.b+1])
                
                # Process children
                prev_end = self.a-1
                for child in self.children:
                    # Left gap
                    if prev_end+1 <= child.a-1:
                        self.maxv = max(self.maxv, max(values[prev_end+1:child.a]))
                    # Child's maximum (after solving)
                    child.solve(values)
                    self.maxv = max(self.maxv, child.maxv)
                    prev_end = child.b
                # Right gap
                if prev_end+1 <= self.b:
                    self.maxv = max(self.maxv, max(values[prev_end+1:self.b+1]))
                
                # Initialize distribution
                self.dist = {self.maxv: 1.0}
                
                # Merge children distributions
                for child in self.children:
                    new_dist = {}
                    for k1, p1 in self.dist.items():
                        for k2, p2 in child.dist.items():
                            key = max(k1, k2)
                            prob = p1 * p2
                            new_dist[key] = new_dist.get(key, 0.0) + prob
                    self.dist = new_dist
                
                # Apply current probability
                if self.p > 0:
                    new_dist = {}
                    for k, p in self.dist.items():
                        new_dist[k+1] = new_dist.get(k+1, 0.0) + p * self.p
                        new_dist[k] = new_dist.get(k, 0.0) + p * (1 - self.p)
                    self.dist = new_dist
                    self.maxv += 1

        # Build interval tree
        segs = [Segment(1, n, 0.0)] + [Segment(l, r, p) for l, r, p in recommendations]
        segs.sort(key=lambda x: (x.a, -(x.b - x.a)))
        
        # Build hierarchy
        stack = [segs[0]]
        for s in segs[1:]:
            while stack and not (stack[-1].a <= s.a and s.b <= stack[-1].b):
                stack.pop()
            if stack:
                stack[-1].children.append(s)
            stack.append(s)
        
        # Solve root
        segs[0].solve(a)
        expectation = sum(k * p for k, p in segs[0].dist.items())
        return expectation

    @staticmethod
    def prompt_func(case):
        input_str = f"{case['n']} {case['q']}\n"
        input_str += " ".join(map(str, case["a"])) + "\n"
        for l, r, p in case["recommendations"]:
            input_str += f"{l} {r} {p:.3f}\n"
        return f"""Calculate the expected maximum money value with exact formatting. Enclose your final answer in [answer] and [/answer].

Input:
{input_str.strip()}

Requirements:
1. Output must have exactly 9 decimal places
2. Use format [answer]X.xxxxxxxxx[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r"\[answer\](.*?)\[/answer\]", output, re.DOTALL)
        if not matches:
            return None
        try:
            return float(matches[-1].strip().split()[0])
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, case):
        if solution is None:
            return False
        expected = case["correct_output"]
        abs_err = abs(solution - expected)
        if abs_err < 1e-6:
            return True
        if expected == 0:
            return abs_err == 0
        rel_err = abs_err / abs(expected)
        return rel_err < 1e-6
