"""# 

### 谜题描述
Logical quantifiers are very useful tools for expressing claims about a set. For this problem, let's focus on the set of real numbers specifically. The set of real numbers includes zero and negatives. There are two kinds of quantifiers: universal (∀) and existential (∃). You can read more about them here.

The universal quantifier is used to make a claim that a statement holds for all real numbers. For example:

  * ∀ x,x<100 is read as: for all real numbers x, x is less than 100. This statement is false. 
  * ∀ x,x>x-1 is read as: for all real numbers x, x is greater than x-1. This statement is true. 



The existential quantifier is used to make a claim that there exists some real number for which the statement holds. For example:

  * ∃ x,x<100 is read as: there exists a real number x such that x is less than 100. This statement is true. 
  * ∃ x,x>x-1 is read as: there exists a real number x such that x is greater than x-1. This statement is true. 



Moreover, these quantifiers can be nested. For example:

  * ∀ x,∃ y,x<y is read as: for all real numbers x, there exists a real number y such that x is less than y. This statement is true since for every x, there exists y=x+1. 
  * ∃ y,∀ x,x<y is read as: there exists a real number y such that for all real numbers x, x is less than y. This statement is false because it claims that there is a maximum real number: a number y larger than every x. 



Note that the order of variables and quantifiers is important for the meaning and veracity of a statement.

There are n variables x_1,x_2,…,x_n, and you are given some formula of the form $$$ f(x_1,...,x_n):=(x_{j_1}<x_{k_1})∧ (x_{j_2}<x_{k_2})∧ ⋅⋅⋅∧ (x_{j_m}<x_{k_m}), $$$

where ∧ denotes logical AND. That is, f(x_1,…, x_n) is true if every inequality x_{j_i}<x_{k_i} holds. Otherwise, if at least one inequality does not hold, then f(x_1,…,x_n) is false.

Your task is to assign quantifiers Q_1,…,Q_n to either universal (∀) or existential (∃) so that the statement $$$ Q_1 x_1, Q_2 x_2, …, Q_n x_n, f(x_1,…, x_n) $$$

is true, and the number of universal quantifiers is maximized, or determine that the statement is false for every possible assignment of quantifiers.

Note that the order the variables appear in the statement is fixed. For example, if f(x_1,x_2):=(x_1<x_2) then you are not allowed to make x_2 appear first and use the statement ∀ x_2,∃ x_1, x_1<x_2. If you assign Q_1=∃ and Q_2=∀, it will only be interpreted as ∃ x_1,∀ x_2,x_1<x_2.

Input

The first line contains two integers n and m (2≤ n≤ 2⋅ 10^5; 1≤ m≤ 2⋅ 10^5) — the number of variables and the number of inequalities in the formula, respectively.

The next m lines describe the formula. The i-th of these lines contains two integers j_i,k_i (1≤ j_i,k_i≤ n, j_i≠ k_i).

Output

If there is no assignment of quantifiers for which the statement is true, output a single integer -1.

Otherwise, on the first line output an integer, the maximum possible number of universal quantifiers.

On the next line, output a string of length n, where the i-th character is \"A\" if Q_i should be a universal quantifier (∀), or \"E\" if Q_i should be an existential quantifier (∃). All letters should be upper-case. If there are multiple solutions where the number of universal quantifiers is maximum, print any.

Examples

Input


2 1
1 2


Output


1
AE


Input


4 3
1 2
2 3
3 1


Output


-1


Input


3 2
1 3
2 3


Output


2
AAE

Note

For the first test, the statement ∀ x_1, ∃ x_2, x_1<x_2 is true. Answers of \"EA\" and \"AA\" give false statements. The answer \"EE\" gives a true statement, but the number of universal quantifiers in this string is less than in our answer.

For the second test, we can show that no assignment of quantifiers, for which the statement is true exists.

For the third test, the statement ∀ x_1, ∀ x_2, ∃ x_3, (x_1<x_3)∧ (x_2<x_3) is true: We can set x_3=max\\{x_1,x_2\}+1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 10;
const int INF = 0x3f3f3f3f;
const long long inf = 0x3f3f3f3f3f3f3f3f;
int n, m;
vector<int> G[N], GG[N];
int vis[N], mn[N], mn2[N];
bool dfs(int u) {
  vis[u] = 1;
  mn[u] = u;
  for (int v : G[u]) {
    if (vis[v] == 1) return false;
    if (!vis[v]) {
      if (!dfs(v)) return false;
    }
    mn[u] = min(mn[u], mn[v]);
  }
  vis[u] = 2;
  return true;
}
bool dfs2(int u) {
  vis[u] = 1;
  mn2[u] = u;
  for (int v : GG[u]) {
    if (vis[v] == 1) return false;
    if (!vis[v]) {
      if (!dfs2(v)) return false;
    }
    mn2[u] = min(mn2[u], mn2[v]);
  }
  vis[u] = 2;
  return true;
}
int main() {
  scanf(\"%d%d\", &n, &m);
  for (int i = 1; i <= m; i++) {
    int u, v;
    scanf(\"%d%d\", &u, &v);
    G[v].push_back(u);
    GG[u].push_back(v);
  }
  for (int i = 1; i <= n; i++)
    if (!vis[i]) {
      if (!dfs(i)) {
        puts(\"-1\");
        return 0;
      }
    }
  memset(vis, 0, sizeof(vis));
  for (int i = 1; i <= n; i++)
    if (!vis[i]) {
      if (!dfs2(i)) {
        puts(\"-1\");
        return 0;
      }
    }
  int ans = 0;
  for (int i = 1; i <= n; i++)
    if (mn[i] >= i && mn2[i] >= i) ans++;
  printf(\"%d\n\", ans);
  for (int i = 1; i <= n; i++)
    printf(\"%c\", mn[i] >= i && mn2[i] >= i ? 'A' : 'E');
  puts(\"\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Equantifierquestionbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10, m_min=1, m_max=20):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, min(self.m_max, n * (n - 1)))
        
        inequalities = []
        for _ in range(m):
            j = random.randint(1, n)
            k = random.randint(1, n)
            while j == k:
                k = random.randint(1, n)
            inequalities.append((j, k))
        
        G = [[] for _ in range(n + 1)]
        GG = [[] for _ in range(n + 1)]
        for j, k in inequalities:
            G[k].append(j)
            GG[j].append(k)
        
        vis = [0] * (n + 1)
        mn = [0] * (n + 1)
        has_cycle = False
        
        def dfs(u):
            nonlocal has_cycle
            if vis[u] == 1:
                has_cycle = True
                return
            if vis[u] == 2:
                return
            vis[u] = 1
            mn_u = u
            for v in G[u]:
                dfs(v)
                if has_cycle:
                    return
                mn_u = min(mn_u, mn[v])
            mn[u] = mn_u
            vis[u] = 2
        
        for i in range(1, n + 1):
            if vis[i] == 0:
                dfs(i)
                if has_cycle:
                    break
        
        if has_cycle:
            return {'n': n, 'm': m, 'inequalities': inequalities, 'answer': -1}
        
        vis2 = [0] * (n + 1)
        mn2 = [0] * (n + 1)
        has_cycle2 = False
        
        def dfs2(u):
            nonlocal has_cycle2
            if vis2[u] == 1:
                has_cycle2 = True
                return
            if vis2[u] == 2:
                return
            vis2[u] = 1
            mn2_u = u
            for v in GG[u]:
                dfs2(v)
                if has_cycle2:
                    return
                mn2_u = min(mn2_u, mn2[v])
            mn2[u] = mn2_u
            vis2[u] = 2
        
        for i in range(1, n + 1):
            if vis2[i] == 0:
                dfs2(i)
                if has_cycle2:
                    break
        
        if has_cycle2:
            return {'n': n, 'm': m, 'inequalities': inequalities, 'answer': -1}
        
        ans = []
        max_universal = 0
        for i in range(1, n + 1):
            if mn[i] >= i and mn2[i] >= i:
                ans.append('A')
                max_universal += 1
            else:
                ans.append('E')
        ans_str = ''.join(ans)
        
        return {
            'n': n,
            'm': m,
            'inequalities': inequalities,
            'answer': ans_str,
            'max_universal': max_universal
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        inequalities = question_case['inequalities']
        
        ineq_text = []
        for j, k in inequalities:
            ineq_text.append(f"x_{j} < x_{k}")
        ineq_text_str = " ∧ ".join(ineq_text)
        
        prompt = f"""
        你是一位逻辑学专家，现在需要解决一个关于量词分配的问题。给定{n}个变量和{m}个不等式，每个不等式是x_j < x_k的形式。你的任务是为每个变量分配量词∀或∃，使得整个命题为真，并且尽可能多地使用∀。如果无解，输出-1。

        具体的不等式如下：
        f(x_1, ..., x_{n}) = {ineq_text_str}

        请输出你的解决方案，将所有变量的量词分配写在一个字符串中，例如'AAE'，并放在[answer][/answer]标签中。
        """
        return prompt.strip()
    
    @staticmethod
    def extract_output(output):
        start = output.rfind('[answer]')
        if start == -1:
            return None
        end = output.find('[/answer]', start)
        if end == -1:
            return None
        answer = output[start + len('[answer]'):end].strip().upper()
        return answer
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        if len(solution) != identity['n']:
            return False
        if not all(c in {'A', 'E'} for c in solution):
            return False
        return solution == identity['answer']
