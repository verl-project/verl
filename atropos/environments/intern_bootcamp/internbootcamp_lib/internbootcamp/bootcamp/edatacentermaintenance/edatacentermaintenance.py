"""# 

### 谜题描述
BigData Inc. is a corporation that has n data centers indexed from 1 to n that are located all over the world. These data centers provide storage for client data (you can figure out that client data is really big!).

Main feature of services offered by BigData Inc. is the access availability guarantee even under the circumstances of any data center having an outage. Such a guarantee is ensured by using the two-way replication. Two-way replication is such an approach for data storage that any piece of data is represented by two identical copies that are stored in two different data centers.

For each of m company clients, let us denote indices of two different data centers storing this client data as ci, 1 and ci, 2.

In order to keep data centers operational and safe, the software running on data center computers is being updated regularly. Release cycle of BigData Inc. is one day meaning that the new version of software is being deployed to the data center computers each day.

Data center software update is a non-trivial long process, that is why there is a special hour-long time frame that is dedicated for data center maintenance. During the maintenance period, data center computers are installing software updates, and thus they may be unavailable. Consider the day to be exactly h hours long. For each data center there is an integer uj (0 ≤ uj ≤ h - 1) defining the index of an hour of day, such that during this hour data center j is unavailable due to maintenance.

Summing up everything above, the condition uci, 1 ≠ uci, 2 should hold for each client, or otherwise his data may be unaccessible while data centers that store it are under maintenance.

Due to occasional timezone change in different cities all over the world, the maintenance time in some of the data centers may change by one hour sometimes. Company should be prepared for such situation, that is why they decided to conduct an experiment, choosing some non-empty subset of data centers, and shifting the maintenance time for them by an hour later (i.e. if uj = h - 1, then the new maintenance hour would become 0, otherwise it would become uj + 1). Nonetheless, such an experiment should not break the accessibility guarantees, meaning that data of any client should be still available during any hour of a day after the data center maintenance times are changed.

Such an experiment would provide useful insights, but changing update time is quite an expensive procedure, that is why the company asked you to find out the minimum number of data centers that have to be included in an experiment in order to keep the data accessibility guarantees.

Input

The first line of input contains three integers n, m and h (2 ≤ n ≤ 100 000, 1 ≤ m ≤ 100 000, 2 ≤ h ≤ 100 000), the number of company data centers, number of clients and the day length of day measured in hours. 

The second line of input contains n integers u1, u2, ..., un (0 ≤ uj < h), j-th of these numbers is an index of a maintenance hour for data center j. 

Each of the next m lines contains two integers ci, 1 and ci, 2 (1 ≤ ci, 1, ci, 2 ≤ n, ci, 1 ≠ ci, 2), defining the data center indices containing the data of client i.

It is guaranteed that the given maintenance schedule allows each client to access at least one copy of his data at any moment of day.

Output

In the first line print the minimum possible number of data centers k (1 ≤ k ≤ n) that have to be included in an experiment in order to keep the data available for any client.

In the second line print k distinct integers x1, x2, ..., xk (1 ≤ xi ≤ n), the indices of data centers whose maintenance time will be shifted by one hour later. Data center indices may be printed in any order.

If there are several possible answers, it is allowed to print any of them. It is guaranteed that at there is at least one valid choice of data centers.

Examples

Input

3 3 5
4 4 0
1 3
3 2
3 1


Output

1
3 

Input

4 5 4
2 1 0 3
4 3
3 2
1 2
1 4
1 3


Output

4
1 2 3 4 

Note

Consider the first sample test. The given answer is the only way to conduct an experiment involving the only data center. In such a scenario the third data center has a maintenance during the hour 1, and no two data centers storing the information of the same client have maintenance at the same hour.

On the other hand, for example, if we shift the maintenance time on hour later for the first data center, then the data of clients 1 and 3 will be unavailable during the hour 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <typename type = int>
using Graph = vector<vector<type>>;
int n, m, h;
Graph<int> g;
vector<int> a;
int cur_num, num_scc;
vector<bool> visited, inStack;
vector<int> num, low, scc_num;
stack<int> s;
Graph<int> scc_to_node;
void initialize() {
  cur_num = num_scc = 0;
  visited.assign(n, false);
  inStack.assign(n, false);
  num.assign(n, 0);
  low.assign(n, 0);
  scc_num.assign(n, 0);
}
void visit(int u) {
  visited[u] = true;
  num[u] = low[u] = cur_num++;
  s.push(u);
  inStack[u] = true;
  for (int v : g[u]) {
    if (!visited[v]) {
      visit(v);
      low[u] = min(low[u], low[v]);
    } else if (inStack[v]) {
      low[u] = min(low[u], low[v]);
    }
  }
  if (num[u] == low[u]) {
    int v;
    do {
      v = s.top();
      s.pop();
      inStack[v] = false;
      scc_num[v] = num_scc;
    } while (u != v);
    num_scc++;
  }
}
void tarjan() {
  initialize();
  for (int u = 0; u < n; ++u) {
    if (!visited[u]) {
      visit(u);
    }
  }
}
Graph<int> contractSCC() {
  vector<set<int>> tmp_g(num_scc, set<int>());
  scc_to_node.assign(num_scc, vector<int>());
  for (int u = 0; u < n; ++u) {
    int scc_u = scc_num[u];
    scc_to_node[scc_u].push_back(u);
    for (int v : g[u]) {
      int scc_v = scc_num[v];
      if (scc_u != scc_v) tmp_g[scc_u].insert(scc_v);
    }
  }
  Graph<int> contraction(num_scc, vector<int>());
  for (int u = 0; u < num_scc; ++u) {
    for (int v : tmp_g[u]) {
      contraction[u].push_back(v);
    }
  }
  return contraction;
}
int main() {
  cin >> n >> m >> h;
  g.assign(n, vector<int>());
  a.assign(n, 0);
  for (int i = 0; i < n; ++i) cin >> a[i];
  for (int i = 0; i < m; ++i) {
    int x, y;
    cin >> x >> y;
    x--;
    y--;
    if ((a[x] + 1) % h == a[y]) {
      g[x].push_back(y);
    }
    if ((a[y] + 1) % h == a[x]) {
      g[y].push_back(x);
    }
  }
  tarjan();
  Graph<int> c = contractSCC();
  int min_scc = -1;
  int min_size = INT_MAX;
  for (int u = 0; u < num_scc; ++u) {
    if (c[u].size() == 0) {
      if (min_size > scc_to_node[u].size()) {
        min_scc = u;
        min_size = scc_to_node[u].size();
      }
    }
  }
  cout << min_size << endl;
  for (int i = 0; i < min_size; ++i) {
    cout << (scc_to_node[min_scc][i] + 1) << (i == min_size - 1 ? \"\n\" : \" \");
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Edatacentermaintenancebootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=10, max_h=24):
        """
        初始化训练场参数，设定数据中心和客户的最大数量及最大小时数
        """
        self.max_n = max_n
        self.max_m = max_m
        self.max_h = max_h

    def case_generator(self):
        """
        生成符合规范的谜题实例，保证至少存在一个有效解
        """
        while True:
            n = random.randint(2, self.max_n)
            m = random.randint(1, self.max_m)
            h = random.randint(2, self.max_h)
            
            # 随机选择至少一个数据中心作为解
            k = random.randint(1, n)
            solution = random.sample(range(1, n+1), k)
            
            # 生成移位后的维护时间（虚拟解）
            v = [random.randint(0, h-1) for _ in range(n)]
            
            # 生成原始维护时间
            u = []
            for i in range(n):
                if (i+1) in solution:
                    u_i = (v[i] - 1) % h
                else:
                    u_i = v[i]
                u.append(u_i)
            
            # 生成有效的客户对
            clients = []
            valid = True
            for _ in range(m):
                retries = 50
                while retries > 0:
                    c1, c2 = random.sample(range(1, n+1), 2)
                    idx1, idx2 = c1-1, c2-1
                    
                    # 检查原始时间冲突
                    if u[idx1] == u[idx2]:
                        retries -= 1
                        continue
                    
                    # 检查解的有效性
                    shift1 = v[idx1] if c1 not in solution else (u[idx1]+1)%h
                    shift2 = v[idx2] if c2 not in solution else (u[idx2]+1)%h
                    if shift1 != shift2:
                        clients.append((c1, c2))
                        break
                    retries -= 1
                else:
                    valid = False
                    break
            
            if valid and len(clients) == m:
                return {
                    'n': n,
                    'm': m,
                    'h': h,
                    'u': u,
                    'clients': clients
                }

    @staticmethod
    def prompt_func(case) -> str:
        problem_desc = f"""## 数据中心维护调度问题

### 背景描述
BigData Inc. 拥有{case['n']}个数据中心（编号1-{case['n']}）和{case['m']}个客户。每个数据中心每天有一个维护时间段（0-{case['h']-1}小时）。客户数据存储在两个不同数据中心，要求这些中心的维护时间在调整后必须不同。

### 任务描述
你需要选择最少数量的数据中心进行维护时间调整（时间+1小时模{case['h']}），使得所有客户的可用性得到保证。

### 输入数据
- 数据中心维护时间：{' '.join(map(str, case['u']))}
- 客户数据存储对：\n""" 
        
        client_pairs = '\n'.join([f"{c1} {c2}" for c1, c2 in case['clients']])
        format_guidance = """
### 输出要求
1. 第一行为调整的数据中心数量k
2. 第二行包含k个不同的数据中心编号

请将答案放置在[answer]和[/answer]标签之间，例如：
[answer]
2
3 5
[/answer]"""
        
        return problem_desc + client_pairs + format_guidance

    @staticmethod
    def extract_output(output):
        import re
        # 查找所有匹配的答案块并取最后一个
        matches = list(re.finditer(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL))
        if not matches:
            return None
        last_match = matches[-1]
        content = last_match.group(1).strip()
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        
        try:
            if len(lines) < 2:
                return None
            k = int(lines[0])
            centers = list(map(int, lines[1].split()))
            if len(centers) == k and len(set(centers)) == k:
                return centers
        except:
            pass
        return None

    @classmethod
    def _verify_correction(cls, solution, case):
        if not solution:
            return False
            
        h = case['h']
        u = case['u']
        solution_set = set(solution)
        
        for c1, c2 in case['clients']:
            # 计算调整后的时间
            t1 = (u[c1-1] + (1 if c1 in solution_set else 0)) % h
            t2 = (u[c2-1] + (1 if c2 in solution_set else 0)) % h
            if t1 == t2:
                return False
        return True
