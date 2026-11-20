"""# 

### 谜题描述
Shichikuji is the new resident deity of the South Black Snail Temple. Her first job is as follows:

There are n new cities located in Prefecture X. Cities are numbered from 1 to n. City i is located x_i km North of the shrine and y_i km East of the shrine. It is possible that (x_i, y_i) = (x_j, y_j) even when i ≠ j.

Shichikuji must provide electricity to each city either by building a power station in that city, or by making a connection between that city and another one that already has electricity. So the City has electricity if it has a power station in it or it is connected to a City which has electricity by a direct connection or via a chain of connections.

  * Building a power station in City i will cost c_i yen; 
  * Making a connection between City i and City j will cost k_i + k_j yen per km of wire used for the connection. However, wires can only go the cardinal directions (North, South, East, West). Wires can cross each other. Each wire must have both of its endpoints in some cities. If City i and City j are connected by a wire, the wire will go through any shortest path from City i to City j. Thus, the length of the wire if City i and City j are connected is |x_i - x_j| + |y_i - y_j| km. 



Shichikuji wants to do this job spending as little money as possible, since according to her, there isn't really anything else in the world other than money. However, she died when she was only in fifth grade so she is not smart enough for this. And thus, the new resident deity asks for your help.

And so, you have to provide Shichikuji with the following information: minimum amount of yen needed to provide electricity to all cities, the cities in which power stations will be built, and the connections to be made.

If there are multiple ways to choose the cities and the connections to obtain the construction of minimum price, then print any of them.

Input

First line of input contains a single integer n (1 ≤ n ≤ 2000) — the number of cities.

Then, n lines follow. The i-th line contains two space-separated integers x_i (1 ≤ x_i ≤ 10^6) and y_i (1 ≤ y_i ≤ 10^6) — the coordinates of the i-th city.

The next line contains n space-separated integers c_1, c_2, ..., c_n (1 ≤ c_i ≤ 10^9) — the cost of building a power station in the i-th city.

The last line contains n space-separated integers k_1, k_2, ..., k_n (1 ≤ k_i ≤ 10^9).

Output

In the first line print a single integer, denoting the minimum amount of yen needed.

Then, print an integer v — the number of power stations to be built.

Next, print v space-separated integers, denoting the indices of cities in which a power station will be built. Each number should be from 1 to n and all numbers should be pairwise distinct. You can print the numbers in arbitrary order.

After that, print an integer e — the number of connections to be made.

Finally, print e pairs of integers a and b (1 ≤ a, b ≤ n, a ≠ b), denoting that a connection between City a and City b will be made. Each unordered pair of cities should be included at most once (for each (a, b) there should be no more (a, b) or (b, a) pairs). You can print the pairs in arbitrary order.

If there are multiple ways to choose the cities and the connections to obtain the construction of minimum price, then print any of them.

Examples

Input


3
2 3
1 1
3 2
3 2 3
3 2 3


Output


8
3
1 2 3 
0


Input


3
2 1
1 2
3 3
23 2 23
3 2 3


Output


27
1
2 
2
1 2
2 3

Note

For the answers given in the samples, refer to the following diagrams (cities with power stations are colored green, other cities are colored blue, and wires are colored red):

<image>

For the first example, the cost of building power stations in all cities is 3 + 2 + 3 = 8. It can be shown that no configuration costs less than 8 yen.

For the second example, the cost of building a power station in City 2 is 2. The cost of connecting City 1 and City 2 is 2 ⋅ (3 + 2) = 10. The cost of connecting City 2 and City 3 is 3 ⋅ (2 + 3) = 15. Thus the total cost is 2 + 10 + 15 = 27. It can be shown that no configuration costs less than 27 yen.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

def mst(mat):
    n = len(mat)
    root = 0
    found = [0]*n
    found[root] = 1
    vec = list(mat[root])
    who = [root]*n
    pairs = []
    for _ in range(n - 1):
        i = min((i for i in range(n) if not found[i]), key = vec.__getitem__)
        pairs.append((who[i], i))
        found[i] = 1
        vec2 = mat[i]
        for j in range(n):
            if vec2[j] < vec[j]:
                vec[j] = vec2[j]
                who[j] = i
    return pairs


inp = [int(x) for x in sys.stdin.read().split()]; ii = 0

n = inp[ii]; ii += 1

X = inp[ii:ii + 2*n: 2]
Y = inp[ii + 1: ii + 2*n: 2]
ii += 2 * n

C = [float(c) for c in inp[ii:ii + n]]; ii += n
K = [float(c) for c in inp[ii:ii + n]]; ii += n

mat = [[0.0]*(n + 1) for _ in range(n + 1)]
for a in range(n):
    for b in range(a + 1, n):
        val = (K[a] + K[b]) * (abs(X[a] - X[b]) + abs(Y[a] - Y[b]))
        mat[a][b] = mat[b][a] = val

for i in range(n):
    mat[i][n] = mat[n][i] = C[i]

cost = 0
power = []
pairs = []
for i,j in mst(mat):
    if i > j:
        i,j = j,i
    cost += mat[i][j]
    if j == n:
        power.append(i)
    else:
        pairs.append((i,j))

print int(cost)
print len(power)
print ' '.join(str(x + 1) for x in power)
print len(pairs)
print '\n' .join('%d %d' % (a+1,b+1) for a,b in pairs)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Dshichikujiandpowergridbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 3)
        self.max_coord = params.get('max_coord', 10**6)
        self.min_c = params.get('min_c', 1)
        self.max_c = params.get('max_c', 10**9)
        self.min_k = params.get('min_k', 1)
        self.max_k = params.get('max_k', 10**9)
    
    def case_generator(self):
        n = self.n
        x = [random.randint(1, self.max_coord) for _ in range(n)]
        y = [random.randint(1, self.max_coord) for _ in range(n)]
        c = [random.randint(self.min_c, self.max_c) for _ in range(n)]
        k = [random.randint(self.min_k, self.max_k) for _ in range(n)]

        n_cities = n
        mat = [[0]*(n_cities +1) for _ in range(n_cities +1)]
        for a in range(n_cities):
            for b in range(a+1, n_cities):
                dx = abs(x[a] - x[b])
                dy = abs(y[a] - y[b])
                val = (k[a] + k[b]) * (dx + dy)
                mat[a][b] = mat[b][a] = val
        for i in range(n_cities):
            mat[i][n_cities] = mat[n_cities][i] = c[i]

        # MST function from reference code
        def mst(mat):
            n_nodes = len(mat)
            root = 0
            found = [False]*n_nodes
            found[root] = True
            vec = mat[root][:]
            who = [root]*n_nodes
            pairs = []
            for _ in range(n_nodes -1):
                min_val = float('inf')
                min_i = -1
                for i in range(n_nodes):
                    if not found[i] and vec[i] < min_val:
                        min_val = vec[i]
                        min_i = i
                i = min_i
                if i == -1:
                    break  # handle disconnected graph, though problem ensures connectivity
                pairs.append((who[i], i))
                found[i] = True
                for j in range(n_nodes):
                    if not found[j] and mat[i][j] < vec[j]:
                        vec[j] = mat[i][j]
                        who[j] = i
            return pairs

        pairs = mst(mat)

        cost = 0
        power_stations = []
        connections = []
        for i, j in pairs:
            if i > j:
                i, j = j, i
            cost += mat[i][j]
            if j == n_cities:
                power_stations.append(i)
            else:
                connections.append((i, j))

        identity = {
            'n': n_cities,
            'x': x,
            'y': y,
            'c': c,
            'k': k,
            'correct_cost': int(cost),
            'correct_power_stations': power_stations,
            'correct_connections': connections
        }
        return identity
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        x = question_case['x']
        y = question_case['y']
        c = question_case['c']
        k = question_case['k']

        problem_input = f"{n}\n"
        for xi, yi in zip(x, y):
            problem_input += f"{xi} {yi}\n"
        problem_input += ' '.join(map(str, c)) + "\n"
        problem_input += ' '.join(map(str, k)) + "\n"

        prompt = f'''Shichikuji needs to provide electricity to all cities with minimal cost. Cities can have power stations or be connected via wires. Wire costs depend on Manhattan distance and city parameters.

Input:
{problem_input}
Output the minimal cost, power stations, and connections as specified. Format your answer within [answer]...[/answer] tags. Example:

[answer]
<total_cost>
<v>
<stations>
<e>
<connections>
[/answer]'''

        return prompt.strip()
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_text = matches[-1].strip()
        lines = [line.strip() for line in answer_text.split('\n') if line.strip()]
        
        if len(lines) < 4:
            return None
        try:
            total_cost = int(lines[0])
            v = int(lines[1])
            if v < 0 or len(lines) < 2 + 1 + 1:
                return None
            power = list(map(int, lines[2].split()))
            if len(power) != v:
                return None
            e = int(lines[3])
            if len(lines) < 4 + e:
                return None
            conns = []
            for i in range(e):
                a, b = map(int, lines[4+i].split())
                conns.append((a, b))
            return {
                'total_cost': total_cost,
                'power_stations': power,
                'connections': conns
            }
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            if solution['total_cost'] != identity['correct_cost']:
                return False

            n = identity['n']
            x = identity['x']
            y = identity['y']
            c = identity['c']
            k = identity['k']

            # Check power stations validity
            power = solution['power_stations']
            if len(power) != len(set(power)) or any(not 1 <= city <= n for city in power):
                return False

            # Check connection validity
            seen = set()
            conns = solution['connections']
            for a, b in conns:
                if a == b or not (1 <= a <= n) or not (1 <= b <= n) or tuple(sorted((a, b))) in seen:
                    return False
                seen.add(tuple(sorted((a, b))))

            # Calculate cost
            c_total = sum(c[i-1] for i in power)
            k_total = 0
            for a, b in conns:
                dx = abs(x[a-1] - x[b-1])
                dy = abs(y[a-1] - y[b-1])
                k_total += (k[a-1] + k[b-1]) * (dx + dy)
            if c_total + k_total != solution['total_cost']:
                return False

            # Check connectivity
            adj = [[] for _ in range(n+1)]
            for a, b in conns:
                adj[a].append(b)
                adj[b].append(a)
            visited = set(power)
            queue = deque(power)
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            return len(visited) == n
        except:
            return False
