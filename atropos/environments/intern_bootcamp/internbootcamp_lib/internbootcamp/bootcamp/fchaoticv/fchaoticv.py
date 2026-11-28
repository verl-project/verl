"""# 

### 谜题描述
[Æsir - CHAOS](https://soundcloud.com/kivawu/aesir-chaos)

[Æsir - V.](https://soundcloud.com/kivawu/aesir-v)

\"Everything has been planned out. No more hidden concerns. The condition of Cytus is also perfect.

The time right now...... 00:01:12......

It's time.\"

The emotion samples are now sufficient. After almost 3 years, it's time for Ivy to awake her bonded sister, Vanessa.

The system inside A.R.C.'s Library core can be considered as an undirected graph with infinite number of processing nodes, numbered with all positive integers (1, 2, 3, …). The node with a number x (x > 1), is directly connected with a node with number (x)/(f(x)), with f(x) being the lowest prime divisor of x.

Vanessa's mind is divided into n fragments. Due to more than 500 years of coma, the fragments have been scattered: the i-th fragment is now located at the node with a number k_i! (a factorial of k_i).

To maximize the chance of successful awakening, Ivy decides to place the samples in a node P, so that the total length of paths from each fragment to P is smallest possible. If there are multiple fragments located at the same node, the path from that node to P needs to be counted multiple times.

In the world of zeros and ones, such a requirement is very simple for Ivy. Not longer than a second later, she has already figured out such a node.

But for a mere human like you, is this still possible?

For simplicity, please answer the minimal sum of paths' lengths from every fragment to the emotion samples' assembly node P.

Input

The first line contains an integer n (1 ≤ n ≤ 10^6) — number of fragments of Vanessa's mind.

The second line contains n integers: k_1, k_2, …, k_n (0 ≤ k_i ≤ 5000), denoting the nodes where fragments of Vanessa's mind are located: the i-th fragment is at the node with a number k_i!.

Output

Print a single integer, denoting the minimal sum of path from every fragment to the node with the emotion samples (a.k.a. node P).

As a reminder, if there are multiple fragments at the same node, the distance from that node to P needs to be counted multiple times as well.

Examples

Input


3
2 1 4


Output


5


Input


4
3 1 4 4


Output


6


Input


4
3 1 4 1


Output


6


Input


5
3 1 4 1 5


Output


11

Note

Considering the first 24 nodes of the system, the node network will look as follows (the nodes 1!, 2!, 3!, 4! are drawn bold):

<image>

For the first example, Ivy will place the emotion samples at the node 1. From here:

  * The distance from Vanessa's first fragment to the node 1 is 1. 
  * The distance from Vanessa's second fragment to the node 1 is 0. 
  * The distance from Vanessa's third fragment to the node 1 is 4. 



The total length is 5.

For the second example, the assembly node will be 6. From here:

  * The distance from Vanessa's first fragment to the node 6 is 0. 
  * The distance from Vanessa's second fragment to the node 6 is 2. 
  * The distance from Vanessa's third fragment to the node 6 is 2. 
  * The distance from Vanessa's fourth fragment to the node 6 is again 2. 



The total path length is 6.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAX_N = 5000;
vector<int> primes;
vector<int> expansions[MAX_N + 1];
vector<int> factorials[MAX_N + 1];
vector<int> cnt_factorials(MAX_N + 1);
int index_of_prime[MAX_N + 1];
int n;
struct Node {
  int cnt = 0;
  long long sub_tree_sum = 0;
  long long sub_tree_cnt = 0;
  vector<pair<Node*, vector<pair<int, int>>>> edges;
};
void get_primes(int n) {
  vector<char> numbers(n + 1, 0);
  for (int i = 2; i <= n; i++)
    if (numbers[i] == false) {
      primes.push_back(i);
      index_of_prime[i] = primes.size() - 1;
      for (int j = i; j <= n; j += i) numbers[j] = true;
    }
}
void expansion(int n) {
  for (int number = 2; number <= n; number++) {
    expansions[number].assign(primes.size(), 0);
    int number_copy = number;
    for (int div = 2; div * div <= number; div++) {
      int cnt = 0;
      while (number_copy % div == 0) {
        cnt++;
        number_copy /= div;
      }
      expansions[number][index_of_prime[div]] += cnt;
    }
    if (number_copy != 1) expansions[number][index_of_prime[number_copy]] += 1;
  }
}
void factorials_expansion(int n) {
  factorials[1].assign(primes.size(), 0);
  for (int number = 2; number <= n; number++) {
    factorials[number] = factorials[number - 1];
    for (int i = 0; i < primes.size(); i++)
      factorials[number][i] += expansions[number][i];
  }
}
int roll(int factorial, int iPrime) {
  while (iPrime >= 0 && factorials[factorial][iPrime] == 0) iPrime--;
  return iPrime;
}
void add_factorial(Node* node, int factorial, int iPrime, int rest) {
  int cpy_iPrime = iPrime;
  if (rest == 0 && iPrime >= 0) iPrime--;
  iPrime = roll(factorial, iPrime);
  if (iPrime == -1) {
    node->cnt += cnt_factorials[factorial];
    return;
  }
  if (cpy_iPrime != iPrime) rest = factorials[factorial][iPrime];
  bool common = false;
  for (int i = 0; i < node->edges.size(); i++) {
    if (node->edges[i].second[0].first == iPrime) {
      int prefix_prime = 0, common_rest = 0;
      while (prefix_prime < node->edges[i].second.size() &&
             node->edges[i].second[prefix_prime].first == iPrime) {
        int rest_in_edge = node->edges[i].second[prefix_prime].second;
        if (rest == rest_in_edge) {
          iPrime = roll(factorial, iPrime - 1);
          if (iPrime == -1) break;
          rest = factorials[factorial][iPrime];
          prefix_prime++;
          continue;
        } else if (rest < rest_in_edge) {
          common_rest = rest;
          break;
        } else if (rest > rest_in_edge) {
          rest -= rest_in_edge;
          prefix_prime++;
        }
      }
      vector<pair<int, int>> left, right;
      left.assign(node->edges[i].second.begin(),
                  node->edges[i].second.begin() + prefix_prime);
      right.assign(node->edges[i].second.begin() + prefix_prime,
                   node->edges[i].second.end());
      if (common_rest != 0) {
        left.push_back({right[0].first, common_rest});
        right[0].second -= common_rest;
        rest = 0;
      }
      if (prefix_prime == node->edges[i].second.size()) {
        add_factorial(node->edges[i].first, factorial, iPrime, rest);
      } else {
        Node* new_node = new Node();
        Node* old_node = node->edges[i].first;
        new_node->edges.push_back({old_node, right});
        node->edges[i].first = new_node;
        node->edges[i].second = left;
        add_factorial(new_node, factorial, iPrime, rest);
      }
      common = true;
      break;
    }
  }
  if (!common) {
    Node* new_node = new Node();
    node->edges.push_back({new_node, vector<pair<int, int>>()});
    node->edges.back().second.push_back({iPrime, rest});
    iPrime--;
    while (iPrime >= 0) {
      if (factorials[factorial][iPrime] != 0)
        node->edges.back().second.push_back(
            {iPrime, factorials[factorial][iPrime]});
      iPrime--;
    }
    add_factorial(new_node, factorial, iPrime, 0);
  }
}
void dfs1(Node* node) {
  for (auto edge : node->edges) {
    dfs1(edge.first);
    int cnt = 0;
    for (auto pr : edge.second) cnt += pr.second;
    node->sub_tree_sum +=
        (long long)cnt * edge.first->sub_tree_cnt + edge.first->sub_tree_sum;
    node->sub_tree_cnt += edge.first->sub_tree_cnt;
  }
  node->sub_tree_cnt += node->cnt;
}
long long answer = 1e18;
void dfs2(Node* node, long long rest, int rest_cnt) {
  answer = min(answer, node->sub_tree_sum + rest);
  long long rest_sum_in_childs = 0;
  int rest_cnt_in_childs = 0;
  for (auto edge : node->edges) {
    rest_sum_in_childs += edge.first->sub_tree_sum;
    rest_cnt_in_childs += edge.first->sub_tree_cnt;
  }
  for (auto edge : node->edges) {
    long long current_cnt =
        rest_cnt + node->sub_tree_cnt - edge.first->sub_tree_cnt;
    int cnt = 0;
    for (auto pr : edge.second) cnt += pr.second;
    long long child_back =
        edge.first->sub_tree_sum + (long long)cnt * edge.first->sub_tree_cnt;
    dfs2(edge.first,
         (rest + node->sub_tree_sum - child_back) + cnt * (current_cnt),
         current_cnt);
  }
}
int main() {
  cin >> n;
  for (int i = 0; i < n; i++) {
    int a;
    cin >> a;
    cnt_factorials[a]++;
  }
  get_primes(MAX_N);
  expansion(MAX_N);
  factorials_expansion(MAX_N);
  Node* root = new Node();
  for (int f = MAX_N; f >= 2; f--) {
    if (cnt_factorials[f] != 0) {
      add_factorial(root, f, primes.size() - 1, factorials[f].back());
    }
  }
  root->cnt = cnt_factorials[0] + cnt_factorials[1];
  dfs1(root);
  dfs2(root, 0, 0);
  cout << answer;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random

class Fchaoticvbootcamp(Basebootcamp):
    def __init__(self):
        pass
    
    def case_generator(self):
        test_cases = [
            {'n': 3, 'k_list': [2, 1, 4], 'correct_answer': 5},
            {'n': 4, 'k_list': [3, 1, 4, 4], 'correct_answer': 6},
            {'n': 4, 'k_list': [3, 1, 4, 1], 'correct_answer': 6},
            {'n': 5, 'k_list': [3, 1, 4, 1, 5], 'correct_answer': 11},
            {'n': 2, 'k_list': [0, 0], 'correct_answer': 0},
            {'n': 1, 'k_list': [5], 'correct_answer': 3},
            {'n': 3, 'k_list': [3, 3, 3], 'correct_answer': 0},
            {'n': 2, 'k_list': [2, 5], 'correct_answer': 5},
            {'n': 4, 'k_list': [0, 1, 2, 3], 'correct_answer': 4},
        ]
        case = random.choice(test_cases)
        return case
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k_list = question_case['k_list']
        input_text = f"{n}\n{' '.join(map(str, k_list))}"
        prompt = f"""As Ivy, determine the optimal node P to minimize the total path length from all Vanessa's mind fragments in the ARC Library system. Each fragment is located at node k_i! (k_i factorial). Nodes form a tree where each node x (x>1) connects to x divided by its smallest prime divisor.

Input:
{input_text}

Output the minimal total path length as an integer. Enclose your answer within [answer] and [/answer] tags."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer')
