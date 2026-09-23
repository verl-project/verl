"""# 

### 谜题描述
Mike is a bartender at Rico's bar. At Rico's, they put beer glasses in a special shelf. There are n kinds of beer at Rico's numbered from 1 to n. i-th kind of beer has ai milliliters of foam on it.

<image>

Maxim is Mike's boss. Today he told Mike to perform q queries. Initially the shelf is empty. In each request, Maxim gives him a number x. If beer number x is already in the shelf, then Mike should remove it from the shelf, otherwise he should put it in the shelf.

After each query, Mike should tell him the score of the shelf. Bears are geeks. So they think that the score of a shelf is the number of pairs (i, j) of glasses in the shelf such that i < j and <image> where <image> is the greatest common divisor of numbers a and b.

Mike is tired. So he asked you to help him in performing these requests.

Input

The first line of input contains numbers n and q (1 ≤ n, q ≤ 2 × 105), the number of different kinds of beer and number of queries.

The next line contains n space separated integers, a1, a2, ... , an (1 ≤ ai ≤ 5 × 105), the height of foam in top of each kind of beer.

The next q lines contain the queries. Each query consists of a single integer integer x (1 ≤ x ≤ n), the index of a beer that should be added or removed from the shelf.

Output

For each query, print the answer for that query in one line.

Examples

Input

5 6
1 2 3 4 6
1
2
3
4
5
1


Output

0
1
3
5
6
2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int arr[1000000], val[1000000], mark[1000000];
int freq[1000000];
void sieve() {
  int i, j;
  for (i = 2; i * i <= 500000; ++i) {
    if (arr[i] == 0) {
      for (j = i + i; j <= 500000; j += i) {
        if (!arr[j]) arr[j] = i;
      }
    }
  }
  for (i = 2; i <= 500000; ++i) {
    if (!arr[i]) arr[i] = i;
  }
}
int main() {
  sieve();
  int tmp, n, q, i, j, k, a, b, c, x, y, z, lim, tot;
  long long int ans;
  scanf(\"%d\", &n);
  scanf(\"%d\", &q);
  for (i = 1; i <= n; ++i) {
    scanf(\"%d\", &val[i]);
  }
  ans = 0ll;
  tot = 0;
  int c1 = 0;
  while (q--) {
    scanf(\"%d\", &x);
    a = val[x];
    vector<int> v;
    while (a != 1) {
      k = arr[a];
      v.push_back(k);
      while (a % k == 0) a /= k;
    }
    lim = (1 << v.size());
    tmp = 0;
    for (i = 1; i < lim; ++i) {
      if (__builtin_popcount(i) & 1) {
        a = 1;
      } else {
        a = -1;
      }
      y = 1;
      for (j = 0; j < v.size(); ++j) {
        if (i & (1 << j)) y *= v[j];
      }
      tmp += (a * freq[y]);
    }
    if (mark[x] == 0) {
      mark[x] = 1;
      ans += (tot - tmp);
      tot++;
      for (i = 1; i < lim; ++i) {
        y = 1;
        for (j = 0; j < v.size(); ++j) {
          if (i & (1 << j)) {
            y *= v[j];
          }
        }
        freq[y]++;
      }
    } else {
      mark[x] = 0;
      if (val[x] == 1)
        ans -= (tot - 1);
      else
        ans -= (tot - tmp);
      tot--;
      for (i = 1; i < lim; ++i) {
        y = 1;
        for (j = 0; j < v.size(); ++j) {
          if (i & (1 << j)) {
            y *= v[j];
          }
        }
        freq[y]--;
      }
    }
    printf(\"%lld\n\", ans);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

def prime_factors(n):
    """返回唯一质因数列表（已排序）"""
    if n == 1:
        return []
    factors = set()
    while n % 2 == 0:
        factors.add(2)
        n = n // 2
    i = 3
    max_i = int(n**0.5) + 1
    while i <= max_i and n > 1:
        while n % i == 0:
            factors.add(i)
            n = n // i
            max_i = int(n**0.5) + 1
        i += 2
    if n > 1:
        factors.add(n)
    return sorted(factors)

def generate_correct_output(n, q, a, queries):
    """生成正确的输出序列"""
    mark = defaultdict(bool)
    freq = defaultdict(int)
    ans = 0
    tot = 0
    output = []
    
    for x in queries:
        val = a[x-1]
        factors = prime_factors(val)
        lim = 1 << len(factors)
        
        # 计算互质的元素数量
        tmp = 0
        for mask in range(1, lim):
            bits = bin(mask).count('1')
            sign = 1 if bits % 2 else -1
            product = 1
            for j in range(len(factors)):
                if mask & (1 << j):
                    product *= factors[j]
            tmp += sign * freq[product]
        
        if not mark[x]:
            # 添加操作
            ans += (tot - tmp)
            tot += 1
            # 更新素数组合频率
            for mask in range(1, lim):
                product = 1
                for j in range(len(factors)):
                    if mask & (1 << j):
                        product *= factors[j]
                freq[product] += 1
            mark[x] = True
        else:
            # 移除操作
            ans -= (tot - 1 - tmp) if val == 1 else (tot - tmp)
            tot -= 1
            # 更新素数组合频率
            for mask in range(1, lim):
                product = 1
                for j in range(len(factors)):
                    if mask & (1 << j):
                        product *= factors[j]
                freq[product] -= 1
                if freq[product] == 0:
                    del freq[product]
            mark[x] = False
        
        output.append(ans)
    
    return output

class Cmikeandfoambootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.max_beer = params.get('max_beer', 5)
        self.max_queries = params.get('max_queries', 8)
        self.max_foam = params.get('max_foam', 20)
    
    def case_generator(self):
        """生成有效测试案例（保证至少有一个解）"""
        while True:
            try:
                n = random.randint(2, self.max_beer)
                q = random.randint(3, self.max_queries)
                a = [random.randint(1, self.max_foam) for _ in range(n)]
                queries = [random.randint(1, n) for _ in range(q)]
                output = generate_correct_output(n, q, a, queries)
                return {
                    'n': n,
                    'q': q,
                    'a': a,
                    'queries': queries,
                    'correct_output': output
                }
            except:
                continue
    
    @staticmethod
    def prompt_func(question_case):
        """生成符合要求的题目描述"""
        problem_desc = (
            "Mike is a bartender at Rico's bar. Your task is to track beer glasses "
            "on a shelf and calculate coprime pairs after each query.\n\n"
            f"Parameters:\n"
            f"- {question_case['n']} beer types\n"
            f"- {question_case['q']} queries\n"
            f"- Foam amounts: {', '.join(map(str, question_case['a']))}\n"
            f"- Query sequence: {', '.join(map(str, question_case['queries']))}\n\n"
            "Rules:\n"
            "1. For each query, toggle the presence of the specified beer type\n"
            "2. After each query, count all (i,j) pairs where i<j and gcd(a_i, a_j)=1\n"
            "3. Output the count immediately after each query\n\n"
            "Output Format:\n"
            "Put each query's result on a separate line within [answer] tags.\n"
            "Example:\n"
            "[answer]\n0\n1\n3\n5\n6\n2\n[/answer]\n"
            "Now provide the answer for the given queries:"
        )
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        """增强答案提取的鲁棒性"""
        answer_pattern = re.compile(r'\[answer\][\s]*(.*?)[\s]*\[/answer\]', re.DOTALL)
        matches = answer_pattern.findall(output)
        
        if not matches:
            return None
        
        # 取最后一个答案块并解析数字
        last_answer = matches[-1].strip()
        valid_numbers = []
        for line in last_answer.split('\n'):
            line = line.strip()
            if line and line.isdigit():
                valid_numbers.append(int(line))
        
        return valid_numbers if valid_numbers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """严格验证答案顺序和数值"""
        correct = identity.get('correct_output', [])
        return solution == correct

# 示例用法
if __name__ == "__main__":
    bootcamp = Cmikeandfoambootcamp()
    case = bootcamp.case_generator()
    print("Generated Case:", case)
    prompt = Cmikeandfoambootcamp.prompt_func(case)
    print("\nPrompt Example:")
    print(prompt)
