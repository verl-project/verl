"""# 

### 谜题描述
This is the easy version of the problem. The difference between the versions is in the constraints on the array elements. You can make hacks only if all versions of the problem are solved.

You are given an array [a_1, a_2, ..., a_n]. 

Your goal is to find the length of the longest subarray of this array such that the most frequent value in it is not unique. In other words, you are looking for a subarray such that if the most frequent value occurs f times in this subarray, then at least 2 different values should occur exactly f times.

An array c is a subarray of an array d if c can be obtained from d by deletion of several (possibly, zero or all) elements from the beginning and several (possibly, zero or all) elements from the end.

Input

The first line contains a single integer n (1 ≤ n ≤ 200 000) — the length of the array.

The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ min(n, 100)) — elements of the array.

Output

You should output exactly one integer — the length of the longest subarray of the array whose most frequent value is not unique. If there is no such subarray, output 0.

Examples

Input


7
1 1 2 2 3 3 3


Output


6

Input


10
1 1 1 5 4 1 3 1 2 2


Output


7

Input


1
1


Output


0

Note

In the first sample, the subarray [1, 1, 2, 2, 3, 3] is good, but [1, 1, 2, 2, 3, 3, 3] isn't: in the latter there are 3 occurrences of number 3, and no other element appears 3 times.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
vector<string> vec_splitter(string s) {
  for (char& c : s) c = c == ',' ? ' ' : c;
  stringstream ss;
  ss << s;
  vector<string> res;
  for (string z; ss >> z; res.push_back(z))
    ;
  return res;
}
void debug_out(vector<string> args, int idx) { cerr << endl; }
template <typename Head, typename... Tail>
void debug_out(vector<string> args, int idx, Head H, Tail... T) {
  if (idx > 0) cerr << \", \";
  stringstream ss;
  ss << H;
  cerr << args[idx] << \" = \" << ss.str();
  debug_out(args, idx + 1, T...);
}
void localTest() {}
const long long N = 2e5 + 5;
long long A[N], has[N];
int main() {
  localTest();
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  long long n;
  cin >> n;
  long long mx = 0, cnt = 0, ele = 0;
  for (long long i = 1; i <= n; ++i) {
    cin >> A[i];
    has[A[i]]++;
    mx = max(mx, has[A[i]]);
  }
  for (long long i = 1; i <= 100; ++i) {
    if (has[i] == mx) {
      ++cnt;
      ele = i;
    }
  }
  if (cnt > 1) {
    cout << n << \"\n\";
  } else {
    long long ans = 0;
    for (long long i = 1; i <= 100; ++i) {
      if (i == ele) {
        continue;
      }
      unordered_map<long long, long long> dp;
      dp[0] = 0;
      long long sum = 0;
      for (long long j = 1; j <= n; ++j) {
        if (A[j] == ele) {
          sum++;
        } else if (A[j] == i) {
          sum--;
        }
        if (dp.find(sum) != dp.end()) {
          ans = max(ans, j - dp[sum]);
        } else {
          dp[sum] = j;
        }
      }
    }
    cout << ans << \"\n\";
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict
import random
import re
from bootcamp import Basebootcamp

class D1frequencyproblemeasyversionbootcamp(Basebootcamp):
    def __init__(self, max_n=100):
        self.max_n = max_n  # 控制生成数组的最大长度

    def case_generator(self):
        n = random.randint(1, self.max_n)
        max_val = min(n, 100)
        a = [random.randint(1, max_val) for _ in range(n)]
        ans = self.calculate_answer(a)
        return {
            'n': n,
            'array': a,
            'answer': ans
        }

    @staticmethod
    def calculate_answer(a):
        """完全对齐参考代码的实现逻辑"""
        freq = defaultdict(int)
        for num in a:
            freq[num] += 1
        if not freq:
            return 0
        
        # 确定最大频率元素
        mx = max(freq.values())
        cnt = sum(1 for v in freq.values() if v == mx)
        ele = next(k for k, v in freq.items() if v == mx)

        # Case 1: 多个元素达到最大频率
        if cnt >= 2:
            return len(a)
        
        # Case 2: 单个最大频率元素时
        max_length = 0
        for candidate in range(1, 101):
            if candidate == ele:
                continue
        
            # 使用前缀和算法查找最长子数组
            prefix_sum = {0: -1}
            current_sum = 0
            for idx, num in enumerate(a):
                if num == ele:
                    current_sum += 1
                elif num == candidate:
                    current_sum -= 1
                
                if current_sum in prefix_sum:
                    max_length = max(max_length, idx - prefix_sum[current_sum])
                else:
                    prefix_sum[current_sum] = idx
        
        return max_length

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        array = ' '.join(map(str, question_case['array']))
        return f"""你是编程竞赛选手，请解决以下问题。给定一个数组，找出最长的子数组，使得该子数组中出现次数最多的元素不是唯一的。输出该子数组长度。

输入格式：
第一行：n (数组长度)
第二行：数组元素，空格分隔

示例输入：
7
1 1 2 2 3 3 3

示例输出：
6

当前问题：
n = {n}
数组为：{array}

请将最终答案放置在[answer]标签内，如：[answer]6[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
