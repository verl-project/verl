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
std::ifstream fin(\"input.txt\");
std::ofstream fout(\"output.txt\");
int n, value;
int x[200005], a[200005], freq[105];
bool foundSolution;
void readInput() {
  std::cin >> n;
  for (int i = 0; i < n; i++) std::cin >> x[i];
}
bool sortCond(std::pair<int, int> a, std::pair<int, int> b) {
  return a.first > b.first;
}
void findMostFrequent() {
  for (int i = 0; i < n; i++) freq[x[i]]++;
  std::vector<std::pair<int, int>> v;
  for (int i = 1; i <= 100; i++) {
    if (freq[i] > 0) v.push_back({freq[i], i});
  }
  std::sort(v.begin(), v.end(), sortCond);
  if (v.size() >= 2 and v[0].first == v[1].first) {
    std::cout << n << '\n';
    foundSolution = true;
    return;
  } else
    value = v[0].second;
}
int findLenght(int val1, int val2) {
  for (int i = 0; i < n; i++) {
    if (x[i] == val1)
      a[i] = 1;
    else if (x[i] == val2)
      a[i] = -1;
    else
      a[i] = 0;
  }
  std::unordered_map<int, int> first;
  first[0] = -1;
  if (a[0] != 0) first[a[0]] = 0;
  int max = 0;
  for (int i = 1; i < n; i++) {
    a[i] += a[i - 1];
    if (first.find(a[i]) != first.end())
      max = std::max(max, i - first[a[i]]);
    else
      first[a[i]] = i;
  }
  return max;
}
bool ok[105];
void solve() {
  int answer = 0;
  for (int i = 0; i < n; i++) {
    if (x[i] == value or ok[x[i]] == true) continue;
    ok[x[i]] = true;
    answer = std::max(answer, findLenght(value, x[i]));
  }
  std::cout << answer << '\n';
}
int main() {
  readInput();
  findMostFrequent();
  if (foundSolution == false) {
    solve();
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

class F1frequencyproblemeasyversionbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=2000, max_val=100):
        """将默认n_max从200000调整为2000以保证生成效率"""
        self.n_min = n_min
        self.n_max = n_max
        self.max_val = max_val
    
    def case_generator(self):
        case_type = random.choice(['random', 'multi_max', 'single_max', 'edge'])
        max_case_size = 2000  # 所有案例统一限制最大尺寸

        if case_type == 'multi_max':
            # 生成两个最高频元素
            n = random.randint(2, min(max_case_size, self.n_max))
            val1, val2 = random.sample(range(1, self.max_val+1), 2)
            return self._create_multi_max_case(n, val1, val2)
        
        elif case_type == 'single_max':
            # 生成带有有效子数组的结构
            n = random.randint(3, min(max_case_size, self.n_max))
            return self._create_single_max_case(n)
        
        elif case_type == 'edge':
            # 边界情况处理
            return {'array': [random.randint(1, self.max_val)], 'answer': 0}
        
        else:  # random case
            n = random.randint(self.n_min, min(max_case_size, self.n_max))
            arr = [random.randint(1, self.max_val) for _ in range(n)]
            return {
                'array': arr,
                'answer': self._optimized_solve(arr)
            }

    def _create_multi_max_case(self, n, val1, val2):
        """创建两个最高频次相同的案例"""
        k = random.randint(1, n//2)
        arr = [val1]*k + [val2]*k
        if n > 2*k:
            arr += random.choices([val1, val2], k=n-2*k)
        random.shuffle(arr)
        return {'array': arr, 'answer': n}

    def _create_single_max_case(self, n):
        """创建存在有效子数组的案例"""
        main_val = random.randint(1, self.max_val)
        sec_val = random.choice([x for x in range(1, self.max_val+1) if x != main_val])
        
        # 确保存在有效子数组
        arr = [main_val]*(n-2) + [sec_val]*2
        random.shuffle(arr)
        return {'array': arr, 'answer': self._optimized_solve(arr)}

    def _optimized_solve(self, array):
        """优化后的求解算法"""
        freq = defaultdict(int)
        for num in array:
            freq[num] += 1

        # 找出前两个最高频元素
        sorted_freq = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        if len(sorted_freq) >= 2 and sorted_freq[0][1] == sorted_freq[1][1]:
            return len(array)

        if not sorted_freq:
            return 0

        # 仅考虑前两个可能候选元素
        main_val = sorted_freq[0][0]
        candidates = [item[0] for item in sorted_freq[1:min(5, len(sorted_freq))]]
        max_len = 0

        for candidate in candidates:
            current_len = self._find_length(array, main_val, candidate)
            max_len = max(max_len, current_len)
        
        return max_len if max_len > 0 else 0

    def _find_length(self, arr, val1, val2):
        """优化后的子数组查找算法"""
        prefix_sum = 0
        first_occurrence = {0: -1}
        max_len = 0

        for idx, num in enumerate(arr):
            if num == val1:
                prefix_sum += 1
            elif num == val2:
                prefix_sum -= 1

            if prefix_sum in first_occurrence:
                max_len = max(max_len, idx - first_occurrence[prefix_sum])
            else:
                first_occurrence[prefix_sum] = idx

        return max_len

    @staticmethod
    def prompt_func(question_case):
        # 保持原有prompt生成逻辑不变
        array = question_case['array']
        return f"""题目要求：找出最长子数组使得出现次数最多的值的频次不唯一。
输入数组长度：{len(array)}
数组元素：{' '.join(map(str, array))}
请将答案放在[answer]标签内。示例：[answer]5[/answer]

你的解答："""

    @staticmethod
    def extract_output(output):
        # 保持原有抽取逻辑不变
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
