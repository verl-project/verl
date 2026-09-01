"""# 

### 谜题描述
Young Teodor enjoys drawing. His favourite hobby is drawing segments with integer borders inside his huge [1;m] segment. One day Teodor noticed that picture he just drawn has one interesting feature: there doesn't exist an integer point, that belongs each of segments in the picture. Having discovered this fact, Teodor decided to share it with Sasha.

Sasha knows that Teodor likes to show off so he never trusts him. Teodor wants to prove that he can be trusted sometimes, so he decided to convince Sasha that there is no such integer point in his picture, which belongs to each segment. However Teodor is lazy person and neither wills to tell Sasha all coordinates of segments' ends nor wills to tell him their amount, so he suggested Sasha to ask him series of questions 'Given the integer point xi, how many segments in Fedya's picture contain that point?', promising to tell correct answers for this questions.

Both boys are very busy studying and don't have much time, so they ask you to find out how many questions can Sasha ask Teodor, that having only answers on his questions, Sasha can't be sure that Teodor isn't lying to him. Note that Sasha doesn't know amount of segments in Teodor's picture. Sure, Sasha is smart person and never asks about same point twice.

Input

First line of input contains two integer numbers: n and m (1 ≤ n, m ≤ 100 000) — amount of segments of Teodor's picture and maximal coordinate of point that Sasha can ask about.

ith of next n lines contains two integer numbers li and ri (1 ≤ li ≤ ri ≤ m) — left and right ends of ith segment in the picture. Note that that left and right ends of segment can be the same point.

It is guaranteed that there is no integer point, that belongs to all segments.

Output

Single line of output should contain one integer number k – size of largest set (xi, cnt(xi)) where all xi are different, 1 ≤ xi ≤ m, and cnt(xi) is amount of segments, containing point with coordinate xi, such that one can't be sure that there doesn't exist point, belonging to all of segments in initial picture, if he knows only this set(and doesn't know n).

Examples

Input

2 4
1 2
3 4


Output

4


Input

4 6
1 3
2 3
4 6
5 6


Output

5

Note

First example shows situation where Sasha can never be sure that Teodor isn't lying to him, because even if one knows cnt(xi) for each point in segment [1;4], he can't distinguish this case from situation Teodor has drawn whole [1;4] segment.

In second example Sasha can ask about 5 points e.g. 1, 2, 3, 5, 6, still not being sure if Teodor haven't lied to him. But once he knows information about all points in [1;6] segment, Sasha can be sure that Teodor haven't lied to him.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
#pragma warning(disable : 4996)
using namespace std;
using ld = long double;
template <class T>
using Table = vector<vector<T>>;
const ld eps = 1e-9;
template <class S, class T>
ostream &operator<<(ostream &os, const pair<S, T> v) {
  os << \"( \" << v.first << \", \" << v.second << \")\";
  return os;
}
template <class T>
ostream &operator<<(ostream &os, const vector<T> &v) {
  for (int i = 0; i < v.size(); i++) {
    if (i > 0) {
      os << \" \";
    }
    os << v[i];
  }
  return os;
}
template <class T>
ostream &operator<<(ostream &os, const vector<vector<T>> &v) {
  for (int i = 0; i < v.size(); i++) {
    if (i > 0) {
      os << endl;
    }
    os << v[i];
  }
  return os;
}
template <class T>
ostream &operator<<(ostream &os, const vector<set<T>> &v) {
  for (int i = 0; i < v.size(); i++) {
    if (i > 0) {
      os << endl;
    }
    os << v[i];
  }
  return os;
}
template <class T>
ostream &operator<<(ostream &os, const set<T> &v) {
  int i = 0;
  for (auto it : v) {
    if (i > 0) {
      os << ' ';
    }
    os << it;
    i++;
  }
  return os;
}
using ll = long long;
vector<int> solve(vector<int> v) {
  vector<int> nums(1000000, 10000000);
  vector<int> anss(v.size());
  nums[0] = 0;
  for (int i = 0; i < v.size(); ++i) {
    auto it = upper_bound(nums.begin(), nums.end(), v[i]);
    anss[i] = it - nums.begin();
    *it = v[i];
  }
  return anss;
}
int main() {
  ios::sync_with_stdio(false);
  int N, M;
  cin >> N >> M;
  vector<int> v(M + 1);
  for (int i = 0; i < N; ++i) {
    int l, r;
    cin >> l >> r;
    v[l - 1]++;
    v[r]--;
  }
  for (int i = 0; i < M; ++i) {
    v[i + 1] += v[i];
  }
  v.pop_back();
  vector<int> ls = solve(v);
  reverse(v.begin(), v.end());
  vector<int> rs = solve(v);
  reverse(rs.begin(), rs.end());
  vector<int> answers(M, 1);
  for (int i = 0; i < M; ++i) {
    answers[i] = ls[i] + rs[i] - 1;
  }
  cout << *max_element(answers.begin(), answers.end()) << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import bisect
from bootcamp import Basebootcamp

class Fteodorisnotaliarbootcamp(Basebootcamp):
    def __init__(self, n=2, m=4):
        """
        Initialize the Fteodorisnotaliarbootcamp with parameters for number of segments and maximum coordinate.
        Default values are n=2 and m=4 as per the first example.
        """
        self.n = n
        self.m = m

    def case_generator(self):
        """
        Generate a puzzle instance (segments) ensuring no common integer point exists in all segments.
        """
        if self.m < 1:
            raise ValueError("m must be at least 1")
        if self.n < 1:
            raise ValueError("n must be at least 1")
        
        segments = []
        if self.m == 1:
            # Only one point, but must have no common point (impossible if n >=1). So this case is invalid.
            # But since m >=1 and n >=1, this can't happen as per problem constraints (guaranteed no common point)
            # For safety, handle edge case
            return {'n': 0, 'm': 1, 'segments': []}
        
        # Choose a split point to divide segments into left and right parts
        split_x = random.randint(1, self.m - 1)
        k = random.randint(0, self.n)  # number of segments on the left

        # Generate left segments (li <= ri <= split_x)
        for _ in range(k):
            li = random.randint(1, split_x)
            ri = random.randint(li, split_x)
            segments.append((li, ri))
        
        # Generate right segments (split_x+1 <= li <= ri <= m)
        for _ in range(self.n - k):
            li = random.randint(split_x + 1, self.m)
            ri = random.randint(li, self.m)
            segments.append((li, ri))
        
        # The case data
        return {
            'n': self.n,
            'm': self.m,
            'segments': segments
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        Convert the generated puzzle instance into a textual problem description with answer format instructions.
        """
        segments_str = '\n'.join([f"{li} {ri}" for li, ri in question_case['segments']])
        prompt = f"""Fteodorisnotaliar has drawn {question_case['n']} segments on integer points within [1, {question_case['m']}]. Sasha wants to determine the maximum number of questions he can ask such that even with the answers, he can't be sure Fteodorisnotaliar isn't lying. 

Problem Details:
- Number of segments (n): {question_case['n']}
- Maximum coordinate (m): {question_case['m']}
Segments (each as 'li ri'):
{segments_str}

Task:
Calculate the largest possible number of questions Sasha can ask (xi, count) where knowing all answers doesn't confirm Fteodorisnotaliar's truthfulness. 

Rules:
1. Each xi must be unique, 1 ≤ xi ≤ {question_case['m']}.
2. Fteodorisnotaliar's segments have no common integer point.
3. Sasha doesn't know n initially.

Output Requirement:
Your answer must be a single integer inside [answer]...[/answer] tags. Example: [answer]4[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        """
        Extract the last occurrence of an answer within [answer] tags.
        """
        import re
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
        """
        Verify if the extracted solution matches the correct maximum value.
        """
        # Helper function to compute the LIS-based solution
        def solve(v):
            max_size = len(v) + 2
            nums = [float('inf')] * max_size
            nums[0] = 0  # Initial setup as per reference C++ code
            anss = []
            for num in v:
                idx = bisect.bisect_right(nums, num)  # Find first index where nums[idx] > num
                anss.append(idx)
                if idx < len(nums) and nums[idx] > num:
                    nums[idx] = num
            return anss

        # Reconstruct the coverage array
        m = identity['m']
        segments = identity['segments']
        v_diff = [0] * (m + 1)
        for li, ri in segments:
            v_diff[li - 1] += 1
            v_diff[ri] -= 1

        # Compute the actual coverage counts
        v = []
        current = 0
        for i in range(m):
            current += v_diff[i]
            v.append(current)

        # Compute ls and rs arrays
        ls = solve(v)
        reversed_v = v[::-1]
        rs_reversed = solve(reversed_v)
        rs = rs_reversed[::-1]

        # Calculate maximum possible answer
        max_answer = 0
        for i in range(len(v)):
            current_val = ls[i] + rs[i] - 1
            if current_val > max_answer:
                max_answer = current_val

        return solution == max_answer
