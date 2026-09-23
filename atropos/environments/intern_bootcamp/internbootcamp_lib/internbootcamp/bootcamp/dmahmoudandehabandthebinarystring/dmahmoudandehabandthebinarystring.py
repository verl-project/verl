"""# 

### 谜题描述
Mahmoud and Ehab are in the fourth stage now.

Dr. Evil has a hidden binary string of length n. He guarantees that there is at least one '0' symbol and at least one '1' symbol in it. Now he wants Mahmoud and Ehab to find a position of any '0' symbol and any '1' symbol. In order to do this, Mahmoud and Ehab can ask Dr. Evil up to 15 questions. They tell Dr. Evil some binary string of length n, and Dr. Evil tells the Hamming distance between these two strings. Hamming distance between 2 binary strings of the same length is the number of positions in which they have different symbols. You can find the definition of Hamming distance in the notes section below.

Help Mahmoud and Ehab find these two positions.

You will get Wrong Answer verdict if 

  * Your queries doesn't satisfy interaction protocol described below. 
  * You ask strictly more than 15 questions and your program terminated after exceeding queries limit. Please note, that you can do up to 15 ask queries and one answer query. 
  * Your final answer is not correct. 

You will get Idleness Limit Exceeded if you don't print anything or if you forget to flush the output, including for the final answer (more info about flushing output below).

If you exceed the maximum number of queries, You should terminate with 0, In this case you'll get Wrong Answer, If you don't terminate you may receive any verdict because you'll be reading from a closed stream .

Input

The first line of input will contain a single integer n (2 ≤ n ≤ 1000) — the length of the hidden binary string.

Output

To print the final answer, print \"! pos0 pos1\" (without quotes), where pos0 and pos1 are positions of some '0' and some '1' in the string (the string is 1-indexed). Don't forget to flush the output after printing the answer!

Interaction

To ask a question use the format \"? s\" (without quotes), where s is a query string. Don't forget to flush the output after printing a query!

After each query you can read a single integer from standard input — the Hamming distance between the hidden string and the query string.

To flush the output you can use:- 

  * fflush(stdout) in C++; 
  * System.out.flush() in Java; 
  * stdout.flush() in Python; 
  * flush(output) in Pascal; 
  * See the documentation for other languages . 



Hacking.

To hack someone just print one binary string with length up to 1000, containing at least one '0' and at least one '1'.

Example

Input

3
2
1
3
2
1
0

Output

? 000
? 001
? 010
? 011
? 100
? 101
! 2 1

Note

Hamming distance definition: <https://en.wikipedia.org/wiki/Hamming_distance>

In the first test case the hidden binary string is 101, The first query is 000, so the Hamming distance is 2. In the second query the hidden string is still 101 and query is 001, so the Hamming distance is 1.

After some queries you find that symbol at position 2 is '0' and symbol at position 1 is '1', so you print \"! 2 1\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 1010;
int n, cnt = 15, pos0, pos1, s;
int a[N];
int query(int l, int r) {
  printf(\"? \");
  int i;
  for (i = 1; i < l; ++i) putchar('0');
  for (; i <= r; ++i) putchar('1');
  for (; i <= n; ++i) putchar('0');
  puts(\"\");
  fflush(stdout);
  --cnt;
  int ret;
  scanf(\"%d\", &ret);
  return ret;
}
void divide(int l, int r) {
  if (l == r) {
    if (!pos0)
      pos0 = l;
    else if (!pos1)
      pos1 = l;
    return;
  }
  int mid = (l + r) >> 1;
  int x = query(l, mid);
  if (x - s == mid - l + 1) {
    pos0 = l;
    divide(mid + 1, r);
  } else if (s - x == mid - l + 1) {
    pos1 = l;
    divide(mid + 1, r);
  } else
    divide(l, mid);
}
int main() {
  scanf(\"%d\", &n);
  int l = 1, r = n;
  s = query(0, 0);
  divide(1, n);
  printf(\"! %d %d\n\", pos0, pos1);
  fflush(stdout);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dmahmoudandehabandthebinarystringbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=1000):
        self.params = {
            'min_n': min_n,
            'max_n': max_n
        }
    
    def case_generator(self):
        n = random.randint(self.params['min_n'], self.params['max_n'])
        # 保证至少各有一个0和1的生成方式
        hidden = ['0'] * n
        ones_pos = random.sample(range(n), k=random.randint(1, n-1))
        for pos in ones_pos:
            hidden[pos] = '1'
        # 确保至少一个0（当全1时替换最后一个）
        if len(ones_pos) == n:
            hidden[-1] = '0'
        # 确保至少一个1（当全0时随机替换）
        if not any(c == '1' for c in hidden):
            hidden[random.randint(0, n-1)] = '1'
        return {'n': n, 'hidden_str': ''.join(hidden)}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""根据以下交互规则找到二进制字符串中的0和1位置：
1. 初始字符串长度：{n}
2. 每次询问格式：? [binary_string]
3. 最终答案格式：! [pos0] [pos1]（1-based索引）
请将最终答案用[answer]标签包裹，示例：[answer]! 2 5[/answer]"""

    @staticmethod
    def extract_output(output):
        # 严格匹配标签内的最后一个答案
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if answer_blocks:
            last_answer = answer_blocks[-1].strip()
            match = re.fullmatch(r'!\s*(\d+)\s+(\d+)', last_answer)
            if match:
                return f"! {match.group(1)} {match.group(2)}"
        
        # 全局匹配严格格式
        matches = re.findall(r'!\s*\d+\s+\d+', output)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 格式验证
        match = re.fullmatch(r'!\s*(\d+)\s+(\d+)', solution.strip())
        if not match:
            return False
        
        pos0, pos1 = map(int, match.groups())
        s = identity['hidden_str']
        n = identity['n']
        
        # 有效性验证
        return (
            1 <= pos0 <= n and
            1 <= pos1 <= n and
            pos0 != pos1 and  # 关键修正点：必须不同位置
            s[pos0-1] == '0' and
            s[pos1-1] == '1'
        )
