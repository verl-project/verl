"""# 

### 谜题描述
Everyone knows that DNA strands consist of nucleotides. There are four types of nucleotides: \"A\", \"T\", \"G\", \"C\". A DNA strand is a sequence of nucleotides. Scientists decided to track evolution of a rare species, which DNA strand was string s initially. 

Evolution of the species is described as a sequence of changes in the DNA. Every change is a change of some nucleotide, for example, the following change can happen in DNA strand \"AAGC\": the second nucleotide can change to \"T\" so that the resulting DNA strand is \"ATGC\".

Scientists know that some segments of the DNA strand can be affected by some unknown infections. They can represent an infection as a sequence of nucleotides. Scientists are interested if there are any changes caused by some infections. Thus they sometimes want to know the value of impact of some infection to some segment of the DNA. This value is computed as follows:

  * Let the infection be represented as a string e, and let scientists be interested in DNA strand segment starting from position l to position r, inclusive. 
  * Prefix of the string eee... (i.e. the string that consists of infinitely many repeats of string e) is written under the string s from position l to position r, inclusive. 
  * The value of impact is the number of positions where letter of string s coincided with the letter written under it. 



Being a developer, Innokenty is interested in bioinformatics also, so the scientists asked him for help. Innokenty is busy preparing VK Cup, so he decided to delegate the problem to the competitors. Help the scientists!

Input

The first line contains the string s (1 ≤ |s| ≤ 105) that describes the initial DNA strand. It consists only of capital English letters \"A\", \"T\", \"G\" and \"C\".

The next line contains single integer q (1 ≤ q ≤ 105) — the number of events.

After that, q lines follow, each describes one event. Each of the lines has one of two formats: 

  * 1 x c, where x is an integer (1 ≤ x ≤ |s|), and c is a letter \"A\", \"T\", \"G\" or \"C\", which means that there is a change in the DNA: the nucleotide at position x is now c. 
  * 2 l r e, where l, r are integers (1 ≤ l ≤ r ≤ |s|), and e is a string of letters \"A\", \"T\", \"G\" and \"C\" (1 ≤ |e| ≤ 10), which means that scientists are interested in the value of impact of infection e to the segment of DNA strand from position l to position r, inclusive. 

Output

For each scientists' query (second type query) print a single integer in a new line — the value of impact of the infection on the DNA.

Examples

Input

ATGCATGC
4
2 1 8 ATGC
2 2 6 TTT
1 4 T
2 2 6 TA


Output

8
2
4


Input

GAGTTGTTAA
6
2 3 4 TATGGTG
1 1 T
1 6 G
2 5 9 AGTAATA
1 10 G
2 2 6 TTGT


Output

0
3
1

Note

Consider the first example. In the first query of second type all characters coincide, so the answer is 8. In the second query we compare string \"TTTTT...\" and the substring \"TGCAT\". There are two matches. In the third query, after the DNA change, we compare string \"TATAT...\"' with substring \"TGTAT\". There are 4 matches.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 7;
int bit[4][12][12][N], n, m, id[307];
string a;
void add(int flag, int start, int gap, int pos, int val) {
  while (pos <= n) bit[flag][start][gap][pos] += val, pos += pos & -pos;
}
int sum(int flag, int start, int gap, int pos) {
  int res = 0;
  while (pos) res += bit[flag][start][gap][pos], pos -= pos & -pos;
  return res;
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  id['A'] = 0;
  id['T'] = 1;
  id['G'] = 2;
  id['C'] = 3;
  cin >> a;
  n = a.size();
  a = string(\"*\") + a;
  for (int i = 1; i <= 10; ++i)
    for (int j = 1; j <= i; ++j)
      for (int k = j; k <= n; k += i) {
        int c = id[a[k]];
        add(c, j, i, k, 1);
      }
  cin >> m;
  while (m--) {
    int op, l, r, x;
    string e;
    cin >> op;
    if (op == 1) {
      cin >> x >> e;
      int c = id[a[x]];
      a[x] = e[0];
      int cc = id[a[x]];
      for (int i = 1; i <= 10; ++i)
        for (int j = 1; j <= i; ++j)
          if ((x - j) % i == 0) {
            add(c, j, i, x, -1);
            add(cc, j, i, x, 1);
          }
    } else {
      cin >> l >> r >> e;
      int res = 0, gap = e.size();
      for (int k = 0; k < e.size(); ++k) {
        int pos = k + l;
        int c = id[e[k]];
        for (int j = 1; j <= gap; ++j)
          if ((pos - j) % gap == 0) {
            res += sum(c, j, gap, r) - sum(c, j, gap, l - 1);
          }
      }
      cout << res << endl;
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ednaevolutionbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        Ednaevolution谜题训练场环境
        参数：
            max_length: 初始Ednaevolution最大长度（默认100）
            max_queries: 最大事件数量（默认10）
            min_queries: 最小事件数量（默认3）
            query_prob: 生成查询事件的概率（默认0.4）
        """
        default_params = {
            'max_length': 100,
            'max_queries': 10,
            'min_queries': 3,
            'query_prob': 0.4,
            'chars': ['A', 'T', 'G', 'C']
        }
        self.params = {**default_params, **params}

    def case_generator(self):
        """生成Ednaevolution谜题实例"""
        # 生成初始Ednaevolution
        s_len = random.randint(1, self.params['max_length'])
        initial_dna = ''.join(random.choices(
            self.params['chars'], 
            k=s_len
        ))
        
        # 生成事件队列
        q = random.randint(
            self.params['min_queries'], 
            self.params['max_queries']
        )
        current_dna = list(initial_dna)
        queries = []
        correct_answers = []
        
        for _ in range(q):
            if random.random() < self.params['query_prob'] and len(current_dna) > 0:
                # 生成查询事件
                l = random.randint(1, len(current_dna))
                r = random.randint(l, len(current_dna))
                e_len = random.randint(1, 10)
                e = ''.join(random.choices(
                    self.params['chars'], 
                    k=e_len
                ))
                
                # 计算匹配数
                match_count = 0
                for i in range(l-1, r):
                    pos_in_e = (i - (l-1)) % len(e)
                    if current_dna[i] == e[pos_in_e]:
                        match_count += 1
                
                queries.append({
                    'type': 2,
                    'l': l,
                    'r': r,
                    'e': e,
                })
                correct_answers.append(match_count)
            else:
                # 生成修改事件
                if len(current_dna) == 0:
                    continue
                x = random.randint(1, len(current_dna))
                c = random.choice(self.params['chars'])
                current_dna[x-1] = c
                queries.append({
                    'type': 1,
                    'x': x,
                    'c': c
                })
        
        return {
            'initial_dna': initial_dna,
            'queries': queries,
            'correct_answers': correct_answers
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        """生成问题提示文本"""
        input_lines = [
            question_case['initial_dna'],
            str(len(question_case['queries']))
        ]
        
        for query in question_case['queries']:
            if query['type'] == 1:
                input_lines.append(f"1 {query['x']} {query['c']}")
            else:
                input_lines.append(f"2 {query['l']} {query['r']} {query['e']}")
        
        problem_text = (
            "科学家正在追踪Ednaevolution演化过程。初始Ednaevolution链为：\n"
            f"{question_case['initial_dna']}\n\n"
            "需要处理以下事件（共{}个）：\n".format(len(question_case['queries'])) +
            "\n".join(input_lines[2:]) + 
            "\n\n请按顺序处理所有事件，对每个查询事件输出匹配次数。\n"
            "将答案按顺序每行一个数字放在[answer]标签内，例如：\n"
            "[answer]\n3\n0\n5\n[/answer]"
        )
        return problem_text

    @staticmethod
    def extract_output(output):
        """从模型输出中提取答案"""
        answer_blocks = re.findall(
            r'\[answer\](.*?)\[/answer\]', 
            output, 
            re.DOTALL
        )
        
        if not answer_blocks:
            return None
            
        last_answer = answer_blocks[-1].strip()
        solutions = []
        for line in last_answer.split('\n'):
            line = line.strip()
            if line and line.isdigit():
                solutions.append(int(line))
                
        return solutions if solutions else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """验证答案正确性"""
        return solution == identity['correct_answers']
