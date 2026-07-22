"""# 

### 谜题描述
Living in Byteland was good enough to begin with, but the good king decided to please his subjects and to introduce a national language. He gathered the best of wise men, and sent an expedition to faraway countries, so that they would find out all about how a language should be designed.

After some time, the wise men returned from the trip even wiser. They locked up for six months in the dining room, after which they said to the king: \"there are a lot of different languages, but almost all of them have letters that are divided into vowels and consonants; in a word, vowels and consonants must be combined correctly.\"

There are very many rules, all of them have exceptions, but our language will be deprived of such defects! We propose to introduce a set of formal rules of combining vowels and consonants, and include in the language all the words that satisfy them.

The rules of composing words are:

  * The letters are divided into vowels and consonants in some certain way;
  * All words have a length of exactly n;
  * There are m rules of the form (pos1, t1, pos2, t2). Each rule is: if the position pos1 has a letter of type t1, then the position pos2 has a letter of type t2.



You are given some string s of length n, it is not necessarily a correct word of the new language. Among all the words of the language that lexicographically not smaller than the string s, find the minimal one in lexicographic order.

Input

The first line contains a single line consisting of letters 'V' (Vowel) and 'C' (Consonant), determining which letters are vowels and which letters are consonants. The length of this string l is the size of the alphabet of the new language (1 ≤ l ≤ 26). The first l letters of the English alphabet are used as the letters of the alphabet of the new language. If the i-th character of the string equals to 'V', then the corresponding letter is a vowel, otherwise it is a consonant.

The second line contains two integers n, m (1 ≤ n ≤ 200, 0 ≤ m ≤ 4n(n - 1)) — the number of letters in a single word and the number of rules, correspondingly.

Next m lines describe m rules of the language in the following format: pos1, t1, pos2, t2 (1 ≤ pos1, pos2 ≤ n, pos1 ≠ pos2, <image> 'V', 'C' }).

The last line contains string s of length n, consisting of the first l small letters of the English alphabet.

It is guaranteed that no two rules are the same.

Output

Print a smallest word of a language that is lexicographically not smaller than s. If such words does not exist (for example, if the language has no words at all), print \"-1\" (without the quotes).

Examples

Input

VC
2 1
1 V 2 C
aa


Output

ab


Input

VC
2 1
1 C 2 V
bb


Output

-1


Input

VCC
4 3
1 C 2 V
2 C 3 V
3 V 4 V
abac


Output

acaa

Note

In the first test word \"aa\" is not a word of the language, but word \"ab\" is.

In the second test out of all four possibilities only word \"bb\" is not a word of a language, but all other words are lexicographically less, so there is no answer.

In the third test, due to the last rule, \"abac\" doesn't belong to the language (\"a\" is a vowel, \"c\" is a consonant). The only word with prefix \"ab\" that meets the given rules is \"abaa\". But it is less than \"abac\", so the answer will be \"acaa\"

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long double pi = 3.14159265359;
template <typename T>
T abs(T x) {
  return x > 0 ? x : -x;
}
template <typename T>
T sqr(T x) {
  return x * x;
}
const int maxn = 405 * 4;
char isV[maxn];
vector<int> g[maxn];
vector<int> gt[maxn];
int enc(int v, int vo, int neg) { return v * 4 + vo * 2 + neg; }
void add(int u, int v) {
  g[u].push_back(v);
  gt[v].push_back(u);
}
void remove(int u, int v) {
  g[u].pop_back();
  gt[v].pop_back();
}
vector<int> order;
int used[maxn];
void dfs(int v) {
  used[v] = 1;
  for (int to : g[v]) {
    if (!used[to]) {
      dfs(to);
    }
  }
  order.push_back(v);
}
int comp[maxn];
int cc = 0;
void dfs2(int v) {
  comp[v] = cc;
  for (int to : gt[v]) {
    if (comp[to] == -1) {
      dfs2(to);
    }
  }
}
bool check() {
  fill(comp, comp + maxn, -1);
  fill(used, used + maxn, 0);
  cc = 0;
  order.clear();
  for (int i = 0; i < maxn; i++) {
    if (!used[i]) {
      dfs(i);
    }
  }
  reverse(order.begin(), order.end());
  for (int v : order) {
    if (comp[v] == -1) {
      cc++;
      dfs2(v);
    }
  }
  for (int i = 0; i < maxn; i++) {
    if (comp[i] == comp[i ^ 1]) {
      return false;
    }
  }
  return true;
}
char alp[26];
int asz;
char s[maxn];
int main() {
  srand(time(NULL));
  scanf(\"%s\n\", alp);
  asz = strlen(alp);
  for (int i = 0; i < asz; i++) {
    isV[i] = alp[i] == 'V';
  }
  int n, m;
  scanf(\"%d %d\n\", &n, &m);
  for (int i = 0; i < n; i++) {
    add(enc(i, 0, 0), enc(i, 1, 1));
    add(enc(i, 0, 1), enc(i, 1, 0));
    add(enc(i, 1, 0), enc(i, 0, 1));
    add(enc(i, 1, 1), enc(i, 0, 0));
  }
  for (int i = 0; i < m; i++) {
    char c1, c2;
    int u, v;
    scanf(\"%d %c %d %c\n\", &u, &c1, &v, &c2);
    u--, v--;
    add(enc(u, c1 == 'V', 0), enc(v, c2 == 'V', 0));
    add(enc(v, c2 == 'V', 1), enc(u, c1 == 'V', 1));
  }
  scanf(\"%s\", s);
  if (!check()) {
    cout << -1 << endl;
    return 0;
  }
  for (int i = n; i >= 0; i--) {
    for (int j = 0; j < i; j++) {
      add(enc(j, isV[s[j] - 'a'], 1), enc(j, isV[s[j] - 'a'], 0));
    }
    bool good = false;
    int kk;
    if (i != n) {
      bool wasV = false, wasC = false;
      for (int k = s[i] - 'a' + 1; k < asz; k++) {
        if (wasV && isV[k]) {
          continue;
        }
        if (wasC && !isV[k]) {
          continue;
        }
        if (isV[k]) {
          wasV = true;
        } else {
          wasC = true;
        }
        add(enc(i, isV[k], 1), enc(i, isV[k], 0));
        if (check()) {
          good = true;
          kk = k;
          break;
        }
        remove(enc(i, isV[k], 1), enc(i, isV[k], 0));
      }
    } else {
      good = check();
    }
    string ans;
    if (good) {
      for (int j = 0; j < i; j++) {
        ans += s[j];
      }
      if (i != n) {
        ans += char('a' + kk);
      }
      vector<pair<int, int>> hh;
      for (int j = i + 1; j < n; j++) {
        bool wasV = false, wasC = false;
        for (int k = 0; k < asz; k++) {
          if (wasV && isV[k]) {
            continue;
          }
          if (wasC && !isV[k]) {
            continue;
          }
          if (isV[k]) {
            wasV = true;
          } else {
            wasC = true;
          }
          add(enc(j, isV[k], 1), enc(j, isV[k], 0));
          hh.push_back(make_pair(enc(j, isV[k], 1), enc(j, isV[k], 0)));
          if (check()) {
            ans += char('a' + k);
            break;
          }
          remove(enc(j, isV[k], 1), enc(j, isV[k], 0));
          hh.pop_back();
        }
      }
      if (ans.size() != n) {
        reverse(hh.begin(), hh.end());
        for (auto x : hh) {
          remove(x.first, x.second);
        }
        if (i != n) {
          remove(enc(i, isV[kk], 1), enc(i, isV[kk], 0));
        }
      } else {
        cout << ans << endl;
        return 0;
      }
    }
    for (int j = i - 1; j >= 0; j--) {
      remove(enc(j, isV[s[j] - 'a'], 1), enc(j, isV[s[j] - 'a'], 0));
    }
  }
  cout << -1 << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from itertools import product
from bootcamp import Basebootcamp

class Cnewlanguagebootcamp(Basebootcamp):
    def __init__(self, max_n=4, max_l=3):
        self.max_n = max_n
        self.max_l = max_l

    def case_generator(self):
        """生成有效谜题实例"""
        while True:
            # 生成字母表配置
            l = random.randint(1, self.max_l)
            alphabet = ''.join(random.choices(['V', 'C'], k=l))
            
            # 生成单词长度（1到max_n）
            n = random.randint(1, self.max_n)
            
            # 生成规则数量
            max_possible_rules = 4 * n * (n-1) if n > 1 else 0
            m = random.randint(0, min(10, max_possible_rules))  # 限制复杂度
            
            # 生成初始字符串
            letters = [chr(ord('a') + i) for i in range(l)]
            s = ''.join(random.choices(letters, k=n))
            
            # 生成规则集合
            rules = set()
            positions = list(range(1, n+1))
            for _ in range(m):
                if len(positions) < 2:
                    break
                pos1 = random.choice(positions)
                pos2 = random.choice([p for p in positions if p != pos1])
                t1 = random.choice(['V', 'C'])
                t2 = random.choice(['V', 'C'])
                rules.add((pos1, t1, pos2, t2))
            rules = list(rules)[:m]

            # 构建案例
            case = {
                'alphabet': alphabet,
                'n': n,
                'm': len(rules),
                'rules': rules,
                's': s
            }
            
            # 计算期望答案
            expected = self.find_min_solution(case)
            if expected != '-1':
                case['expected_answer'] = expected
                return case

    @staticmethod
    def find_min_solution(case):
        """暴力搜索验证解法"""
        l = len(case['alphabet'])
        valid_letters = [chr(ord('a') + i) for i in range(l)]
        n = case['n']
        target = case['s']
        
        # 生成所有可能候选（>= target）
        candidates = []
        for chars in product(valid_letters, repeat=n):
            candidate = ''.join(chars)
            if candidate >= target:
                candidates.append(candidate)
        candidates.sort()
        
        # 检查每个候选是否满足规则
        for candidate in candidates:
            # 预处理类型映射
            type_map = []
            for c in candidate:
                idx = ord(c) - ord('a')
                type_map.append(case['alphabet'][idx])
            
            # 验证所有规则
            valid = True
            for rule in case['rules']:
                pos1, t1, pos2, t2 = rule
                p1 = pos1 - 1
                p2 = pos2 - 1
                if p1 >= n or p2 >= n:
                    continue
                if type_map[p1] == t1 and type_map[p2] != t2:
                    valid = False
                    break
            if valid:
                return candidate
        return '-1'

    @staticmethod
    def prompt_func(question_case) -> str:
        """生成问题描述"""
        vowels = []
        cons = []
        for i, t in enumerate(question_case['alphabet']):
            letter = chr(ord('a') + i)
            if t == 'V':
                vowels.append(letter)
            else:
                cons.append(letter)
        
        # 构建规则描述
        rule_lines = []
        for i, rule in enumerate(question_case['rules'], start=1):
            rule_str = f"{i}. If position {rule[0]} is {rule[1]}, then position {rule[2]} must be {rule[3]}"
            rule_lines.append(rule_str)
        
        rules_section = "\n".join(rule_lines) if rule_lines else "No additional rules"

        return f"""You are developing a language validator for the new Byteland language. The specifications are:

Alphabet classification (first {len(question_case['alphabet'])} letters):
- Vowels: {', '.join(vowels) if vowels else 'None'}
- Consonants: {', '.join(cons) if cons else 'None'}

Language rules:
1. All words must have exactly {question_case['n']} characters
{rules_section}

Given the candidate word "{question_case['s']}", find the lexicographically smallest valid word that is not smaller than it. If no such word exists, output "-1".

Format your answer as: [answer]your_answer_here[/answer]"""

    @staticmethod
    def extract_output(output):
        """抽取答案"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """验证答案正确性"""
        return solution == identity.get('expected_answer', '-1')
