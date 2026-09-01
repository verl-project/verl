"""# 

### 谜题描述
Little Elephant loves Furik and Rubik, who he met in a small city Kremenchug.

The Little Elephant has two strings of equal length a and b, consisting only of uppercase English letters. The Little Elephant selects a pair of substrings of equal length — the first one from string a, the second one from string b. The choice is equiprobable among all possible pairs. Let's denote the substring of a as x, and the substring of b — as y. The Little Elephant gives string x to Furik and string y — to Rubik.

Let's assume that f(x, y) is the number of such positions of i (1 ≤ i ≤ |x|), that xi = yi (where |x| is the length of lines x and y, and xi, yi are the i-th characters of strings x and y, correspondingly). Help Furik and Rubik find the expected value of f(x, y).

Input

The first line contains a single integer n (1 ≤ n ≤ 2·105) — the length of strings a and b. The second line contains string a, the third line contains string b. The strings consist of uppercase English letters only. The length of both strings equals n.

Output

On a single line print a real number — the answer to the problem. The answer will be considered correct if its relative or absolute error does not exceed 10 - 6.

Examples

Input

2
AB
BA


Output

0.400000000


Input

3
AAB
CAA


Output

0.642857143

Note

Let's assume that we are given string a = a1a2... a|a|, then let's denote the string's length as |a|, and its i-th character — as ai.

A substring a[l... r] (1 ≤ l ≤ r ≤ |a|) of string a is string alal + 1... ar.

String a is a substring of string b, if there exists such pair of integers l and r (1 ≤ l ≤ r ≤ |b|), that b[l... r] = a.

Let's consider the first test sample. The first sample has 5 possible substring pairs: (\"A\", \"B\"), (\"A\", \"A\"), (\"B\", \"B\"), (\"B\", \"A\"), (\"AB\", \"BA\"). For the second and third pair value f(x, y) equals 1, for the rest it equals 0. The probability of choosing each pair equals <image>, that's why the answer is <image> · 0 +  <image> · 1 +  <image> · 1 +  <image> · 0 +  <image> · 0 =  <image> =  0.4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
import copy	
import os


def main(cin):
	n = int(cin.readline().strip())
	a = cin.readline().strip()
	b = cin.readline().strip()
	total = 0.0
	for i in range(n):
		total+=(i+1)*(i+1)
	f = 0.0
	s = [0 for i in range(30)]
	for i in range(n):
		s[ord(a[i])-ord('A')]+= i+1
		f+= s[ord(b[i])-ord('A')] * (n-i)
	s = [0 for i in range(30)]
	for i in reversed(range(n)):
		f+= s[ord(b[i])-ord('A')] * (i+1)
		s[ord(a[i])-ord('A')]+= n-i
	print f/total


if __name__ == \"__main__\":
	cin = sys.stdin
	if (os.path.exists('best.txt')):
		cin = open('best.txt')
	main(cin)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Elittleelephantandfurikandrubikbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=100):
        """
        参数优化为更合理的测试范围
        """
        if n_min < 1:
            raise ValueError("Minimum string length must be at least 1")
        self.n_min = n_min
        self.n_max = n_max
    
    def case_generator(self):
        """生成完全随机的测试案例，包含边缘情况"""
        n = random.randint(self.n_min, self.n_max)
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # 随机生成字符串时包含全同字符的情况
        if random.random() < 0.2:  # 20%概率生成全同字符
            char = random.choice(chars)
            a = char * n
            b = char * n
        else:
            a = ''.join(random.choices(chars, k=n))
            b = ''.join(random.choices(chars, k=n))
        
        return {
            'n': n,
            'a': a.upper(),
            'b': b.upper()
        }

    @staticmethod
    def prompt_func(question_case):
        """增强问题描述格式规范"""
        n = question_case['n']
        a = question_case['a']
        b = question_case['b']
        return f"""## 题目描述

你需要解决小象的概率计算问题：

给定两个长度为{n}的大写字母字符串：
字符串a: {a}
字符串b: {b}

请计算所有等长子串对的字符匹配位置的期望值。

## 输入格式
第一行：整数n
第二行：字符串a
第三行：字符串b

## 输出要求
输出一个浮点数，精确到小数点后9位（参考示例格式）

## 示例
输入：
2
AB
BA

输出：
0.400000000

请在[ANSWER]标签内给出答案，例如：
[ANSWER]0.123456789[/ANSWER]

当前题目输入：
{n}
{a}
{b}

请计算答案："""

    @staticmethod
    def extract_output(output):
        """增强格式提取鲁棒性"""
        # 匹配科学计数法和常规小数
        patterns = [
            r'\[ANSWER\]([\d]+\.[\d]{9})\D*?\[/ANSWER\]',  # 标准格式
            r'answer[\s:]*([\d\.e+-]+)',  # 无标签格式
            r'[\d]+\.[\d]{4,}'  # 自由格式
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    val = float(matches[-1])
                    # 统一转换为标准小数格式
                    return float("{0:.9f}".format(val).rstrip('0').rstrip('.') if '.' in "{0:.9f}".format(val) else val)
                except:
                    continue
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """增强精度验证逻辑"""
        try:
            # 计算标准答案
            n = identity['n']
            a = identity['a']
            b = identity['b']
            correct = cls.calculate_expected(n, a, b)
            
            # 标准化处理
            solution = round(solution, 9)
            correct_rounded = round(correct, 9)
            
            # 直接比较9位小数
            return abs(solution - correct_rounded) < 1e-10
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    @staticmethod
    def calculate_expected(n, a, b):
        """优化算法实现"""
        total = (n * (n + 1) * (2*n + 1)) // 6  # 数学公式优化计算总和
        
        freq = defaultdict(int)
        current_sum = 0
        for i in range(n):
            # 前向扫描
            a_char = a[i]
            freq[a_char] += (i + 1)
            current_sum += freq.get(b[i], 0) * (n - i)
        
        # 反向扫描
        freq.clear()
        for i in reversed(range(n)):
            # 使用字典优化字符索引
            b_char = b[i]
            current_sum += freq.get(b_char, 0) * (i + 1)
            a_char = a[i]
            freq[a_char] += (n - i)
        
        return round(current_sum / total, 9)
