"""# 

### 谜题描述
Bob is an avid fan of the video game \"League of Leesins\", and today he celebrates as the League of Leesins World Championship comes to an end! 

The tournament consisted of n (n ≥ 5) teams around the world. Before the tournament starts, Bob has made a prediction of the rankings of each team, from 1-st to n-th. After the final, he compared the prediction with the actual result and found out that the i-th team according to his prediction ended up at the p_i-th position (1 ≤ p_i ≤ n, all p_i are unique). In other words, p is a permutation of 1, 2, ..., n.

As Bob's favorite League player is the famous \"3ga\", he decided to write down every 3 consecutive elements of the permutation p. Formally, Bob created an array q of n-2 triples, where q_i = (p_i, p_{i+1}, p_{i+2}) for each 1 ≤ i ≤ n-2. Bob was very proud of his array, so he showed it to his friend Alice.

After learning of Bob's array, Alice declared that she could retrieve the permutation p even if Bob rearranges the elements of q and the elements within each triple. Of course, Bob did not believe in such magic, so he did just the same as above to see Alice's respond.

For example, if n = 5 and p = [1, 4, 2, 3, 5], then the original array q will be [(1, 4, 2), (4, 2, 3), (2, 3, 5)]. Bob can then rearrange the numbers within each triple and the positions of the triples to get [(4, 3, 2), (2, 3, 5), (4, 1, 2)]. Note that [(1, 4, 2), (4, 2, 2), (3, 3, 5)] is not a valid rearrangement of q, as Bob is not allowed to swap numbers belong to different triples.

As Alice's friend, you know for sure that Alice was just trying to show off, so you decided to save her some face by giving her any permutation p that is consistent with the array q she was given. 

Input

The first line contains a single integer n (5 ≤ n ≤ 10^5) — the size of permutation p.

The i-th of the next n-2 lines contains 3 integers q_{i, 1}, q_{i, 2}, q_{i, 3} (1 ≤ q_{i, j} ≤ n) — the elements of the i-th triple of the rearranged (shuffled) array q_i, in random order. Remember, that the numbers within each triple can be rearranged and also the positions of the triples can be rearranged.

It is guaranteed that there is at least one permutation p that is consistent with the input. 

Output

Print n distinct integers p_1, p_2, …, p_n (1 ≤ p_i ≤ n) such that p is consistent with array q. 

If there are multiple answers, print any. 

Example

Input


5
4 3 2
2 3 5
4 1 2


Output


1 4 2 3 5 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
dic = {}
l = []
cnt = [0 for i in range(n + 1)]
for i in range(n-2):
	l.append(map(int, raw_input().split()))
	l[-1].sort()
	cnt[l[-1][0]] += 1
	cnt[l[-1][1]] += 1
	cnt[l[-1][2]] += 1
	if (l[-1][0], l[-1][1]) in dic:
		dic[(l[-1][0], l[-1][1])].append(l[-1][2])
	else:
		dic[(l[-1][0], l[-1][1])] = [l[-1][2]]

	if (l[-1][1], l[-1][0]) in dic:
		dic[(l[-1][1], l[-1][0])].append(l[-1][2])
	else:
		dic[(l[-1][1], l[-1][0])] = [l[-1][2]]

	if (l[-1][2], l[-1][1]) in dic:
		dic[(l[-1][2], l[-1][1])].append(l[-1][0])
	else:
		dic[(l[-1][2], l[-1][1])] = [l[-1][0]]

	if (l[-1][1], l[-1][2]) in dic:
		dic[(l[-1][1], l[-1][2])].append(l[-1][0])
	else:
		dic[(l[-1][1], l[-1][2])] = [l[-1][0]]

	if (l[-1][0], l[-1][2]) in dic:
		dic[(l[-1][0], l[-1][2])].append(l[-1][1])
	else:
		dic[(l[-1][0], l[-1][2])] = [l[-1][1]]

	if (l[-1][2], l[-1][0]) in dic:
		dic[(l[-1][2], l[-1][0])].append(l[-1][1])
	else:
		dic[(l[-1][2], l[-1][0])] = [l[-1][1]]

start = 0
end = 0
for i in range(1, n + 1):
	if cnt[i] == 1:
		start = i 
		break
cnt = [0 for i in range(n + 1)]
ans = []
for i in l:
	if start in i:
		if start == i[0]:
			ans.append(i[0])
			ans.append(i[1])
			ans.append(i[2])
		elif start == i[1]:
			ans.append(i[1])
			ans.append(i[0])
			ans.append(i[2])
		else:
			ans.append(i[2])
			ans.append(i[1])
			ans.append(i[0])
		break
for i in ans:
	cnt[i] = 1
tmpans = list(ans)
for i in range(n):
	for j in dic[(tmpans[-2], tmpans[-1])]:
		if cnt[j] == 0:
			cnt[j] = 1
			tmpans.append(j)
			break
if len(tmpans) == n:
	print ' '.join([str(x) for x in tmpans])
else:
	ans[1], ans[2] = ans[2], ans[1]
	cnt = [0 for i in range(n + 1)]
	for i in ans:
		cnt[i] = 1
	tmpans = ans[:]
	for i in range(n):
		for j in dic[(tmpans[-2], tmpans[-1])]:
			if cnt[j] == 0:
				cnt[j] = 1
				tmpans.append(j)
				break
	print ' '.join([str(x) for x in tmpans])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
import re
from bootcamp import Basebootcamp

class Cleagueofleesinsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', random.randint(5, 10))
        self.min_n = params.get('min_n', 5)
        self.max_n = params.get('max_n', 10)
        
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        p = list(range(1, n+1))
        random.shuffle(p)
        
        triples = []
        for i in range(n-2):
            triplet = p[i:i+3]
            random.shuffle(triplet)
            triples.append(tuple(triplet))
        
        random.shuffle(triples)
        return {'n': n, 'triples': triples}
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['n'])] + \
            [' '.join(map(str, t)) for t in question_case['triples']]
        input_example = '\n'.join(input_lines)
        
        return f"""你正在解决排列重建问题。给定打乱顺序的三元组，请重建原始排列。

输入格式要求：
- 第一行为整数n
- 后续n-2行为三个空格分隔的整数

当前输入数据：
{input_example}

请输出任意满足条件的排列，格式为空格分隔的数字，并将最终答案放在[answer]标签内。例如：
[answer]1 2 3 4 5[/answer]"""  # 修复字符串结尾的引号对齐问题

    @staticmethod
    def extract_output(output):
        patterns = [
            r'\[answer\]([\d\s]+?)\[/answer\]',  # 严格匹配数字和空格
            r'(?:\n|^)(\d+(?:\s+\d+)+)(?:\n|$)'  # 匹配纯数字序列
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output)
            if matches:
                last_match = matches[-1].strip()
                try:
                    return list(map(int, last_match.split()))
                except:
                    continue
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证基础参数
        n = identity['n']
        if not solution or len(solution) != n:
            return False
        if set(solution) != set(range(1, n+1)):
            return False
        
        # 构建有效三元组集合
        expected = defaultdict(int)
        for t in identity['triples']:
            key = tuple(sorted(t))
            expected[key] += 1
        
        # 验证解的三元组
        actual = defaultdict(int)
        for i in range(len(solution)-2):
            triplet = tuple(sorted(solution[i:i+3]))
            actual[triplet] += 1
        
        # 比较多重集合
        return actual == expected
