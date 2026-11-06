"""# 

### 谜题描述
Ryouko is an extremely forgetful girl, she could even forget something that has just happened. So in order to remember, she takes a notebook with her, called Ryouko's Memory Note. She writes what she sees and what she hears on the notebook, and the notebook became her memory.

Though Ryouko is forgetful, she is also born with superb analyzing abilities. However, analyzing depends greatly on gathered information, in other words, memory. So she has to shuffle through her notebook whenever she needs to analyze, which is tough work.

Ryouko's notebook consists of n pages, numbered from 1 to n. To make life (and this problem) easier, we consider that to turn from page x to page y, |x - y| pages should be turned. During analyzing, Ryouko needs m pieces of information, the i-th piece of information is on page ai. Information must be read from the notebook in order, so the total number of pages that Ryouko needs to turn is <image>.

Ryouko wants to decrease the number of pages that need to be turned. In order to achieve this, she can merge two pages of her notebook. If Ryouko merges page x to page y, she would copy all the information on page x to y (1 ≤ x, y ≤ n), and consequently, all elements in sequence a that was x would become y. Note that x can be equal to y, in which case no changes take place.

Please tell Ryouko the minimum number of pages that she needs to turn. Note she can apply the described operation at most once before the reading. Note that the answer can exceed 32-bit integers.

Input

The first line of input contains two integers n and m (1 ≤ n, m ≤ 105).

The next line contains m integers separated by spaces: a1, a2, ..., am (1 ≤ ai ≤ n).

Output

Print a single integer — the minimum number of pages Ryouko needs to turn.

Examples

Input

4 6
1 2 3 4 3 2


Output

3


Input

10 5
9 4 3 8 8


Output

6

Note

In the first sample, the optimal solution is to merge page 4 to 3, after merging sequence a becomes {1, 2, 3, 3, 3, 2}, so the number of pages Ryouko needs to turn is |1 - 2| + |2 - 3| + |3 - 3| + |3 - 3| + |3 - 2| = 3.

In the second sample, optimal solution is achieved by merging page 9 to 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# Enter your code here. Read input from STDIN. Print output to STDOUT# ===============================================================================================
# importing some useful libraries.
from __future__ import division, print_function
from fractions import Fraction
import sys
import os
from io import BytesIO, IOBase
from itertools import *
import bisect
from heapq import *
from math import ceil, floor
from copy import *
from collections import deque, defaultdict
from collections import Counter as counter  # Counter(list)  return a dict with {key: count}
from itertools import combinations  # if a = [1,2,3] then print(list(comb(a,2))) -----> [(1, 2), (1, 3), (2, 3)]
from itertools import permutations as permutate
from bisect import bisect_left as bl
from operator import *
# If the element is already present in the list,

# the left most position where element has to be inserted is returned.
from bisect import bisect_right as br
from bisect import bisect

# If the element is already present in the list,
# the right most position where element has to be inserted is returned

# ==============================================================================================
# fast I/O region

BUFSIZE = 8192
from sys import stderr


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = \"x\" in file.mode or \"r\" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b\"\n\") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))
        self.read = lambda: self.buffer.read().decode(\"ascii\")
        self.readline = lambda: self.buffer.readline().decode(\"ascii\")


def print(*args, **kwargs):
    \"\"\"Prints the values to a stream, or to sys.stdout by default.\"\"\"
    sep, file = kwargs.pop(\"sep\", \" \"), kwargs.pop(\"file\", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop(\"end\", \"\n\"))
    if kwargs.pop(\"flush\", False):
        file.flush()


if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

# inp = lambda: sys.stdin.readline().rstrip(\"\r\n\")

# ===============================================================================================
### START ITERATE RECURSION ###
from types import GeneratorType


def iterative(f, stack=[]):
    def wrapped_func(*args, **kwargs):
        if stack: return f(*args, **kwargs)
        to = f(*args, **kwargs)
        while True:
            if type(to) is GeneratorType:
                stack.append(to)
                to = next(to)
                continue
            stack.pop()
            if not stack: break
            to = stack[-1].send(to)
        return to

    return wrapped_func


#### END ITERATE RECURSION ####
###########################
# Sorted list
class SortedList:
    def __init__(self, iterable=[], _load=200):
        \"\"\"Initialize sorted list instance.\"\"\"
        values = sorted(iterable)
        self._len = _len = len(values)
        self._load = _load
        self._lists = _lists = [values[i:i + _load] for i in range(0, _len, _load)]
        self._list_lens = [len(_list) for _list in _lists]
        self._mins = [_list[0] for _list in _lists]
        self._fen_tree = []
        self._rebuild = True

    def _fen_build(self):
        \"\"\"Build a fenwick tree instance.\"\"\"
        self._fen_tree[:] = self._list_lens
        _fen_tree = self._fen_tree
        for i in range(len(_fen_tree)):
            if i | i + 1 < len(_fen_tree):
                _fen_tree[i | i + 1] += _fen_tree[i]
        self._rebuild = False

    def _fen_update(self, index, value):
        \"\"\"Update `fen_tree[index] += value`.\"\"\"
        if not self._rebuild:
            _fen_tree = self._fen_tree
            while index < len(_fen_tree):
                _fen_tree[index] += value
                index |= index + 1

    def _fen_query(self, end):
        \"\"\"Return `sum(_fen_tree[:end])`.\"\"\"
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        x = 0
        while end:
            x += _fen_tree[end - 1]
            end &= end - 1
        return x

    def _fen_findkth(self, k):
        \"\"\"Return a pair of (the largest `idx` such that `sum(_fen_tree[:idx]) <= k`, `k - sum(_fen_tree[:idx])`).\"\"\"
        _list_lens = self._list_lens
        if k < _list_lens[0]:
            return 0, k
        if k >= self._len - _list_lens[-1]:
            return len(_list_lens) - 1, k + _list_lens[-1] - self._len
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        idx = -1
        for d in reversed(range(len(_fen_tree).bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < len(_fen_tree) and k >= _fen_tree[right_idx]:
                idx = right_idx
                k -= _fen_tree[idx]
        return idx + 1, k

    def _delete(self, pos, idx):
        \"\"\"Delete value at the given `(pos, idx)`.\"\"\"
        _lists = self._lists
        _mins = self._mins
        _list_lens = self._list_lens

        self._len -= 1
        self._fen_update(pos, -1)
        del _lists[pos][idx]
        _list_lens[pos] -= 1

        if _list_lens[pos]:
            _mins[pos] = _lists[pos][0]
        else:
            del _lists[pos]
            del _list_lens[pos]
            del _mins[pos]
            self._rebuild = True

    def _loc_left(self, value):
        \"\"\"Return an index pair that corresponds to the first position of `value` in the sorted list.\"\"\"
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._mins

        lo, pos = -1, len(_lists) - 1
        while lo + 1 < pos:
            mi = (lo + pos) >> 1
            if value <= _mins[mi]:
                pos = mi
            else:
                lo = mi

        if pos and value <= _lists[pos - 1][-1]:
            pos -= 1

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value <= _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def _loc_right(self, value):
        \"\"\"Return an index pair that corresponds to the last position of `value` in the sorted list.\"\"\"
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._mins

        pos, hi = 0, len(_lists)
        while pos + 1 < hi:
            mi = (pos + hi) >> 1
            if value < _mins[mi]:
                hi = mi
            else:
                pos = mi

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value < _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def add(self, value):
        \"\"\"Add `value` to sorted list.\"\"\"
        _load = self._load
        _lists = self._lists
        _mins = self._mins
        _list_lens = self._list_lens

        self._len += 1
        if _lists:
            pos, idx = self._loc_right(value)
            self._fen_update(pos, 1)
            _list = _lists[pos]
            _list.insert(idx, value)
            _list_lens[pos] += 1
            _mins[pos] = _list[0]
            if _load + _load < len(_list):
                _lists.insert(pos + 1, _list[_load:])
                _list_lens.insert(pos + 1, len(_list) - _load)
                _mins.insert(pos + 1, _list[_load])
                _list_lens[pos] = _load
                del _list[_load:]
                self._rebuild = True
        else:
            _lists.append([value])
            _mins.append(value)
            _list_lens.append(1)
            self._rebuild = True

    def discard(self, value):
        \"\"\"Remove `value` from sorted list if it is a member.\"\"\"
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_right(value)
            if idx and _lists[pos][idx - 1] == value:
                self._delete(pos, idx - 1)

    def remove(self, value):
        \"\"\"Remove `value` from sorted list; `value` must be a member.\"\"\"
        _len = self._len
        self.discard(value)
        if _len == self._len:
            raise ValueError('{0!r} not in list'.format(value))

    def pop(self, index=-1):
        \"\"\"Remove and return value at `index` in sorted list.\"\"\"
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        value = self._lists[pos][idx]
        self._delete(pos, idx)
        return value

    def bisect_left(self, value):
        \"\"\"Return the first index to insert `value` in the sorted list.\"\"\"
        pos, idx = self._loc_left(value)
        return self._fen_query(pos) + idx

    def bisect_right(self, value):
        \"\"\"Return the last index to insert `value` in the sorted list.\"\"\"
        pos, idx = self._loc_right(value)
        return self._fen_query(pos) + idx

    def count(self, value):
        \"\"\"Return number of occurrences of `value` in the sorted list.\"\"\"
        return self.bisect_right(value) - self.bisect_left(value)

    def __len__(self):
        \"\"\"Return the size of the sorted list.\"\"\"
        return self._len

    def __getitem__(self, index):
        \"\"\"Lookup value at `index` in sorted list.\"\"\"
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        return self._lists[pos][idx]

    def __delitem__(self, index):
        \"\"\"Remove value at `index` from sorted list.\"\"\"
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        self._delete(pos, idx)

    def __contains__(self, value):
        \"\"\"Return true if `value` is an element of the sorted list.\"\"\"
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_left(value)
            return idx < len(_lists[pos]) and _lists[pos][idx] == value
        return False

    def __iter__(self):
        \"\"\"Return an iterator over the sorted list.\"\"\"
        return (value for _list in self._lists for value in _list)

    def __reversed__(self):
        \"\"\"Return a reverse iterator over the sorted list.\"\"\"
        return (value for _list in reversed(self._lists) for value in reversed(_list))

    def __repr__(self):
        \"\"\"Return string representation of sorted list.\"\"\"
        return 'SortedList({0})'.format(list(self))


# ===============================================================================================
# some shortcuts

mod = 1000000007


def testcase(t):
    for p in range(t):
        solve()


def pow(x, y, p):
    res = 1  # Initialize result
    x = x % p  # Update x if it is more , than or equal to p
    if (x == 0):
        return 0
    while (y > 0):
        if ((y & 1) == 1):  # If y is odd, multiply, x with result
            res = (res * x) % p

        y = y >> 1  # y = y/2
        x = (x * x) % p
    return res


from functools import reduce


def factors(n):
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))


def gcd(a, b):
    if a == b: return a
    while b > 0: a, b = b, a % b
    return a


# discrete binary search
# minimise:
# def search():
#     l = 0
#     r = 10 ** 15
#
#     for i in range(200):
#         if isvalid(l):
#             return l
#         if l == r:
#             return l
#         m = (l + r) // 2
#         if isvalid(m) and not isvalid(m - 1):
#             return m
#         if isvalid(m):
#             r = m + 1
#         else:
#             l = m
#     return m

# maximise:
# def search():
#     l = 0
#     r = 10 ** 15
#
#     for i in range(200):
#         # print(l,r)
#         if isvalid(r):
#             return r
#         if l == r:
#             return l
#         m = (l + r) // 2
#         if isvalid(m) and not isvalid(m + 1):
#             return m
#         if isvalid(m):
#             l = m
#         else:
#             r = m - 1
#     return m


##############Find sum of product of subsets of size k in a array
# ar=[0,1,2,3]
# k=3
# n=len(ar)-1
# dp=[0]*(n+1)
# dp[0]=1
# for pos in range(1,n+1):
#     dp[pos]=0
#     l=max(1,k+pos-n-1)
#     for j in range(min(pos,k),l-1,-1):
#         dp[j]=dp[j]+ar[pos]*dp[j-1]
# print(dp[k])

def prefix_sum(ar):  # [1,2,3,4]->[1,3,6,10]
    return list(accumulate(ar))


def suffix_sum(ar):  # [1,2,3,4]->[10,9,7,4]
    return list(accumulate(ar[::-1]))[::-1]


def N():
    return int(inp())


dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]


def YES():
    print(\"YES\")


def NO():
    print(\"NO\")


def Yes():
    print(\"Yes\")


def No():
    print(\"No\")


# =========================================================================================
from collections import defaultdict


def numberOfSetBits(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


class MergeFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.num_sets = n
        # self.lista = [[_] for _ in range(n)]

    def find(self, a):
        to_update = []
        while a != self.parent[a]:
            to_update.append(a)
            a = self.parent[a]
        for b in to_update:
            self.parent[b] = a
        return self.parent[a]

    def merge(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.num_sets -= 1
        self.parent[b] = a
        self.size[a] += self.size[b]
        # self.lista[a] += self.lista[b]
        # self.lista[b] = []

    def set_size(self, a):
        return self.size[self.find(a)]

    def __len__(self):
        return self.num_sets


def lcm(a, b):
    return abs((a // gcd(a, b)) * b)


# #
# to find factorial and ncr
# tot = 100005
# mod = 10**9 + 7
# fac = [1, 1]
# finv = [1, 1]
# inv = [0, 1]
#
# for i in range(2, tot + 1):
#     fac.append((fac[-1] * i) % mod)
#     inv.append(mod - (inv[mod % i] * (mod // i) % mod))
#     finv.append(finv[-1] * inv[-1] % mod)


def comb(n, r):
    if n < r:
        return 0
    else:
        return fac[n] * (finv[r] * finv[n - r] % mod) % mod


def inp(): return sys.stdin.readline().rstrip(\"\r\n\")  # for fast input


def out(var): sys.stdout.write(str(var))  # for fast output, always take string


def lis(): return list(map(int, inp().split()))


def stringlis(): return list(map(str, inp().split()))


def sep(): return map(int, inp().split())


def strsep(): return map(str, inp().split())


def fsep(): return map(float, inp().split())


def nextline(): out(\"\n\")  # as stdout.write always print sring.


def arr1d(n, v):
    return [v] * n


def arr2d(n, m, v):
    return [[v] * m for _ in range(n)]


def arr3d(n, m, p, v):
    return [[[v] * p for _ in range(m)] for i in range(n)]


def ceil(a, b):
    return (a + b - 1) // b


# co-ordinate compression
# ma={s:idx for idx,s in enumerate(sorted(set(l+r)))}

# mxn=100005
# lrg=[0]*mxn
# for i in range(2,mxn-3):
#     if (lrg[i]==0):
#         for j in range(i,mxn-3,i):
#             lrg[j]=i


def solve():
    n,m=sep()
    ar=lis()
    adj=[[] for _ in range(n+1)]
    for i in range(m):
        if(i-1>=0):
            if(ar[i]!=ar[i-1]):
                adj[ar[i]].append(ar[i-1])
        if (i + 1 < m):
            if (ar[i] != ar[i + 1]):
                adj[ar[i]].append(ar[i + 1])
    totscore=0
    for i in range(1,m):
        totscore+=abs(ar[i]-ar[i-1])
    redscore=0
    for i in range(1,n+1):
        adj[i].sort()
        curscore=0
        l=(len(adj[i]))
        if l==0:continue

        med=adj[i][(l)//2]
        besscore=0
        for j in adj[i]:
            curscore+=abs(i-j)
            besscore+=abs(med-j)
        redscore=max(redscore,curscore-besscore)

    print(min(totscore-redscore,totscore))






solve()
# testcase(int(inp()))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import Dict, Any

class Cryoukosmemorynotebootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_m=20):
        """
        初始化训练场参数，默认最大页面数为20，最大序列长度为20。
        """
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self) -> Dict[str, Any]:
        """
        生成谜题实例，包含n, m和页面序列a。
        """
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        a = [random.randint(1, n) for _ in range(m)]
        return {
            'n': n,
            'm': m,
            'a': a
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        将谜题实例转换为详细的自然语言问题描述。
        """
        n = question_case['n']
        m = question_case['m']
        a = ' '.join(map(str, question_case['a']))
        problem_text = f"""你是Ryouko，需要解决一个关于记忆笔记本页面优化的问题。你的笔记本共有{n}页，编号从1到{n}。你需要按顺序查阅以下页面序列（共{m}次）：{a}。

每次翻页的代价是当前页与目标页的绝对差值。你最多可以执行一次合并操作：选择一个页面x合并到页面y，使得所有在序列中的x都会被替换为y。请计算经过最优合并后最小的总翻页代价。

例如，当输入为：
4 6
1 2 3 4 3 2
时，合并页面4到3后，总代价为3，因此最终答案为3。

你的任务是解决以下具体案例：
- 笔记本总页数n = {n}
- 序列长度m = {m}
- 访问序列a = {a}

请将最终答案的整数值放置在[answer]和[/answer]标签之间。例如：[answer]42[/answer]"""
        return problem_text
    
    @staticmethod
    def extract_output(output: str):
        """
        从模型输出中提取最后一个[answer]标签包裹的整数。
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution: int, identity: Dict[str, Any]) -> bool:
        """
        验证答案是否符合计算出的最小翻页数。
        """
        try:
            # 计算正确答案
            n = identity['n']
            m = identity['m']
            a = identity['a']
            correct = cls.compute_min_turns(n, m, a)
            return solution == correct
        except:
            return False
    
    @staticmethod
    def compute_min_turns(n: int, m: int, a: list) -> int:
        """
        计算给定案例的最小翻页数。
        """
        if m <= 1:
            return 0
        
        original_cost = sum(abs(a[i] - a[i-1]) for i in range(1, m))
        adj = [[] for _ in range(n+1)]  # 邻接关系存储
        
        # 构建邻接关系
        for i in range(m):
            current = a[i]
            if i > 0 and a[i-1] != current:
                adj[current].append(a[i-1])
            if i < m-1 and a[i+1] != current:
                adj[current].append(a[i+1])
        
        max_reduction = 0
        for page in range(1, n+1):
            neighbors = adj[page]
            if not neighbors:
                continue
            
            # 计算原始总代价
            original_sum = sum(abs(page - x) for x in neighbors)
            
            # 计算最优合并后的代价
            sorted_neighbors = sorted(neighbors)
            median_index = len(sorted_neighbors) // 2
            median = sorted_neighbors[median_index]
            optimized_sum = sum(abs(median - x) for x in sorted_neighbors)
            
            # 更新最大减少量
            current_reduction = original_sum - optimized_sum
            if current_reduction > max_reduction:
                max_reduction = current_reduction
        
        return original_cost - max_reduction
