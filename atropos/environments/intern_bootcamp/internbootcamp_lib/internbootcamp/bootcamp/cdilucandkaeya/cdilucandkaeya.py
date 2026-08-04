"""# 

### 谜题描述
The tycoon of a winery empire in Mondstadt, unmatched in every possible way. A thinker in the Knights of Favonius with an exotic appearance.

This time, the brothers are dealing with a strange piece of wood marked with their names. This plank of wood can be represented as a string of n characters. Each character is either a 'D' or a 'K'. You want to make some number of cuts (possibly 0) on this string, partitioning it into several contiguous pieces, each with length at least 1. Both brothers act with dignity, so they want to split the wood as evenly as possible. They want to know the maximum number of pieces you can split the wood into such that the ratios of the number of occurrences of 'D' to the number of occurrences of 'K' in each chunk are the same.

Kaeya, the curious thinker, is interested in the solution for multiple scenarios. He wants to know the answer for every prefix of the given string. Help him to solve this problem!

For a string we define a ratio as a:b where 'D' appears in it a times, and 'K' appears b times. Note that a or b can equal 0, but not both. Ratios a:b and c:d are considered equal if and only if a⋅ d = b⋅ c. 

For example, for the string 'DDD' the ratio will be 3:0, for 'DKD' — 2:1, for 'DKK' — 1:2, and for 'KKKKDD' — 2:4. Note that the ratios of the latter two strings are equal to each other, but they are not equal to the ratios of the first two strings.

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 1000). Description of the test cases follows.

The first line of each test case contains an integer n (1 ≤ n ≤ 5 ⋅ 10^5) — the length of the wood.

The second line of each test case contains a string s of length n. Every character of s will be either 'D' or 'K'.

It is guaranteed that the sum of n over all test cases does not exceed 5 ⋅ 10^5.

Output

For each test case, output n space separated integers. The i-th of these numbers should equal the answer for the prefix s_{1},s_{2},...,s_{i}.

Example

Input


5
3
DDK
6
DDDDDD
4
DKDK
1
D
9
DKDKDDDDK


Output


1 2 1 
1 2 3 4 5 6 
1 1 1 2 
1 
1 1 1 2 1 2 1 1 3 

Note

For the first test case, there is no way to partition 'D' or 'DDK' into more than one block with equal ratios of numbers of 'D' and 'K', while you can split 'DD' into 'D' and 'D'.

For the second test case, you can split each prefix of length i into i blocks 'D'.

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
        self.writable = \"A\" in file.mode or \"r\" not in file.mode
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
    for A in args:
        if not at_start:
            file.write(sep)
        file.write(str(A))
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
        self._lists = _lists = [values[start:start + _load] for start in range(0, _len, _load)]
        self._list_lens = [len(_list) for _list in _lists]
        self._mins = [_list[0] for _list in _lists]
        self._fen_tree = []
        self._rebuild = True

    def _fen_build(self):
        \"\"\"Build a fenwick tree instance.\"\"\"
        self._fen_tree[:] = self._list_lens
        _fen_tree = self._fen_tree
        for start in range(len(_fen_tree)):
            if start | start + 1 < len(_fen_tree):
                _fen_tree[start | start + 1] += _fen_tree[start]
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
        A = 0
        while end:
            A += _fen_tree[end - 1]
            end &= end - 1
        return A

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
        \"\"\"Return number of ofinansurrences of `value` in the sorted list.\"\"\"
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


def YES():
    print(\"YES\")


def NO():
    print(\"NO\")


def Yes():
    print(\"Yes\")


def No():
    print(\"No\")


def pow(A, B, p):
    res = 1  # Initialize result
    A = A % p  # Update A if it is more , than or equal to p
    if (A == 0):
        return 0
    while (B > 0):
        if ((B & 1) == 1):  # If B is odd, multiply, A with result
            res = (res * A) % p

        B = B >> 1  # B = B/2
        A = (A * A) % p
    return res


from functools import reduce


def numberOfSetBits(n):
    n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
    n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
    n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
    n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
    n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
    n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32)  # This last & isn't strictly necessary.
    return n


def factors(n):
    return set(reduce(list.__add__,
                      ([start, n // start] for start in range(1, int(n ** 0.5) + 1) if n % start == 0)))


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


def gcd(a, b):
    if a == b: return a
    while b > 0: a, b = b, a % b
    return a


def lcm(a, b):
    return abs((a // gcd(a, b)) * b)


inf = float(\"inf\")

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


##two pointer method


# l=0
# for r in range(n):
#     add(r)
#     while(not ok(l,r)):#l,r included
#         remove(l)
#         l+=1
#     #[l,r] is valid
#     if(ok()):
#         do()


# #==========================


# r=-1
# for l in range(n):
#     while (r + 1 < l):
#         r=l-1
#         reset state
#
#
#
#     while(r+1<n and  ok_to_include_r+1()):
#         add(r)
#         r+=1
#     #[l,r) is valid
#     if(ok()):
#         do()
#     remove(l)


# #############################


# discrete binary search
# minimise:
# def search(l,r):
#     ans=inf
#     while(l<=r):
#         mid=(r-l)//2 + l
#         if(check(mid)):
#             ans=min(ans,mid)
#             r=mid-1
#         else:
#             l=mid+1
#
#     return ans

# maximise:
# def search(l,r):
#
#     ans=-1
#     while(l<=r):
#         mid=l+(r-l)//2
#         if(check(mid)):
#             ans=max(ans,mid)
#             l=mid+1
#         else:
#             r=mid-1
#
#     return ans


# =========================================================================================
from collections import defaultdict

# #
# to find factorial and ncr
# tot = 2005
# mod = 998244353
# fac = [1, 1]
# finv = [1, 1]
# inv = [0, 1]
#
# for start in range(2, tot + 1):
#     fac.append((fac[-1] * start) % mod)
#     inv.append(mod - (inv[mod % start] * (mod // start) % mod))
#     finv.append(finv[-1] * inv[-1] % mod)


def comb(n, r):
    if (r == 0 or r == n): return 1
    if n < r:
        return 0
    else:
        return fac[n] * (finv[r] * finv[n - r] % mod) % mod


#
# from functools import lru_cache
# p=3
# def ncr_small_mod_p(n,r):
#     ans=1
#     while(n>0):
#         x=n%p
#         y=r%p
#         n//=p
#         r//=p
#         ans*=comb(x,y)
#         ans%=p
#     return ans


#

def inp(): return sys.stdin.readline().rstrip(\"\r\n\")  # for fast input


def N():
    return int(inp())


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
    return [[[v] * p for _ in range(m)] for sta in range(n)]


def ceil(a, b):
    return (a + b - 1) // b


# co-ordinate compression
# ma={s:idx for idx,s in enumerate(sorted(set(l+r)))}

# mxn=100005
# lrg=[0]*mxn
# for start in range(2,mxn-3):
#     if (lrg[start]==0):
#         for j in range(start,mxn-3,start):
#             lrg[j]=start

test_count = 1


def testcase(t):
    global test_count
    for p in range(t):
        global test_count
        # print(\"Case #{}:\".format(test_count), end=\" \")
        solve()
        test_count += 1


def solve():
    n=N()
    s=inp()
    d=0
    k=0
    co=defaultdict(int)
    ans=[]
    for i in range(n):
        if(s[i]==\"D\"):
            d+=1
        else:
            k+=1
        if(d*k >0):
            g=gcd(d,k)
            td=d//g
            tk=k//g
        else:
            if(d>0):
                td=1
                tk=0
            else:
                tk=1
                td=0
        ans.append(co[(td,tk)]+1)
        co[(td,tk)]+=1
    print(*ans)











#solve()
testcase(N())
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from math import gcd
from collections import defaultdict

class Cdilucandkaeyabootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        # 参数处理增强：允许动态配置字符串长度范围和特殊模式概率
        self.min_length = params.get('min_length', 1)
        self.max_length = params.get('max_length', 10)
        self.special_prob = params.get('special_prob', 0.2)  # 全D/K的概率
        
    def case_generator(self):
        # 生成更丰富的测试用例（包含全D、全K、混合情况）
        if random.random() < self.special_prob:
            # 生成特殊模式
            char = random.choice(['D', 'K'])
            n = random.randint(self.min_length, self.max_length)
            s = char * n
        else:
            # 正常随机生成
            n = random.randint(self.min_length, self.max_length)
            s = ''.join(random.choices(['D','K'], k=n))
        
        # 严格遵循问题示例的数据结构
        return {'n': n, 's': s}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        # 构造精确匹配问题描述的prompt
        return f"""给定长度为{question_case['n']}的字符串s={question_case['s']}，对每个前缀s[1..i]（1 ≤ i ≤ {question_case['n']}），输出最大分割块数使得每块的D:K比例相同。将答案用空格分隔放在[answer]标签内，如：[answer]1 2 3[/answer]"""
    
    @staticmethod
    def extract_output(output):
        # 增强版答案提取：处理多种格式变体
        pattern = r'''
            \[answer\]       # 起始标签
            \s*              # 允许前置空白
            ((?:             # 捕获组：匹配数字序列
                \d+          # 数字
                (?:\s+|,|;)* # 允许空格、逗号、分号分隔
            )+) 
            \s*              # 允许后置空白
            \[/answer\]      # 结束标签
        '''
        matches = re.findall(pattern, output, re.VERBOSE | re.IGNORECASE)
        if not matches:
            return None
        
        # 规范化数字序列
        last_match = re.sub(r'[^0-9\s]', ' ', matches[-1])  # 替换非数字字符为空格
        normalized = ' '.join(last_match.strip().split())   # 合并多余空格
        return normalized
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 强化验证逻辑
        try:
            # 步骤1：解析输入
            s = identity['s']
            n = identity['n']
            
            # 步骤2：计算正确答案
            ratio_counter = defaultdict(int)
            correct_answers = []
            d_count = k_count = 0
            
            for char in s:
                d_count += (char == 'D')
                k_count += (char == 'K')
                
                # 计算最简比例
                if d_count == 0 and k_count == 0:
                    current_ratio = (0, 0)  # 理论上不可能出现
                elif k_count == 0:
                    current_ratio = (1, 0)
                elif d_count == 0:
                    current_ratio = (0, 1)
                else:
                    divisor = gcd(d_count, k_count)
                    current_ratio = (d_count//divisor, k_count//divisor)
                
                # 当前可分割数 = 该比例出现次数（包含当前）
                ratio_counter[current_ratio] += 1
                correct_answers.append(str(ratio_counter[current_ratio]))
            
            # 步骤3：验证格式和内容
            expected = ' '.join(correct_answers)
            return solution.strip() == expected
            
        except Exception as e:
            print(f"Validation Error: {str(e)}")
            return False
