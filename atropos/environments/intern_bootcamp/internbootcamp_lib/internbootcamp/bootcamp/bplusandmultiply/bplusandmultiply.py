"""# 

### 谜题描述
There is an infinite set generated as follows:

  * 1 is in this set. 
  * If x is in this set, x ⋅ a and x+b both are in this set. 



For example, when a=3 and b=6, the five smallest elements of the set are:

  * 1, 
  * 3 (1 is in this set, so 1⋅ a=3 is in this set), 
  * 7 (1 is in this set, so 1+b=7 is in this set), 
  * 9 (3 is in this set, so 3⋅ a=9 is in this set), 
  * 13 (7 is in this set, so 7+b=13 is in this set). 



Given positive integers a, b, n, determine if n is in this set.

Input

The input consists of multiple test cases. The first line contains an integer t (1≤ t≤ 10^5) — the number of test cases. The description of the test cases follows.

The only line describing each test case contains three integers n, a, b (1≤ n,a,b≤ 10^9) separated by a single space.

Output

For each test case, print \"Yes\" if n is in this set, and \"No\" otherwise. You can print each letter in any case.

Example

Input


5
24 3 5
10 3 6
2345 1 4
19260817 394 485
19260817 233 264


Output


Yes
No
Yes
No
Yes

Note

In the first test case, 24 is generated as follows:

  * 1 is in this set, so 3 and 6 are in this set; 
  * 3 is in this set, so 9 and 8 are in this set; 
  * 8 is in this set, so 24 and 13 are in this set. 



Thus we can see 24 is in this set.

The five smallest elements of the set in the second test case is described in statements. We can see that 10 isn't among them.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# Enter your code here. Read input from STDIN. Print output to STDOUT# ===============================================================================================
# importing some useful libraries
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
# large=1000000000000000035000061
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
    n,a,b=sep()
    now=1
    while(1):
        tempo=n-now
        if(tempo<0):
            No()
            return
        if(tempo%b==0):
            Yes()
            return
        now*=a
        if(a==1):
            No()
            return







#solve()
testcase(N())
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Bplusandmultiplybootcamp(Basebootcamp):
    def __init__(self, a_min=1, a_max=100, b_min=1, b_max=100, n_min=1, n_max=10**9, seed=None):
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.n_min = n_min
        self.n_max = n_max
        self.rng = random.Random(seed)
    
    def case_generator(self):
        # 强制生成20% a=1的案例以覆盖特殊情况
        if self.rng.random() < 0.2:
            a = 1
        else:
            a = self.rng.randint(self.a_min, self.a_max)
        
        b = self.rng.randint(self.b_min, self.b_max)
        
        # 当a=1时，确保生成有效的正例和反例
        if a == 1:
            if self.rng.random() < 0.5:
                # 生成有效n=1 + k*b
                k = self.rng.randint(0, 10)
                n = 1 + k * b
            else:
                # 生成无效n
                while True:
                    n = self.rng.randint(self.n_min, self.n_max)
                    if (n - 1) % b != 0:
                        break
        else:
            # 生成可能合法的普通案例
            steps = self.rng.randint(0, 5)
            now = 1
            for _ in range(steps):
                if self.rng.random() < 0.5:
                    now *= a
                else:
                    now += b
            n = self.rng.choice([
                now * a**self.rng.randint(0,3) + b*self.rng.randint(0,10),
                self.rng.randint(self.n_min, self.n_max)
            ])
        
        return {'n': n, 'a': a, 'b': b}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        b = question_case['b']
        return f"""你正在解决一个数学谜题，需要判断数字{n}是否属于特定无限集合。集合生成规则如下：
1. 初始元素为1
2. 如果x在集合中，则x*a和x+b也在集合中

请判断给定数值n={n}（a={a}, b={b}）是否属于该集合，并将答案（Yes/No）包裹在[answer]标签中，如：[answer]Yes[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
        last_match = matches[-1].strip().lower()
        return last_match.capitalize() if last_match in ('yes', 'no') else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        a = identity['a']
        b = identity['b']
        expected = 'Yes' if cls.check_in_set(n, a, b) else 'No'
        return solution.strip().lower() == expected.lower()
    
    @staticmethod
    def check_in_set(n, a, b):
        if n == 1:
            return True
        max_multiplier = 0
        current = 1
        while current <= n:
            if (n - current) % b == 0:
                return True
            if a == 1:
                return False  # 避免无限循环
            prev = current
            current *= a
            if current == prev:  # 防止a=1时的无限循环
                break
        return False
