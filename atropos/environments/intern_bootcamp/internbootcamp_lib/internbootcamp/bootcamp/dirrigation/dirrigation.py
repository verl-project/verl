"""# 

### 谜题描述
Misha was interested in water delivery from childhood. That's why his mother sent him to the annual Innovative Olympiad in Irrigation (IOI). Pupils from all Berland compete there demonstrating their skills in watering. It is extremely expensive to host such an olympiad, so after the first n olympiads the organizers introduced the following rule of the host city selection.

The host cities of the olympiads are selected in the following way. There are m cities in Berland wishing to host the olympiad, they are numbered from 1 to m. The host city of each next olympiad is determined as the city that hosted the olympiad the smallest number of times before. If there are several such cities, the city with the smallest index is selected among them.

Misha's mother is interested where the olympiad will be held in some specific years. The only information she knows is the above selection rule and the host cities of the first n olympiads. Help her and if you succeed, she will ask Misha to avoid flooding your house.

Input

The first line contains three integers n, m and q (1 ≤ n, m, q ≤ 500 000) — the number of olympiads before the rule was introduced, the number of cities in Berland wishing to host the olympiad, and the number of years Misha's mother is interested in, respectively.

The next line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ m), where a_i denotes the city which hosted the olympiad in the i-th year. Note that before the rule was introduced the host city was chosen arbitrarily.

Each of the next q lines contains an integer k_i (n + 1 ≤ k_i ≤ 10^{18}) — the year number Misha's mother is interested in host city in.

Output

Print q integers. The i-th of them should be the city the olympiad will be hosted in the year k_i.

Examples

Input


6 4 10
3 1 1 1 2 2
7
8
9
10
11
12
13
14
15
16


Output


4
3
4
2
3
4
1
2
3
4


Input


4 5 4
4 4 5 1
15
9
13
6


Output


5
3
3
3

Note

In the first example Misha's mother is interested in the first 10 years after the rule was introduced. The host cities these years are 4, 3, 4, 2, 3, 4, 1, 2, 3, 4.

In the second example the host cities after the new city is introduced are 2, 3, 1, 2, 3, 5, 1, 2, 3, 4, 5, 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
from __future__ import division, print_function

import operator as op
import os
import sys
from bisect import bisect_left, bisect_right, insort
from io import BytesIO, IOBase
from itertools import chain, repeat, starmap

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip
else:
    from functools import reduce


def main():
    n, m, q = map(int, input().split())
    a = [int(x) for x in input().split()]

    counter = [0] * m
    for ai in a:
        counter[ai - 1] += 1

    count_order = sorted(range(m), key=lambda x: counter[x])
    years = [0] * m
    for i in range(m - 1):
        diff = counter[count_order[i + 1]] - counter[count_order[i]]
        years[i + 1] = years[i] + diff * (i + 1)

    max_k = max(counter) * m - sum(counter)
    k = [int(input()) - n for _ in range(q)]
    k_order = sorted(range(q), key=lambda x: k[x])

    sortedlist = SortedList()
    res, idx = [0] * q, 0

    for i in range(q):
        ki = k[k_order[i]]
        if ki > max_k:
            res[k_order[i]] = ((ki - max_k - 1) % m) + 1
        else:
            while years[idx] < ki:
                sortedlist.add(count_order[idx])
                idx += 1
            ki -= years[idx - 1] + 1
            ki %= idx
            res[k_order[i]] = sortedlist[ki] + 1

    print(*res, sep='\n')


class SortedList:
    def __init__(self, iterable=[], _load=200):
        \"\"\"Initialize sorted list instance.\"\"\"
        values = sorted(iterable)
        self._len = _len = len(values)
        self._load = _load
        self._lists = _lists = [
            values[i:i + _load] for i in range(0, _len, _load)
        ]
        self._mins = [_list[0] for _list in _lists]
        self._list_lens = [len(_list) for _list in _lists]
        self._fen_tree = []
        self._balanced = False

    def _fen_init(self):
        \"\"\"Initialize a fenwick tree instance.\"\"\"
        self._fen_tree[:] = self._list_lens
        _fen_tree = self._fen_tree
        for i in range(len(_fen_tree)):
            if i | i + 1 < len(_fen_tree):
                _fen_tree[i | i + 1] += _fen_tree[i]
        self._balanced = True

    def _fen_update(self, index, value):
        \"\"\"Update `fen_tree[index] += value`.\"\"\"
        if self._balanced:
            _fen_tree = self._fen_tree
            while index < len(_fen_tree):
                _fen_tree[index] += value
                index |= index + 1

    def _fen_query(self, end):
        \"\"\"Return `sum(_fen_tree[:end])`.\"\"\"
        if not self._balanced:
            self._fen_init()

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
        if not self._balanced:
            self._fen_init()

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
            self._balanced = False

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
                _mins.insert(pos + 1, _list[_load])
                _list_lens.insert(pos + 1, len(_list) - _load)
                _list_lens[pos] = _load
                del _list[_load:]
                self._balanced = False
        else:
            _lists.append([value])
            _mins.append(value)
            _list_lens.append(1)
            self._balanced = False

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

    def pop(self, index=-1):
        \"\"\"Remove and return value at `index` in sorted list.\"\"\"
        pos, idx = self._fen_findkth(index if 0 <= index else self._len -
                                     index)
        value = self._lists[pos][idx]
        self._delete(pos, idx)
        return value

    def __len__(self):
        \"\"\"Return the size of the sorted list.\"\"\"
        return self._len

    def __getitem__(self, index):
        \"\"\"Lookup value at `index` in sorted list.\"\"\"
        pos, idx = self._fen_findkth(index if 0 <= index else self._len +
                                     index)
        return self._lists[pos][idx]

    def __delitem__(self, index):
        \"\"\"Remove value at `index` from sorted list.\"\"\"
        pos, idx = self._fen_findkth(index if 0 <= index else self._len +
                                     index)
        self._delete(pos, idx)

    def __contains__(self, value):
        \"\"\"Return true if `value` is an element of the sorted list.\"\"\"
        _lists = self._lists
        pos, idx = self._loc_left(value)
        return _lists and idx < len(_lists[pos]) and _lists[pos][idx] == value

    def __iter__(self):
        \"\"\"Return an iterator over the sorted list.\"\"\"
        return (value for _list in self._lists for value in _list)

    def __reversed__(self):
        \"\"\"Return a reverse iterator over the sorted list.\"\"\"
        return (value for _list in self._lists[::-1] for value in _list[::-1])

    def __repr__(self):
        \"\"\"Return string representation of sorted list.\"\"\"
        return 'SortedList({0})'.format(
            [value for _list in self._lists for value in _list])


# region fastio

BUFSIZE = 8192


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

input = lambda: sys.stdin.readline().rstrip(\"\r\n\")

# endregion

if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import bisect
import re
from bootcamp import Basebootcamp

class Dirrigationbootcamp(Basebootcamp):
    def __init__(self, m_max=1000, n_max=1000):
        """
        初始化城市数量上限和前n届数量上限
        """
        super().__init__()
        self.m_max = max(1, m_max)   # 确保最小值为1
        self.n_max = max(1, n_max)   # 确保最小值为1
    
    def case_generator(self):
        """
        生成完整的问题实例，包含输入参数和预计算的正确答案
        """
        import random
        m = random.randint(1, self.m_max)  # m至少为1
        n = random.randint(1, self.n_max)  # n至少为1
        a = [random.randint(1, m) for _ in range(n)]  # 保证城市编号有效
        k_val = random.randint(n + 1, n + 10**6)  # 确保k > n
        
        # 生成完整的问题描述字典
        case = {
            'n': n,
            'm': m,
            'a': a,
            'k': k_val,
            'answer': self._compute_answer(n, m, a, k_val)  # 预存正确答案
        }
        return case  # 返回单一字典

    def _compute_answer(self, n, m, a, k_val):
        """计算正确答案的核心算法"""
        if m == 0 or not a:
            return None
        
        counter = [0] * m
        for ai in a:
            counter[ai-1] += 1
        
        # 双重排序键：先按次数升序，再按编号升序
        count_order = sorted(range(m), key=lambda x: (counter[x], x))
        years = [0] * m
        
        # 计算阶段边界
        for i in range(m-1):
            if i+1 >= len(count_order):
                break
            diff = counter[count_order[i+1]] - counter[count_order[i]]
            years[i+1] = years[i] + diff * (i + 1)
        
        # 计算最大有效年份
        max_k = max(counter) * m - sum(counter)
        query_k = k_val - n
        
        # 处理超大规模年份
        if query_k > max_k:
            return ((query_k - max_k - 1) % m) + 1
        else:
            # 二分查找阶段边界
            idx = bisect.bisect_right(years, query_k) - 1
            # 获取候选城市列表
            candidates = sorted(count_order[:idx+1])
            # 计算最终位置
            return candidates[(query_k - years[idx] - 1) % len(candidates)] + 1
    
    @staticmethod
    def prompt_func(question_case):
        """生成符合规范的问题描述"""
        return (
            f"## 奥林匹克主办城市选择问题\n\n"
            f"**已知条件**\n"
            f"- 前 {question_case['n']} 届主办城市：{question_case['a']}\n"
            f"- 共有 {question_case['m']} 个候选城市（编号1-{question_case['m']}）\n\n"
            f"**选择规则**\n"
            f"1. 从第 {question_case['n']+1} 届开始，每年选择历史上主办次数最少的城市\n"
            f"2. 若有多个城市次数相同，选择编号最小的\n\n"
            f"**查询请求**\n"
            f"请计算第 {question_case['k']} 届的主办城市编号，并将答案放置于[answer]和[/answer]之间\n\n"
            f"**答案格式示例**\n"
            f"[answer]3[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        """从模型输出中严格提取答案"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """严格验证答案，identity即case_generator的输出"""
        return solution == identity.get('answer', None)

# 单元测试保障
if __name__ == "__main__":
    # 测试案例1：官方样例输入1
    test_case1 = {
        'n': 6,
        'm': 4,
        'a': [3,1,1,1,2,2],
        'k': 16,
        'answer': 4
    }
    assert Dirrigationbootcamp._verify_correction(4, test_case1)
    
    # 测试案例2：边界值测试
    test_case2 = {
        'n': 1,
        'm': 1,
        'a': [1],
        'k': 2,
        'answer': 1
    }
    assert Dirrigationbootcamp._verify_correction(1, test_case2)
    
    print("所有基本测试通过")
