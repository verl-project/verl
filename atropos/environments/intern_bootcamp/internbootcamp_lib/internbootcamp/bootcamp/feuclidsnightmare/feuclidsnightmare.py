"""# 

### 谜题描述
You may know that Euclid was a mathematician. Well, as it turns out, Morpheus knew it too. So when he wanted to play a mean trick on Euclid, he sent him an appropriate nightmare. 

In his bad dream Euclid has a set S of n m-dimensional vectors over the Z_2 field and can perform vector addition on them. In other words he has vectors with m coordinates, each one equal either 0 or 1. Vector addition is defined as follows: let u+v = w, then w_i = (u_i + v_i) mod 2. 

Euclid can sum any subset of S and archive another m-dimensional vector over Z_2. In particular, he can sum together an empty subset; in such a case, the resulting vector has all coordinates equal 0.

Let T be the set of all the vectors that can be written as a sum of some vectors from S. Now Euclid wonders the size of T and whether he can use only a subset S' of S to obtain all the vectors from T. As it is usually the case in such scenarios, he will not wake up until he figures this out. So far, things are looking rather grim for the philosopher. But there is hope, as he noticed that all vectors in S have at most 2 coordinates equal 1. 

Help Euclid and calculate |T|, the number of m-dimensional vectors over Z_2 that can be written as a sum of some vectors from S. As it can be quite large, calculate it modulo 10^9+7. You should also find S', the smallest such subset of S, that all vectors in T can be written as a sum of vectors from S'. In case there are multiple such sets with a minimal number of elements, output the lexicographically smallest one with respect to the order in which their elements are given in the input. 

Consider sets A and B such that |A| = |B|. Let a_1, a_2, ... a_{|A|} and b_1, b_2, ... b_{|B|} be increasing arrays of indices elements of A and B correspondingly. A is lexicographically smaller than B iff there exists such i that a_j = b_j for all j < i and a_i < b_i.

Input

In the first line of input, there are two integers n, m (1 ≤ n, m ≤ 5 ⋅ 10^5) denoting the number of vectors in S and the number of dimensions. 

Next n lines contain the description of the vectors in S. In each of them there is an integer k (1 ≤ k ≤ 2) and then follow k distinct integers x_1, ... x_k (1 ≤ x_i ≤ m). This encodes an m-dimensional vector having 1s on coordinates x_1, ... x_k and 0s on the rest of them.

Among the n vectors, no two are the same.

Output

In the first line, output two integers: remainder modulo 10^9+7 of |T| and |S'|. In the second line, output |S'| numbers, indices of the elements of S' in ascending order. The elements of S are numbered from 1 in the order they are given in the input.

Examples

Input


3 2
1 1
1 2
2 2 1


Output


4 2
1 2 


Input


2 3
2 1 3
2 1 2


Output


4 2
1 2 


Input


3 5
2 1 2
1 3
1 4


Output


8 3
1 2 3 

Note

In the first example we are given three vectors: 

  * 10 
  * 01 
  * 11 



It turns out that we can represent all vectors from our 2-dimensional space using these vectors: 

  * 00 is a sum of the empty subset of above vectors; 
  * 01 = 11 + 10, is a sum of the first and third vector; 
  * 10 = 10, is just the first vector; 
  * 11 = 10 + 01, is a sum of the first and the second vector. 



Hence, T = \{00, 01, 10, 11\}. We can choose any two of the three vectors from S and still be able to obtain all the vectors in T. In such a case, we choose the two vectors which appear first in the input. Since we cannot obtain all vectors in T using only a single vector from S, |S'| = 2 and S' = \{10, 01\} (indices 1 and 2), as set \{1, 2 \} is lexicographically the smallest. We can represent all vectors from T, using only vectors from S', as shown below: 

  * 00 is a sum of the empty subset; 
  * 01 = 01 is just the second vector; 
  * 10 = 10 is just the first vector; 
  * 11 = 10 + 01 is a sum of the first and the second vector. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#   Author: yumtam
#   Created at: 2020-12-31 01:27

from __future__ import division, print_function
_interactive = False

def main():
    n, m = input_as_list()
    A = [input_with_offset(-1)[1:] for _ in range(n)]

    uf = DisjointSetUnion(m)
    dims = 0
    for e in A:
        if len(e) == 1:
            x = e[0]
            if not uf.is_done(x):
                uf.set_done(x)
                dims += 1
        elif len(e) == 2:
            x, y = e
            if uf.find(x) != uf.find(y):
                dx, dy = uf.is_done(x), uf.is_done(y)
                both = dx and dy
                done = dx or dy
                uf.union(x, y)
                if not both:
                    dims += 1
                if done:
                    uf.set_done(x)

    print(pow(2, dims, MOD), dims)

    uf = DisjointSetUnion(m)
    dims_orig = dims
    dims = 0
    ans = []
    for i, e in enumerate(A, start=1):
        if len(e) == 1:
            x = e[0]
            if not uf.is_done(x):
                uf.set_done(x)
                dims += 1
                ans += [i]
        elif len(e) == 2:
            x, y = e
            if uf.find(x) != uf.find(y):
                dx, dy = uf.is_done(x), uf.is_done(y)
                both = dx and dy
                done = dx or dy
                uf.union(x, y)
                if not both:
                    dims += 1
                    ans += [i]
                if done:
                    uf.set_done(x)
        if dims == dims_orig:
            break
    print(*ans, sep=' ')


# Constants
INF = float('inf')
MOD = 10**9+7

# Python3 equivalent names
import os, sys, itertools
if sys.version_info[0] < 3:
    input = raw_input
    range = xrange

    filter = itertools.ifilter
    map = itertools.imap
    zip = itertools.izip

# print-flush in interactive problems
if _interactive:
    flush = sys.stdout.flush
    def printf(*args, **kwargs):
        print(*args, **kwargs)
        flush()

# Debug print, only works on local machine
LOCAL = \"LOCAL_\" in os.environ
debug_print = (print) if LOCAL else (lambda *x, **y: None)

# Fast IO
if (not LOCAL) and (not _interactive):
    from io import BytesIO
    from atexit import register
    sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
    sys.stdout = BytesIO()
    register(lambda: os.write(1, sys.stdout.getvalue()))
    input = lambda: sys.stdin.readline().rstrip('\r\n')

# Some utility functions(Input, N-dimensional lists, ...)
def input_as_list():
    return [int(x) for x in input().split()]

def input_with_offset(o):
    return [int(x)+o for x in input().split()]

def input_as_matrix(n, m):
    return [input_as_list() for _ in range(n)]

def array_of(f, *dim):
    return [array_of(f, *dim[1:]) for _ in range(dim[0])] if dim else f()

# Start of external code templates...
class DisjointSetUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.num_sets = n
        self.done = [0] * n

    def is_done(self, a):
        return self.done[self.find(a)]

    def set_done(self, a):
        self.done[self.find(a)] = 1

    def find(self, a):
        acopy = a
        while a != self.parent[a]:
            a = self.parent[a]
        while acopy != a:
            self.parent[acopy], acopy = a, self.parent[acopy]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a

            self.num_sets -= 1
            self.parent[b] = a
            self.size[a] += self.size[b]

    def set_size(self, a):
        return self.size[self.find(a)]

    def __len__(self):
        return self.num_sets

# End of external code templates.

main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class DisjointSetUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.num_sets = n
        self.done = [0] * n  # 0: not done, 1: done

    def find(self, a):
        acopy = a
        while a != self.parent[a]:
            a = self.parent[a]
        while acopy != a:
            self.parent[acopy], acopy = a, self.parent[acopy]
        return a

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a
            self.parent[b] = a
            self.size[a] += self.size[b]
            self.num_sets -= 1

    def is_done(self, a):
        return self.done[self.find(a)]

    def set_done(self, a):
        self.done[self.find(a)] = 1

class Feuclidsnightmarebootcamp(Basebootcamp):
    def __init__(self, min_m=2, max_m=5, min_n=1, max_n=10, **kwargs):
        super().__init__(**kwargs)
        self.min_m = min_m
        self.max_m = max_m
        self.min_n = min_n
        self.max_n = max_n

    def case_generator(self):
        m = random.randint(self.min_m, self.max_m)
        all_vectors = []
        # Generate all possible single-coordinate vectors
        for x in range(1, m + 1):
            all_vectors.append({'k': 1, 'x': [x]})
        # Generate all possible two-coordinate vectors, sorted
        for x in range(1, m + 1):
            for y in range(x + 1, m + 1):
                all_vectors.append({'k': 2, 'x': [x, y]})
        max_possible_n = len(all_vectors)
        if max_possible_n == 0:
            max_possible_n = 1
        n_min = max(self.min_n, 1)
        n_max = min(self.max_n, max_possible_n)
        if n_min > n_max:
            n_min = 1
            n_max = max_possible_n
        n = random.randint(n_min, n_max)
        selected_vectors = random.sample(all_vectors, n)
        random.shuffle(selected_vectors)
        vectors_0based = []
        for vec in selected_vectors:
            if vec['k'] == 1:
                x = vec['x'][0]
                vectors_0based.append([x - 1])
            else:
                x, y = vec['x']
                vectors_0based.append([x - 1, y - 1])
        # Compute correct size_mod
        uf1 = DisjointSetUnion(m)
        dims = 0
        for e in vectors_0based:
            if len(e) == 1:
                x = e[0]
                if not uf1.is_done(x):
                    uf1.set_done(x)
                    dims += 1
            elif len(e) == 2:
                x, y = e
                if uf1.find(x) != uf1.find(y):
                    dx = uf1.is_done(x)
                    dy = uf1.is_done(y)
                    both = dx and dy
                    done = dx or dy
                    uf1.union(x, y)
                    if not both:
                        dims += 1
                    if done:
                        uf1.set_done(x)
        size_mod = pow(2, dims, MOD)
        # Compute ans list
        uf2 = DisjointSetUnion(m)
        current_dims = 0
        ans = []
        for i, e in enumerate(vectors_0based, start=1):
            if current_dims == dims:
                break
            if len(e) == 1:
                x = e[0]
                if not uf2.is_done(x):
                    uf2.set_done(x)
                    current_dims += 1
                    ans.append(i)
            elif len(e) == 2:
                x, y = e
                if uf2.find(x) != uf2.find(y):
                    dx = uf2.is_done(x)
                    dy = uf2.is_done(y)
                    both = dx and dy
                    done = dx or dy
                    uf2.union(x, y)
                    if not both:
                        current_dims += 1
                        ans.append(i)
                    if done:
                        uf2.set_done(x)
        # Ensure the indices are sorted in the answer
        ans = sorted(ans)
        return {
            'n': n,
            'm': m,
            'vectors': selected_vectors,
            'correct_size': size_mod,
            'correct_indices': ans
        }

    @staticmethod
    def prompt_func(question_case):
        vectors_desc = []
        for vec in question_case['vectors']:
            k = vec['k']
            coords = ' '.join(map(str, vec['x']))
            vectors_desc.append(f"{k} {coords}")
        vectors_str = '\n'.join(vectors_desc)
        prompt = f"""You are the mathematician Euclid, tasked with solving a vector space problem over Z₂. Given a set S of {question_case['n']} vectors in {question_case['m']}-dimensional space, where each vector has at most 2 coordinates set to 1, determine two things:

1. The size of the set T, which consists of all possible vectors obtainable by summing subsets of S (modulo 10⁹+7).
2. The smallest subset S' of S such that every vector in T can be expressed as a sum of elements from S'. If multiple such subsets exist, choose the lexicographically smallest one based on their original order in the input.

Input:
The first line contains two integers n and m.
The next n lines describe each vector with a value k followed by k distinct coordinates (1-based).

Output:
Two lines. The first line contains |T| mod 10⁹+7 and the size of S'. The second line lists the indices of S' in ascending order.

Sample Input:
{question_case['n']} {question_case['m']}
{vectors_str}

Enclose your final answer within [answer] and [/answer] tags. For example:

[answer]
4 2
1 2
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        size_line = lines[0].split()
        if len(size_line) != 2:
            return None
        try:
            size = int(size_line[0])
            s_prime_size = int(size_line[1])
        except ValueError:
            return None
        indices_line = lines[1].split()
        try:
            indices = list(map(int, indices_line))
            if len(indices) != s_prime_size:
                return None
        except ValueError:
            return None
        return {
            'size': size,
            's_prime_size': s_prime_size,
            'indices': indices
        }

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        if solution['size'] != identity['correct_size']:
            return False
        if solution['s_prime_size'] != len(identity['correct_indices']):
            return False
        if solution['indices'] != identity['correct_indices']:
            return False
        return True
