"""# 

### 谜题描述
On the competitive programming platform CodeCook, every person has a rating graph described by an array of integers a of length n. You are now updating the infrastructure, so you've created a program to compress these graphs.

The program works as follows. Given an integer parameter k, the program takes the minimum of each contiguous subarray of length k in a.

More formally, for an array a of length n and an integer k, define the k-compression array of a as an array b of length n-k+1, such that $$$b_j =min_{j≤ i≤ j+k-1}a_i$$$

For example, the 3-compression array of [1, 3, 4, 5, 2] is [min\{1, 3, 4\}, min\{3, 4, 5\}, min\{4, 5, 2\}]=[1, 3, 2].

A permutation of length m is an array consisting of m distinct integers from 1 to m in arbitrary order. For example, [2,3,1,5,4] is a permutation, but [1,2,2] is not a permutation (2 appears twice in the array) and [1,3,4] is also not a permutation (m=3 but there is 4 in the array).

A k-compression array will make CodeCook users happy if it will be a permutation. Given an array a, determine for all 1≤ k≤ n if CodeCook users will be happy after a k-compression of this array or not.

Input

The first line contains a single integer t (1≤ t≤ 10^4) — the number of test cases.

The first line of the description of each test case contains a single integer n (1≤ n≤ 3⋅ 10^5) — the length of the array.

The second line of the description of each test case contains n integers a_1,…,a_n (1≤ a_i≤ n) — the elements of the array.

It is guaranteed, that the sum of n for all test cases does not exceed 3⋅ 10^5.

Output

For each test case, print a binary string of length n. 

The k-th character of the string should be 1 if CodeCook users will be happy after a k-compression of the array a, and 0 otherwise. 

Example

Input


5
5
1 5 3 4 2
4
1 3 2 1
5
1 3 3 3 2
10
1 2 3 4 5 6 7 8 9 10
3
3 3 2


Output


10111
0001
00111
1111111111
000

Note

In the first test case, a=[1, 5, 3, 4, 2].

  * The 1-compression of a is [1, 5, 3, 4, 2] and it is a permutation. 
  * The 2-compression of a is [1, 3, 3, 2] and it is not a permutation, since 3 appears twice. 
  * The 3-compression of a is [1, 3, 2] and it is a permutation. 
  * The 4-compression of a is [1, 2] and it is a permutation. 
  * The 5-compression of a is [1] and it is a permutation. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
if sys.subversion[0] == \"PyPy\":
    import io, atexit
    sys.stdout = io.BytesIO()
    atexit.register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))
    
    sys.stdin = io.BytesIO(sys.stdin.read())
    input = lambda: sys.stdin.readline().rstrip()

RS = raw_input
RI = lambda x=int: map(x,RS().split())
RN = lambda x=int: x(RS())
''' ...................................................................... '''
class SegTree:
    ''' PU-RQ '''
 
    def __init__(self,arr,n):
        ''' Builds in O(N) '''
        
        self.N = n
        self.tree = [float('inf')]*(2*n)
 
        for i in xrange(n):     # Leaf Node
            self.tree[n+i] = arr[i]
 
        for i in xrange(n-1,0,-1):  # Parent Node
            self.tree[i] = min(self.tree[2*i] , self.tree[2*i+1])
 
    
    def update(self, i, newValue):
        ''' 0-Based Indexing '''
        
        i += self.N     # Points to leaf node (i)
 
        self.tree[i] = newValue
 
        while i>1:
            i/=2        # Points to parent node
            self.tree[i] = min(self.tree[2*i] , self.tree[2*i+1])
 
 
    def query(self, L, R):
        ''' Query in interval [L,R] (0-Based Indexing) '''
        
        # Leaf Nodes
        L += self.N
        R += self.N + 1
 
        res = float('inf')
        while L<R:
            if L&1:
                res = min(res,self.tree[L])
                #res += self.tree[L]     # Right Child
                L += 1
                
            if R&1:
                R -= 1
                res = min(res,self.tree[R])
                #res += self.tree[R]     # Left Child
                
            L/=2; R/=2
        return res

    
for _ in xrange(RN()):
    n = RN()
    arr = RI()
    seg = SegTree(arr,n)
    
    ans = [0]*n
    lo,hi = 0,n-1

    val = 1
    
    for k in xrange(n,1,-1):
        m = seg.query(lo,hi)

        if m==val:
            ans[k-1]=1

            if arr[lo]==m: lo+=1
            elif arr[hi]==m: hi-=1
            else: break
        else:
            break

        val+=1


    if len(set(arr))==n: ans[0]=1
        
    print ''.join(map(str,ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
from bootcamp import Basebootcamp

class Dratingcompressionbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_n = params.get('max_n', 10)
        self.min_n = params.get('min_n', 1)
    
    def case_generator(self):
        """生成多样化测试案例，包含：完全随机、保证有效k=1、强制全无效等类型"""
        case_type = random.choice(['random', 'valid_k1', 'invalid_all'])
        n = random.randint(self.min_n, self.max_n)
        
        if case_type == 'valid_k1':
            # 生成保证k=1有效的案例（数组本身就是排列）
            a = list(range(1, n+1))
            random.shuffle(a)
        elif case_type == 'invalid_all':
            # 生成所有k都无效的案例（数组元素全相同）
            a = [1] * n
        else:
            # 完全随机生成
            a = [random.randint(1, n) for _ in range(n)]
        
        correct_answer = self.optimized_solve(n, a)
        return {
            'n': n,
            'a': a,
            'correct_answer': correct_answer,
            'case_type': case_type  # 用于验证时追踪
        }

    def optimized_solve(self, n, a):
        """准确高效的解法实现"""
        answer = ['0'] * n
        
        # 预处理k=1的情况
        k1_valid = (sorted(a) == list(range(1, n+1)))
        answer[0] = '1' if k1_valid else '0'
        
        # 预处理每个位置的next smaller元素
        next_smaller = [n] * n
        prev_smaller = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and a[i] < a[stack[-1]]:
                next_smaller[stack.pop()] = i
            prev_smaller[i] = stack[-1] if stack else -1
            stack.append(i)
        
        # 统计每个元素作为最小值的影响范围
        min_intervals = {}
        for i in range(n):
            left = prev_smaller[i] + 1
            right = next_smaller[i] - 1
            min_intervals[a[i]] = max(min_intervals.get(a[i], 0), right - left + 1)
        
        # 根据定理：当且仅当存在元素只能在窗口大小>=某个值时出现
        for m in range(1, n):
            max_k = n - m + 1
            if m in min_intervals and min_intervals[m] >= m:
                for k in range(max(1, m), max_k+1):
                    if k <= min_intervals[m]:
                        answer[k-1] = '1'
        
        # 最终验证每个k的结果
        for k in range(1, n+1):
            m = n - k + 1
            if m < 1:
                continue
            if answer[k-1] == '1':
                # 二次验证确保正确性
                window_min = self.sliding_window_min(a, k)
                if not self.is_permutation(window_min, m):
                    answer[k-1] = '0'
        return ''.join(answer)

    @staticmethod
    def sliding_window_min(arr, k):
        """精确计算滑动窗口的最小值"""
        dq = deque()
        result = []
        for i, num in enumerate(arr):
            while dq and arr[dq[-1]] >= num:
                dq.pop()
            dq.append(i)
            
            if dq[0] == i - k:
                dq.popleft()
            
            if i >= k - 1:
                result.append(arr[dq[0]])
        return result

    @staticmethod
    def is_permutation(nums, m):
        """验证是否为1~m的排列"""
        return len(nums) == m and set(nums) == set(range(1, m+1)) and len(set(nums)) == m

    @staticmethod
    def prompt_func(question_case) -> str:
        a_str = ' '.join(map(str, question_case['a']))
        n = question_case['n']
        return f"""给定长度为{n}的数组：[{a_str}]
请对k=1到k={n}依次判断：
1. 计算k-compression数组（每个元素是连续k个元素的最小值）
2. 检查该数组是否是1到(n-k+1)的排列

输出：长度为{n}的二进制字符串，第k位为1表示有效。答案置于[answer][/answer]中。例如：[answer]1010[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 格式验证
        if not solution or len(solution) != identity['n']:
            return False
        
        # 逻辑验证（兼容可能存在多个正确解的情况）
        expected = identity['correct_answer']
        
        # 检查每个有效位的合理性
        for k in range(1, identity['n']+1):
            if solution[k-1] == '1' and expected[k-1] == '0':
                return False
            if identity['case_type'] == 'invalid_all' and '1' in solution:
                return False
        return solution == expected
