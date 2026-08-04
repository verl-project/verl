"""# 

### 谜题描述
Nezzar loves the game osu!.

osu! is played on beatmaps, which can be seen as an array consisting of distinct points on a plane. A beatmap is called nice if for any three consecutive points A,B,C listed in order, the angle between these three points, centered at B, is strictly less than 90 degrees.

<image> Points A,B,C on the left have angle less than 90 degrees, so they can be three consecutive points of a nice beatmap; Points A',B',C' on the right have angle greater or equal to 90 degrees, so they cannot be three consecutive points of a nice beatmap.

Now Nezzar has a beatmap of n distinct points A_1,A_2,…,A_n. Nezzar would like to reorder these n points so that the resulting beatmap is nice.

Formally, you are required to find a permutation p_1,p_2,…,p_n of integers from 1 to n, such that beatmap A_{p_1},A_{p_2},…,A_{p_n} is nice. If it is impossible, you should determine it.

Input

The first line contains a single integer n (3 ≤ n ≤ 5000).

Then n lines follow, i-th of them contains two integers x_i, y_i (-10^9 ≤ x_i, y_i ≤ 10^9) — coordinates of point A_i.

It is guaranteed that all points are distinct.

Output

If there is no solution, print -1.

Otherwise, print n integers, representing a valid permutation p.

If there are multiple possible answers, you can print any.

Example

Input


5
0 0
5 0
4 2
2 1
3 0


Output


1 2 5 3 4

Note

Here is the illustration for the first test:

<image>

Please note that the angle between A_1, A_2 and A_5, centered at A_2, is treated as 0 degrees. However, angle between A_1, A_5 and A_2, centered at A_5, is treated as 180 degrees.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include<bits/stdc++.h>
#define pb push_back
#define fi first
#define se second
#define all(v) (v).begin(), (v).end()
using namespace std;
typedef long long LL;
typedef pair<int, int> pii;
typedef pair<LL, LL> pil;
const int MAX_N = 5005;
const LL MOD = 1000000007LL;
const int inf = 0x3fffffff;
int n;
bool chk[MAX_N];
pil arr[MAX_N];
LL get_dst(int u, int v){
	return (arr[u].fi - arr[v].fi) * (arr[u].fi - arr[v].fi)
			+ (arr[u].se - arr[v].se) * (arr[u].se - arr[v].se);
}
int get_far(int pre){
	int res = pre;
	for(int i=1;i<=n;i++){
		if(chk[i])	continue;
		if(get_dst(pre, res) < get_dst(pre, i))	res = i;
	}
	return res;
}
int main(){
	scanf(\"%d\",&n);
	for(int i=1;i<=n;i++){
		scanf(\"%lld %lld\",&arr[i].fi, &arr[i].se);
	}
	int pre = 1;
	chk[pre] = true;
	printf(\"%d \",pre);
	for(int i=2;i<=n;i++){
		int curr = get_far(pre);
		printf(\"%d \",curr);
		chk[pre = curr] = true;
	}
	printf(\"\n\");
	return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cnezzarandnicebeatmapbootcamp(Basebootcamp):
    def __init__(self, n=5):
        """
        初始化训练场参数，允许自定义点的数量n。
        """
        self.n = n
    
    def case_generator(self):
        """
        生成n个不同的点，并使用贪心算法生成有效排列。确保每个生成的实例都有解。
        """
        while True:
            n = self.n
            points = []
            # 生成n个不重复的点
            while len(points) < n:
                x = random.randint(-10**9, 10**9)
                y = random.randint(-10**9, 10**9)
                if (x, y) not in points:
                    points.append((x, y))
            # 生成排列并验证
            try:
                permutation = self.generate_permutation(n, points)
                if self._verify_correction(permutation, {'n': n, 'points': points}):
                    return {'n': n, 'points': points}
            except:
                continue
    
    @staticmethod
    def generate_permutation(n, points):
        """
        参考解题算法生成排列，正确处理1-based到0-based的索引转换。
        """
        used = [False] * (n + 1)  # 使用1-based索引
        permutation = []
        pre = 1  # 初始化为第一个点（1-based）
        used[pre] = True
        permutation.append(pre)
        
        for _ in range(n - 1):
            max_dist = -1
            curr = pre
            for i in range(1, n + 1):  # 遍历所有1-based索引
                if not used[i]:
                    # 计算当前点（pre）到候选点i的距离
                    dx = points[i-1][0] - points[pre-1][0]  # 转换为0-based索引
                    dy = points[i-1][1] - points[pre-1][1]
                    dist = dx * dx + dy * dy
                    if dist > max_dist:
                        max_dist = dist
                        curr = i
            permutation.append(curr)
            used[curr] = True
            pre = curr
        return permutation
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        生成符合题目要求的详细问题描述，明确答案格式。
        """
        n = question_case['n']
        points = question_case['points']
        problem = [
            "Nezzar wants to reorder points to form a 'nice' beatmap where each triplet has an angle <90° at the center point.",
            f"Given {n} distinct points:"
        ]
        for idx, (x, y) in enumerate(points, 1):
            problem.append(f"Point {idx}: ({x}, {y})")
        problem.append(
            "Output a valid permutation (space-separated numbers) or -1 if impossible.\n"
            "Put your answer within [answer] and [/answer], e.g., [answer]1 2 3[/answer]."
        )
        return '\n'.join(problem)
    
    @staticmethod
    def extract_output(output):
        """
        严格提取最后一个[answer]标签内的答案。
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if last_match == '-1':
            return [-1]
        try:
            return list(map(int, last_match.split()))
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证排列是否满足所有连续三点角度小于90度。
        """
        if solution == [-1]:
            return False  # 该训练场生成的实例保证有解
        
        n = identity['n']
        points = identity['points']
        if len(solution) != n or set(solution) != set(range(1, n+1)):
            return False
        
        # 检查每个连续三点A,B,C的向量点积
        for i in range(n - 2):
            a, b, c = solution[i], solution[i+1], solution[i+2]
            ax, ay = points[a-1]
            bx, by = points[b-1]
            cx, cy = points[c-1]
            # 向量BA和BC
            ba_x = ax - bx
            ba_y = ay - by
            bc_x = cx - bx
            bc_y = cy - by
            dot_product = ba_x * bc_x + ba_y * bc_y
            if dot_product <= 0:
                return False
        return True
