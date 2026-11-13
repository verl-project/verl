"""# 

### 谜题描述
This is the easy version of the problem. The only difference is that in this version q = 1. You can make hacks only if both versions of the problem are solved.

There is a process that takes place on arrays a and b of length n and length n-1 respectively. 

The process is an infinite sequence of operations. Each operation is as follows: 

  * First, choose a random integer i (1 ≤ i ≤ n-1). 
  * Then, simultaneously set a_i = min\left(a_i, \frac{a_i+a_{i+1}-b_i}{2}\right) and a_{i+1} = max\left(a_{i+1}, \frac{a_i+a_{i+1}+b_i}{2}\right) without any rounding (so values may become non-integer). 

See notes for an example of an operation.

It can be proven that array a converges, i. e. for each i there exists a limit a_i converges to. Let function F(a, b) return the value a_1 converges to after a process on a and b.

You are given array b, but not array a. However, you are given a third array c. Array a is good if it contains only integers and satisfies 0 ≤ a_i ≤ c_i for 1 ≤ i ≤ n.

Your task is to count the number of good arrays a where F(a, b) ≥ x for q values of x. Since the number of arrays can be very large, print it modulo 10^9+7.

Input

The first line contains a single integer n (2 ≤ n ≤ 100).

The second line contains n integers c_1, c_2 …, c_n (0 ≤ c_i ≤ 100).

The third line contains n-1 integers b_1, b_2, …, b_{n-1} (0 ≤ b_i ≤ 100).

The fourth line contains a single integer q (q=1).

The fifth line contains q space separated integers x_1, x_2, …, x_q (-10^5 ≤ x_i ≤ 10^5).

Output

Output q integers, where the i-th integer is the answer to the i-th query, i. e. the number of good arrays a where F(a, b) ≥ x_i modulo 10^9+7.

Example

Input


3
2 3 4
2 1
1
-1


Output


56

Note

The following explanation assumes b = [2, 1] and c=[2, 3, 4] (as in the sample).

Examples of arrays a that are not good: 

  * a = [3, 2, 3] is not good because a_1 > c_1; 
  * a = [0, -1, 3] is not good because a_2 < 0. 



One possible good array a is [0, 2, 4]. We can show that no operation has any effect on this array, so F(a, b) = a_1 = 0.

Another possible good array a is [0, 1, 4]. In a single operation with i = 1, we set a_1 = min((0+1-2)/(2), 0) and a_2 = max((0+1+2)/(2), 1). So, after a single operation with i = 1, a becomes equal to [-1/2, 3/2, 4]. We can show that no operation has any effect on this array, so F(a, b) = -1/2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#pragma GCC optimize(\"Ofast\")
#pragma GCC optimize(\"unroll-loops\")
#pragma GCC target(\"sse,sse2,sse3,ssse3,abm,mmx,tune=native\")
#include<vector>
#include<iostream>
#include<stack>
#include<cmath>
#include<algorithm>
#include<set>
#include<map>
#include<string>
#include<tuple>
#include<bitset>
#include<queue>
#include<unordered_map>
#include<random>
#include<ctime>
//#include<complex>
#include<numeric>
typedef long long ll;
typedef long double ld;
typedef unsigned short us;
typedef unsigned long long ull;
//typedef complex<double> base;
using namespace std;
ll gcd(ll i, ll j) {
	if (j == 0)return i;
	else return gcd(j, i % j);
}
#ifdef _DEBUG
int __builtin_popcount(int x) { return x ? (__builtin_popcount(x >> 1) + (x & 1)) : 0; }
#endif
template<typename T> inline T getint() {
	T val = 0;
	char c;

	bool neg = false;
	while ((c = getchar()) && !(c >= '0' && c <= '9')) {
		neg |= c == '-';
	}

	do {
		val = (val * 10) + c - '0';
	} while ((c = getchar()) && (c >= '0' && c <= '9'));

	return val * (neg ? -1 : 1);
}
//#define int long long
const ll INF = 1e9 + 100;
const int mod = 1000000007;
const ld eps = 1e-6, pi = acosl(-1);
const ll maxN = 10210, maxT = 10010, A = 179, mid = 150;
mt19937 mt_rand(time(0));
ll bp(ll et, ll b) {
	b %= mod - 1;
	ll res = 1;
	for (int i = 30; i >= 0; --i) {
		res = (res * res) % mod;
		if ((b & (1 << i)) != 0)res = (res * et) % mod;
	}
	return res;
}
void panic() {
	cout << \"-1\n\";
	exit(0);
}
int pl(const int& a, const int& b) {
	int r = a + b;
	if (r >= mod)r -= mod;
	return r;
}
vector<ll>c, b;
map<ll, ll>mp;
int n;
bool check_0(ll x) {
	ll d = 0, s = 0;
	bool ff = 1;
	for (int i = 0; i < n; ++i) {
		if (i) {
			d += b[i - 1];
			s += d;
		}
		ll vv = x * (i + 1) + s;
		ff &= vv <= 0;
	}
	return ff;
}
bool check_1(ll x) {
	ll d = 0, s = 0;
	bool ff = 0;
	ll sum_s = 0;
	for (int i = 0; i < n; ++i) {
		sum_s += c[i];
		if (i) {
			d += b[i - 1];
			s += d;
		}
		ll vv = x * (i + 1) + s;
		ff |= sum_s < vv;
	}
	return ff;
}
ll get_r(ll x) {
	if (mp.count(x))
		return mp[x];
	if (check_0(x)) {
		ll r = 1;
		for (auto x : c) {
			r = (r * (x + 1)) % mod;
		}
		return mp[x] = r;
	}
	if (check_1(x)) {
		return mp[x] = 0;
	}
	ll d = 0, s = 0;
	vector<vector<ll>>dp(n + 1, vector<ll>(maxN));
	dp[0][0] = 1;
	for (int i = 0; i < n; ++i) {
		if (i) {
			d += b[i - 1];
			s += d;
		}
		ll vv = x * (i + 1) + s;
		ll cc = c[i];
		for (int j = maxN - cc - 1; j >= 0; --j) {
			for (int k = j + cc; k >= j && k >= vv; --k) {
				dp[i + 1][k] = pl(dp[i + 1][k], dp[i][j]);
			}
		}
	}
	ll ans = 0;
	for (int i = 0; i < maxN; ++i)
		ans = pl(ans, dp[n][i]);
	return mp[x] = ans;
}
void solve() {
	cin >> n;
	c.assign(n, 0);
	b.assign(n - 1, 0);
	for (auto& x : c)
		cin >> x;
	for (auto& x : b)
		cin >> x;
	int q;
	cin >> q;
	while (q--) {
		ll x;
		cin >> x;
		cout << get_r(x) << \"\n\";
	}
}
int32_t main()
{
	ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
	cout.precision(20);
	//srand(time(0));
#ifdef _DEBUG
	freopen(\"input.txt\", \"r\", stdin); freopen(\"output.txt\", \"w\", stdout);
#else
	//freopen(\"gymnasts.in\", \"r\", stdin); freopen(\"gymnasts.out\", \"w\", stdout);
#endif
	int t = 1;
	//cin >> t;
	while (t--) {
		solve();
	}
	return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class E1convergingarrayeasyversionbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 2)
        self.n_max = params.get('n_max', 5)
        self.c_min = params.get('c_min', 0)
        self.c_max = params.get('c_max', 100)
        self.b_min = params.get('b_min', 0)
        self.b_max = params.get('b_max', 100)
        self.x_min = params.get('x_min', -100000)
        self.x_max = params.get('x_max', 100000)
    
    def case_generator(self):
        # 多策略生成有效案例
        generation_strategies = [
            self._generate_simple_case,
            self._generate_zero_c_case,
            self._generate_max_b_case,
            self._generate_negative_x_case
        ]
        
        for strategy in generation_strategies:
            case = strategy()
            if self._validate_case(case):
                return case
        
        return self._default_case()

    def _generate_simple_case(self):
        n = random.randint(2, 3)
        c = [random.randint(5, 10) for _ in range(n)]
        b = [random.randint(0, 2) for _ in range(n-1)]
        x = random.randint(-10, 0)
        return {'n':n, 'c':c, 'b':b, 'x':x}

    def _generate_zero_c_case(self):
        n = 3
        c = [0] + [random.randint(0, 5) for _ in range(n-1)]
        b = [random.randint(0, 5) for _ in range(n-1)]
        x = random.randint(-5, 0)
        return {'n':n, 'c':c, 'b':b, 'x':x}

    def _generate_max_b_case(self):
        n = random.randint(2, 4)
        c = [random.randint(50, 100) for _ in range(n)]
        b = [100] * (n-1)
        x = random.randint(-100, 0)
        return {'n':n, 'c':c, 'b':b, 'x':x}

    def _generate_negative_x_case(self):
        n = random.randint(2, 3)
        c = [random.randint(10, 100) for _ in range(n)]
        b = [random.randint(0, 10) for _ in range(n-1)]
        x = random.randint(-1000, -100)
        return {'n':n, 'c':c, 'b':b, 'x':x}

    def _validate_case(self, case):
        try:
            result = self._get_r(case['x'], case['n'], case['c'], case['b'])
            return result >= 0
        except:
            return False

    def _default_case(self):
        return {
            'n': 3,
            'c': [2, 3, 4],
            'b': [2, 1],
            'x': -1
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        # ...保持原有prompt结构，优化问题描述...
        problem_desc = f"""
Calculate the number of valid arrays for given parameters. Put the final answer within [answer] tags.

Problem Parameters:
n = {question_case['n']}
c = {' '.join(map(str, question_case['c']))}
b = {' '.join(map(str, question_case['b']))}
x = {question_case['x']}

Constraints:
- Array a must satisfy 0 ≤ a_i ≤ c_i for all i
- The answer should be modulo {MOD}
- Format answer as: [answer]N[/answer] where N is the computed value

Example Valid Response:
[answer]56[/answer]
"""
        return problem_desc

    @staticmethod
    def extract_output(output):
        # 增强模式匹配鲁棒性
        matches = re.findall(r'\[answer\s*\][\n\s]*(-?\d+)[\n\s]*\[/answer\s*\]', output, re.IGNORECASE)
        if not matches:
            return None
        try:
            return int(matches[-1].strip()) % MOD
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = cls._get_r(
                identity['x'],
                identity['n'],
                identity['c'],
                identity['b']
            ) % MOD
            return solution == expected
        except:
            return False

    # 保持核心算法不变，增加防御性编程
    @staticmethod
    def _check_0(x, n, b):
        d = s = 0
        for i in range(n):
            if i > 0:
                d += b[i-1]
                s += d
            if x*(i+1) + s > 0:
                return False
        return True

    @staticmethod
    def _check_1(x, n, c, b):
        sum_s = 0
        d = s = 0
        for i in range(n):
            sum_s += c[i]
            if i > 0:
                d += b[i-1]
                s += d
            if sum_s < x*(i+1) + s:
                return True
        return False

    @classmethod
    def _compute_dp(cls, x, n, c, b):
        maxN = 10210
        dp = [[0]*maxN for _ in range(n+1)]
        dp[0][0] = 1
        d = s = 0

        for i in range(n):
            if i > 0:
                d += b[i-1]
                s += d
            current_v = x*(i+1) + s
            max_c = c[i]

            # 动态规划优化
            for j in range(maxN):
                if dp[i][j] == 0:
                    continue
                
                min_val = max(current_v, j)
                max_val = min(j + max_c, maxN-1)

                if min_val > max_val:
                    continue

                # 批量更新区间
                dp[i+1][min_val] = (dp[i+1][min_val] + dp[i][j]) % MOD
                if max_val + 1 < maxN:
                    dp[i+1][max_val+1] = (dp[i+1][max_val+1] - dp[i][j]) % MOD

            # 前缀和优化
            prefix = 0
            for j in range(maxN):
                prefix = (prefix + dp[i+1][j]) % MOD
                dp[i+1][j] = prefix

        return sum(dp[n]) % MOD

    @classmethod
    def _get_r(cls, x, n, c, b):
        # 添加输入验证
        if any(ci < 0 for ci in c):
            raise ValueError("Invalid c array")
        if any(bi < 0 for bi in b):
            raise ValueError("Invalid b array")

        if cls._check_0(x, n, b):
            product = 1
            for ci in c:
                product = (product * (ci + 1)) % MOD
            return product
        if cls._check_1(x, n, c, b):
            return 0
        return cls._compute_dp(x, n, c, b)
