"""# 

### 谜题描述
Cirno has prepared n arrays of length n each. Each array is a permutation of n integers from 1 to n. These arrays are special: for all 1 ≤ i ≤ n, if we take the i-th element of each array and form another array of length n with these elements, the resultant array is also a permutation of n integers from 1 to n. In the other words, if you put these n arrays under each other to form a matrix with n rows and n columns, this matrix is a [Latin square](https://en.wikipedia.org/wiki/Latin_square).

Afterwards, Cirno added additional n arrays, each array is a permutation of n integers from 1 to n. For all 1 ≤ i ≤ n, there exists at least one position 1 ≤ k ≤ n, such that for the i-th array and the (n + i)-th array, the k-th element of both arrays is the same. Notice that the arrays indexed from n + 1 to 2n don't have to form a Latin square. 

Also, Cirno made sure that for all 2n arrays, no two arrays are completely equal, i. e. for all pair of indices 1 ≤ i < j ≤ 2n, there exists at least one position 1 ≤ k ≤ n, such that the k-th elements of the i-th and j-th array are different.

Finally, Cirno arbitrarily changed the order of 2n arrays.

AquaMoon calls a subset of all 2n arrays of size n good if these arrays from a Latin square.

AquaMoon wants to know how many good subsets exist. Because this number may be particularly large, find it modulo 998 244 353. Also, she wants to find any good subset. Can you help her?

Input

The input consists of multiple test cases. The first line contains a single integer t (1 ≤ t ≤ 100) — the number of test cases.

The first line of each test case contains a single integer n (5 ≤ n ≤ 500).

Then 2n lines followed. The i-th of these lines contains n integers, representing the i-th array.

It is guaranteed, that the sum of n over all test cases does not exceed 500.

Output

For each test case print two lines.

In the first line, print the number of good subsets by modulo 998 244 353.

In the second line, print n indices from 1 to 2n — indices of the n arrays that form a good subset (you can print them in any order). If there are several possible answers — print any of them.

Example

Input


3
7
1 2 3 4 5 6 7
2 3 4 5 6 7 1
3 4 5 6 7 1 2
4 5 6 7 1 2 3
5 6 7 1 2 3 4
6 7 1 2 3 4 5
7 1 2 3 4 5 6
1 2 3 4 5 7 6
1 3 4 5 6 7 2
1 4 5 6 7 3 2
1 5 6 7 4 2 3
1 6 7 5 2 3 4
1 7 6 2 3 4 5
1 7 2 3 4 5 6
5
4 5 1 2 3
3 5 2 4 1
1 2 3 4 5
5 2 4 1 3
3 4 5 1 2
2 3 4 5 1
1 3 5 2 4
4 1 3 5 2
2 4 1 3 5
5 1 2 3 4
6
2 3 4 5 6 1
3 1 2 6 4 5
6 1 2 3 4 5
5 6 1 3 2 4
4 3 6 5 2 1
5 6 1 2 3 4
4 5 6 1 2 3
3 4 5 6 1 2
1 2 3 4 5 6
2 5 4 1 6 3
3 2 5 4 1 6
1 4 3 6 5 2


Output


1
1 2 3 4 5 6 7
2
1 3 5 6 10
4
1 3 6 7 8 9

Note

In the first test case, the number of good subsets is 1. The only such subset is the set of arrays with indices 1, 2, 3, 4, 5, 6, 7.

In the second test case, the number of good subsets is 2. They are 1, 3, 5, 6, 10 or 2, 4, 7, 8, 9.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <cstdio>
#include <vector>
#include <set>
using namespace std;
const int mod = 998244353;
set<int> e[510][510];
int a[1010][510], d[1010][1010], u[1010];
vector<int> col[2], g[1010];
void dfs(int v, int b) {
	u[v] = -2;
	col[b].push_back(v);
	for (int i = 0; i < g[v].size(); i++) {
		int w = g[v][i];
		if (u[w] == -1) dfs(w, !b);
	}
}
void solve() {
	int n;
	scanf(\"%d\", &n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			e[i][j].clear();
		}
	}
	for (int i = 0; i < 2 * n; i++) {
		for (int j = 0; j < 2 * n; j++) {
			d[i][j] = 0;
		}
	}
	for (int i = 0; i < 2 * n; i++) {
		for (int j = 0; j < n; j++) {
			scanf(\"%d\", &a[i][j]);
			--a[i][j];
			e[j][a[i][j]].insert(i);
		}
		u[i] = -1;
	}
	vector<int> ad;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (e[i][j].size() == 1) {
				int x = *e[i][j].begin();
				if (u[x] != 1) {
					ad.push_back(x);
					u[x] = 1;
				}
				e[i][j].clear();
			}
		}
	}
	while (ad.size()) {
		set<int> de;
		for (int i = 0; i < ad.size(); i++) {
			for (int j = 0; j < n; j++) {
				int x = a[ad[i]][j];
				e[j][x].erase(ad[i]);
				de.insert(e[j][x].begin(), e[j][x].end());
				e[j][x].clear();
			}
		}
		ad.clear();
		vector<int> v(de.begin(), de.end());
		for (int i = 0; i < v.size(); i++) {
			u[v[i]] = 0;
			for (int j = 0; j < n; j++) {
				int x = a[v[i]][j];
				e[j][x].erase(v[i]);
				if (e[j][x].size() == 1) {
					int y = *e[j][x].begin();
					if (u[y] != 1) {
						ad.push_back(y);
						u[y] = 1;
					}
					e[j][x].clear();
				}
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			vector<int> w(e[i][j].begin(), e[i][j].end());
			for (int k = 0; k < w.size(); k++) {
				for (int l = k + 1; l < w.size(); l++) {
					d[w[k]][w[l]] = d[w[l]][w[k]] = 1;
				}
			}
		}
	}
	for (int i = 0; i < 2 * n; i++) {
		g[i].clear();
		for (int j = 0; j < 2 * n; j++) {
			if (d[i][j]) g[i].push_back(j);
		}
	}
	int ans = 1;
	for (int i = 0; i < 2 * n; i++) {
		if (u[i] != -1) continue;
		col[0].clear();
		col[1].clear();
		dfs(i, 0);
		int t = col[0].size() < col[1].size();
		for (int k = 0; k < 2; k++) {
			for (int j = 0; j < col[k].size(); j++) {
				u[col[k][j]] = k == t;
			}
		}
		if (col[0].size() == col[1].size()) {
			ans += ans;
			if (ans >= mod) ans -= mod;
		}
	}
	printf(\"%d\n\", ans);
	for (int i = 0; i < 2 * n; i++) {
		if (u[i]) printf(\"%d \", i + 1);
	}
	puts(\"\");
}
int main() {
	int T;
	scanf(\"%d\", &T);
	while (T--) {
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

class Caquamoonandpermutationsbootcamp(Basebootcamp):
    def __init__(self, n=5):
        self.n = n
        self.mod = 998244353

    def case_generator(self):
        n = self.n
        
        # Generate original Latin square (cyclic)
        original_latin = []
        for i in range(n):
            row = [(i + j) % n + 1 for j in range(n)]
            original_latin.append(row)
        
        # Generate additional arrays with conflict constraints
        additional = []
        for i in range(n):
            original_row = original_latin[i]
            while True:
                # 确保每个additional数组与原拉丁方对应i行有一个共同元素
                k = random.randint(0, n-1)
                fixed_value = original_row[k]
                other_values = list(range(1, n+1))
                other_values.remove(fixed_value)
                random.shuffle(other_values)
                
                # Build new row with fixed_value at position k
                new_row = []
                for pos in range(n):
                    if pos == k:
                        new_row.append(fixed_value)
                    else:
                        new_row.append(other_values.pop())
                
                # Check uniqueness
                if new_row not in original_latin and new_row not in additional:
                    additional.append(new_row)
                    break
        
        # Combine and shuffle with index tracking
        combined = [(row, True) for row in original_latin] + [(row, False) for row in additional]
        random.shuffle(combined)
        
        # Build final data structure
        shuffled_arrays = []
        correct_indices = []
        for idx, (row, is_original) in enumerate(combined, 1):
            shuffled_arrays.append(row)
            if is_original:
                correct_indices.append(idx)
        
        return {
            'n': n,
            'arrays': shuffled_arrays,
            'correct_count': 1,  # Only original Latin square valid
            'correct_indices_set': set(correct_indices)
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        arrays = question_case['arrays']
        arrays_str = '\n'.join(' '.join(map(str, row)) for row in arrays)
        
        return f'''Solve the Latin square puzzle. 

Problem:
Given 2n arrays (n={n}) forming a Latin square and additional arrays with element overlaps. All arrays are distinct.

Task:
1. Calculate the number of valid Latin square subsets (mod 998244353)
2. Provide indices of one valid subset (1-based)

Input:
{arrays_str}

Format answer as:
[answer]
<count>
<space-separated indices>
[/answer]'''

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_match = matches[-1].strip()
        lines = [line.strip() for line in last_match.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return None
        
        try:
            count = int(lines[0])
            indices = list(map(int, lines[1].split()))
            return (count, indices)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        count, indices = solution
        n = identity['n']
        correct_count = identity['correct_count']
        correct_indices = identity['correct_indices_set']
        
        # Validate format
        if len(indices) != n or len(set(indices)) != n:
            return False
        if any(not (1 <= x <= 2*n) for x in indices):
            return False
        
        # Check Latin square properties
        selected_arrays = [identity['arrays'][i-1] for i in indices]
        
        # Check rows
        for row in selected_arrays:
            if sorted(row) != list(range(1, n+1)):
                return False
        
        # Check columns
        for col in range(n):
            column = [row[col] for row in selected_arrays]
            if sorted(column) != list(range(1, n+1)):
                return False
        
        # Verify correctness
        return count == correct_count and set(indices) == correct_indices
