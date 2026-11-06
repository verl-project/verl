"""# 

### 谜题描述
This is an interactive problem!

Nastia has a hidden permutation p of length n consisting of integers from 1 to n. You, for some reason, want to figure out the permutation. To do that, you can give her an integer t (1 ≤ t ≤ 2), two different indices i and j (1 ≤ i, j ≤ n, i ≠ j), and an integer x (1 ≤ x ≤ n - 1). 

Depending on t, she will answer: 

  * t = 1: max{(min{(x, p_i)}, min{(x + 1, p_j)})}; 
  * t = 2: min{(max{(x, p_i)}, max{(x + 1, p_j)})}. 



You can ask Nastia at most ⌊ \frac {3 ⋅ n} { 2} ⌋ + 30 times. It is guaranteed that she will not change her permutation depending on your queries. Can you guess the permutation?

Input

The input consists of several test cases. In the beginning, you receive the integer T (1 ≤ T ≤ 10 000) — the number of test cases.

At the beginning of each test case, you receive an integer n (3 ≤ n ≤ 10^4) — the length of the permutation p.

It's guaranteed that the permutation is fixed beforehand and that the sum of n in one test doesn't exceed 2 ⋅ 10^4.

Interaction

To ask a question, print \"? t i j x\" (t = 1 or t = 2, 1 ≤ i, j ≤ n, i ≠ j, 1 ≤ x ≤ n - 1) Then, you should read the answer.

If we answer with −1 instead of a valid answer, that means you exceeded the number of queries or made an invalid query. Exit immediately after receiving −1 and you will see the Wrong Answer verdict. Otherwise, you can get an arbitrary verdict because your solution will continue to read from a closed stream.

To print the answer, print \"! p_1 p_2 … p_{n} (without quotes). Note that answering doesn't count as one of the ⌊ \frac {3 ⋅ n} {2} ⌋ + 30 queries.

After printing a query or printing the answer, do not forget to output end of line and flush the output. Otherwise, you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++; 
  * System.out.flush() in Java; 
  * flush(output) in Pascal; 
  * stdout.flush() in Python; 
  * See the documentation for other languages. 



Hacks

To hack the solution, use the following test format.

The first line should contain a single integer T (1 ≤ T ≤ 10 000) — the number of test cases.

For each test case in the first line print a single integer n (3 ≤ n ≤ 10^4) — the length of the hidden permutation p.

In the second line print n space-separated integers p_1, p_2, …, p_n (1 ≤ p_i ≤ n), where p is permutation.

Note that the sum of n over all test cases should not exceed 2 ⋅ 10^4.

Example

Input


2
4

3

2

5

3

Output


? 2 4 1 3

? 1 2 4 2

! 3 1 4 2

? 2 3 4 2

! 2 5 3 4 1

Note

Consider the first test case.

The hidden permutation is [3, 1, 4, 2].

We print: \"? 2 4 1 3\" and get back min{(max{(3, p_4}), max{(4, p_1)})} = 3.

We print: \"? 1 2 4 2\" and get back max{(min{(2, p_2)}, min{(3, p_4)})} = 2.

Consider the second test case.

The hidden permutation is [2, 5, 3, 4, 1].

We print: \"? 2 3 4 2\" and get back min{(max{(2, p_3}), max{(3, p_4)})} = 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include \"bits/stdc++.h\"
#pragma GCC optimize(\"Ofast\")
#pragma GCC target(\"avx,avx2,fma\")
// #pragma GCC optimization (\"O3\")
// #pragma GCC optimization (\"unroll-loops\")
// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
// using namespace __gnu_pbds;
//#include <boost/multiprecision/cpp_int.hpp>
//using namespace boost::multiprecision;
using namespace std;

#define all(c) (c).begin(),(c).end()
// #define endl \"\n\"
#define ff first
#define ss second
#define allr(c) (c).rbegin(),(c).rend()
#define ifr(i, begin, end) for (__typeof(end) i = (begin) - ((begin) > (end)); i != (end) - ((begin) > (end)); i += 1 - 2 * ((begin) > (end)))
#define pof pop_front
#define pob pop_back
#define pb emplace_back
#define pf emplace_front
#define fstm(m,n,r) m.reserve(n);m.max_load_factor(r)
#define mp make_pair
#define mt make_tuple
#define inf LLONG_MAX
#define os tree<int, null_type,less<int>, rb_tree_tag,tree_order_statistics_node_update>
//order_of_key (k) : Number of items strictly smaller than k .
//find_by_order(k) : K-th element in a set (counting from zero).
const double PI = acos(-1);
typedef complex<double> cd;
typedef long long ll;
ll gcd(){return 0ll;} template<typename T,typename... Args> T gcd(T a,Args... args) { return __gcd(a,(__typeof(a))gcd(args...)); }
typedef map<ll,ll> mll;
typedef map<string,ll> msll;
typedef unordered_map<ll,ll> umap;
typedef vector<ll> vll;
typedef pair<ll,ll> pll;
typedef long double ld;
#define mod 1000000007 

const int N = 1e5 + 2;

int ask(int t, int i, int j, int x) {
    int r;
    if(t == 1) t = 2; else t = 1;
    cout<<\"? \"<<t<<\" \"<<i+1<<\" \"<<j+1<<\" \"<<x<<endl;
    cin>>r;
    return r;
}

void answer(vector<int> &a) {
    cout<<\"! \";
    for(auto &it: a)
        cout<<it<<\" \";
    cout<<endl;
}

void Solve() {
    int n;
    cin>>n;
    int x, y;
    vector<int> a(n,0);
    for(int i=0; i+1<n; i+=2) {
        x = ask(1, i, i+1, 1);
        y = ask(2, i, i+1, n-1);

        if(x == 2 && ask(2, i, i+1, 1)==1) {
            a[i+1] = 1;
            if(y == n-1 && ask(1, i, i+1, n-1)==n)
                a[i] = n;
            else
                a[i] = y;
            continue;
        }

        if(y == n-1 && ask(1, i, i+1, n-1) == n) {
            a[i] = n;
            a[i+1] = x;
            continue;
        }

        if(ask(1, i+1, i, x) == x+1) {
            a[i] = x;
            a[i+1] = y;
        } else {
            a[i] = y;
            a[i+1] = x;
        }
    }
    if(n&1) {
        vector<bool> vis(n+1, 0);
        for(int i=0; i<n-1; i++)
            vis[a[i]] = 1;
        for(int i=1; i<n+1; i++) if(!vis[i]) {
            a[n-1] = i;
            break;
        }
    }
    answer(a);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    // #ifndef ONLINE_JUDGE
    //     freopen(\"input.txt\",\"r\",stdin);
    // #endif
    
    int T;cin>>T;while(T--) 
    {
        Solve();
    }
    return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cnastiaandahiddenpermutationbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=10):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        p = list(range(1, n + 1))
        random.shuffle(p)
        queries = self.simulate_queries(p)
        return {
            'n': n,
            'permutation': p,
            'queries': queries
        }
    
    def simulate_queries(self, p):
        queries = []
        n = len(p)
        
        def ask(t, i, j, x):
            i_index = i - 1  # Convert to 0-based
            j_index = j - 1
            pi = p[i_index]
            pj = p[j_index]
            if t == 1:
                res = max(min(x, pi), min(x + 1, pj))
            elif t == 2:
                res = min(max(x, pi), max(x + 1, pj))
            else:
                res = -1
            queries.append((t, i, j, x, res))
            return res
        
        a = [0] * n
        for i in range(0, n - 1, 2):
            x = ask(2, i + 1, (i + 1) + 1, 1)
            y = ask(1, i + 1, (i + 1) + 1, n - 1)
            
            if x == 2 and ask(1, i + 1, (i + 1) + 1, 1) == 1:
                a[i + 1] = 1
                if y == n - 1 and ask(2, i + 1, (i + 1) + 1, n - 1) == n:
                    a[i] = n
                else:
                    a[i] = y
                continue
            
            if y == n - 1 and ask(2, i + 1, (i + 1) + 1, n - 1) == n:
                a[i] = n
                a[i + 1] = x
                continue
            
            check = ask(2, (i + 1) + 1, i + 1, x)
            if check == x + 1:
                a[i] = x
                a[i + 1] = y
            else:
                a[i] = y
                a[i + 1] = x
        
        if n % 2 == 1:
            last = set(range(1, n + 1)) - set(a[:n-1])
            a[-1] = last.pop()
        
        return queries
    
    @staticmethod
    def prompt_func(question_case):
        queries = question_case['queries']
        queries_text = []
        for t, i, j, x, res in queries:
            queries_text.append(f"? {t} {i} {j} {x} → {res}")
        queries_str = "\n".join(queries_text)
        return f"""Nastia has a hidden permutation of length {question_case['n']}. The following queries were made and their responses are given:

{queries_str}

Determine the hidden permutation and provide your answer as "! p1 p2 ... pn" enclosed within [answer] tags. For example: [answer]! 1 2 3 4[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'!([\d\s]+)', output)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.split()))
        except:
            return None
        return solution

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['permutation']
