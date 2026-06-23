"""# 

### 谜题描述
This is the hard version of the problem. The only difference is that here 2≤ k≤ 100. You can make hacks only if both the versions of the problem are solved.

This is an interactive problem!

Every decimal number has a base k equivalent. The individual digits of a base k number are called k-its. Let's define the k-itwise XOR of two k-its a and b as (a + b)mod k.

The k-itwise XOR of two base k numbers is equal to the new number formed by taking the k-itwise XOR of their corresponding k-its. The k-itwise XOR of two decimal numbers a and b is denoted by a⊕_{k} b and is equal to the decimal representation of the k-itwise XOR of the base k representations of a and b. All further numbers used in the statement below are in decimal unless specified.

You have hacked the criminal database of Rockport Police Department (RPD), also known as the Rap Sheet. But in order to access it, you require a password. You don't know it, but you are quite sure that it lies between 0 and n-1 inclusive. So, you have decided to guess it. Luckily, you can try at most n times without being blocked by the system. But the system is adaptive. Each time you make an incorrect guess, it changes the password. Specifically, if the password before the guess was x, and you guess a different number y, then the system changes the password to a number z such that x⊕_{k} z=y. Guess the password and break into the system.

Input

The first line of input contains a single integer t (1≤ t≤ 10 000) denoting the number of test cases. t test cases follow.

The first line of each test case contains two integers n (1≤ n≤ 2⋅ 10^5) and k (2≤ k≤ 100).

It is guaranteed that the sum of n over all test cases does not exceed 2⋅ 10^5.

Interaction

For each test case, first read two integers n and k. Then you may ask up to n queries.

For each query, print a single integer y (0≤ y≤ 2⋅ 10^7). Let the current password be x. After that, read an integer r.

If x=y, you will read r=1 and the test case is solved. You must then continue solving the remaining test cases.

Else, you will read r=0. At this moment the password is changed to a number z such that x⊕_{k} z=y.

After printing a query, do not forget to output the end of line and flush the output. Otherwise, you will get the Idleness limit exceeded verdict.

To do this, use:

  * fflush(stdout) or cout.flush() in C++; 
  * System.out.flush() in Java; 
  * flush(output) in Pascal; 
  * stdout.flush() in Python; 
  * see documentation for other languages. 



If you ask an invalid query or exceed n queries, you will read r=-1 and you will receive the Wrong Answer verdict. Make sure to exit immediately to avoid unexpected verdicts.

Note that the interactor is adaptive. That is, the original password is not fixed in the beginning and may depend on your queries. But it is guaranteed that at any moment there is at least one initial password such that all the answers to the queries are consistent.

Hacks:

To use hacks, use the following format of tests:

The first line should contain a single integer t (1≤ t≤ 10 000) — the number of test cases.

The first and only line of each test case should contain two integers n (1≤ n≤ 2⋅ 10^5) and k (2≤ k≤ 100) denoting the number of queries and the base respectively. The optimal original password is automatically decided by the adaptive interactor.

You must ensure that the sum of n over all test cases does not exceed 2⋅ 10^5.

Example

Input


2
5 2

0

0

1
5 3

0

0

1


Output


3

4

5


1

4

6

Note

Test Case 1:

In this case, the hidden password is 2.

The first query is 3. It is not equal to the current password. So, 0 is returned, and the password is changed to 1 since 2⊕_2 1=3.

The second query is 4. It is not equal to the current password. So, 0 is returned, and the password is changed to 5 since 1⊕_2 5=4.

The third query is 5. It is equal to the current password. So, 1 is returned, and the job is done.

Test Case 2:

In this case, the hidden password is 3.

The first query is 1. It is not equal to the current password. So, 0 is returned, and the password is changed to 7 since 3⊕_3 7=1. [3=(10)_3, 7=(21)_3, 1=(01)_3 and (10)_3⊕_3 (21)_3 = (01)_3].

The second query is 4. It is not equal to the current password. So, 0 is returned, and the password is changed to 6 since 7⊕_3 6=4. [7=(21)_3, 6=(20)_3, 4=(11)_3 and (21)_3⊕_3 (20)_3 = (11)_3].

The third query is 6. It is equal to the current password. So, 1 is returned, and the job is done.

Note that these initial passwords are taken just for the sake of explanation. In reality, the grader might behave differently because it is adaptive.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
//#include <bits/stdc++.h>
//#define int long long int
//#define double long double
//#define endl '\n'
//#define all(c) c.begin(),c.end()
//#define mp(x,y) make_pair(x,y)
//#define eb emplace_back
//#define tr(k,st,en) for(int k = st; k <= en ; k++)
//#define trb(k,en,st) for(int k = en; k >= st ; k--)
//#define test int TN; cin>>TN; tr(T,1,TN)
//#define mxe(c) max_element(all(c))
//#define mne(c) min_element(all(c))
//using namespace std;
//
//template<typename A, typename B> ostream& operator<<(ostream &os, const pair<A, B> &p) { return os << '(' << p.first << \", \" << p.second << ')'; }
//template<typename T_container, typename T = typename enable_if<!is_same<T_container, string>::value, typename T_container::value_type>::type> ostream& operator<<(ostream &os, const T_container &v) { os << '{'; string sep; for (const T &x : v) os << sep << x, sep = \", \"; return os << '}'; }
//
//void dbg_out() { cerr << endl; }
//template<typename Head, typename... Tail> void dbg_out(Head H, Tail... T) { cerr << ' ' << H; dbg_out(T...); }
//
//#ifdef LOCAL
//#define dbg(...) cerr << \"(\" << #__VA_ARGS__ << \"):\", dbg_out(__VA_ARGS__)
//#else
//#define dbg(...)
//#endif
//
//typedef pair<int, int> pr;
//typedef vector<int> vi;
//typedef vector<vi> vvi;
//typedef vector<pr> vpr;
//
////const int mod = 1e9+7;
////const int inf = 9e18;
//
//double func(double c, double m, double p, double v) {
//    if(c <= 0.0000000000001 and m <= 0.0000000000001) return 1;
//    double ret = 0;
//    if(c > 0.0000000000001) {
//    if(c <= v + 0.00000000001) {
//        double de = (c/2.0);
//        ret += c * (func(0, m + de, p+de, v) + 1);
//    } else {
//        double de = ((v - c)/2.0);
//        ret += c * (func(v, m + de, p+de, v) + 1);
//    }
//    }
//    if(m > 0.0000000000001){
//    if(m <= v + 0.00000000001) {
//        double de = (m/2.0);
//        ret += m * (func(c + de, 0 , p+de, v) + 1);
//    } else {
//        double de = ((v - m)/2.0);
//        ret += m * (func(c + de, v, p+de, v) + 1);
//    }
//    }
//    return ret;
//}
//
//int32_t main(){
//    std::ios::sync_with_stdio(false);
//    cin.tie(NULL); cout.tie(NULL);
////    cout<<\"----START-----\"<<endl;
//    test{
//        double a,b,c,d;
//        cin>>a>>b>>c>>d;
//        cout<<func(a, b, c, d)<<endl;
//    }
//    return 0;
//}

#include <bits/stdc++.h>
#define int long long int
#define double long double
#define all(c) c.begin(),c.end()
#define mp(x,y) make_pair(x,y)
#define eb emplace_back
#define tr(k,st,en) for(int k = st; k <= en ; k++)
#define trb(k,en,st) for(int k = en; k >= st ; k--)
#define test int TN; cin>>TN; tr(T,1,TN)
#define mxe(c) max_element(all(c))
#define mne(c) min_element(all(c))
using namespace std;

template<typename A, typename B> ostream& operator<<(ostream &os, const pair<A, B> &p) { return os << '(' << p.first << \", \" << p.second << ')'; }
template<typename T_container, typename T = typename enable_if<!is_same<T_container, string>::value, typename T_container::value_type>::type> ostream& operator<<(ostream &os, const T_container &v) { os << '{'; string sep; for (const T &x : v) os << sep << x, sep = \", \"; return os << '}'; }

void dbg_out() { cerr << endl; }
template<typename Head, typename... Tail> void dbg_out(Head H, Tail... T) { cerr << ' ' << H; dbg_out(T...); }

#ifdef LOCAL
#define dbg(...) cerr << \"(\" << #__VA_ARGS__ << \"):\", dbg_out(__VA_ARGS__)
#else
#define dbg(...)
#endif

typedef pair<int, int> pr;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<pr> vpr;

//const int mod = 1e9+7;
//const int inf = 9e18;

int k;
int sub(int a, int b) {
    vector<int> t1, t2;
    while(a > 0 ) {
        t1.eb(a%k);
        a/=k;
    }
    while(b > 0) {
        t2.eb(b%k);
        b/=k;
    }
    while(t2.size() < t1.size()) {
        t2.eb(0);
    }
    while(t1.size() < t2.size()) {
        t1.eb(0);
    }
    int ret = 0;
    int x = 1;
    vector<int> tm;
    for(int i=0;i<t2.size();i++) {
        tm.eb((t1[i] - t2[i] + k)%k);
    }
    for(int i=0;i<t2.size();i++) {
        ret += tm[i] * x;
        x *= k;
    }
    return ret;
}

int add(int a, int b) {
    vector<int> t1, t2;
    while(a > 0 ) {
        t1.eb(a%k);
        a/=k;
    }
    while(b > 0) {
        t2.eb(b%k);
        b/=k;
    }
    if(t1.size() < t2.size()) {
        swap(t1, t2);
    }
    while(t2.size() < t1.size()) {
        t2.eb(0);
    }
    int ret = 0;
    int x = 1;
    vector<int> tm;
    for(int i=0;i<t2.size();i++) {
        tm.eb((t2[i] + t1[i])%k);
    }
    for(int i=0;i<t2.size();i++) {
        ret += tm[i] * x;
        x *= k;
    }
    return ret;
}


void solve(){
    int n;
    cin>>n>>k;
    int x=0;
    tr(i,0,n-1){
        int orig=i;
        int toask= i%2 == 0 ? add(x, orig) : sub(x, orig);
        cout<<toask<<endl;
        int r;
        cin>>r;
        if(r==1) return;
        x = sub(toask, x);
    }
}
int32_t main(){
    std::ios::sync_with_stdio(false);
    cin.tie(NULL); cout.tie(NULL);
//    cout<<\"----START-----\"<<endl;
    test{
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

class D2rpdandrapsheethardversionbootcamp(Basebootcamp):
    def __init__(self, max_n=10, k_min=2, k_max=100):
        self.max_n = max_n
        self.k_min = k_min
        self.k_max = k_max
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        k = random.randint(self.k_min, self.k_max)
        x0 = random.randint(0, n - 1) if n > 0 else 0
        return {
            'n': n,
            'k': k,
            'x0': x0
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        prompt = f"""You are trying to hack into a secure system. The password is initially between 0 and {n-1} (inclusive). You can make up to {n} guesses. 

Each time you guess incorrectly, the password changes using the k-itwise XOR operation in base {k}. Specifically, if the current password is x and you guess y, the new password becomes z such that x ⊕_{k} z = y. 

The k-itwise XOR is computed by adding each corresponding pair of digits in base {k} and taking the result modulo {k}. 

Your task is to determine a sequence of guesses that will find the correct password within {n} attempts. Provide your guesses as a comma-separated list of integers within [answer] and [/answer]. For example: [answer]3,4,5[/answer].

Ensure each guess is an integer between 0 and 20000000."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.strip().split(',')))
            return solution
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list) or len(solution) == 0:
            return False
        n = identity['n']
        k = identity['k']
        x0 = identity['x0']
        current_x = x0
        for i, y in enumerate(solution):
            if i >= n:
                return False
            if not (0 <= y <= 2 * 10**7):
                return False
            if y == current_x:
                return True
            current_x = cls.subtract_kits(y, current_x, k)
        return False
    
    @staticmethod
    def subtract_kits(a, b, k):
        def get_digits(num):
            digits = []
            if num == 0:
                return [0]
            while num > 0:
                digits.append(num % k)
                num = num // k
            return digits
        
        a_digits = get_digits(a)
        b_digits = get_digits(b)
        max_len = max(len(a_digits), len(b_digits))
        a_digits += [0] * (max_len - len(a_digits))
        b_digits += [0] * (max_len - len(b_digits))
        result_digits = [(ad - bd) % k for ad, bd in zip(a_digits, b_digits)]
        result = 0
        for i, d in enumerate(result_digits):
            result += d * (k ** i)
        return result
