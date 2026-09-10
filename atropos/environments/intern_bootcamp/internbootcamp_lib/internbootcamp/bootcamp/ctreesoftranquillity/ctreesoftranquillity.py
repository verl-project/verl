"""# 

### 谜题描述
Soroush and Keshi each have a labeled and rooted tree on n vertices. Both of their trees are rooted from vertex 1.

Soroush and Keshi used to be at war. After endless decades of fighting, they finally became allies to prepare a Codeforces round. To celebrate this fortunate event, they decided to make a memorial graph on n vertices.

They add an edge between vertices u and v in the memorial graph if both of the following conditions hold: 

  * One of u or v is the ancestor of the other in Soroush's tree. 
  * Neither of u or v is the ancestor of the other in Keshi's tree. 



Here vertex u is considered ancestor of vertex v, if u lies on the path from 1 (the root) to the v.

Popping out of nowhere, Mashtali tried to find the maximum clique in the memorial graph for no reason. He failed because the graph was too big. 

Help Mashtali by finding the size of the maximum clique in the memorial graph.

As a reminder, clique is a subset of vertices of the graph, each two of which are connected by an edge.

Input

The first line contains an integer t (1≤ t≤ 3 ⋅ 10^5) — the number of test cases. The description of the test cases follows.

The first line of each test case contains an integer n (2≤ n≤ 3 ⋅ 10^5).

The second line of each test case contains n-1 integers a_2, …, a_n (1 ≤ a_i < i), a_i being the parent of the vertex i in Soroush's tree.

The third line of each test case contains n-1 integers b_2, …, b_n (1 ≤ b_i < i), b_i being the parent of the vertex i in Keshi's tree.

It is guaranteed that the given graphs are trees.

It is guaranteed that the sum of n over all test cases doesn't exceed 3 ⋅ 10^5.

Output

For each test case print a single integer — the size of the maximum clique in the memorial graph.

Example

Input


4
4
1 2 3
1 2 3
5
1 2 3 4
1 1 1 1
6
1 1 1 1 2
1 2 1 2 2
7
1 1 3 4 4 5
1 2 1 4 2 5


Output


1
4
1
3

Note

In the first and third test cases, you can pick any vertex.

In the second test case, one of the maximum cliques is \{2, 3, 4, 5\}.

In the fourth test case, one of the maximum cliques is \{3, 4, 6\}.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#pragma GCC optimize(\"Ofast\")
#include<bits/stdc++.h>
#define FOR(i,a,b) for(int i=a;i<=b;++i)
#define PII pair<int,int>
#define ll long long
#define pb push_back
#define sz(x) (int)(x.size())
#define rd (rand()<<16^rand())
#define db double
#define gc (_p1==_p2&&(_p2=(_p1=_buf)+fread(_buf,1,100000,stdin),_p1==_p2)?EOF:*_p1++)
using namespace std;
char _buf[100000],*_p1=_buf,*_p2=_buf;
inline int gi()
{
	int x=0,f=1;
	char ch=gc;
	while(ch<'0'||ch>'9')
	{
		if(ch=='-')f=-1;
		ch=gc;
	}
	while(ch>='0'&&ch<='9')
	{
		x=(x<<3)+(x<<1)+(ch^48);
		ch=gc;
	}
	return (f==1)?x:-x;
}
inline int max(int a,int b){return a>b?a:b;}
inline int min(int a,int b){return a<b?a:b;}
const int maxn=3e5+5;
int sum,ans,n;
struct BIT
{
	int c[maxn];
	inline void update(int x,int v)
	{
		while(x<=n)c[x]+=v,x+=x&-x;
	}
	inline void clear()
	{
		FOR(i,1,n)c[i]=0;
	}
	inline int query(int x)
	{
		int ret=0;
		while(x)ret+=c[x],x-=x&-x;
		return ret;
	}
}tree,tree2;
vector<int>e[maxn],e2[maxn];
int dep[maxn],dfn[maxn],low[maxn],tot,fa[maxn][20],siz[maxn],tp[maxn],son[maxn],Log[maxn];
inline void dfs(int u)
{
	siz[u]=1;dep[u]=dep[fa[u][0]]+1;
	FOR(i,1,Log[dep[u]])fa[u][i]=fa[fa[u][i-1]][i-1];
	for(int v:e[u])
	{
		if(v==fa[u][0])continue;
		fa[v][0]=u,dfs(v),siz[u]+=siz[v];
		if(siz[v]>siz[son[u]])son[u]=v;
	}
}
inline void dfs2(int u,int topf)
{
	tp[u]=topf,dfn[u]=++tot;
	if(son[u])dfs2(son[u],topf);
	for(int v:e[u])
	{
		if(v==fa[u][0]||v==son[u])continue;
		dfs2(v,v);
	}
	low[u]=tot;
}
inline void init()
{
	sum=ans=0;
	FOR(i,1,n)e[i].clear(),e2[i].clear();
	FOR(i,1,n)dep[i]=dfn[i]=low[i]=fa[i][0]=siz[i]=tp[i]=son[i]=0;
	FOR(i,1,n)FOR(j,0,19)fa[i][j]=0;
	//FOR(i,1,n)cout<<tree.c[i]<<' ';cout<<endl;
	//FOR(i,1,n)cout<<tree2.c[i]<<' ';cout<<endl;
	tree.clear(),tree2.clear();
	tot=0;
}
inline void input()
{
	n=gi();
	FOR(i,2,n)
	{
		int u=gi();
		e2[i].pb(u),e2[u].pb(i);
	}
	FOR(i,2,n)
	{
		int u=gi();
		e[i].pb(u),e[u].pb(i);
	}
	dfs(1);
	dfs2(1,1);
}
inline int get(int u)
{
	for(int i=Log[dep[u]];i>=0;--i)
	{
		if(fa[u][i]!=0&&tree.query(dfn[fa[u][i]]))u=fa[u][i];
	}
	return u;
}
inline void dfs3(int u,int fa)
{
	vector<pair<int,int>>now;
	bool ok=0;
	int t=sum;
	if(tree2.query(low[u])-tree2.query(dfn[u]-1)==0)
	{
		tree2.update(dfn[u],1);ok=1;
		if(tree.query(dfn[u]))
		{
			int x=get(u);
			now.pb({x,1});
			tree.update(dfn[x],-1);
			tree.update(low[x]+1,1);
		}
		else sum++;
		tree.update(dfn[u],1);
		tree.update(low[u]+1,-1);
		now.pb({u,-1});
	}
	ans=max(ans,sum);
	for(int v:e2[u])
	{
		if(v==fa)continue;
		dfs3(v,u);
	}
	if(ok)tree2.update(dfn[u],-1);
	for(auto x:now)
	{
		int y=x.first,v=x.second;
		tree.update(dfn[y],v);
		tree.update(low[y]+1,-v);
	}
	sum=t;
}
inline void solve()
{
	dfs3(1,0);
	printf(\"%d\n\",ans);
}
int main()
{
	int T=1;
	T=gi();
	FOR(i,2,maxn-1)Log[i]=Log[i>>1]+1;
	while(T--)
	{
		init();
		input();
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
from itertools import combinations
from bootcamp import Basebootcamp

class Ctreesoftranquillitybootcamp(Basebootcamp):
    def __init__(self, max_n=5):
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(2, self.max_n)
        # 生成合法的树结构 [关键修正点]
        a_parents = [random.randint(1, i) for i in range(1, n)]
        b_parents = [random.randint(1, i) for i in range(1, n)]

        # 构建祖先关系字典 [增加调试日志]
        def safe_build_parents(parents):
            ancestor_dict = {1: set()}
            for node in range(2, n+1):
                path = set()
                current = parents[node-2]
                path.add(current)
                while current != 1:
                    current = parents[current-2] if current >=2 else 1
                    path.add(current)
                ancestor_dict[node] = path
            return ancestor_dict

        s_ancestors = safe_build_parents(a_parents)
        k_ancestors = safe_build_parents(b_parents)

        # 构建邻接矩阵 [变量名修正]
        adj = {u: set() for u in range(1, n+1)}
        for u in range(1, n+1):
            for v in range(1, n+1):
                if u == v: continue
                cond1 = (u in s_ancestors[v]) or (v in s_ancestors[u])
                cond2 = (u not in k_ancestors[v]) and (v not in k_ancestors[u])
                if cond1 and cond2:
                    adj[u].add(v)

        # 寻找最大团 [修正变量覆盖bug]
        max_size = 1
        nodes = list(range(1, n+1))
        for k in range(min(n, 5), 0, -1):
            for subset in combinations(nodes, k):
                valid = True
                # 修改循环变量名为u和v
                for u, v in combinations(subset, 2):
                    if v not in adj[u]:
                        valid = False
                        break
                if valid:
                    return {  # [关键修正点：保证返回原始数据]
                        'n': n,
                        'a': a_parents,  # 保持列表类型
                        'b': b_parents,  # 保持列表类型 
                        'correct_answer': k
                    }
        return {
            'n': n,
            'a': a_parents,
            'b': b_parents,
            'correct_answer': 1
        }
    
    @staticmethod
    def prompt_func(question_case):
        # 添加类型检查确保输入正确
        assert isinstance(question_case['a'], list), "Invalid a type"
        assert isinstance(question_case['b'], list), "Invalid b type"
        
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        b = ' '.join(map(str, question_case['b']))
        
        return f"""You are a programming expert. Solve the problem and put the answer within [answer] tags.

Problem:
Two rooted trees (both rooted at 1) define edge conditions:
1. In Soroush's tree: u is ancestor of v or vice versa
2. In Keshi's tree: neither is ancestor of the other

Input:
n = {n}
Soroush's parents (a_2..a_n): {a}
Keshi's parents (b_2..b_n): {b}

Output the maximum clique size. Put your final answer between [answer] and [/answer]."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
