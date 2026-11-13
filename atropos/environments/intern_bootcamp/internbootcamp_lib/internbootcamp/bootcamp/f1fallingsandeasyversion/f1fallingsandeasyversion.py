"""# 

### 谜题描述
This is the easy version of the problem. The difference between the versions is the constraints on a_i. You can make hacks only if all versions of the problem are solved.

Little Dormi has recently received a puzzle from his friend and needs your help to solve it. 

The puzzle consists of an upright board with n rows and m columns of cells, some empty and some filled with blocks of sand, and m non-negative integers a_1,a_2,…,a_m (0 ≤ a_i ≤ n). In this version of the problem, a_i will be equal to the number of blocks of sand in column i.

When a cell filled with a block of sand is disturbed, the block of sand will fall from its cell to the sand counter at the bottom of the column (each column has a sand counter). While a block of sand is falling, other blocks of sand that are adjacent at any point to the falling block of sand will also be disturbed and start to fall. Specifically, a block of sand disturbed at a cell (i,j) will pass through all cells below and including the cell (i,j) within the column, disturbing all adjacent cells along the way. Here, the cells adjacent to a cell (i,j) are defined as (i-1,j), (i,j-1), (i+1,j), and (i,j+1) (if they are within the grid). Note that the newly falling blocks can disturb other blocks.

In one operation you are able to disturb any piece of sand. The puzzle is solved when there are at least a_i blocks of sand counted in the i-th sand counter for each column from 1 to m.

You are now tasked with finding the minimum amount of operations in order to solve the puzzle. Note that Little Dormi will never give you a puzzle that is impossible to solve.

Input

The first line consists of two space-separated positive integers n and m (1 ≤ n ⋅ m ≤ 400 000).

Each of the next n lines contains m characters, describing each row of the board. If a character on a line is '.', the corresponding cell is empty. If it is '#', the cell contains a block of sand.

The final line contains m non-negative integers a_1,a_2,…,a_m (0 ≤ a_i ≤ n) — the minimum amount of blocks of sand that needs to fall below the board in each column. In this version of the problem, a_i will be equal to the number of blocks of sand in column i.

Output

Print one non-negative integer, the minimum amount of operations needed to solve the puzzle.

Examples

Input


5 7
#....#.
.#.#...
#....#.
#....##
#.#....
4 1 1 1 0 3 1


Output


3


Input


3 3
#.#
#..
##.
3 1 1


Output


1

Note

For example 1, by disturbing both blocks of sand on the first row from the top at the first and sixth columns from the left, and the block of sand on the second row from the top and the fourth column from the left, it is possible to have all the required amounts of sand fall in each column. It can be proved that this is not possible with fewer than 3 operations, and as such the answer is 3. Here is the puzzle from the first example.

<image>

For example 2, by disturbing the cell on the top row and rightmost column, one can cause all of the blocks of sand in the board to fall into the counters at the bottom. Thus, the answer is 1. Here is the puzzle from the second example.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
/*
v:r:1BRBI.7J.  jIJ27IUYUuri:7:::::::v7::ri..::irRBb71P2qKbu7i.::... :vIKr:i:rRS:::XXJur:iLIsS7LLrr1r
r .:JBRBL.:r.  ...:iiuIIr:.r:..... .iir:.:.  .:2I7i.2g7Pgd2IJ:..     rYr .:..i:...i5vsi:i2j11r7Yiir:
i.:rSRgZXi7irir:::r7ivZ5ii5s.r7iii::i:Yv::i.......i7bSJRdqDbX7ii:....:qr..:ir:irr::s2j7r1q15b77uv7r:
:..sgi7Xs:irirr71SP1J1U::i7rv2rr7s7Y::i7i:r:.:...iii.:DB.UZXs7iii:..  .iir::irr7iiij7JuY1qYssrr7 77:
. .LB:.qRi:iri7vMgg2XEXiBUY7LMqr517Yi:r7r:i:..:.::ii:YBg.PDSUr::i::.. .ir7Yriii:iirvi:I1KqsYjLu7rrvr
d1irRYiBBYsKSu7SSXDUUdZv5Bg17qQRPPg5ri7vi:.:::.:.  :rIq1.LU5L:::::::..:r:rrrir:::i:7:.:u55LJj1Y777rr
QDX:i77RBDuuEs7L:.rr1PgsLdBQZX5::rL:.rj7ri:::..::.iri:i:r7vv:...::::..ii::i::::.:::rv: .vU2YuY7irr7i
. .:....v1i:rrrr7vr:i::i.7Qj.rr    .:.:v7::irYPQBBQQZDdKPgEd5u77:.::..ir::.vKUUUr:..:: :j2uI1uvriii:
:.:.:::.::::7rvrr7Li... . ...... .:...:::.1gBBBQBQQgggMMBBBQQRQgPY:  .ir7vJSPKdX1vr:rirLJ7Lvv77rrrr:
. ........ .       . ....      ... .    igBQgMgMggDDDQQBMgRQggZPbEP7 .7s7rr7777i:riiiiiYvv7v7vvYvYLr
U7jsjLYvYvLvvvv77777v77r7r7r7r7rriri:.rgBQgDRMQRQggEDPSJv75ZRggqqPZZXJsr:.:::....::..::7JYJYsLLvLvLr
5UIX5SIS5XIS55I5ISIX5X5SISIXIS2S22jv2BBBMMQQggPX1Jv7r::.:i71qgZZbPPgdKKSi.:i.:.irirsYsvY7rir::::7Yvr
vrv77777v7777777777r7r7r7r7rrrrrr:.vBBQQBQg5sii::.:...:...::rjZDDEbPMbKdq::rL1X151uii:::riiir7vvvvsr
i:iiiririiiiiiiiiiiiiiii:i:i:i:::.jBQQQBdsi:...:::::::::::::.:7dgMdPdgPXEMYYs1vL7LLrr77vvsYsYsvYvvvr
:.:::::::::::::.:.:.:.:.:.:...:. rBQBBgv:.....:::::::::::.:::::7gRMPPdDKKgE:::::::irjLJssvYvLvvvLvYr
:.::::::::::::::::::::::::::.:. .BBBBPi::ri:.:::::::.........:::vRRMPEZbXPB7 :::::.::::r7vLsvsvYvYLr
i.::::::::::::::::::::::::::::. QBQBbr2ZBQBgKvr:i:i:rvuU5SIs7:::iXBggZZbPXgM:.::i:i:::::ii7YjsjLsYjr
:.:::::::::::::::::::::::::::: IBBQQv1Jr:i7uU27riri7ubZgddqPX5ri:vMBMDggPbKBv:iir7riii::......71uU1v
:.:::::::::::::::::::::::::::..QBQBPr::777vi7vvi:i7L1vrirr:::ivriiPBQDDRDEEg1.i::::::::::.:...:::::.
i.:::::::::::::::::::::::::::..BQBBL:1BBQjvSLuvi.:r1UuEBIJPPvi:iiivBQgEMDgDMi .........::i:i::......
:.:.:::::::::::::::::::::::::.:BBQQ:LPZDEiijYr7:.:rr7PBQv.rEM2vrr:ibBRQZRgMMs:rr7r7r7rrr7rrr7r7rrr2:
i:rrriiriiiiirrriririririririi7BBDviiiirrsYriii:.::::r7v7YriiL7rirrPBBQRRRgQ1i77r7r7r777777v7v7rYQ7
rivvvrr:iiriiri:iiiiiiriiirir:..  :rv::::::::i:.:::...:irrir:.::r:.:iSBBBRBQI:r7v7v7v77rrirr7riuBBi
rij2Js7r:irL77i:::::::::::::...:7viri::i::.:i:...:::...:...r7..:7s..  .1BQbBPrr7r7r77vLJvrirri:UBBBR
:.r7r:iYuiirrri.:::.:.:.:::...iv7..:rrvi:.iri::::.:ii::.:.:vu...rBU.... 7BPDj.::ii::....iiiii:i. .7I
i..::::7v777r7ii:::::::i:i...rv:..71uZU::iivMg5KQQ77ii::::i1s...rBP...:. KQE1rir:r::....    .:ri
i.::::rLY7Ysuvv77rr77rrJ5Iv..:vr:.JK75jiiii:1u25IPUrrii:irYu:.ii5gv...:. rRDPIKZv7r::ii...:.  .vr.:
r:::iLYjLJvjsJv7r777rvr7SSjvi:vsi:.5L117ii::.......::rirv1j:.ir1Pi .:L7:.:DMPIPD:::rri.....r7: .:::.
r:iirrriiiriiirir7Yvv77ruUUjJrrYv:..r227irvUZEPPqJrirrrvur..:vuv:.:iY1YLYJBgggBP .i::.:....:v1Ji....
rirs177Lr7rrrvrirr:77r:7SbPPJs:77i:..7XUvXEMZEPEEgDQQP1s:..irjr.:7J2u1uY:2BRQBBM..::::........:svi..
UYUuUvrj1Lvv7rriL7ri7L7LDEZgX7u7vrri:.rX5rr7Yj2u1Lvr7J7..:irv7irYu1u2J77PBBBBBBQ:::ir...::::i..:i::.
JsujLvrsJjvvrrr7rr7LvsvYvuIZqL2UUUL7ri.:LsvvvsJUus77r:.:i7JUYLLuLvvsvvsvLMKZg2vi.::iriirrL7vvi.  .:.
2u5Jsv7rs12vv7Y7LJu1JYYriir1jYq2U1uLri:..:7ri:::::ii..ir7vsvYLussLvi7jK:i:  .....:rrvsI1uJYLsvriii:
ggD5sY7L7vssr7jYjUL7iiii:i:i:Y5S1I1J7rii:..:ri:i::..:irr7L7vvLvjJsr1Qqui::..   ..:r2PKsYvJIqSPZSjSS1
:.:irLL77rv777s77rv77ri:::r1vrJ5PII1Jv7ri:..:1X7:.::ir77vvLYJYJvJ17QBSJrvr:.......  :7Ss2PIs7LPjU1IL
i:rv77vv7777vv7rrvv7i:.:::2Zs7rrLPIuYs7vrr::.iL.:iirr77vvYYJjjJuY77BX7ubL::i::.:...:..7BZPvr7I2jJUui
i.irrr7vsvL777r::::::.:ii:7Ijr7i.YqjLvY77rr:::::iirrrrv7vvsJU2u71PBRjLDX::i::::..:i::..Kq7iisu2U51vr
:.irrri7sjLLs7::rYv7rr::ii7Pj7rrrrr2vv77ii::r7:::::irrrv77L5IIsKQBQK7Z5::i:::::i7r::::..75vYYvi7777D
:.rrrrri7sv7vi:12jUJsJJiii7Qsivrrii7sii:::::J1::::::::ir7rPbXIqEQRD7DX::i:::::i7rr7i:::.:RBMDj7vvYXQ
::rririririiiii77Yu1jYIIii:dEr7Y:rr7i....::i5Br:::.....:ijBdKddgDg1Eb::i:i:i::rrrrr7r:::..uP222q1LYv
i.rrrrrrr:ii:iriii7jX5U2uii7QbrLsi7::.:.:::7ZB2:i::.:....iQgggP5gqdMi:iri:i::7vrriiir7i::. LI1I21Yuv
i::rrrrr:i:i:rrrrriYs5KX2sir5RP7jr:::::iiiiKPQKriii:::::..:BBX7ZZdMs:ivr:i::ujrir777r7r::::......:.
i:ririi:::i:rrrr77vj2uu5qXvi5PQ7::::iirrrr2ZDMQLririi::::..:S2dRbDE::7r:ir:Ljrvvv77ri:::riusvr7r7sr.
i:rriiiiii:ir7r7rrivY21UPdIrP2:::::iirr77uZdBQQQ7rrrii::::.. IBgXMv:rr:iirr5u1J5YL7ri:isYiYBIvvvvj1J
:.:::.....:rr7v7vrrs2Lu2qQPrr:::::iiii77jdUURBQBRrririi::::.. YZQdri1viivrPbPSsvrivvuLirIj:P1iiiiir:
uL122IU5Iq7777r7v1IPXKPQQQL:.::::iiir77JX2:EPBRQBdrriiii::::.. :RPi7SPi7UZgbJ7iiiiis5EIr:viiKsvv7Lv7
DqdbbPdPdb1LsvY7JY15dPQBP:.:::::iirr77sUDu:dMgBQBBXr7irii:::::. .rvuSXr2MZurirJ5Pu7::rqPKv7.1BQRQRQR
X5SK5KSS5P1v7suIuurrDBbr..:::i:iirrvvu1PPb77q1QBBg5J77iiii::::::..:IKqsDZsr1PgRMPEZEj:.r77i.:BBBBBBB
S152II55PP7irirsUuUKDs..::::i:iir7YsUIdbbEur7r7U11uS2v7rrii::::::...1QREXIEZd1v:..::YJi.. .i:77rrir:
2uU5UIII7L7LsIJuSgQq:..::::iiri77sjXXEZPXPI1IdqPK5s5s177rrii::::::.. JQgdZPK7::i::::.::ivKgPriii:ri:
5J52I2K7:ivJIuUU5Li.::::i:iirr7vjuX5YLv5IuIvIqgdKus7LqU77rrii:i:::::. :gMS2vrirri:::i7KDdKPvrirr7EY:
5UIS2SK7:r:iir7vi:::::iiirirrvvJjXIi::iq5U5vr5qP55775ZqJ77rrrrii::::.. .iiii:i:i:i:rJUsr:i7Jsvirrii:
d5dPbPMjir7vL7r:::i:iirirrrrvLuu2DUiiiiJbUS5r7JEXsrUEKdXsvv7rrrii::::::..:iiiiii77Li:.::::irvv7rrir:
svLY7LJ7i7r7ri:i:i:iirirr77vvsjIPQurrrii2qKEriYPvrvEKPPDKJ7v77rriii::::::..irv777JjUYr::::rL:777iiri
r:7r7vrirrrii:i:iiiirrrr77LvJJ2PgD5iY7ri7PPP1:UIrr5PdbgZb2u7v77rrrrii::::::.:7Lri:::ir7i::iB2irrrrr:
ZdZgR2r7iiiiiiiiiiirrrr7vLvJuIPgdMX7svrriKKPI:2XJPPZdZEPuUXj7L77rrrrii::::::.:irr77rii:i::.PBg7vsJvr
1uZQbr7rriiiririrrrr77vLYs11SPMEgMEvuvvrrXD2IrIXBMQEEPdX1sU5JLsvv77rriiii:::::::irv7L77rrri..rXBDuYr
S2XL:::::::::iiiirr7r77LvJuXIDgggRDL7vri.vKuU2j5bPqdqPPP5Y7L1Y7777rrii::::...........::::::rri.:vYLr*/
//This disgusting code is written by Juruo D0zingbear
//Please don't hack me! 0v0 <3
/*
_/_/_/_/    _/_/_/_/_/  _/_/_/
_/      _/      _/    _/      _/
_/      _/      _/    _/      _/
_/      _/      _/    _/      _/
_/      _/      _/    _/  _/  _/
_/      _/  _/  _/    _/    _/_/
_/_/_/_/      _/_/     _/_/_/_/_/

_/_/_/_/    _/    _/  _/      _/
_/      _/   _/  _/   _/_/  _/_/
_/      _/    _/_/    _/ _/_/ _/
_/      _/     _/     _/  _/  _/
_/      _/    _/_/    _/      _/
_/      _/   _/  _/   _/      _/
_/_/_/_/    _/    _/  _/      _/

_/_/_/_/_/ _/_/_/_/_/ _/_/_/_/_/
    _/         _/     _/
    _/         _/     _/
    _/         _/     _/_/_/_/
    _/         _/     _/
    _/         _/     _/
    _/     _/_/_/_/_/ _/_/_/_/_/

_/_/_/_/_/ _/_/_/_/_/ _/_/_/_/_/
    _/         _/     _/
    _/         _/     _/
    _/         _/     _/_/_/_/
    _/         _/     _/
    _/         _/     _/
    _/     _/_/_/_/_/ _/_/_/_/_/
*/
#include<bits/stdc++.h>
#define LL long long
#define pb push_back
#define mp make_pair
#define pii pair<int,int>
using namespace std;
namespace IO
{
    const int sz=1<<15;
    char inbuf[sz],outbuf[sz];
    char *pinbuf=inbuf+sz;
    char *poutbuf=outbuf;
    inline char _getchar()
    {
        if (pinbuf==inbuf+sz)fread(inbuf,1,sz,stdin),pinbuf=inbuf;
        return *(pinbuf++);
    }
    inline void _putchar(char x)
    {
        if (poutbuf==outbuf+sz)fwrite(outbuf,1,sz,stdout),poutbuf=outbuf;
        *(poutbuf++)=x;
    }
    inline void flush()
    {
        if (poutbuf!=outbuf)fwrite(outbuf,1,poutbuf-outbuf,stdout),poutbuf=outbuf;
    }
}
inline int read(){
	int v=0,f=1;
	char c=getchar();
	while (c<'0' || c>'9'){
		if (c=='-') f=-1;
		c=getchar();
	}
	while (c>='0' && c<='9'){
		v=v*10+c-'0';
		c=getchar();
	}
	return v*f;
}

const int Maxn=400005;
const int dx[]={0,0,1,-1};
const int dy[]={1,-1,0,0};
int n,m,A[Maxn];
vector<char> V[Maxn];
vector<int> pos[Maxn];
vector<int> G[Maxn],nG[Maxn],vG[Maxn];
int col[Maxn],vis[Maxn],O[Maxn],ko,bad[Maxn],kc,I[Maxn];
void dfs1(int x){
	vis[x]=1;
	for (int i=0;i<G[x].size();i++){
		int v=G[x][i];
		if (!vis[v]){
			dfs1(v);
		}
	}
	O[ko++]=x;
}
void dfs2(int x,int C){
	col[x]=C;vis[x]=1;
	for (int i=0;i<nG[x].size();i++){
		int v=nG[x][i];
		if (!vis[v]){
			dfs2(v,C);
		}
	}
}
void Kos(){
	for (int i=0;i<n*m;i++){
		for (int j=0;j<G[i].size();j++){
			//cerr<<i<<' '<<G[i][j]<<endl;
			nG[G[i][j]].pb(i);
		}
	}
	
	memset(vis,0,sizeof(vis));
	
	for (int i=0;i<n*m;i++){
		if (!vis[i] && !bad[i]){
			dfs1(i);
		}
	}
	memset(vis,0,sizeof(vis));
	
	for (int i=ko-1;i>=0;i--){
		int x=O[i];
		if (!vis[x]){
			dfs2(x,++kc);
		}
	}
	
	for (int i=0;i<n*m;i++){
		for (int j=0;j<G[i].size();j++){
			int x=col[i],y=col[G[i][j]];
			if (x!=y) vG[x].pb(y),I[y]=1;
		}
	}
	int ans=0;
	for (int i=1;i<=kc;i++){
		if (!I[i]) ans++;
	}
	printf(\"%d\n\",ans);
}
int Lim[Maxn];
int par[Maxn];
int main(){
	n=read();m=read();
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
			char ch=getchar();
			while (ch=='\n' || ch==' ') ch=getchar();
			V[i].pb(ch);
			if (ch=='#'){
				pos[j].pb(i);
			}
		}
	}
	for (int i=0;i<m;i++){
		scanf(\"%d\",&A[i]);
		assert(pos[i].size()>=A[i]);
		if (!A[i]) {
			Lim[i]=-1;
			continue;
		}
		reverse(pos[i].begin(),pos[i].end());
		Lim[i]=pos[i][A[i]-1];
		reverse(pos[i].begin(),pos[i].end());
	}
	
//	cerr<<123<<endl;
	
	for (int i=0;i<m;i++){
		if (pos[i].empty()) continue;
		
		for (int j=0;j<pos[i].size()-1;j++){
			int A=pos[i][j]*m+i,B=pos[i][j+1]*m+i;
			G[A].pb(B);
			if (pos[i][j]==pos[i][j+1]-1) G[B].pb(A);
		}
		
		for (int j=0;j<pos[i].size();j++){
			int A=pos[i][j]*m+i;
			
			if (i){
				int p=lower_bound(pos[i-1].begin(),pos[i-1].end(),pos[i][j])-pos[i-1].begin();
				if (p!=pos[i-1].size()){
					p=pos[i-1][p];
					G[A].pb(p*m+(i-1));	
				}
				
			} 
			if (i!=m-1){
				int p=lower_bound(pos[i+1].begin(),pos[i+1].end(),pos[i][j])-pos[i+1].begin();
				if (p==pos[i+1].size()) continue;
				p=pos[i+1][p];
				G[A].pb(p*m+(i+1));
			}
		}
	}
//	cerr<<123<<endl;
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
			if (V[i][j]=='.') bad[i*m+j]=true;
		}
	}
	Kos();
}
/*
5 7
#..###.
.#.#...
#....#.
#....##
#######
4 1 1 1 0 3 1
*/
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict

class F1fallingsandeasyversionbootcamp(Basebootcamp):
    def __init__(self, n=5, m=7, a=None):
        if a is None:
            a = [4, 1, 1, 1, 0, 3, 1]
        self.n = n
        self.m = m
        self.a = a.copy()
    
    def case_generator(self):
        n = self.n
        m = self.m
        a = self.a
        
        cols = {}
        for j in range(m):
            if a[j] == 0:
                cols[j] = set()
                continue
            # Ensure a[j] <= n
            if a[j] > n:
                a[j] = n
            sand_rows = random.sample(range(n), a[j])
            cols[j] = set(sand_rows)
        
        grid = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append('#' if i in cols[j] else '.')
            grid.append(''.join(row))
        
        return {
            'grid': grid,
            'a': a
        }
    
    @staticmethod
    def prompt_func(question_case):
        grid = question_case['grid']
        a = question_case['a']
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        prompt = (
            "You are presented with a sand puzzle. The board has {n} rows and {m} columns. "
            "Each cell is either empty ('.') or contains a sand block ('#'). When you disturb a sand block, "
            "it falls down the column, disturbing all adjacent sand blocks (up, down, left, right), which then also fall. "
            "Your goal is to determine the minimum number of sand blocks you need to disturb so that each column i has at least {a_i} sand blocks in the counter below.\n\n"
        )
        prompt += "The board is as follows:\n"
        for row in grid:
            prompt += f"{row}\n"
        prompt += (
            "The required sand blocks for each column are: {a_str}\n\n"
            "Please provide the minimum number of operations needed. Place your answer within [answer] tags."
        )
        
        # Fix the a[i] issue by using proper string formatting
        a_str = ", ".join(map(str, a))
        prompt = prompt.format(
            n=n,
            m=m,
            a_i=" (for each column i, the required sand blocks are {})".format(a_str),
            a_str=a_str
        )
        
        return prompt
    
    @staticmethod
    def extract_output(output):
        # Use a more robust regular expression to extract the answer
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        return int(matches[-1])
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        grid = identity['grid']
        a = identity['a']
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        sand_blocks = []
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '#':
                    sand_blocks.append((i, j))
        
        if not sand_blocks:
            return solution == 0
        
        idx_map = {(i, j): idx for idx, (i, j) in enumerate(sand_blocks)}
        total = len(sand_blocks)
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for idx, (i, j) in enumerate(sand_blocks):
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m and grid[ni][nj] == '#':
                    if (ni, nj) in idx_map:
                        neighbor_idx = idx_map[(ni, nj)]
                        if neighbor_idx != idx:
                            graph[idx].append(neighbor_idx)
                            reverse_graph[neighbor_idx].append(idx)
        
        visited = [False] * total
        order = []
        
        def dfs(u):
            stack = [(u, False)]
            while stack:
                node, processed = stack.pop()
                if processed:
                    order.append(node)
                    continue
                if visited[node]:
                    continue
                visited[node] = True
                stack.append((node, True))
                for v in graph[node]:
                    if not visited[v]:
                        stack.append((v, False))
        
        for i in range(total):
            if not visited[i]:
                dfs(i)
        
        visited = [False] * total
        component = [0] * total
        current_component = 0
        
        def reverse_dfs(u, label):
            stack = [u]
            visited[u] = True
            component[u] = label
            while stack:
                node = stack.pop()
                for v in reverse_graph[node]:
                    if not visited[v]:
                        visited[v] = True
                        component[v] = label
                        stack.append(v)
        
        for node in reversed(order):
            if not visited[node]:
                reverse_dfs(node, current_component)
                current_component += 1
        
        in_degree = defaultdict(int)
        for u in range(total):
            for v in graph[u]:
                if component[u] != component[v]:
                    in_degree[component[v]] += 1
        
        count = 0
        for i in range(current_component):
            if in_degree[i] == 0:
                count += 1
        
        return solution == count
