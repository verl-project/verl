"""# 

### 谜题描述
Anya has bought a new smartphone that uses Berdroid operating system. The smartphone menu has exactly n applications, each application has its own icon. The icons are located on different screens, one screen contains k icons. The icons from the first to the k-th one are located on the first screen, from the (k + 1)-th to the 2k-th ones are on the second screen and so on (the last screen may be partially empty).

Initially the smartphone menu is showing the screen number 1. To launch the application with the icon located on the screen t, Anya needs to make the following gestures: first she scrolls to the required screen number t, by making t - 1 gestures (if the icon is on the screen t), and then make another gesture — press the icon of the required application exactly once to launch it.

After the application is launched, the menu returns to the first screen. That is, to launch the next application you need to scroll through the menu again starting from the screen number 1.

All applications are numbered from 1 to n. We know a certain order in which the icons of the applications are located in the menu at the beginning, but it changes as long as you use the operating system. Berdroid is intelligent system, so it changes the order of the icons by moving the more frequently used icons to the beginning of the list. Formally, right after an application is launched, Berdroid swaps the application icon and the icon of a preceding application (that is, the icon of an application on the position that is smaller by one in the order of menu). The preceding icon may possibly be located on the adjacent screen. The only exception is when the icon of the launched application already occupies the first place, in this case the icon arrangement doesn't change.

Anya has planned the order in which she will launch applications. How many gestures should Anya make to launch the applications in the planned order? 

Note that one application may be launched multiple times.

Input

The first line of the input contains three numbers n, m, k (1 ≤ n, m, k ≤ 105) — the number of applications that Anya has on her smartphone, the number of applications that will be launched and the number of icons that are located on the same screen.

The next line contains n integers, permutation a1, a2, ..., an — the initial order of icons from left to right in the menu (from the first to the last one), ai — is the id of the application, whose icon goes i-th in the menu. Each integer from 1 to n occurs exactly once among ai.

The third line contains m integers b1, b2, ..., bm(1 ≤ bi ≤ n) — the ids of the launched applications in the planned order. One application may be launched multiple times.

Output

Print a single number — the number of gestures that Anya needs to make to launch all the applications in the desired order.

Examples

Input

8 3 3
1 2 3 4 5 6 7 8
7 8 1


Output

7


Input

5 4 2
3 1 5 2 4
4 4 4 4


Output

8

Note

In the first test the initial configuration looks like (123)(456)(78), that is, the first screen contains icons of applications 1, 2, 3, the second screen contains icons 4, 5, 6, the third screen contains icons 7, 8. 

After application 7 is launched, we get the new arrangement of the icons — (123)(457)(68). To launch it Anya makes 3 gestures. 

After application 8 is launched, we get configuration (123)(457)(86). To launch it Anya makes 3 gestures. 

After application 1 is launched, the arrangement of icons in the menu doesn't change. To launch it Anya makes 1 gesture.

In total, Anya makes 7 gestures.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
[n, m, k] = [int(x) for x in raw_input().split(\" \")]

app_po = [0] * 10 ** 6
po_app = [0] * 10 ** 6
data = [int(x) for x in raw_input().split(\" \")]
for i in range(len(data)):
    app_po[data[i]] = i + 1
    po_app[i + 1] = data[i]

q = [int(x) for x in raw_input().split(\" \")]

ans = 0
for x in q:
    po = app_po[x]
    ans += po / k
    if po % k != 0:
        ans += 1

    if po != 1:
        tempapp = po_app[po - 1]
        po_app[po - 1] = x
        app_po[x] = po - 1
        po_app[po] = tempapp
        app_po[tempapp] = po

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Canyaandsmartphonebootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=20, m_min=1, m_max=20, k_min=1, k_max=20):
        """
        参数调整说明：
        - 允许k=1的边界情况
        - 支持n=1的极端情况
        - 增加n/m/k的生成范围
        """
        self.n_min = max(n_min, 1)
        self.n_max = max(n_max, self.n_min)
        self.m_min = max(m_min, 1)
        self.m_max = max(m_max, self.m_min)
        self.k_min = max(k_min, 1)
        self.k_max = max(k_max, self.k_min)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        k = random.randint(self.k_min, min(self.k_max, n))  # 确保k不超过n
        
        # 生成1~n的随机排列
        a = list(range(1, n+1))
        random.shuffle(a)
        
        # 生成可能包含重复的启动序列
        b = [random.choice(a) for _ in range(m)]
        
        return {
            'n': n,
            'm': m,
            'k': k,
            'a': a.copy(),  # 防止后续修改影响原始数据
            'b': b.copy()
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        prompt = f"""你是Berdroid系统的测试工程师，需要计算Anya按照指定顺序启动应用所需的总手势次数。规则如下：

1. 屏幕划分：共有{question_case['n']}个应用，每屏显示{question_case['k']}个图标，按从左到右顺序排列。例如：
   - 第1屏：1~{question_case['k']}号位置
   - 第2屏：{question_case['k']+1}~{2*question_case['k']}号位置
   - （最后一屏可能不满）

2. 启动流程：
   a) 当前显示第1屏
   b) 要启动位于t屏的应用，需要滚动(t-1)次+点击1次，共t次手势
   c) 每次启动后自动回到第1屏

3. 动态调整规则：
   - 启动应用后，该应用会与前一个位置的应用交换（若当前不在第1位）
   - 例：应用在位置5→启动后与位置4的应用交换
   - 交换可能跨屏幕发生（如位置4和5在不同屏幕仍然交换）

输入格式：
第一行：n m k
第二行：a1 a2 ... an （初始图标顺序）
第三行：b1 b2 ... bm （启动顺序）

请根据以下数据计算总手势次数，将最终答案用[answer]标签包裹：

输入：
{question_case['n']} {question_case['m']} {question_case['k']}
{' '.join(map(str, question_case['a']))}
{' '.join(map(str, question_case['b']))}

示例答案格式：[answer]42[/answer]"""
        return prompt
    
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
        try:
            n = identity['n']
            m = identity['m']
            k = identity['k']
            a = identity['a']
            b = identity['b']
        except KeyError:
            return False
        
        # 初始化数据结构
        app_pos = {app: i+1 for i, app in enumerate(a)}
        pos_app = [0] * (n + 2)  # 位置从1开始
        for idx, app in enumerate(a):
            pos_app[idx+1] = app
        
        total_gestures = 0
        
        for app_id in b:
            current_pos = app_pos[app_id]
            # 计算手势次数
            screen = (current_pos - 1) // k
            total_gestures += (screen + 1)  # 滚动screen次 + 点击1次
            
            # 处理位置交换
            if current_pos > 1:
                prev_pos = current_pos - 1
                prev_app = pos_app[prev_pos]
                
                # 更新映射关系
                app_pos[app_id] = prev_pos
                app_pos[prev_app] = current_pos
                pos_app[current_pos] = prev_app
                pos_app[prev_pos] = app_id
        
        try:
            return int(solution) == total_gestures
        except:
            return False
