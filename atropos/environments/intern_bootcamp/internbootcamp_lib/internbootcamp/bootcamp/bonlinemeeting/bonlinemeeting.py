"""# 

### 谜题描述
Nearly each project of the F company has a whole team of developers working on it. They often are in different rooms of the office in different cities and even countries. To keep in touch and track the results of the project, the F company conducts shared online meetings in a Spyke chat.

One day the director of the F company got hold of the records of a part of an online meeting of one successful team. The director watched the record and wanted to talk to the team leader. But how can he tell who the leader is? The director logically supposed that the leader is the person who is present at any conversation during a chat meeting. In other words, if at some moment of time at least one person is present on the meeting, then the leader is present on the meeting.

You are the assistant director. Given the 'user logged on'/'user logged off' messages of the meeting in the chronological order, help the director determine who can be the leader. Note that the director has the record of only a continuous part of the meeting (probably, it's not the whole meeting).

Input

The first line contains integers n and m (1 ≤ n, m ≤ 105) — the number of team participants and the number of messages. Each of the next m lines contains a message in the format:

  * '+ id': the record means that the person with number id (1 ≤ id ≤ n) has logged on to the meeting. 
  * '- id': the record means that the person with number id (1 ≤ id ≤ n) has logged off from the meeting. 



Assume that all the people of the team are numbered from 1 to n and the messages are given in the chronological order. It is guaranteed that the given sequence is the correct record of a continuous part of the meeting. It is guaranteed that no two log on/log off events occurred simultaneously.

Output

In the first line print integer k (0 ≤ k ≤ n) — how many people can be leaders. In the next line, print k integers in the increasing order — the numbers of the people who can be leaders.

If the data is such that no member of the team can be a leader, print a single number 0.

Examples

Input

5 4
+ 1
+ 2
- 2
- 1


Output

4
1 3 4 5 

Input

3 2
+ 1
- 2


Output

1
3 

Input

2 4
+ 1
- 1
+ 2
- 2


Output

0


Input

5 6
+ 1
- 1
- 3
+ 3
+ 4
- 4


Output

3
2 3 5 

Input

2 4
+ 1
- 2
+ 2
- 1


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
char str[2];
int n, m, a[101000], b[101000], ans, l[101000], v[101000], s[101000];
int main() {
  scanf(\"%d%d\", &n, &m);
  for (int i = 1; i <= m; i++) {
    scanf(\"%s%d\", str, &b[i]);
    a[i] = (str[0] == '+' ? 1 : -1);
  }
  for (int i = 1; i <= m; i++) {
    if (a[i] < 0 && !l[b[i]]) s[0]++;
    s[i] = a[i];
    l[b[i]] = i;
  }
  for (int i = 1; i <= m; i++) s[i] += s[i - 1];
  for (int i = 0; i <= m; i++) s[i] = (s[i] > 0);
  for (int i = 1; i <= m; i++) s[i] += s[i - 1];
  memset(l, 0, sizeof l);
  for (int i = 1; i <= m; i++) {
    if (a[i] > 0 &&
        (!l[b[i]] && s[i - 1] > 0 || l[b[i]] && s[i - 1] - s[l[b[i]] - 1] > 0))
      v[b[i]] = 1;
    l[b[i]] = i;
  }
  for (int i = 1; i <= n; i++)
    if (l[i] && a[l[i]] < 0 && s[m] - s[l[i] - 1] > 0) v[i] = 1;
  for (int i = 1; i <= n; i++) ans += !v[i];
  printf(\"%d\n\", ans);
  for (int i = 1; i <= n; i++)
    if (!v[i]) printf(\"%d \", i);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Bonlinemeetingbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'n_min': params.get('n_min', 2),
            'n_max': params.get('n_max', 10),
            'm_min': params.get('m_min', 3),
            'm_max': params.get('m_max', 20),
        }
    
    def case_generator(self):
        n = random.randint(self.params['n_min'], self.params['n_max'])
        m = random.randint(self.params['m_min'], self.params['m_max'])
        
        messages = []
        user_states = defaultdict(bool)  # False表示离线
        
        for _ in range(m):
            user_id = random.randint(1, n)
            current_state = user_states[user_id]
            
            # 自动生成合法操作
            op = '-' if current_state else '+'
            messages.append(f"{op} {user_id}")
            user_states[user_id] = not current_state
        
        # 确保最后所有用户都离线
        for user in list(user_states.keys()):
            if user_states[user]:
                messages.append(f"- {user}")
                user_states[user] = False
        
        expected = self.solve_leader(n, messages)
        return {
            'n': n,
            'm': len(messages),  # 更新实际消息数
            'messages': messages,
            'expected': expected
        }
    
    @staticmethod
    def solve_leader(n, messages):
        m = len(messages)
        a = [0]*(m+1)  # 操作数组（1-based）
        b = [0]*(m+1)  # 用户数组（1-based）
        
        # 解析操作
        for i in range(1, m+1):
            op, id_str = messages[i-1].split()
            a[i] = 1 if op == '+' else -1
            b[i] = int(id_str)
        
        # 第一遍处理：初始化s数组
        l = defaultdict(int)  # 记录用户最后一次操作位置
        s = [0]*(m+2)  # 前缀和数组
        
        for i in range(1, m+1):
            user = b[i]
            # 处理首次登出但之前未登录的情况
            if a[i] == -1 and l[user] == 0:
                s[0] += 1  # 初始未在线但收到登出
            s[i] = a[i]
            l[user] = i
        
        # 计算在线人数前缀和
        for i in range(1, m+1):
            s[i] += s[i-1]
        
        # 转换为在线状态标记（1在线，0离线）
        for i in range(m+1):
            s[i] = 1 if s[i] > 0 else 0
        
        # 转换为累计在线时间
        for i in range(1, m+1):
            s[i] += s[i-1]
        
        # 第二遍处理：验证候选者
        l = defaultdict(int)  # 重置记录
        v = [0]*(n+1)  # 违规标记
        
        for i in range(1, m+1):
            user = b[i]
            if a[i] == 1:  # 登录事件
                violation = False
                if l[user] == 0:  # 首次登录
                    if s[i-1] > 0:  # 登录前已有在线
                        violation = True
                else:  # 非首次登录
                    prev = l[user]
                    if (s[i-1] - s[prev-1]) > 0:  # 两次登录之间有其他人
                        violation = True
                
                if violation:
                    v[user] = 1
            l[user] = i  # 更新最后操作位置
        
        # 检查最后一次登出后的状态
        for user in range(1, n+1):
            last_op_idx = l[user]
            if last_op_idx != 0 and a[last_op_idx] == -1:  # 最后操作是登出
                if (s[m] - s[last_op_idx-1]) > 0:  # 登出后仍有其他人
                    v[user] = 1
        
        # 收集未违规的候选人
        leaders = [user for user in range(1, n+1) if v[user] == 0]
        return sorted(leaders)
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        messages = question_case['messages']
        
        prompt = (
            "你是F公司的助理董事，需要根据会议记录确定可能的团队领导者。领导者的定义是：在任何时刻，只要至少有一人在线，领导者必须在线。\n\n"
            "输入格式：\n"
            f"第一行包含两个整数n和m（{n} {m}），表示团队成员数和消息数。\n"
            "接下来m行每行格式为'+ id'或'- id'，表示用户id的登录/登出记录。\n\n"
            "输出要求：\n"
            "第一行为可能的领导者数量k，第二行为按升序排列的k个ID。若无可能领导者，仅输出0。\n\n"
            "当前会议记录：\n" +
            '\n'.join(messages) + "\n\n"
            "请将最终答案放置在[answer]和[/answer]标记之间，示例如下：\n"
            "[answer]\n"
            "2\n"
            "3 5\n"
            "[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 匹配最后一个answer块
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        # 处理答案内容
        answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        
        try:
            if not lines:
                return None
            
            # 处理0的情况
            if lines[0] == '0':
                return [] if len(lines) == 1 else None  # 严格格式检查
            
            k = int(lines[0])
            # 检查数量一致性
            if len(lines) < 2 or k == 0:
                return None
            
            ids = list(map(int, lines[1].split()))
            if len(ids) != k or sorted(ids) != ids:
                return None  # 数量或顺序错误
            
            return ids
        except (ValueError, IndexError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected']
        # 预期是空列表表示0个候选人
        if not expected:
            return solution == []
        # 比较ID列表是否完全一致
        return solution == expected
