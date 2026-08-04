"""# 

### 谜题描述
Let's define a string <x> as an opening tag, where x is any small letter of the Latin alphabet. Each opening tag matches a closing tag of the type </x>, where x is the same letter.

Tegs can be nested into each other: in this case one opening and closing tag pair is located inside another pair.

Let's define the notion of a XML-text: 

  * an empty string is a XML-text
  * if s is a XML-text, then s'=<a>+s+</a> also is a XML-text, where a is any small Latin letter 
  * if s1, s2 are XML-texts, then s1+s2 also is a XML-text



You are given a XML-text (it is guaranteed that the text is valid), your task is to print in the following form: 

  * each tag (opening and closing) is located on a single line 
  * print before the tag 2 * h spaces, where h is the level of the tag's nestedness. 

Input

The input data consists on the only non-empty string — the XML-text, its length does not exceed 1000 characters. It is guaranteed that the text is valid. The text contains no spaces.

Output

Print the given XML-text according to the above-given rules.

Examples

Input

&lt;a&gt;&lt;b&gt;&lt;c&gt;&lt;/c&gt;&lt;/b&gt;&lt;/a&gt;


Output

&lt;a&gt;
  &lt;b&gt;
    &lt;c&gt;
    &lt;/c&gt;
  &lt;/b&gt;
&lt;/a&gt;


Input

&lt;a&gt;&lt;b&gt;&lt;/b&gt;&lt;d&gt;&lt;c&gt;&lt;/c&gt;&lt;/d&gt;&lt;/a&gt;


Output

&lt;a&gt;
  &lt;b&gt;
  &lt;/b&gt;
  &lt;d&gt;
    &lt;c&gt;
    &lt;/c&gt;
  &lt;/d&gt;
&lt;/a&gt;

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import re
import sys

s=sys.stdin.readline()
result = re.findall('<(\w)>|<(/\w)>',s)
indent = 0
for match in result:
    if (match[1]==''):
        print ' '*indent+'<'+match[0]+\">\"
        indent = indent+2
    else:
        indent = indent-2
        print ' '*indent + '<'+match[1]+\">\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Bsimplexmlbootcamp(Basebootcamp):
    def __init__(self, max_depth=3, max_children=2):
        self.params = {
            'max_depth': max_depth,
            'max_children': max_children  # 控制每个节点最大子节点数
        }
    
    def case_generator(self):
        def build_xml(current_depth):
            # 保证至少生成一个根标签
            if current_depth > self.params['max_depth']:
                return ''
            
            tag = random.choice('abcdefghijklmnopqrstuvwxyz')
            xml = [f'<{tag}>']
            
            # 随机决定是否生成子节点
            if current_depth < self.params['max_depth'] and random.random() < 0.8:
                num_children = random.randint(1, self.params['max_children'])
                for _ in range(num_children):
                    xml.append(build_xml(current_depth + 1))
            
            xml.append(f'</{tag}>')
            return ''.join(xml)
        
        # 确保根标签有效
        xml_str = build_xml(0)
        return {'xml': xml_str}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        xml_input = question_case['xml']
        return f"""请严格按照以下规则格式化给定的Bsimplexml文本：

格式规范：
1. 每个标签（包括开标签和闭标签）必须独占一行
2. 行首缩进使用2个空格乘以当前嵌套层级：
   - 根标签层级为0（无缩进）
   - 子标签层级逐层递增
   - 闭标签与对应开标签保持相同层级
3. 标签格式保持原始大小写（均为小写字母）

输入示例：
输入：<a><b><c></c></b><d></d></a>
正确输出：
<a>
  <b>
    <c>
    </c>
  </b>
  <d>
  </d>
</a>

需要处理的Bsimplexml：
{xml_input}

请将最终结果包裹在[answer]标签内。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if matches:
            # 清理首尾空白和空行
            content = matches[-1].strip()
            return '\n'.join([line.rstrip() for line in content.split('\n') if line.strip()])
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 解析原始Bsimplexml结构
            tags = []
            stack = []
            for match in re.finditer(r'</?([a-z])>', identity['xml']):
                is_closing = match.group(0).startswith('</')
                char = match.group(1)
                tags.append((is_closing, char))
                
                # 验证标签匹配
                if not is_closing:
                    stack.append(char)
                else:
                    if not stack or stack.pop() != char:
                        return False

            # 生成标准答案
            expected = []
            indent_level = 0
            for is_closing, char in tags:
                if is_closing:
                    indent_level -= 1
                
                line = ' ' * (indent_level * 2) + f'<{"/" if is_closing else ""}{char}>'
                expected.append(line)
                
                if not is_closing:
                    indent_level += 1

            # 对比用户答案
            user_lines = solution.split('\n')
            return user_lines == expected
        except:
            return False
