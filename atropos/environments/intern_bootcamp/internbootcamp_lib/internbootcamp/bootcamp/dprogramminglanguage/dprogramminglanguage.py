"""# 

### 谜题描述
Recently, Valery have come across an entirely new programming language. Most of all the language attracted him with template functions and procedures. Let us remind you that templates are tools of a language, designed to encode generic algorithms, without reference to some parameters (e.g., data types, buffer sizes, default values).

Valery decided to examine template procedures in this language in more detail. The description of a template procedure consists of the procedure name and the list of its parameter types. The generic type T parameters can be used as parameters of template procedures.

A procedure call consists of a procedure name and a list of variable parameters. Let's call a procedure suitable for this call if the following conditions are fulfilled: 

  * its name equals to the name of the called procedure; 
  * the number of its parameters equals to the number of parameters of the procedure call; 
  * the types of variables in the procedure call match the corresponding types of its parameters. The variable type matches the type of a parameter if the parameter has a generic type T or the type of the variable and the parameter are the same. 



You are given a description of some set of template procedures. You are also given a list of variables used in the program, as well as direct procedure calls that use the described variables. For each call you need to count the number of procedures that are suitable for this call.

Input

The first line contains a single integer n (1 ≤ n ≤ 1000) — the number of template procedures. The next n lines contain the description of the procedures specified in the following format:

\"void procedureName (type_1, type_2, ..., type_t)\" (1 ≤ t ≤ 5), where void is the keyword, procedureName is the procedure name, type_i is the type of the next parameter. Types of language parameters can be \"int\", \"string\", \"double\", and the keyword \"T\", which denotes the generic type.

The next line contains a single integer m (1 ≤ m ≤ 1000) — the number of used variables. Next m lines specify the description of the variables in the following format:

\"type variableName\", where type is the type of variable that can take values \"int\", \"string\", \"double\", variableName — the name of the variable.

The next line contains a single integer k (1 ≤ k ≤ 1000) — the number of procedure calls. Next k lines specify the procedure calls in the following format:

\"procedureName (var_1, var_2, ..., var_t)\" (1 ≤ t ≤ 5), where procedureName is the name of the procedure, var_i is the name of a variable.

The lines describing the variables, template procedures and their calls may contain spaces at the beginning of the line and at the end of the line, before and after the brackets and commas. Spaces may be before and after keyword void. The length of each input line does not exceed 100 characters. The names of variables and procedures are non-empty strings of lowercase English letters and numbers with lengths of not more than 10 characters. Note that this is the only condition at the names. Only the specified variables are used in procedure calls. The names of the variables are distinct. No two procedures are the same. Two procedures are the same, if they have identical names and identical ordered sets of types of their parameters.

Output

On each of k lines print a single number, where the i-th number stands for the number of suitable template procedures for the i-th call.

Examples

Input

4
void f(int,T)
void  f(T, T)
 void foo123   ( int,  double,  string,string  ) 
  void  p(T,double)
3
int a
 string    s
double x123 
5
f(a,  a)
  f(s,a   )
foo   (a,s,s)
 f  (  s  ,x123)
proc(a)


Output

2
1
0
1
0


Input

6
void f(string,double,int)
void f(int)
   void f  ( T  )
void procedure(int,double)
void f  (T, double,int)   
void f(string, T,T)
4
 int a
 int x
string  t
double  val  
5
f(t, a, a)
f(t,val,a)
f(val,a, val)
 solve300(val, val)
f  (x)


Output

1
3
0
0
2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
n = int(raw_input().strip())
template = {}
func = {}
var = {}
for i in xrange(n):
    a = raw_input().strip().split(' ',1)[1].replace(' ','').split('(')
    if template.get(a[0],-1) == -1:
        template[a[0]] = []
    template[a[0]].append (a[1].strip(')').split(','))

n = int(raw_input().strip())
for i in xrange(n):
    x = raw_input().strip().split()
    var[x[1]] = x[0]
n = int(raw_input().strip())
for i in xrange(n):
    a = raw_input().strip().split('(')
    func[a[0].strip()] = a[1].strip(')').replace(' ','').split(',')
    res = 0
    for k in template.keys():
        p = template[k]
        if k==a[0].strip():
            for pp in p:
                fp = func[a[0].strip()]
                if len(pp)==len(fp):
                    for i in xrange(len(pp)):
                        if pp[i]!='T' and pp[i]!=var[fp[i]]:
                            break
                    else:
                        res +=1
    print res
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

class Dprogramminglanguagebootcamp(Basebootcamp):
    def __init__(self, max_procedures=10, max_variables=10, max_calls=10, procedure_name_length=5, var_name_length=5, t_probability=0.3):
        self.max_procedures = max_procedures
        self.max_variables = max_variables
        self.max_calls = max_calls
        self.procedure_name_length = procedure_name_length
        self.var_name_length = var_name_length
        self.t_probability = t_probability

    def case_generator(self):
        def generate_name(length, prefix=''):
            chars = string.ascii_lowercase + string.digits
            return prefix + ''.join(random.choice(chars) for _ in range(length))
        
        # 生成变量
        m = random.randint(1, self.max_variables)
        variables = {}
        for _ in range(m):
            while True:
                var_name = generate_name(self.var_name_length, 'var_')
                if var_name not in variables:
                    break
            variables[var_name] = random.choice(['int', 'string', 'double'])
        
        # 生成调用时确保部分调用有匹配过程
        k = random.randint(1, self.max_calls)
        calls = []
        for _ in range(k):
            # 生成调用时，参数数量关联过程参数
            call_name = generate_name(self.procedure_name_length) if random.random() < 0.3 else None
            params_count = random.randint(1, 5)
            available_vars = list(variables.keys())
            vars_list = [random.choice(available_vars) for _ in range(params_count)] if available_vars else []
            calls.append({'name': call_name, 'vars': vars_list})
        
        # 生成过程，部分针对调用生成
        existing_procedures = set()
        procedures = []
        
        # 随机生成基础过程
        base_procedure_count = random.randint(0, self.max_procedures)
        for _ in range(base_procedure_count):
            name = generate_name(self.procedure_name_length)
            params_count = random.randint(1, 5)
            params = []
            for _ in range(params_count):
                if random.random() < self.t_probability:
                    param = 'T'
                else:
                    param = random.choice(['int', 'string', 'double'])
                params.append(param)
            key = (name, tuple(params))
            if key not in existing_procedures:
                existing_procedures.add(key)
                procedures.append({'name': name, 'params': params})
        
        # 为部分调用生成匹配过程
        for call in calls:
            if call['name'] is None or random.random() > 0.6:
                continue  # 不处理未命名调用或随机跳过
            var_types = [variables[var] for var in call['vars']]
            # 生成匹配参数类型的过程
            for _ in range(random.randint(0, 2)):  # 每个调用生成0-2个匹配过程
                params = []
                for t in var_types:
                    if random.random() < self.t_probability:
                        params.append('T')
                    else:
                        params.append(t)
                key = (call['name'], tuple(params))
                if key not in existing_procedures:
                    existing_procedures.add(key)
                    procedures.append({'name': call['name'], 'params': params})
        
        # 最终确保过程名称多样性
        procedure_names = list({proc['name'] for proc in procedures})
        for call in calls:
            if call['name'] is None and procedure_names:
                call['name'] = random.choice(procedure_names)
        
        return {
            'procedures': procedures,
            'variables': variables,
            'calls': calls
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = []
        input_lines.append(str(len(question_case['procedures'])))
        for proc in question_case['procedures']:
            # 模拟输入空格
            spaces_before = ' ' * random.randint(0, 2)
            spaces_after = ' ' * random.randint(0, 2)
            params = [f"{' ' * random.randint(0,1)}{p}{' ' * random.randint(0,1)}" for p in proc['params']]
            input_lines.append(f"void{spaces_before}{proc['name']}{spaces_after}({','.join(params)})".replace(' ', ' '))
        input_lines.append(str(len(question_case['variables'])))
        for var_name, var_type in question_case['variables'].items():
            input_lines.append(f"{var_type}{' ' * random.randint(1,3)}{var_name}")
        input_lines.append(str(len(question_case['calls'])))
        for call in question_case['calls']:
            spacer = ' ' * random.randint(0, 2)
            params = [f"{spacer}{var}{spacer}" for var in call['vars']]
            input_lines.append(f"{call['name']}({','.join(params)})")
        input_example = '\n'.join(input_lines)
        
        prompt = f"""你是编程竞赛的参赛者，需要解决一个关于模板过程调用的问题。请仔细阅读问题描述，并按照要求输出答案。

问题描述：

给定一组模板过程、变量列表和一系列过程调用，对于每个调用，统计有多少个模板过程适合该调用。

模板过程的条件如下：
1. 名称与调用名称相同。
2. 参数数量相同。
3. 每个参数的类型为T或者与实际变量类型相同。

输入格式：
- 第一行是整数n，表示模板过程的数量。
- 接下来的n行，每行描述一个模板过程，格式为："void 过程名 (参数类型列表)"，参数类型可以是int、string、double或T。
- 接下来一行是整数m，表示变量的数量。
- 接下来的m行，每行描述一个变量，格式为："类型 变量名"，类型是int、string、double中的一个。
- 接下来一行是整数k，表示调用的数量。
- 接下来的k行，每行描述一个调用，格式为："过程名 (变量列表)"。

输出格式：
输出k行，每行是对应调用的适合模板过程的数量。

请根据以下输入数据，编写程序解决问题。将你的答案放在[answer]标签内，每个结果占一行。

输入数据：
{input_example}

请将答案按顺序放在[answer]和[/answer]标签之间，例如：
[answer]
0
1
2
[/answer]

请确保你的输出格式正确，否则将无法得到分数。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = [line.strip() for line in last_match.split('\n') if line.strip()]
        solution = []
        for line in lines:
            try:
                solution.append(int(line))
            except ValueError:
                continue
        return solution if solution else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = []
        variables = identity['variables']
        for call in identity['calls']:
            call_name = call['name']
            var_names = call['vars']
            if not var_names:
                expected.append(0)
                continue
            var_types = [variables[name] for name in var_names]
            count = 0
            for proc in identity['procedures']:
                if proc['name'] != call_name:
                    continue
                if len(proc['params']) != len(var_names):
                    continue
                match = True
                for p_type, v_type in zip(proc['params'], var_types):
                    if p_type != 'T' and p_type != v_type:
                        match = False
                        break
                if match:
                    count += 1
            expected.append(count)
        return solution == expected
