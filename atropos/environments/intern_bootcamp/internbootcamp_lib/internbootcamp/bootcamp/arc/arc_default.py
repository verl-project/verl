import sys
import re
import json
from pathlib import Path
from typing import Tuple
import random

sys.path.insert(0, Path(__file__).parent.parent.parent.as_posix())

from internbootcamp.libs.re_arc.main import get_generators, get_verifiers, get_rng_difficulty, get_pso_difficulty, demo_generator, generate_dataset, demo_dataset, evaluate_verifiers_on_original_tasks
from bootcamp.base import Basebootcamp

Grid = Tuple[Tuple[int]]

def list_to_tuple(l: list) -> Tuple:
    """递归地将列表转换为元组"""
    return tuple(list_to_tuple(item) if isinstance(item, list) else item for item in l)

def tuple_to_list(t: Tuple) -> list:
    """递归地将元组转换为列表"""
    return [tuple_to_list(item) if isinstance(item, tuple) else item for item in t]

template = """
### **ARC Puzzle Simple Question Template**

1. **Problem Description**  
   - There is a logical relationship between the input and output grids. The goal is to deduce the rule and solve the test grid.

2. **Example Explanation**  
{examples}

3. **Test Grid**  
   **Input**:
```arcmatrix
[
{test_input}
]
```
**Output**:  
?
"""

example_template = """
- Example {index}:  
  **Input**:  
```arcmatrix
[
{input}
]
```  
  **Output**:  
```arcmatrix
[
{output}
]
```
"""

def generate_arc_puzzle(examples, test_case):
    """
    Generates an ARC puzzle question.
    
    :param examples: List of dicts, each containing "input" and "output" fields.
    :param test_case: Dict containing "input" (grid for the test case).
    :return: Formatted puzzle string.
    """
    # Generate the examples section dynamically
    examples_section = ""
    for i, example in enumerate(examples, start=1):
        examples_section += example_template.format(
            index=i,
            input=',\n'.join([str(list(x)) for x in example["input"]]),
            output=',\n'.join([str(list(x)) for x in example["output"]])
        )
    
    # Format the full template
    return template.format(
        examples=examples_section.strip(),
        test_input=',\n'.join([str(list(x)) for x in test_case])
    )

class Arcbootcamp(Basebootcamp):
    
    # 类变量，所有类方法共享
    verifiers_mapper = get_verifiers()
    def __init__(self, task_key_file: str = None):
        task_key_file = "/".join(__file__.split('/')[:-4]) + "/" + task_key_file
        task_key = [json.loads(f) for f in open(task_key_file, 'r').readlines()]
        self.task_key = random.choice(task_key)['key']
        self.generators = get_generators()
        self.hint_examples = []
        self.current_example = None


    def case_generator(self):
        if self.task_key not in self.generators:
            raise ValueError(f"Task key '{self.task_key}' not found in generators.")
        generator = self.generators[self.task_key]
        self.current_example = generator(0, 1)
        
        for _ in range(3):
            self.hint_examples.append(generator(0, 1))
        input_grid = self.current_example['input']
        return  {'input_grid': input_grid, 'task_key': self.task_key}
    

    def prompt_func(self, identity) -> str:
        """
        Process the input_data and return the processed prompt.
        """
        return generate_arc_puzzle(self.hint_examples, identity['input_grid'])
    
    @staticmethod
    def extract_output(output:str)->Grid:
        """
        Extract the output from the solution.
        """
        json_objects = re.findall(r'\[\s*\[\s*.*?\s*\]\s*\]', output, re.DOTALL)
        json_obj = None
        for item in reversed(json_objects):
            try:
                json_obj = json.loads(item)
                if isinstance(json_obj, list) and all(isinstance(i, list) for i in json_obj):
                    return list_to_tuple(json_obj)
            except json.JSONDecodeError:
                continue
        return list_to_tuple(json_obj)
    
    @classmethod
    def _verify_correction(cls, solution:Grid, identity: dict) -> bool:
        """
        Verify the correction of the solution.
        
        Ensure all parameters are 'Grid' type.
        """
        if "std_ans" in identity and type(identity["std_ans"]) == str and list_to_tuple(json.loads(identity["std_ans"])) == solution:
            # 如果提供了答案，直接比较答案
            return True
        if "std_ans" in identity and type(identity["std_ans"]) == list and list_to_tuple(identity["std_ans"]) == solution:
            return True
        input_grid, task_key = identity['input_grid'], identity['task_key']
        if type(input_grid) == str:
            input_grid = list_to_tuple(json.loads(input_grid))
        else:
            input_grid = list_to_tuple(input_grid)
        verifier = cls.verifiers_mapper[task_key]  # 使用类变量 verifiers
        std_ans = verifier(input_grid)
        return std_ans == solution
    
    
    
if __name__ == "__main__":
    
    
    def extract_output(output):
        """
        Extract the output from the solution.
        """
        json_objects = re.findall(r'\[\s*\[\s*.*?\s*\]\s*\]', output, re.DOTALL)
        json_obj = None
        for item in reversed(json_objects):
            try:
                json_obj = json.loads(item)
                if isinstance(json_obj, list) and all(isinstance(i, list) for i in json_obj):
                    return json_obj
            except json.JSONDecodeError:
                continue
        return json_obj
    
    # Unit Test
    import jsonlines
    with jsonlines.open('your test file path.jsonl') as reader:
        test_data = list(reader)

    test_item = test_data[0]
    output = test_item['output']
    test_identity = {
        'input_grid': tuple_to_list(extract_output(test_item['prompt'])), # 这里是一个字符串
        'task_key': test_item['task_id']
    }
    print(test_identity)
    res = Arcbootcamp.verify_score(output, test_identity,format_score=0.1)
    print(res)
    # 1.0
    print('-------------------')
    test_identity = {
        'input_grid': tuple_to_list(extract_output(test_item['prompt'])), # 
        'task_key': "3bd67248", # 换一个其他的id,
    }
    print(test_identity)
    res = Arcbootcamp.verify_score(output, test_identity,format_score=0.1)
    print(res)
