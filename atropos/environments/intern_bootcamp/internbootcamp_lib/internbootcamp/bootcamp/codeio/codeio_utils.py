import json
import subprocess
import sys
import re
from pympler import asizeof
from decimal import Decimal

import shortuuid

import os
import signal
import math


refcode_template = """Tip: Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

<<<<refcode>>>>"""

output_pred_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following input:

<<<<input>>>>

Can you predict the output without writing any code? Please reason and put your final answer in the following json format: {"output": <your output>}, where <your output> should strictly match the the output requirement as specified."""

input_pred_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following output:

<<<<output>>>>

Can you predict a feasible input without writing any code? Please reason and put your final answer in the following json format: {"input": <your input>}, where <your input> should be a dictionary, even if the there is only one input variable, with keys strictly match the input variables' names as specified."""

solution_prefix="""from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
import signal
from string import ascii_lowercase, ascii_uppercase
from math import floor, log2, log10, sqrt, hypot, comb, gcd, ceil, inf, isqrt, lcm, factorial, dist
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge, heapreplace
from functools import reduce, lru_cache, cache, cmp_to_key
from random import randrange, shuffle
from operator import itemgetter, sub, or_, xor, and_
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import datetime
import string
import math
from math import atan2, pi
import collections
import bisect
import heapq
from heapq import nlargest
import functools
import random
from random import randint
import itertools
import operator
import re
import json
import numpy as np
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
import sys
from itertools import pairwise"""

build_testcases_prompt_advanced="""Task Overview:

Given a code file (not be not python), you need to organize a main function in python and generate a problem based on this function. The request includes the following components:

1. Main Function:
  - Refer to the provided code file to build your solution function to solve the problem; it should include the main logic of the code file.
  - If some self-defined modules are imported, please keep them as they are instead of write a placeholder for them.
  - The function should be named as `main_solution`; you can do this by either renaming some functions in the code file or calling multiple of them in the new main function.
  - The `main_solution` function must have JSON serializable input and output variables.
      If no input variables are needed in the reference code file, please adjust the code to make sure input variables always exist, like rewriting the code for one specific case to general cases.
      If some input variables in the original code file are not JSON serializable (like set, tuple, np.array, functions, self-defined objects, etc.), you need to convert the JSON serializable inputs of the `main_solution` function to the original input variables at the beginning of the function.
      If some output variables in the original code file are not JSON serializable (like set, tuple, np.array, functions, self-defined objects, etc.), you must convert them to JSON serializable outputs at the end of the `main_solution` function before returning.
      The size of both input and output variables should be reasonable, like less than 1KB.
      Try to avoid too complex input and output variables, like too many nested structures or extremely large numbers or floats with too many decimals.
  - The `main_solution` function should return the final output instead of printing it.
  - Please remove all plotting code and only keep the core solution code as we never want to return plots. Also remove all print statements, and writing to files.
  - Please always define the `main_solution` function at the end of this part, before this, you must prepare all the necessary code by referring to the code file to make sure the `main_solution` function can run correctly.

2. Input Output Description:
  - You need to provide clear descriptions of the input and output variables in the `main_solution` function.
  - In the descriptions, you should include the type of each variable and a brief explanation of its meaning.
    For example, if the variable is a dictionary, you should specify the key names and the object type and meaning corresponding to each key's value. In short, please make sure the input and output requirements are very clear and unambiguous.
    For example, if the variable is a string, you need to specify what format the string should be presented (like what is the separator to link multiple items in the string), open-ended string inputs and outputs are not allowed.
  
3. Input Generator:
  - You need to provide a function named `input_generator` that generates the input arguments for the `main_solution` function.
  - The `input_generator` function should not require any input arguments, and each time it is called, it should return a set of input arguments that meet the requirements of the `main_solution` function.
  - The output of `input_generator` should always be a dictionary because we always call by `**kwargs` in the `main_solution` function.
  - Add some randomness in the `input_generator` function to ensure the input arguments are different each time it is called.
  - Please try to make the generated input arguments as reasonable as possible, try to avoid generating too complex or too trivial input variables, also the size of the variables should be reasonable, like less than 1KB.
  
4. Problem Statement:
  - Based on the `main_solution` function, you need to create a problem that is related to the provided code.
  - Please avoid writing contents such as "implement a function", "write a function" or "implement a system" in the problem, but instead, describe the background and requirements to present a non-programming problem and you must have a wh-question in your problem.
  - You should clearly denote the input variable names (but not an exact or specific value) in your problem statement, and clearly ask for the returned value, to be consistent with the `main_solution` function.
  - You do not need to include again the input and output variable requirements or any examples in this part.

---------

Your final output should be like this:
## Main Function
```python
# import necessary packages
import ...
from ...

# all class and function definitions in the code file, if any
# they will be used in the main_solution function, you may need to modify them to meet the requirements of the main_solution function (optional)
class ...
def ...

# main function
def main_solution(arg1, arg2, ...):
  # all input arguments of the main_solution function should be json serializable (no self-defined objects, functions, np.array, set, tuple, etc.)
  # if you need to invoke functions that require non-json serializable inputs, like those defined in the reference code file, you need to convert them (optional)
  ...
  # return, the returned value must be json serializable (no self-defined objects, functions, np.array, set, tuple, etc.)
  return ...
```
## Input Output Description
Input:
  `arg1` (type): description
  `arg2` (type): description
  ...
Output:
  `return` (type): description
## Input Generator
```python
def input_generator():
  # generate input arguments for the main_solution function
  ...
  return {"arg1": ..., "arg2": ..., ...}
```
## Problem Statement
... (with a wh-question, the input variables names should be in the questions) ...

---------

Here is the code file you need to process:
[Code Start]
<<<<code>>>>
[Code End]"""

# 在函数外部，预编译正则表达式
exception_pattern = re.compile(
    r"Traceback \(most recent call last\):\s*(?:.*\n)+([a-zA-Z_][a-zA-Z0-9_]*):\s*(.+)",
    re.MULTILINE
)

template_check_input = """{solution_prefix}

{refcode}

def is_close(pred, target, tol=0.001):
    if isinstance(pred, dict) and isinstance(target, dict):
        if pred.keys() != target.keys():
            return False
        return all(is_close(pred[k], target[k], tol) for k in pred)

    elif isinstance(pred, list) and isinstance(target, list):
        if len(pred) != len(target):
            return False
        return all(is_close(p, t, tol) for p, t in zip(pred, target))

    elif isinstance(pred, (int, float)) and isinstance(target, (int, float)):
        if isinstance(pred, float) or isinstance(target, float):
            # if we have non number, like nan, inf, we should not compare them
            if math.isnan(pred) or math.isnan(target) or math.isinf(pred) or math.isinf(target):
                return False
            return (abs(pred - target) <= tol * abs(target)) and (int(pred) == int(target))
        return pred == target
        
    else:
        return pred == target

def diy_check_input_output():
    iiiioooo = {io}

    input_xx = iiiioooo['input']  # should be a json object
    output_xx = iiiioooo['output']  # should be a json object

    warning_string = "[Mismatch] Your input is not feasible! Given the output <<<<3>>>>, your predicted input is <<<<1>>>>, which actually gets a wrong output as <<<<2>>>>"

    string_iii = json.dumps(input_xx)
    string_ooo = json.dumps(output_xx).strip()

    execed_output = None

    if not {bypass}:
        if isinstance(input_xx, dict):
            execed_output = {funcname}(**input_xx)
        else:
            execed_output = {funcname}(input_xx)
    else:
        execed_output = {funcname}(input_xx)

    string_eee = json.dumps(execed_output).strip()

    cond1 = string_ooo == string_eee
    cond2 = is_close(execed_output, output_xx)
    
    assert cond1 or cond2, warning_string.replace(
        "<<<<1>>>>", string_iii).replace("<<<<2>>>>", string_eee).replace("<<<<3>>>>", string_ooo)

diy_check_input_output()
"""

def sub_extract_last_complete_json(s):
    if '```json' not in s:
        # Stack to keep track of opening and closing braces
        stack = []
        last_json_start = None
        last_json_str = None
        
        for i, char in enumerate(s):
            if char == '{':
                stack.append(i)
                if last_json_start is None:
                    last_json_start = i
            elif char == '}':
                if stack:
                    start = stack.pop()
                    if not stack:
                        # Complete JSON object found
                        last_json_str = s[last_json_start:i+1]
                        last_json_start = None
    else:
        # find the last ```json
        last_json_start = s.rfind('```json')
        last_json_end = s.find('```', last_json_start+len('```json'))
        last_json_str = s[last_json_start+7:last_json_end].strip()

    # Load the last JSON object

    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            # replace 'False', 'True' to 'false', 'true'
            last_json_str = last_json_str.replace("False", "false").replace("True", "true").replace("None", "null")
            try:
                return json.loads(last_json_str.replace("\n", ""))
            except json.JSONDecodeError:
                pass
    return None

def extract_last_complete_json(s):
    res = sub_extract_last_complete_json(s)
    if res is None:
        s = s.replace("\{","{").replace("\}","}").replace('(','[').replace(')',']')
        res = sub_extract_last_complete_json(s)
    if res is None and "\\boxed{" in s:
        boxstart = s.rfind("\\boxed{")+len("\\boxed{")
        boxend = s.rfind("}",boxstart)
        boxcontent = s[boxstart:boxend]
        processed_box_content = boxcontent.replace("\\\\","\\").replace("\\{","{").replace("\\}","}").replace('\\left','').replace('\\right','')
        res = sub_extract_last_complete_json(processed_box_content)
    return res

def strict_check_size(obj):
    # Check if object size is less than 1024 bytes
    if asizeof.asizeof(obj) >= 1024: 
        return False

    # Check for dict type
    if isinstance(obj, dict):
        if len(obj) >= 20:  # Check dict has fewer than 20 key-value pairs
            return False
        # Recursively check keys and values
        for k, v in obj.items():
            if not strict_check_size(k) or not strict_check_size(v):
                return False

    # Check for list, tuple, or set
    elif isinstance(obj, (list, tuple, set)):
        if len(obj) >= 20:  # Check if the length is less than 20
            return False
        # Recursively check each element
        for item in obj:
            if not strict_check_size(item):
                return False

    # Check for string
    elif isinstance(obj, str):
        if len(obj) >= 100:  # Check if string length is less than 100 characters
            return False

    # elif isinstance(obj, float):
    #     d = Decimal(str(obj))
    #     if d.as_tuple().exponent < -3:
    #         return False

    # Other objects - check size in bytes
    else:
        if asizeof.asizeof(obj) >= 128:  # Check if object size is less than 128 bytes
            return False

    # If all checks are passed, return True
    return True

def combine(mainbody,funcname, args, output_file="output.json"):
    return solution_prefix+'\n\n\n'+mainbody+'\n\n\n'+exec_part.replace('<<function_name>>',funcname).replace('<<args>>',args).replace('<<output_file>>',"\""+output_file+"\"")

def get_output(mainbody, funcname, args, debug=False):
    uid = shortuuid.uuid()
    pyfilename = "./temp/solutions/solution."+uid+".py"
    outputfilename = "./temp/solutions/output."+uid+".json"

    if not os.path.exists(pyfilename):
        os.makedirs(os.path.dirname(pyfilename), exist_ok=True)

    solution_py = combine(mainbody,funcname, args, outputfilename)

    if debug:
        print('=================')
        print(solution_py)
        print('=================')

    # start a commend to run the code
    # write it into a file

    with open(pyfilename, 'w') as f:
        f.write(solution_py)
    
    # run the code
    # if error, raise ValueError
    subprocess.run(["python3", pyfilename], check=True)

    # read the output.json
    with open(outputfilename, 'r') as f:
        output = f.read()
    return output

def extract_last_python(text):
    posstart = text.rfind("```python")
    if posstart == -1:
        return None
    posstart+=len("```python")
    posend = text.find("```", posstart)
    if posend == -1:
        return None
    return text[posstart:posend].strip()

def extract_input(ss):
    last_json = extract_last_complete_json(ss)
    if last_json is None:
        return None
    if isinstance(last_json,dict):
        inputx = last_json.get('input',None)
    else:
        inputx = last_json
    return inputx

def check_input_legacy(refcode, io, funcname, 
                solution_prefix="", 
                used_python_path="x", 
                run_path = "x",
                runtime_limit=5,
                bypass=False,
                ):
    strbypass = "True" if bypass else "False"

    runnablepycode = template_check_input.format(
        solution_prefix=solution_prefix, refcode=refcode, io=io, funcname=funcname, bypass=strbypass
    )

    result_dict = {}

    try:
        # Run the code with a timeout of 5 seconds
        result = subprocess.run(
            [used_python_path, '-'],
            input=runnablepycode,
            stdout=subprocess.DEVNULL,     # Discard standard output
            stderr=subprocess.PIPE,        # Capture standard error
            text=True,
            timeout=runtime_limit,
            cwd = run_path
        )
        if result.returncode == 0:
            # Success
            result_dict['status'] = 'success'
            result_dict['message'] = 'Feasible input!'
        else:
            # Error occurred
            stderr = result.stderr

            # Attempt to extract the specific exception type and message
            exception_type = None
            exception_message = None

            # Pattern to match Python traceback exceptions
            match = exception_pattern.search(stderr)

            if match:
                # Extract exception type and message
                exception_type = match.group(1)
                exception_message = match.group(2).strip()

                # Special handling for AssertionError
                if exception_type == 'AssertionError':
                    result_dict['status'] = 'AssertionError'
                    result_dict['message'] = exception_message or 'An assertion error occurred.'
                else:
                    result_dict['status'] = 'exception'
                    result_dict['exception_type'] = exception_type
                    result_dict['message'] = exception_message
            else:
                # If pattern matching fails, return the last line as the error message
                lines = stderr.strip().splitlines()
                if lines:
                    last_line = lines[-1]
                    result_dict['status'] = 'exception'
                    result_dict['message'] = last_line.strip()
                else:
                    result_dict['status'] = 'error'
                    result_dict['message'] = 'An unknown error occurred.'
    except subprocess.TimeoutExpired:
        # Timeout
        result_dict['status'] = 'timeout'
        result_dict['message'] = f'Code execution time exceeded the limit {runtime_limit} seconds, may encounter infinite loop.'
    except Exception as e:
        # Other exceptions
        result_dict['status'] = 'exception'
        result_dict['message'] = str(e)
    finally:
        pass

    return result_dict

def check_input(refcode, io, funcname, 
                solution_prefix="", 
                used_python_path="x", 
                run_path="x",
                runtime_limit=5,
                bypass=False,
                ):


    # Define the exception pattern if not already defined
    exception_pattern = re.compile(r'Traceback \(most recent call last\):.*\n([\w\.]+):\s+(.*)', re.DOTALL)

    strbypass = "True" if bypass else "False"

    runnablepycode = template_check_input.format(
        solution_prefix=solution_prefix, refcode=refcode, io=io, funcname=funcname, bypass=strbypass
    )

    result_dict = {}

    # Cross-platform process creation and termination functions
    # if sys.platform == 'win32':
    #     # Windows
    #     def start_process(*args, **kwargs):
    #         return subprocess.Popen(
    #             *args,
    #             **kwargs,
    #             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    #         )
    #     def kill_process(process):
    #         try:
    #             process.send_signal(signal.CTRL_BREAK_EVENT)
    #         except Exception:
    #             process.kill()
    # else:

    # Unix/Linux
    def start_process(*args, **kwargs):
        return subprocess.Popen(
            *args,
            **kwargs,
            preexec_fn=os.setsid
        )
    def kill_process(process):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            process.kill()

    process = None

    try:
        # Start the process
        process = start_process(
            [used_python_path, '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=run_path,
        )

        try:
            # Communicate with the process
            stdout_data, stderr_data = process.communicate(
                input=runnablepycode,
                timeout=runtime_limit
            )
        except subprocess.TimeoutExpired:
            # Timeout occurred, kill the process group
            kill_process(process)
            stdout_data, stderr_data = process.communicate()
            result_dict['status'] = 'timeout'
            result_dict['message'] = f'Code execution time exceeded the limit of {runtime_limit} seconds; may have encountered an infinite loop.'
            return result_dict
        except Exception as e:
            # Kill the process group
            kill_process(process)
            stdout_data, stderr_data = process.communicate()
            result_dict['status'] = 'exception'
            result_dict['message'] = str(e)
            return result_dict
        else:
            # Process completed
            returncode = process.returncode
            if returncode == 0:
                # Success
                result_dict['status'] = 'success'
                result_dict['message'] = 'Feasible input!'
            else:
                # Error occurred
                stderr = stderr_data

                # Attempt to extract the specific exception type and message
                exception_type = None
                exception_message = None

                # Pattern to match Python traceback exceptions
                match = exception_pattern.search(stderr)

                if match:
                    # Extract exception type and message
                    exception_type = match.group(1)
                    exception_message = match.group(2).strip()

                    # Special handling for AssertionError
                    if exception_type == 'AssertionError':
                        result_dict['status'] = 'AssertionError'
                        result_dict['message'] = exception_message or 'An assertion error occurred.'
                    else:
                        result_dict['status'] = 'exception'
                        result_dict['exception_type'] = exception_type
                        result_dict['message'] = exception_message
                else:
                    # If pattern matching fails, return the last line as the error message
                    lines = stderr.strip().splitlines()
                    if lines:
                        last_line = lines[-1]
                        result_dict['status'] = 'exception'
                        result_dict['message'] = last_line.strip()
                    else:
                        result_dict['status'] = 'error'
                        result_dict['message'] = 'An unknown error occurred.'
    finally:
        # Ensure that the process is terminated and resources are cleaned up
        if process is not None:
            try:
                kill_process(process)
            except Exception:
                pass
            # Wait for the process to terminate to prevent zombies
            process.wait()
            # Close any open file descriptors
            process.stdout.close() if process.stdout else None
            process.stderr.close() if process.stderr else None
            process.stdin.close() if process.stdin else None

    return result_dict

def is_close(pred, target, tol=0.001):
    if isinstance(pred, dict) and isinstance(target, dict):
        if pred.keys() != target.keys():
            return False
        return all(is_close(pred[k], target[k], tol) for k in pred)

    elif isinstance(pred, list) and isinstance(target, list):
        if len(pred) != len(target):
            return False
        return all(is_close(p, t, tol) for p, t in zip(pred, target))

    elif isinstance(pred, (int, float)) and isinstance(target, (int, float)):
        #try:
        if isinstance(pred, float) or isinstance(target, float):
            # if we have non number, like nan, inf, we should not compare them
            if math.isnan(pred) or math.isnan(target) or math.isinf(pred) or math.isinf(target):
                return False
            return (abs(pred - target) <= tol * abs(target)) and (int(pred) == int(target))
        return pred == target
        # except:
        #     return False
    else:
        return pred == target