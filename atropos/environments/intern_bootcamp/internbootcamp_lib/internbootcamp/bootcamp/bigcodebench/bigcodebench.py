from internbootcamp.bootcamp.base import Basebootcamp
import re
import tempfile
import subprocess
import os
from datasets import load_dataset

class BigCodeBenchbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)

    def case_generator(self):
        pass
    
    @staticmethod
    def prompt_func(case) -> str:
        def add_python_marker(text: str) -> str:
            marker = "```"
            insert_position = text.find(marker)
            
            if insert_position == -1:
                return text  # 如果没有 ```，则返回原文本
            
            return text[:insert_position + len(marker)] + "python " + text[insert_position + len(marker):]
        return add_python_marker(case['instruct_prompt'])
    
    @staticmethod
    def extract_output(output):
        match = re.search(r'```python(.*?)```', output, re.DOTALL)
        if match:
            solution = match.group(1).strip()
            return solution
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity) -> bool:
        test_code = identity['test']
        timeout = 30
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. 生成 task_func 代码文件
            task_file = os.path.join(temp_dir, "solution.py")
            with open(task_file, "w", encoding="utf-8") as f:
                f.write(solution)
            
            # 2. 生成 unittest 测试文件
            # test_code = "from task import task_func\n" + test_code
            test_file = os.path.join(temp_dir, "test_task.py")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(test_code)
            
            # 3. 运行 unittest
            process = subprocess.Popen(
                ["python", "-m", "unittest", "test_task.py"],
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return 0  # 进程超时，返回 0 分
            process.stdout.close()  # 关闭管道，释放资源
            process.stderr.close()
            # 4. 解析测试结果
            match_ran = re.search(r'Ran (\d+) tests?', stderr)
            match_failed = re.search(r'failures=(\d+)', stderr)
            match_error = re.search(r'errors=(\d+)', stderr)
            
            total_tests = int(match_ran.group(1)) if match_ran else 0
            failed_tests = int(match_failed.group(1)) if match_failed else 0
            error_tests = int(match_error.group(1)) if match_error else 0
            passed_tests = total_tests - failed_tests - error_tests
            pass_rate = float(passed_tests) / float(total_tests) if total_tests > 0 else 0.0
        
        return pass_rate