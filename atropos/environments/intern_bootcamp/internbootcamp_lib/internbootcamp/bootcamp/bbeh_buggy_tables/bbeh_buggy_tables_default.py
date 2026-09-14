import json
import os
from typing import Dict, Any, List
from internbootcamp.bootcamp.base import Basebootcamp
from internbootcamp.libs.bbeh_buggy_tables.bbeh_buggy_tables_generator import BBEHBuggyTablesGenerator
from internbootcamp.libs.bbeh_buggy_tables.bbeh_buggy_tables_solver import BBEHBuggyTablesSolver
from internbootcamp.libs.bbeh_buggy_tables.bbeh_buggy_tables_validor import BBEHBuggyTablesValidator


class BBEHBuggyTablesbootcamp(Basebootcamp):
    # 类级别的属性
    _generator = None
    _solver = None
    _validator = None
    _task_file_path = f'{os.path.dirname(os.path.abspath(__file__))}/../../libs/bbeh_buggy_tables/task.json'
    _generator = BBEHBuggyTablesGenerator(_task_file_path)
    _solver = BBEHBuggyTablesSolver()
    _validator = BBEHBuggyTablesValidator()

    # 实例级别的引用
    generator = _generator
    solver = _solver
    validator = _validator

    @classmethod
    def case_generator(cls) -> Dict[str, Any]:
        """生成一个新的BBEH Buggy Tables示例"""
        if cls._generator is None:
            raise RuntimeError("Generator not initialized. Create an instance of BBEHBuggyTablesbootcamp first.")
        return cls._generator.generate_example()

    @staticmethod
    def extract_output(output: str) -> float:
        """从模型输出中提取答案"""
        try:
            numbers = [float(s) for s in output.split() if s.replace('.', '').isdigit()]
            return numbers[-1] if numbers else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, answer: float, identity: Dict[str, Any]) -> bool:
        """验证答案是否正确"""
        if cls._solver is None:
            raise RuntimeError("Solver not initialized. Create an instance of BBEHBuggyTablesbootcamp first.")
        expected_result = cls._solver.solve(identity)
        return abs(answer - expected_result) < 1e-6 if answer is not None and expected_result is not None else False

    @staticmethod
    def prompt_func(identity: Dict[str, Any]) -> str:
        """生成提示语"""
        table_str = json.dumps(identity['input']['table'], indent=2)
        prompt = f"""你是一个擅长处理有bug的表格数据的助手。请解决以下BBEH Buggy Tables问题:

表格数据:
{table_str}

Bug描述: {identity['input']['bug_description']}

查询: {identity['input']['query']}

请根据给定的信息修复表格,并执行查询。给出最终的数值结果,保留两位小数。

请按以下格式输出你的答案:
最终答案: [你的数值结果]
"""
        return prompt


if __name__ == '__main__':
    try:
               
        # 创建实例以初始化类级别的属性
        bootcamp = BBEHBuggyTablesbootcamp()

        # 使用类方法生成示例
        identity = BBEHBuggyTablesbootcamp.case_generator()

        if identity is None:
            print("Error: Failed to generate example")
            exit(1)

        # 使用静态方法生成提示
        print(BBEHBuggyTablesbootcamp.prompt_func(identity))

        # 使用实例方法求解
        correct_solution = bootcamp.solver.solve(identity)

        fake_output = f"""经过分析和计算,
        最终答案: {correct_solution}
        """
        print(fake_output)

        # 使用静态方法提取答案
        extracted_answer = BBEHBuggyTablesbootcamp.extract_output(fake_output)
        print("Extracted answer:", extracted_answer)

        # 使用类方法验证
        print("Is it correct?", BBEHBuggyTablesbootcamp._verify_correction(extracted_answer, identity))

    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback

        traceback.print_exc()
