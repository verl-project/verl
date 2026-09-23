from internbootcamp.bootcamp.base import Basebootcamp
import traceback
import re
import json
from .codeio_utils import *
import os
from datasets import load_dataset

python_path = "python"
run_path = "./python_tmp"


def evaluate_codeio(completion, reference_answer):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    
    # print("completion",completion)
    # print("******************")
    # print("reference_answer",reference_answer)
    # print("******************")
    if not  os.path.exists(run_path):
        #print(f"文件夹 '{run_path}' 不存在，正在创建...")
        try:
            os.makedirs(run_path)
            #print(f"文件夹 '{run_path}' 已成功创建！")
        except Exception as e:
            print(f"创建文件夹时发生错误: {str(e)}")
    assert type(reference_answer) == str
    
    reference_answer = json.loads(reference_answer)
    
    last_json = extract_last_complete_json(completion)
    if last_json is None:
        format_correctness = False
        return False , format_correctness

    if reference_answer['io_pred'] == "output":
        if not isinstance(last_json, dict):
            return False, False
        if "output" not in last_json:
            return False, False
        pred_output = last_json["output"]
        # print("pred_output",pred_output)
        # print("reference_answer[output]",reference_answer["output"])
        acc = is_close(pred_output, reference_answer["output"])
        if acc:
            return True, True
        else:
            return False, True
    elif  reference_answer['io_pred'] == "input":
        if not isinstance(last_json, dict):
            return False, False
        if "input" not in last_json:
            return False, False
        pred_input = last_json["input"]
        
        candio = {'input': pred_input, 'output': reference_answer['output']}
        res = check_input(reference_answer['refcode'], candio, reference_answer['funcname'], solution_prefix=solution_prefix, runtime_limit=5, used_python_path = python_path, run_path=run_path)
        if res['status'] == 'success':
            return True, True
        else:
            return False, True



def compute_score(solution_str, ground_truth, method='strict'):
    """
    Scoring function for instruction following task
    """

    reference_answer = ground_truth['gt']

    answer = solution_str
    
    if answer is None:
        correctness = False
        return correctness

    correctness , _ = evaluate_codeio(answer, reference_answer=reference_answer)
    
    assert isinstance(correctness, bool), f"correctness is not bool, is {type(correctness)}"

    return correctness


class CodeIObootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)


    # no case generator for instruction following
    def case_generator(self):
        pass
    
    @staticmethod
    def prompt_func(case) -> str:
        return case['prompt']
    
    @staticmethod
    def extract_output(output):
        return output
    
    @classmethod
    def _verify_correction(cls, solution, identity) -> bool:
        correctness = compute_score(solution_str=solution, ground_truth=identity)
        return float(correctness)