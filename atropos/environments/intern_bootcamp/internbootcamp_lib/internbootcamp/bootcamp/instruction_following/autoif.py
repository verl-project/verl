from internbootcamp.bootcamp.base import Basebootcamp

import traceback
import re
import json
import langdetect
from datasets import load_dataset

def run_code(code_str, response):
    code_str += "\nresult = evaluate(response)"
    safe_globals = {"print": print, "len": len, "range": range, "re": re, "json": json, "langdetect": langdetect}
    safe_locals = {"response": response}
    try:
        exec(code_str, safe_globals, safe_locals)
        result = safe_locals.get("result")
        return result
    except Exception as e:
        error_details = traceback.format_exc()
        print("Execution Error:\n", error_details)
        return False

def compute_score(solution_str, ground_truth, method='strict'):
    """
    Scoring function for instruction following task
    """

    eval_code = ground_truth['eval_code']

    answer = solution_str
    
    if answer is None:
        correctness = False
        return correctness

    correctness = run_code(code_str=eval_code, response=answer)
    
    assert isinstance(correctness, bool), f"correctness is not bool, is {type(correctness)}"

    return correctness


class AutoIFbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        
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