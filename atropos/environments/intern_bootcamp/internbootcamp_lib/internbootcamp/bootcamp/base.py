import re
import json


class Basebootcamp:
    """
    Base class for bootcamp implementations.
    
    A bootcamp is a class that contains the logic to verify the solution of a task.
    """
    
    @staticmethod
    def prompt_func(question_ori) -> str:
        """
        Process the input_data and return the processed prompt.
        
        Args:
            question_ori: The question to be processed.
        
        Returns:
            str: The processed prompt.
        """
        raise NotImplementedError
    
    @staticmethod
    def extract_output(output):
        """
        Extract the output from the solution.
        
        Args:
            output: Model output to be processed.
        
        Returns:
            The processed output.
        """
        raise NotImplementedError
    
    
    @classmethod
    def _verify_correction(cls,solution,identity)->bool:
        """
        Verify the correction of the solution.
        """
        raise NotImplementedError
    
    

    @classmethod
    def verify_score(cls, model_output, identity: dict, format_score=0, short_penalty=True, short_threshold=100, format_penalty=True) -> float:
        """
        Verify the output against the ground truth.
        
        Args:
            output: The model output to be verified.
            identity: Some rules or parameters to be used in the verification.
            format_score: Whether to give a score for the format of the output.
        
        Returns:
            float: The score of the output.
        """
        score = 0. 
        if short_penalty and len(model_output) < short_threshold:
            # if the output is too short, consider it incorrect
            return score
        if format_penalty and "</think>" not in model_output:
            return score
        try:
            extract_solution = cls.extract_output(model_output)
            if extract_solution is None:
                return score
            else:
                score = format_score # 必须在这里就给format_score 赋值！否则后面verify_correction如果报错，format_score就没有赋值
            judge =  cls._verify_correction(extract_solution, identity)
            if type(judge) == bool and judge:
                score = 1.
            else:
                assert type(judge) == float or type(judge) == int
                score = float(judge)
                
        except Exception as e:
            # print("Error in verify_score:", e)
            pass
        return score




# class BaseV2bootcamp(Basebootcamp):
#     @classmethod
#     def verify_score(cls, model_output, identity: str, format_score=0.1, short_penalty=True, short_threshold=100) -> float:
#         """
#         Verify the output against the ground truth.
        
#             float: The score of the output.
#         """
#         score = 0. 
#         if short_penalty and len(model_output) < short_threshold:
#             # if the output is too short, consider it incorrect
#             return score
#         identity = json.loads(identity)
#         try:
#             extract_solution = cls.extract_output(model_output)
#         except Exception as e:
#             # print("Error in verify_score:", e)
#             pass
#         return score
#         return score
