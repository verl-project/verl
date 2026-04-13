def compute_score_router(data_source, solution_str, ground_truth, extra_info=None):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from verl.utils.reward_score import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["math", "math-500", "numinamath"]:
        from verl.utils.reward_score import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["aqua_rat", "ai2_arc"]:
        from verl.utils.reward_score import multiple_choice
        res = multiple_choice.compute_score(solution_str, ground_truth)
    elif data_source in ["strategyQA"]:
        from verl.utils.reward_score import truefalse
        res = truefalse.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
