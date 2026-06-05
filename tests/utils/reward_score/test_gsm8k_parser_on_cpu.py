from verl.utils.reward_score import gsm8k


def test_flexible_parser_strips_sentence_period_from_integer_answer():
    assert gsm8k.extract_solution("The answer is 308.", method="flexible") == "308"
    assert gsm8k.compute_score("The answer is 308.", "308", method="flexible") == 1.0


def test_flexible_parser_preserves_decimal_answers():
    assert gsm8k.extract_solution("The answer is 3.14", method="flexible") == "3.14"
    assert gsm8k.compute_score("The answer is 3.14", "3.14", method="flexible") == 1.0


def test_flexible_parser_treats_equivalent_numeric_surfaces_as_correct():
    assert gsm8k.extract_solution("The answer is 15.00", method="flexible") == "15.00"
    assert gsm8k.compute_score("The answer is 15.00", "15", method="flexible") == 1.0


def test_flexible_parser_prefers_boxed_numeric_answer():
    completion = "We compute everything carefully. Final result is \\boxed{10}. This took 50 minutes."
    assert gsm8k.extract_solution(completion, method="flexible") == "10"
    assert gsm8k.compute_score(completion, "10", method="flexible") == 1.0


def test_flexible_parser_falls_back_to_last_number_without_box():
    completion = "First estimate 4, then revise. The answer is 12."
    assert gsm8k.extract_solution(completion, method="flexible") == "12"
    assert gsm8k.compute_score(completion, "12", method="flexible") == 1.0


def test_strict_parser_normalizes_trailing_period():
    assert gsm8k.extract_solution("#### 308.", method="strict") == "308"
    assert gsm8k.compute_score("#### 308.", "308", method="strict") == 1.0
