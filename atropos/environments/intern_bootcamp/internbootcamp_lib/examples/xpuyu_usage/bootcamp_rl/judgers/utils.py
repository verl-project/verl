# flake8: noqa
# isort: skip_file

import multiprocessing
import re
from math import isclose
from typing import Optional, Union
from collections import defaultdict, Counter

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def extract_answer(pred_str: str, execute: bool = False) -> str:
    if re.search("\\boxed|boxed|\\box|box", pred_str):
        answer = re.split("\\boxed|boxed|\\box|box", pred_str)[-1]
        if len(answer) == 0:
            return ""
        elif answer[0] == "{":
            stack = 1
            a = ""
            for c in answer[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = answer.split("$")[0].strip()
    elif re.search("[Tt]he (final )?answer is:?", pred_str):
        a = re.split("[Tt]he (final )?answer is:?", pred_str)[-1].strip().rstrip(".")
    else:  # use the last number
        pred = re.findall(r"-?\d*\.?\d+", pred_str.replace(",", ""))
        if len(pred) >= 1:
            a = pred[-1]
        else:
            a = ""
    choice = re.findall(r"([A-E]):\s*(.*)", a)
    if len(choice) > 0:
        for option, content in choice:
            a = option
    choice = re.findall(r"\(([A-E])\)\s*(.*)", a)
    if len(choice) > 0:
        for option, content in choice:
            a = option

    a = re.split(r"=|\\approx|≈", a)[-1]

    # multiple lines
    answer = ""
    preds = re.split("\n", a)
    for pred in preds:
        if "\\begin{align" in pred or pred.endswith(":"):
            continue
        if pred != "" and pred[0] == ":":
            pred = pred[1:]
        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]
        if pred != "" and pred[-1] == "/":
            pred = pred[:-1]
        pred = strip_string(pred)
        pred = re.sub(r"^[a-zA-Z0-9]+[\)]\s*", "", pred)
        for p in pred.split("{}"):
            if p != "":
                pred = p
                break

        pred = re.sub(r"^\{([A-Z])\}|\(([A-Z])\)", r"\1\2", pred)
        if pred != "":
            answer = pred
            break
    return answer


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def extract_answer(pred_str: str, execute: bool = False) -> str:
    if re.search("\boxed|boxed", pred_str):
        answer = re.split("\boxed|boxed", pred_str)[-1]
        if len(answer) == 0:
            return ""
        elif answer[0] == "{":
            stack = 1
            a = ""
            for c in answer[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = answer.split("$")[0].strip()
    elif re.search("[Tt]he (final )?answer is:?", pred_str):
        a = re.split("[Tt]he (final )?answer is:?", pred_str)[-1].strip().rstrip(".")
    elif pred_str.startswith("```python") and execute:
        # fall back to program
        from lagent import get_tool

        a = get_tool("IPythonInteractive").exec(pred_str).value or ""
    else:  # use the last number
        pred = re.findall(r"-?\d*\.?\d+", pred_str.replace(",", ""))
        if len(pred) >= 1:
            a = pred[-1]
        else:
            a = ""
    # multiple lines
    pred = a.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred


def is_digit(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    tolerance: float = 1e-4,
    timeout: bool = False,
) -> bool:
    """Exact match of math if and only if:

    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(",", ""))
            reference = float(str(reference).replace(",", ""))
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=tolerance):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        (prediction.startswith("[") and prediction.endswith("]"))
        and (reference.startswith("[") and reference.endswith("]"))
        or (prediction.startswith("(") and prediction.endswith(")"))
        and (reference.startswith("(") and reference.endswith(")"))
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def math_equal_process(param):
    if param[-2] is None:
        return False
    return math_equal(param[-2], param[-1])


def symbolic_equal(a, b):

    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except Exception:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def math_majority_vote(answers: list, majority: Optional[int] = None):
    # threshold = len(answers) // 2 + 1
    ans2cnt, ans2idx = Counter(), defaultdict(list)
    for i, ans in enumerate(answers):
        if isinstance(ans, str) and ans.strip():
            for key in ans2cnt.keys():
                if math_equal(ans, key):
                    ans2cnt[key] += 1
                    ans2idx[key].append(i)
                    break
            else:
                ans2cnt[ans] += 1
                ans2idx[ans].append(i)
    if ans2cnt:
        maj, cnt = ans2cnt.most_common(1)[0]
        if maj and cnt >= (majority or 1):
            return maj, ans2idx[maj]
    return None, []
