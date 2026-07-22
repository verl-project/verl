# Copyright (c) InternLM. All rights reserved.
import random
import re
import time
from typing import List, Optional, Tuple

import requests

from .base_judger import BaseJudger, JudgeStatus, MessageItem, Reward, register_judger
from .utils import extract_answer, math_equal


@register_judger("math_judger")
class MathJudger(BaseJudger):
    verify_prompt = """You are a helpful assistant who evaluates the correctness and quality of models' outputs.
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    Just return the letters \"A\" or \"B\", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>:
    {question}
    <Original Question End>


    <Gold Target Begin>:
    {gold_answer}
    <Gold Target End>


    <Predicted Answer Begin>:
    {answer}
    <Predicted End>


    Judging the correctness of candidates' answers:"""

    def __init__(
        self,
        hosts: List[str],
        max_retries: int = 1,
        retry_delay: float = 1.0,
        stop_word="<|im_end|>",
        thinking_finish_words=["<conclude>", "**Final Answer**", "</think>"],
    ):
        super().__init__()
        self.hosts = hosts
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stop_word = stop_word
        self.thinking_finish_words = thinking_finish_words

        self.host_ip_idx = random.randint(0, len(hosts) - 1)
        self.model_name = requests.get(
            f"http://{self.hosts[self.host_ip_idx]}/v1/models",
            headers={"Authorization": "Bearer "},
        ).json()["data"][0]["id"]

    def on_data_received(
        self,
        prompt_messages: List[MessageItem],
        completion_messages: List[MessageItem],
        metadata: dict,
    ) -> JudgeStatus:
        question = prompt_messages[-1]["content"]
        response = completion_messages[-1]["content"]
        question_type = metadata.get("question_type", None)
        gold_answer = metadata["gold_answer"]
        if not response.strip().endswith(self.stop_word):
            # If the response does not end with the stop word, it is not a complete response, treat as incorrect
            return JudgeStatus(
                ok=True,
                handle={
                    "question": question,
                    "question_type": question_type,
                    "response": response,
                    "gold_answer": gold_answer,
                    "verify_label": False,
                },
            )

        for thinking_finish_word in self.thinking_finish_words:
            if thinking_finish_word in response:
                response = response.split(thinking_finish_word)[-1]

        response = response.replace(self.stop_word, "")

        # first try to extract and verify with rule, if correct, return
        extracted_answer, verify_label = self._extract_and_verify_with_logic(
            response, gold_answer
        )
        if verify_label is True:
            return JudgeStatus(
                ok=True,
                handle={
                    "question": question,
                    "question_type": question_type,
                    "response": response,
                    "gold_answer": gold_answer,
                    "verify_label": verify_label,
                },
            )

        # then try to evaluate with model
        res_string, verify_label = self._evaluate_answer_with_llm(
            question, question_type, response, gold_answer
        )
        return JudgeStatus(
            ok=True,
            handle={
                "question": question,
                "question_type": question_type,
                "response": response,
                "gold_answer": gold_answer,
                "verify_label": verify_label,
            },
        )

    def on_reward_required(
        self, status: JudgeStatus, timeout: Optional[float] = None
    ) -> Reward:
        if status.handle is None:
            return None
        if status.handle["verify_label"] is not None:
            return 1.0 if status.handle["verify_label"] else -1.0
        return None

    def _evaluate_answer_with_llm(
        self, question: str, question_type: str, answer: str, gold_answer: str
    ) -> Tuple[str, bool]:
        for i in range(self.max_retries):
            host = self.hosts[self.host_ip_idx]
            self.host_ip_idx = (self.host_ip_idx + 1) % len(self.hosts)
            prompt = self.verify_prompt.format(
                "", "", question=question, answer=answer, gold_answer=gold_answer
            )
            try:
                res = requests.post(
                    f"http://{host}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        "temperature": 0.0,
                        "top_p": 0.8,
                        "top_k": 20,
                        "repetition_penalty": 1.05,
                        "max_tokens": 100,
                        "stop": ["<|im_end|>", "<|endoftext|>"],
                    },
                )
                res_string = res.json()["choices"][0]["message"]["content"]
                print(f"Evaluate result: {res_string}")
                verify_label = self._verify_from_string(res_string)
                if verify_label is None:
                    raise ValueError(
                        f"Evaluate result is None, judger prediction: {res_string}"
                    )
                return res_string, verify_label

            except Exception as e:
                print(f"Error verifying answer: {e}")
                time.sleep(self.retry_delay)
                continue
        print(f"Failed to verify answer after {self.max_retries} retries.")
        return None, None

    def _verify_from_string(self, verification: str):
        if "A" in verification and "B" not in verification:
            label = True
        elif "B" in verification and "A" not in verification:
            label = False
        else:  # judger model failed to predict A or B
            label = None
        return label

    def _extract_and_verify_with_logic(
        self, response: str, gold_answer: str
    ) -> Tuple[str, bool]:
        extracted_answer = extract_answer(response)
        verify_label = math_equal(extracted_answer, gold_answer)
        return extracted_answer, verify_label
