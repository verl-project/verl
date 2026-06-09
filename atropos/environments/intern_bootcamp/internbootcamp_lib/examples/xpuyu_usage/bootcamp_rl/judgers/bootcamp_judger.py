# Copyright (c) InternLM. All rights reserved.
import random
import re
import time
from typing import List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

import requests
import internbootcamp

from .base_judger import BaseJudger, JudgeStatus, MessageItem, Reward, register_judger



@register_judger("bootcamp_judger")
class bootcampJudger(BaseJudger):
    def __init__(
        self,
        stop_word="<|im_end|>",
        format_score=0,
        format_penalty=True,
        short_penalty=True,
        short_threshold=128,

    ):
        super().__init__()
        self.stop_word = stop_word
        self.format_score = format_score
        self.format_penalty = format_penalty
        self.short_penalty = short_penalty
        self.short_threshold = short_threshold
        
    def on_data_received(
        self,
        prompt_messages: List[MessageItem],
        completion_messages: List[MessageItem],
        metadata: dict,  # 存在数据集对应的字段里面，想存啥都可以，自己解析出来就行
    ) -> JudgeStatus:
        question = prompt_messages[-1]["content"]
        response = completion_messages[-1]["content"]
        identity = metadata["ground_truth"]
        data_source = metadata["data_source"]
        verify_label = None
        if not response.strip().endswith(self.stop_word):
            # If the response does not end with the stop word, it is not a complete response, treat as incorrect
            verify_label = False
        return JudgeStatus(
            ok=True,
            handle={
                "data_source": data_source,
                "question": question,
                "response": response,
                "identity": identity,
                "verify_label": verify_label,
            },
        )
    
    def on_reward_required( # 把judger的判断结果转成reward的score
        self, status: JudgeStatus, timeout: Optional[float] = None
    ) -> Reward:
        if status.handle["verify_label"] is False:
            score = 0.0
            return score
        # 把judger的判断结果转成reward的score
        data_source = status.handle["data_source"]
        response = status.handle["response"]
        identity = status.handle["identity"]
        prompt = status.handle["question"]
        bootcamp_cls= getattr(internbootcamp, data_source[0].upper() + data_source[1:] + "bootcamp")
        try:
            score = bootcamp_cls.verify_score(response,identity,format_score=self.format_score,format_penalty=self.format_penalty,short_penalty=self.short_penalty,short_threshold=self.short_threshold)
        except:
            score = bootcamp_cls.verify_score(response,identity,format_score=self.format_score)
        return score
        # print(f"[Debug] Prompt: {prompt}")
        # print(f"[Debug]: score: {score}, response: {response}")
        # if type(score) == int:
        #     assert score >= 0 and score <= 1
        #     return score
        # return 0
