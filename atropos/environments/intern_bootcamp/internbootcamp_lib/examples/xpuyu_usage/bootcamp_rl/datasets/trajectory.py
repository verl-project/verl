# Copyright (c) InternLM. All rights reserved.
import json
import random

import numpy as np
import torch
from xtuner._lite import get_logger
from xtuner._lite.algorithms.sft.dataset import SftCollator

logger = get_logger()


class InferDataset(torch.utils.data.Dataset):

    def __init__(self, prompts_input_ids, responses_ids, message_data, metadata):
        super().__init__()

        assert (
            len(prompts_input_ids)
            == len(responses_ids)
            == len(message_data)
            == len(metadata)
        ), f"The length of prompts_input_ids, responses_ids, message_data, metadata should be the same, but got {len(prompts_input_ids)}, {len(responses_ids)}, {len(message_data)}, {len(metadata)}"
        self.prompts_input_ids = prompts_input_ids
        self.responses_ids = responses_ids
        self.message_data = message_data
        self.metadata = metadata

    def __len__(self):
        return len(self.prompts_input_ids)

    def __getitem__(self, item):

        prompt_input_ids = self.prompts_input_ids[item]
        response_ids = self.responses_ids[item]
        num_prefill_tokens = len(prompt_input_ids)

        input_ids = prompt_input_ids + response_ids
        labels = [-100] * (num_prefill_tokens - 1) + response_ids + [-100]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "num_tokens": len(input_ids),
            "message_data": self.message_data[item],
            "metadata": self.metadata[item],
        }


class TrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        self._num_action_tokens = 0
        self._num_total_tokens = 0
        self._trajectories = []

    @property
    def num_action_tokens(self):
        return self._num_action_tokens.item()

    @property
    def num_total_tokens(self):
        return self._num_total_tokens

    def update(self, trajectories):
        num_total_tokens = 0
        num_action_tokens = 0
        for data in trajectories:
            labels = np.array(data["labels"])
            num_total_tokens += labels.size
            num_action_tokens += (labels >= 0).sum()

        self._num_action_tokens = num_action_tokens
        self._num_total_tokens = num_total_tokens

        self._trajectories = trajectories

    def dump_jsonl(self, path, tokenizer, debug=False):

        with open(path, "w", encoding="utf8") as f:
            for data in self._trajectories:
                json_line = {
                    "sequence": (
                        data["sequence_text"]
                        if "sequence_text" in data
                        else tokenizer.decode(data["input_ids"])
                    ),
                    "num_tokens": data["num_tokens"],
                }
                json_line["judger_reward"] = data["judger_reward"]
                json_line["judger_advantage"] = data["judger_advantage"]

                if debug:
                    json_line["input_ids"] = data["input_ids"]
                    json_line["labels"] = data["labels"]

                json_str = json.dumps(json_line, ensure_ascii=False)
                f.write(json_str + "\n")

    def dump_log(self, path, tokenizer, debug=False):

        with open(path, "w", encoding="utf8") as f:
            for data in self._trajectories:
                log_string = f"[sequence]:\n{data['sequence_text'] if 'sequence_text' in data else tokenizer.decode(data['input_ids'])}\n\n"
                log_string += f"[num_tokens]: {data['num_tokens']}\n"
                log_string += f"[judger_reward]: {data['judger_reward']}\n"
                log_string += f"[judger_advantage]: {data['judger_advantage']}\n"
                f.write(log_string + "\n\n=======================\n")

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, item):

        return self._trajectories[item]


class TrajectoryDatasetWithFilter(TrajectoryDataset):
    def __init__(self, repeat_k=1, only_keep_1_pair=True):
        super().__init__()
        self.repeat_k = repeat_k
        self.only_keep_1_pair = only_keep_1_pair

    def update(self, trajectories):
        # split trajectories into k groups: (a, a, b, b, c, c) -> [(a, a), (b, b), (c, c)]
        groups = [
            trajectories[i : i + self.repeat_k]
            for i in range(0, len(trajectories), self.repeat_k)
        ]
        keeped_trajectories = []
        for group in groups:
            correctness = [1 if data["judger_reward"] == 1 else 0 for data in group]
            correct = [data for data in group if data["judger_reward"] == 1]
            incorrect = [data for data in group if data["judger_reward"] != 1]
            pass_rate = sum(correctness) / len(correctness)
            if self.only_keep_1_pair:
                if pass_rate == 1 or pass_rate == 0:
                    continue
                # max keep 1 correct and 1 incorrect
                correct = random.choice(correct)
                incorrect = random.choice(incorrect)
                correct["pass_rate"] = pass_rate
                incorrect["pass_rate"] = pass_rate
                keeped_trajectories.append(correct)
                keeped_trajectories.append(incorrect)
            else:
                if pass_rate == 1 or pass_rate == 0:
                    continue
                for data in group:
                    data["pass_rate"] = pass_rate
                    keeped_trajectories.append(data)

        super().update(keeped_trajectories)


class TrajectoryCollator(SftCollator):

    def __call__(self, instances):

        data = super().__call__(instances)
        data["judger_rewards"] = [item["judger_reward"] for item in instances]
        data["judger_advantages"] = [item["judger_advantage"] for item in instances]
        if "pass_rate" in instances[0]:
            data["pass_rate"] = [item["pass_rate"] for item in instances]
        return data
