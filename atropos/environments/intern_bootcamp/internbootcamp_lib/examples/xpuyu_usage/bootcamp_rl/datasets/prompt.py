# Copyright (c) InternLM. All rights reserved.
import json

import time
import numpy as np
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from xtuner._lite import get_logger

logger = get_logger()


def load_hf_datasets(repo, split="train"):
    dataset = load_dataset(repo, split=split)
    converted_ds = []
    for sample in dataset:
        converted_ds.append(
            {
                "pass_rate": sample["pass_rate"],
                "message_data": [{"role": "user", "content": sample["question"]}],
                "metadata": {
                    "data_source": "math",  # for the router to know which judger to use
                    "gold_answer": sample["gold_answer"],
                },
            }
        )
    logger.info(f"Loaded {len(converted_ds)} samples from {repo}")
    return converted_ds


def load_jsonl_datasets(file_path):
    subsample_ratio = 1.0
    if "::" in file_path:
        file_path, subsample_ratio = file_path.split("::")
        subsample_ratio = float(subsample_ratio)
    with open(file_path, "r") as f:
        lines = f.readlines()
    datasets = []
    for line in lines:
        sample = json.loads(line)
        if "message_data" not in sample:
            datasets.append(
                {
                    "pass_rate": sample["pass_rate"],
                    "message_data": [{"role": "user", "content": sample["question"]}],
                    "metadata": {
                        "data_source": "math",  # for the router to know which judger to use
                        "gold_answer": sample["gold_answer"],
                    },
                }
            )
        else:
            datasets.append(sample)
    if subsample_ratio < 1.0:
        np.random.seed(0)
        datasets = np.random.choice(
            datasets, int(len(datasets) * subsample_ratio), replace=False
        ).tolist()

    logger.info(f"Loaded {len(datasets)} samples from {file_path}")
    return datasets


def balance_difficulty_with_cfg(dataset, difficulty_balance_cfg):
    balanced_dataset = []
    for sample in dataset:
        pass_rate = sample["pass_rate"]
        for (low, high), repeat in difficulty_balance_cfg:
            if low <= pass_rate < high:
                balanced_dataset.extend([sample] * repeat)
                break
    logger.info(
        f"After difficulty balancing, the dataset size is {len(balanced_dataset)}"
    )
    return balanced_dataset


class bootcampPromptDataset(Dataset):
    def __init__(self, path, tokenizer, difficulty_balance_cfg=None):
        if isinstance(path, str):
            path = [path]
        dataset = []
        for p in path:
            if p.endswith(".jsonl"):
                dataset.extend(load_jsonl_datasets(p))
            else:
                dataset.extend(load_hf_datasets(p))
        if difficulty_balance_cfg:
            dataset = balance_difficulty_with_cfg(dataset, difficulty_balance_cfg)
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        input_ids = self.tokenizer.apply_chat_template(
            sample["message_data"], add_generation_prompt=True
        )
        sample["input_ids"] = input_ids
        sample["labels"] = input_ids
        sample["num_tokens"] = len(input_ids)
        return sample


class PromptCollator:

    def __init__(self, pad_token_id=0, ignore_id=-100, pack_batch=False):
        self.pack_batch = pack_batch
        self.pad_token_id = pad_token_id
        self.ignore_id = ignore_id

    def __call__(self, instances):

        _instances = []
        for ins in instances:
            if isinstance(ins, list):
                _instances.extend(ins)
            else:
                _instances.append(ins)

        instances = _instances

        input_ids = []
        labels = []
        num_tokens = []
        metadatas = []
        message_datas = []

        for data in instances:

            input_ids.append(torch.LongTensor(data["input_ids"]))
            labels.append(torch.LongTensor(data["labels"]))
            metadatas.append(data["metadata"])
            message_datas.append(data["message_data"])

            if isinstance(data["num_tokens"], int):
                num_tokens.append(data["num_tokens"])
            else:
                num_tokens.extend(data["num_tokens"])

        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        num_tokens = torch.IntTensor(num_tokens)

        if len(instances) > 1 and self.pack_batch:

            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            labels = torch.cat(labels, dim=0).unsqueeze(0)
            attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)

        elif len(instances) > 1 and not self.pack_batch:

            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id
            )
            labels = pad_sequence(
                labels, batch_first=True, padding_value=self.ignore_id
            )
            attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            attention_mask = torch.stack(attention_mask)

        if input_ids.shape != labels.shape:
            logger.error(f"[instances] {instances}")
            logger.error(f"[num_tokens] {num_tokens}")
            logger.error(f"[input_ids] {input_ids}")
            logger.error(f"[labels] {labels}")
            raise RuntimeError(
                "The shape of input_ids and labels must be "
                f"equal, but  found {input_ids.shape} and "
                f"{labels.shape}."
            )
        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "num_tokens": num_tokens,
            "attention_mask": attention_mask.bool(),
            "metadata": metadatas,
            "message_data": message_datas,
        }

        return data_dict

class InfiniteDataLoaderIter:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self._epoch = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            logger.info(f"Dataloader epoch {self._epoch} finished. Start a new epoch.")
            self._epoch += 1
            if hasattr(self.dataloader, 'sampler') and hasattr(
                    self.dataloader.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self.dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data
