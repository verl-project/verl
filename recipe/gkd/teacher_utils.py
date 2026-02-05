# Copyright 2025 Individual Contributor: furunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for teacher model knowledge distillation.

Functions:
    get_teacher_knowledge: Retrieve teacher model's top-k predictions and log probabilities.
"""

import time
from types import SimpleNamespace

import torch

from verl import DataProto

teacher_topk_logps_padded, teacher_topk_indices_padded = None, None


def get_teacher_knowledge(batch: DataProto, teacher_client, n_server_workers=1, is_async=False):
    """
    Retrieve teacher model's top-k predictions and log probabilities for knowledge distillation.

    Args:
        batch (DataProto): Input batch containing input_ids and attention_mask
        teacher_client: Client for communicating with teacher model
        n_server_workers (int): Number of parallel workers for teacher model inference
        is_async (bool): Whether to use asynchronous processing

    Returns:
        If is_async=True: SimpleNamespace with get() method to process futures
        If is_async=False: Processed DataProto containing teacher knowledge

    Raises:
        RuntimeError: If teacher model request fails
    """

    input_ids = []
    attention_mask = batch.batch["attention_mask"].to(torch.bool)
    # response_length = batch.meta_info["response_length"]

    for ids, mask in zip(batch.batch["input_ids"], attention_mask, strict=False):
        input_ids.append(ids[mask].tolist())

    all_teacher_topk_logps = []
    all_teacher_topk_indices = []

    batch_size = len(input_ids)
    assert batch_size % n_server_workers == 0
    micro_batch_size = batch_size // n_server_workers
    futures = []
    tik1 = time.time()
    tok1 = tik1

    def cb(future):
        nonlocal tok1
        tok1 = max(tok1, time.time())

    for i in range(0, batch_size, micro_batch_size):
        fut = teacher_client.submit(input_ids[i : i + micro_batch_size])
        fut.add_done_callback(cb)
        futures.append(fut)

    def handle_futures():
        for future in futures:
            try:
                _, teacher_topk_logps, teacher_topk_indices = future.result()
            except Exception as e:
                raise RuntimeError(f"Teacher request failed: {e}") from e

            all_teacher_topk_logps.extend(teacher_topk_logps)
            all_teacher_topk_indices.extend(teacher_topk_indices)

        tik2 = time.time()
        # teacher_topk_logps = [x.to(params_dtype) for x in all_teacher_topk_logps]
        # teacher_topk_indices = [x.to(params_dtype) for x in all_teacher_topk_indices]
        teacher_topk_logps, teacher_topk_indices = all_teacher_topk_logps, all_teacher_topk_indices

        real_seq_lens = torch.tensor([x.size(0) for x in teacher_topk_logps], dtype=torch.int32)

        topk = teacher_topk_logps[0].size(-1)

        logp_dtype = teacher_topk_logps[0].dtype
        idx_dtype = teacher_topk_indices[0].dtype
        teacher_knowledge_shape = list(batch.batch["input_ids"].shape) + [topk]

        global teacher_topk_logps_padded, teacher_topk_indices_padded
        if (
            teacher_topk_logps_padded is None
            or teacher_topk_logps_padded.dtype != logp_dtype
            or teacher_topk_logps_padded.shape != torch.Size(teacher_knowledge_shape)
        ):
            teacher_topk_logps_padded = torch.zeros(*teacher_knowledge_shape, dtype=logp_dtype)
        else:
            teacher_topk_logps_padded.zero_()

        if (
            teacher_topk_indices_padded is None
            or teacher_topk_indices_padded.dtype != idx_dtype
            or teacher_topk_indices_padded.shape != torch.Size(teacher_knowledge_shape)
        ):
            teacher_topk_indices_padded = torch.zeros(*teacher_knowledge_shape, dtype=idx_dtype)
        else:
            teacher_topk_indices_padded.zero_()

        batch_size = attention_mask.size(0)
        sequence_length = attention_mask.size(1)
        for i in range(batch_size):
            num_valid_positions = attention_mask[i].sum().item()
            teacher_seq_len = teacher_topk_logps[i].size(0)

            if teacher_seq_len == num_valid_positions:
                # only_response=False: teacher returns logprobs for full sequence
                # Fill at all attention_mask positions (left-aligned)
                teacher_topk_logps_padded[i][attention_mask[i]] = teacher_topk_logps[i]
                teacher_topk_indices_padded[i][attention_mask[i]] = teacher_topk_indices[i]
            else:
                # only_response=True: teacher returns logprobs only for response tokens
                # Need to right-align: place at the END of valid positions
                # Find the valid positions and fill from the right
                valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
                # Take the last teacher_seq_len positions
                response_positions = valid_positions[-teacher_seq_len:]
                teacher_topk_logps_padded[i][response_positions] = teacher_topk_logps[i]
                teacher_topk_indices_padded[i][response_positions] = teacher_topk_indices[i]

        output_batch = DataProto.from_single_dict(
            data={"real_seq_lens": real_seq_lens},
        )

        output_batch.non_tensor_batch.update(
            {
                "teacher_topk_logps": teacher_topk_logps_padded.numpy(),
                "teacher_topk_indices": teacher_topk_indices_padded.numpy(),
            }
        )

        tok2 = time.time()

        output_batch.meta_info["timing"] = {"get_teacher_knowledge": (tok1 - tik1) + (tok2 - tik2)}

        return output_batch

    if is_async:
        return SimpleNamespace(get=handle_futures)
    else:
        return handle_futures()


if __name__ == "__main__":
    batch = DataProto.load_from_disk("gen_batch_output")
    from teacher import TeacherClient

    teacher_client = TeacherClient(server_ip="10.215.192.141", server_port=15555)
    output_batch = get_teacher_knowledge(batch, 2, teacher_client)
    output_batch_chunks = output_batch.chunk(2)

    for data in output_batch_chunks:
        topk = data.meta_info["topk"]
        seq_lens = data.batch["seq_lens"]
        teacher_topk_logps = data.batch["teacher_topk_logps"].view(-1, topk)
        teacher_topk_indices = data.batch["teacher_topk_indices"].view(-1, topk)

        attention_mask = data.batch["attention_mask"]
        batch_size, sequence_length = attention_mask.size(0), attention_mask.size(1)
        teacher_topk_logps_padded = torch.zeros(batch_size, sequence_length, topk, dtype=teacher_topk_logps.dtype)
        teacher_topk_indices_padded = torch.zeros(batch_size, sequence_length, topk, dtype=teacher_topk_indices.dtype)

        teacher_topk_logps_padded[attention_mask] = teacher_topk_logps[: seq_lens.sum()]
        teacher_topk_indices_padded[attention_mask] = teacher_topk_indices[: seq_lens.sum()]

        data.batch["teacher_topk_logps"] = teacher_topk_logps_padded
        data.batch["teacher_topk_indices"] = teacher_topk_indices_padded

        assert (data.batch["teacher_topk_logps"] == data.batch["teacher_topk_logps_padded"]).all()
        assert (data.batch["teacher_topk_indices"] == data.batch["teacher_topk_indices_padded"]).all()