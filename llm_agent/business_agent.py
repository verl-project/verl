import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import torch
import numpy as np
from tensordict import TensorDict
from tqdm.auto import tqdm

from verl import DataProto
from verl.deepsearch_agent.search_online import SearchTool
from verl.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    ProfilerConfig,
    log_gpu_memory_usage,
    simple_timer,
    marked_timer,
)
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max


def check_which_tag_comes_later(text: str) -> str:
    web_search_pos = text.rfind("<web_search>")
    answer_pos = text.rfind("<answer>")

    if web_search_pos == -1 and answer_pos == -1:
        return "none"

    if web_search_pos == -1:
        return "answer"
    if answer_pos == -1:
        return "web_search"

    if web_search_pos > answer_pos:
        return "web_search"
    elif answer_pos > web_search_pos:
        return "answer"
    else:
        return "same_position"


def split_queries(query_str: str) -> list[str]:
    queries = re.split(r"[;;\n]", query_str)
    queries = [q.strip() for q in queries if q.strip()]
    queries = [str(i)[:40] for i in queries if len(str(i)) > 2]
    return queries


def web_search_tool(query):
    try:
        tool = SearchTool(max_cut_len=2000)
        result = tool.forward(query)
        return result
    except Exception as e:
        return None


def batch_query_search(query_list: list[str]):
    res = {}
    filtered_queries = [query for query in query_list if query.strip()]
    if not filtered_queries:
        return res

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(web_search_tool, query): query for query in filtered_queries
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Web Searching"
        ):
            query = futures[future]
            try:
                search_result = future.result()
            except Exception:
                continue
            if search_result is not None:
                res[query] = search_result
    return res


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        max_rounds: int = 2,  # 默认最多2轮
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.max_rounds = max_rounds

    def run_llm_loop(self, gen_batch):
        timeing_logs = {}

        status_log_data = {
            index: {
                "raw_prompt_ids": gen_batch.non_tensor_batch["raw_prompt_ids"][index],
                "raw_prompt_text": self.tokenizer.decode(
                    gen_batch.non_tensor_batch["raw_prompt_ids"][index],
                    skip_special_tokens=False,
                ),
                "model_gen_str": [],
            }
            for index in range(len(gen_batch.batch.get("input_ids")))
        }

        raw_user_inputs = [v.get("raw_prompt_text") for k, v in status_log_data.items()]

        # 保存原始batch
        gen_batch_copy = DataProto(**{
            "batch": gen_batch.batch,
            "non_tensor_batch": gen_batch.non_tensor_batch,
            "meta_info": gen_batch.meta_info,
        })

        with marked_timer("agent_loop/all", timeing_logs, color="olive"):
            # s1
            # 第一轮使用原始输入
            gen_batch.meta_info["stop"] = [
                "<|endoftext|>",
                "</web_search>",
                "<|im_end|>",
            ]
            model_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            # 提取web_search内容并调用tool，然后拼接检索结果
            gen_texts = self.convert_batch_to_texts(model_output)
            for index in range(len(gen_texts)):
                status_log_data[index]["model_gen_str"].append(gen_texts[index])

            with marked_timer("agent_loop/web_search_s1", timeing_logs, color="olive"):
                gen_texts_with_search_s1 = self._append_search_results(gen_texts)

            # 原始输入+检索结果，发送给模型，让他生成。
            gen_batch_continue_s1 = self.build_gen_batch_from_texts(
                gen_texts_with_search_s1, gen_batch_copy
            )
            gen_batch_continue_s1.meta_info["stop"] = [
                "<|endoftext|>",
                "</web_search>",
                "<|im_end|>",
                "</answer>",
            ]
            model_output_continue_s1 = self.actor_rollout_wg.generate_sequences(
                gen_batch_continue_s1
            )
            gen_texts_continue_s1 = self.convert_batch_to_texts(
                model_output_continue_s1
            )
            for index in range(len(gen_texts_continue_s1)):
                status_log_data[index]["model_gen_str"].append(
                    gen_texts_continue_s1[index]
                )

            # s2
            # 判断生成的内容是否含有<answer>标签，含有则流程只需1轮即可结束；如果标签是<web_search>，则继续下一轮；如果都没有，则视为失败结束。

            need_next_round = []
            texts_need_search = []
            for index, gen_text in enumerate(gen_texts_continue_s1):
                last_tag = check_which_tag_comes_later(gen_text)

                if last_tag == "answer":
                    # 当前轮成功结束
                    need_next_round.append(False)
                elif last_tag == "web_search":
                    # 需要继续下一轮
                    need_next_round.append(True)
                    texts_need_search.append(gen_text)
                else:
                    # 当前轮失败
                    need_next_round.append(False)

            # 批量处理检索-收集需要检索的样本
            with marked_timer("agent_loop/web_search_s2", timeing_logs, color="olive"):
                gen_texts_with_search_s2 = self._append_search_results(
                    texts_need_search
                )

            # s3
            # 在之前的基础上，再次对需要检索的部分，在生成，发送给模型，让他生成。
            # bug: 因为要求均匀分割，所以需要长度填充到原来的长度,就拿最初的输入文本来填充
            pad_str_list = [
                good_text
                for good_text, need in zip(raw_user_inputs, need_next_round)
                if not need
            ]
            gen_batch_continue_s2 = self.build_gen_batch_from_texts(
                gen_texts_with_search_s2 + pad_str_list, gen_batch_copy
            )
            gen_batch_continue_s2.meta_info["stop"] = [
                "<|endoftext|>",
                "</web_search>",
                "<|im_end|>",
                "</answer>",
            ]
            model_output_continue_s2 = self.actor_rollout_wg.generate_sequences(
                gen_batch_continue_s2
            )
            gen_texts_continue_s2 = self.convert_batch_to_texts(
                model_output_continue_s2
            )[: len(gen_texts_with_search_s2)]
            start_index = 0
            for index in range(len(need_next_round)):
                if need_next_round[index]:
                    status_log_data[index]["model_gen_str"].append(
                        gen_texts_continue_s2[start_index]
                    )
                    start_index += 1

            # with open("data/debug/status_log_data.json", "w") as fout:
            #     json.dump(status_log_data, fout, indent=4, ensure_ascii=False)

            user_input_texts = [
                v.get("raw_prompt_text") for k, v in status_log_data.items()
            ]
            user_input_tokenizer = self.tokenizer(
                user_input_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
                padding_side="left",
            )
            response_texts = [
                v.get("model_gen_str")[-1][len(v.get("raw_prompt_text")) :]
                for k, v in status_log_data.items()
            ]
            response_tokenizer = self.tokenizer(
                response_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
                padding_side="right",
            )

            concat_mask = torch.cat(
                (
                    user_input_tokenizer["attention_mask"],
                    response_tokenizer["attention_mask"],
                ),
                dim=1,
            )

            concat_input_ids = torch.cat(
                (user_input_tokenizer["input_ids"], response_tokenizer["input_ids"]),
                dim=1,
            )

            concat_mask = self._mask_observations(concat_input_ids, concat_mask)

            # breakpoint()

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        # timeing_logs_topk_ratio, timeing_logs_min, timeing_logs_max = topk_reduce_ratio_min_max(
        #     timeing_logs["agent_loop/all"]
        # )
        # timeing_logs = reduce_timing(timeing_logs)
        # timeing_logs.update(
        #     {
        #         "agent_loop/max": timeing_logs_max,
        #         "agent_loop/min": timeing_logs_min,
        #         "agent_loop/topk_ratio": timeing_logs_topk_ratio,
        #     }
        # )
        final_data = DataProto.from_dict(
            tensors={
                "attention_mask": concat_mask,
                "input_ids": torch.cat(
                    (
                        user_input_tokenizer["input_ids"],
                        response_tokenizer["input_ids"],
                    ),
                    dim=1,
                ),
                "position_ids": self._generate_position_ids(concat_mask),
                "prompts": user_input_tokenizer["input_ids"],
                "responses": response_tokenizer["input_ids"],
            },
            non_tensors={
                **gen_batch.non_tensor_batch,
                # "status_log_data": np.array(status_log_data, dtype=object),
            },
            meta_info={
                "original_meta_info": gen_batch.meta_info,
                # **gen_batch.non_tensor_batch,
                "status_log_data": status_log_data,
                "timing": timeing_logs,
            },
        )

        # final_data.save_to_disk(
        #     f"/data2/huzheng/hz_deepxsearch/verl_agent_train/data/debug/final_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
        # )
        return final_data

    def _mask_observations(self, input_ids, attention_mask):
        """屏蔽<observation> xxx </observation>部分的attention_mask"""
        import torch

        masked_attention = attention_mask.clone()

        # 定义标记序列
        target_seq = torch.tensor(
            [151645, 198, 151644, 77091, 198],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        start_marker = torch.tensor(
            [27, 77960, 29], dtype=input_ids.dtype, device=input_ids.device
        )
        end_marker = torch.tensor(
            [77960, 1339], dtype=input_ids.dtype, device=input_ids.device
        )

        # 对batch中每个样本处理
        for batch_idx in range(input_ids.shape[0]):
            data = input_ids[batch_idx]
            mask = masked_attention[batch_idx]

            # 查找第一次出现目标序列的位置
            target_pos = -1
            seq_len = len(target_seq)

            for i in range(len(data) - seq_len + 1):
                if torch.equal(data[i : i + seq_len], target_seq):
                    target_pos = i
                    break

            # 如果找到目标序列
            if target_pos != -1:
                # 规则1: 将目标序列及其前面的所有token设为0
                mask[: target_pos + seq_len] = 0

                # 规则2: 在目标序列后面查找并处理<observation>到</observation>的区间
                search_start = target_pos + seq_len
                i = search_start

                while i <= len(data) - len(start_marker):
                    # 查找start_marker <observation>
                    if i + len(start_marker) <= len(data) and torch.equal(
                        data[i : i + len(start_marker)], start_marker
                    ):
                        start_pos = i

                        # 从start_marker位置开始查找end_marker </observation>
                        j = start_pos + len(start_marker)
                        end_pos = -1

                        while j <= len(data) - len(end_marker):
                            if torch.equal(data[j : j + len(end_marker)], end_marker):
                                end_pos = j
                                break
                            j += 1

                        # 如果找到完整的区间，将整个区间设为0
                        if end_pos != -1:
                            mask[start_pos : end_pos + len(end_marker)] = 0
                            i = end_pos + len(end_marker)
                        else:
                            # 如果没找到end_marker，只将start_marker设为0
                            mask[start_pos : start_pos + len(start_marker)] = 0
                            i = start_pos + len(start_marker)
                    else:
                        i += 1

        return masked_attention

    def _generate_position_ids(self, attention_mask):
        """根据attention_mask生成position_ids"""
        import torch

        position_ids = torch.zeros_like(attention_mask, dtype=torch.long)

        for batch_idx in range(attention_mask.shape[0]):
            mask = attention_mask[batch_idx]
            # 累积计数有效位置
            position_ids[batch_idx] = torch.cumsum(mask, dim=0) - 1
            # 将mask为0的位置的position_id也设为0
            position_ids[batch_idx][mask == 0] = 0

        return position_ids

    def _append_search_results(self, gen_texts: list[str]) -> list[str]:
        """对生成的文本进行检索并拼接结果"""
        search_results = self.web_search(gen_texts)
        texts_with_search = []

        for gen_text, search_res in zip(gen_texts, search_results, strict=True):
            if gen_text.endswith("</web_search>\n\n"):
                texts_with_search.append(
                    gen_text + f"""<observation> {search_res}</observation>\n\n"""
                )
            elif gen_text.endswith("</web_search>"):
                texts_with_search.append(
                    gen_text + f"""\n\n<observation> {search_res}</observation>\n\n"""
                )
            elif gen_text.endswith("</web_search>\n\n<|endoftext|>"):
                texts_with_search.append(
                    gen_text[: -len("<|endoftext|>")]
                    + f"""<observation> {search_res}</observation>\n\n"""
                )
            else:
                # 如果没有正确的结束标签,直接使用原文本
                texts_with_search.append(gen_text)

        return texts_with_search

    def build_gen_batch_from_texts(self, texts: list[str], original_batch):
        """将文本列表转换回 DataProto 格式的 batch"""
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            padding_side="left",
        )
        position_ids = tokenized["attention_mask"].cumsum(dim=1) - 1
        position_ids = position_ids.masked_fill(tokenized["attention_mask"] == 0, 0)
        tokenized["position_ids"] = position_ids

        new_batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "position_ids": tokenized["position_ids"],
        }
        prompt_token_idss = np.array(
            [
                self.tokenizer(
                    texts[i],
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                    padding_side="left",
                )
                .get("input_ids")
                .tolist()[0]
                for i in range(len(texts))
            ],
            dtype=object,
        )

        gen_batch = DataProto.from_dict(
            tensors=new_batch,
            non_tensors={"raw_prompt_ids": prompt_token_idss},
            meta_info=original_batch.meta_info.copy(),
        )
        return gen_batch

    def web_search(self, gen_texts: list[str]) -> list[str]:
        all_querys = []
        for text in gen_texts:
            query_list = split_queries(self.extract_web_search_content(text))
            all_querys.append(query_list)
        print("All querys:")
        print(all_querys)

        unique_querys = sum(all_querys, [])
        unique_querys = list(set(unique_querys))  # 去重
        search_results = batch_query_search(unique_querys)

        search_results_list = [
            {query: str(search_results.get(query, "")) for query in querys}
            for querys in all_querys
        ]
        return search_results_list

    def convert_batch_to_texts(self, gen_batch):
        all_texts = []
        for index in range(len(gen_batch.batch.get("input_ids"))):
            text = self.tokenizer.decode(
                [
                    value
                    for idx, value in enumerate(gen_batch.batch.get("input_ids")[index])
                    if gen_batch.batch.get("attention_mask")[index][idx] == 1
                ],
                skip_special_tokens=False,
            )
            if text.endswith("<|endoftext|>"):
                text = text[: -len("<|endoftext|>")]
            all_texts.append(text)
        return all_texts

    def extract_sepcific_content(self, text: str, tag: str):
        """在一段文本中的找到最后面一对的start_tag和end_tag之间的内容"""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        end_index = text.rfind(end_tag)
        if end_index == -1:
            return ""

        search_text = text[:end_index]
        start_index = search_text.rfind(start_tag)

        if start_index != -1:
            return text[start_index + len(start_tag) : end_index]
        else:
            return ""

    def extract_web_search_content(self, text: str) -> str:
        """Extract content between <web_search> and </web_search> tags"""
        return self.extract_sepcific_content(text, "web_search")
