# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Pairwise Comparison Reward Manager
实现基于两两对比的奖励机制
"""

from collections import defaultdict
from typing import Any
import torch
import itertools
import numpy as np
import os
import json
import asyncio
import re
from datetime import datetime

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("pairwise_comparison")
class PairwiseComparisonRewardManager(AbstractRewardManager):
    """基于两两对比的奖励管理器"""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        """
        初始化配对比较奖励管理器
        
        Args:
            tokenizer: 用于解码的tokenizer
            num_examine: 调试时打印的批次数
            compute_score: 自定义评分函数
            reward_fn_key: 数据源键名
            comparison_model: OpenAI比较模型配置
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        
        # 比较相关配置
        self.comparison_model = kwargs.get("comparison_model", None)
        
        # 初始化比较模型
        self._init_comparison_model()
    
    def _init_comparison_model(self):
        """初始化自定义推理模型比较器"""
        try:
            from .openai_comparator import OpenAIComparator
            
            # 从环境变量或配置中获取API密钥
            import os
            
            # 从环境变量获取API key
            novita_api_key = os.getenv("NOVITA_API_KEY")
            
            # 获取配置参数
            if hasattr(self, 'comparison_model') and self.comparison_model:
                api_key = self.comparison_model.get("api_key", novita_api_key)
                model_name = self.comparison_model.get("model_name", "deepseek/deepseek-r1-distill-qwen-14b")
                api_base = self.comparison_model.get("api_base", "https://api.novita.ai/openai")
                max_retries = self.comparison_model.get("max_retries", 3)
                max_concurrent = self.comparison_model.get("max_concurrent", 10)
                max_concurrent_groups = self.comparison_model.get("max_concurrent_groups", 8)
            else:
                api_key = novita_api_key
                model_name = "deepseek/deepseek-r1-distill-qwen-14b"
                api_base = "https://api.novita.ai/openai"
                max_retries = 3
                max_concurrent = 10
                max_concurrent_groups = 8
            
            # 创建group级别的信号量
            self.group_semaphore = asyncio.Semaphore(max_concurrent_groups)
            
            self.comparator = OpenAIComparator(
                api_key=api_key,
                model_name=model_name,
                api_base=api_base,
                max_retries=max_retries,
                max_concurrent=max_concurrent
            )
            print(f"🚀 Using custom reasoning model for pairwise comparison")
            print(f"   - Model: {model_name}")
            print(f"   - API Base: {api_base}")
            print(f"   - Max Concurrent per Group: {max_concurrent}")
            print(f"   - Max Concurrent Groups: {max_concurrent_groups}")
            
        except ImportError as e:
            raise RuntimeError(f"Failed to import OpenAI comparator: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI comparator: {e}")
    
    def _extract_answer_from_response(self, response: str) -> str:
        """
        从response中提取最终答案
        
        Args:
            response: 模型生成的完整回答
            
        Returns:
            str: 提取的答案，如果找不到则返回""
        """
        # 优先匹配 #### 格式的答案
        answer_pattern = r'####\s*(.+?)(?:\n|$)'
        match = re.search(answer_pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 次优匹配 \boxed{...} 格式
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, response)
        if match:
            return match.group(1).strip()
        
        # 最后尝试匹配最后一个数字或简单表达式
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and (re.search(r'\d+', line) or '=' in line):
                # 提取等号后的内容或最后的数字
                if '=' in line:
                    return line.split('=')[-1].strip()
                # 提取最后的数字
                number_match = re.findall(r'\d+(?:\.\d+)?', line)
                if number_match:
                    return number_match[-1]
        
        return ""
    
    def _normalize_answer(self, answer: str) -> str:
        """
        标准化答案格式以便比较
        
        Args:
            answer: 原始答案字符串
            
        Returns:
            str: 标准化后的答案
        """
        if not answer:
            return ""
        
        # 移除常见的格式符号和空格
        answer = answer.replace('$', '').replace(',', '').strip()
        
        # 如果是分数，尝试转换为小数
        if '/' in answer and len(answer.split('/')) == 2:
            try:
                parts = answer.split('/')
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator != 0:
                    return str(numerator / denominator)
            except:
                pass
        
        # 如果是数字，标准化格式
        try:
            # 尝试转换为浮点数再转回字符串以标准化
            num = float(answer)
            if num.is_integer():
                return str(int(num))
            else:
                return str(num)
        except:
            pass
        
        # 对于非数字答案，只是清理空格和大小写
        return answer.lower().strip()
    
    def _check_answer_correctness(self, response_text: str, ground_truth: str) -> tuple[bool, str, str]:
        """
        檢查單個回答是否正確
        
        Args:
            response_text: 模型生成的回答文本
            ground_truth: 正確答案
            
        Returns:
            tuple: (is_correct, extracted_answer, normalized_answer)
        """
        if not ground_truth:
            return False, "", ""
        
        # 提取並標準化答案
        extracted_answer = self._extract_answer_from_response(response_text)
        normalized_answer = self._normalize_answer(extracted_answer)
        normalized_gt = self._normalize_answer(ground_truth)
        
        # 比較是否正確
        is_correct = normalized_answer == normalized_gt
        
        return is_correct, extracted_answer, normalized_answer
    
    async def _compute_win_rate_async(self, prompt: str, responses: list[str], ground_truth: str) -> dict:
        """
        使用OpenAI comparator计算responses相比ground truth的胜率
        
        Args:
            prompt: 原始问题
            responses: 模型生成的所有回答
            ground_truth: 正确答案（ground truth）
            
        Returns:
            dict: 包含胜率统计信息
        """
        if not ground_truth:
            print(f"⚠️ No ground truth provided")
            return {
                "win_rate": 0.0,
                "win_count": 0,
                "total_count": len(responses),
                "win_indices": []
            }
        
        # 提取ground truth的HTML内容
        ground_truth_html = self._extract_html_from_response(ground_truth)
        
        win_indices = []
        win_count = 0
        
        print(f"🔍 Win Rate Check Debug:")
        print(f"   - Prompt: '{prompt[:100]}...'")
        print(f"   - Ground Truth: '{ground_truth[:100]}...'")
        print(f"   - Ground Truth HTML: '{ground_truth_html[:100]}...'")
        print(f"   - Total Responses: {len(responses)}")
        
        # 异步比较每个response与ground truth
        comparison_tasks = []
        for i, response in enumerate(responses):
            # 提取response的HTML内容
            response_html = self._extract_html_from_response(response)
            print(f"   Response {i} HTML: '{response_html[:100]}...'")
            
            # 创建异步比较任务（response vs ground_truth）
            task = self.comparator.compare_responses_async(prompt, response_html, ground_truth_html)
            comparison_tasks.append((i, task))
        
        # 执行所有比较任务
        print(f"🔄 Executing {len(comparison_tasks)} comparisons with ground truth...")
        for i, task in comparison_tasks:
            try:
                winner = await task  # compare_responses_async返回int，不是dict
                
                # winner: 0=tie, 1=response更好, 2=ground_truth更好
                # 如果response获胜（winner==1），则计为胜利
                is_win = (winner == 1)
                
                print(f"   Response {i} vs GT: Winner={winner}, Win={is_win}")
                
                if is_win:
                    win_indices.append(i)
                    win_count += 1
                    
            except Exception as e:
                print(f"   ⚠️ Comparison error for response {i}: {e}")
        
        win_rate = win_count / len(responses) if responses else 0.0
        print(f"   📊 Final win rate vs GT: {win_rate:.3f} ({win_count}/{len(responses)})")
        
        return {
            "win_rate": win_rate,
            "win_count": win_count,
            "total_count": len(responses),
            "win_indices": win_indices,
            "ground_truth": ground_truth
        }
    
    async def _conduct_tournament(self, prompt: str, responses: list[str], group_id: str = None) -> dict[int, float]:
        """
        进行tournament比赛，返回每个response的得分，并保存详细对比结果
        
        Args:
            prompt: 输入问题
            responses: 所有回答列表
            group_id: 用于创建保存文件夹的ID
            
        Returns:
            dict: {response_index: score}
        """
        n_responses = len(responses)
        win_matrix = np.zeros((n_responses, n_responses))
        
        print(f"🏆 Starting tournament with {n_responses} responses")
        
        # 创建保存文件夹
        debug_dir = None
        if group_id:
            debug_dir = f"./pairwise_debug/group_{group_id}"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"📁 Saving comparison details to: {debug_dir}")
        
        # 创建所有比较任务
        total_comparisons = n_responses * (n_responses - 1) // 2
        comparison_tasks = []
        
        print(f"🚀 Creating {total_comparisons} concurrent comparison tasks...")
        
        # 创建异步比较任务
        for i, j in itertools.combinations(range(n_responses), 2):
            task = self._compare_pair_async(prompt, responses[i], responses[j], i, j, debug_dir)
            comparison_tasks.append(task)
        
        # 并发执行所有比较任务
        print(f"⚡ Starting concurrent OpenAI API calls...")
        comparison_results = await asyncio.gather(*comparison_tasks)
        
        print(f"✅ Completed {len(comparison_results)} concurrent comparisons")
        
        # 处理结果并更新矩阵
        comparison_details = []
        for result in comparison_results:
            i, j, winner, comparison_info = result
            comparison_details.append(comparison_info)
            
            # 更新胜负矩阵
            if winner == 1:  # response1 wins
                win_matrix[i, j] = 1.0
                win_matrix[j, i] = 0.0
            elif winner == 2:  # response2 wins
                win_matrix[i, j] = 0.0
                win_matrix[j, i] = 1.0
            else:  # winner == 0, tie (fallback case) - 平局时两个都是0分
                win_matrix[i, j] = 0.0
                win_matrix[j, i] = 0.0
        
        # 计算每个response的总得分
        scores = {}
        for i in range(n_responses):
            total_wins = np.sum(win_matrix[i, :])
            # 归一化得分：赢的比赛数 / 总比赛数
            scores[i] = total_wins / (n_responses - 1)
        
        # 打印排名结果
        sorted_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print("🏅 Tournament Results (Response Index: Score):")
        for rank, (idx, score) in enumerate(sorted_by_score, 1):
            print(f"   Rank {rank}: Response {idx} -> {score:.3f}")
        
        # 保存汇总信息文件
        if debug_dir:
            info_data = {
                "group_id": group_id,
                "prompt": prompt,
                "responses": responses,
                "tournament_results": {
                    "rankings": [{"rank": rank, "response_idx": idx, "score": score} 
                               for rank, (idx, score) in enumerate(sorted_by_score, 1)],
                    "win_matrix": win_matrix.tolist(),
                    "total_comparisons": total_comparisons
                },
                "comparison_summary": comparison_details,
                "timestamp": datetime.now().isoformat()
            }
            
            info_file = os.path.join(debug_dir, "info.json")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Saved tournament info to: {info_file}")
        
        return scores

    def _extract_html_from_response(self, response: str) -> str:
        """
        从response中提取HTML代码（<article...>...</article>标签内容）
        支持多种格式：```html代码块、直接article标签等
        
        Args:
            response: 原始回答文本
            
        Returns:
            str: 提取的HTML代码，如果没有找到则返回原始response
        """
        import re
        
        if not response:
            return ''
        
        # 1. 优先尝试提取```html代码块中的内容
        html_block_pattern = r'```html\s*(.*?)\s*```'
        html_match = re.search(html_block_pattern, response, re.DOTALL | re.IGNORECASE)
        if html_match:
            return html_match.group(1).strip()
        
        # 2. 尝试直接匹配article标签（支持各种属性格式）
        # 匹配 <article...>...</article>，支持任意属性
        article_pattern = r'<article[^>]*>.*?</article>'
        article_match = re.search(article_pattern, response, re.DOTALL | re.IGNORECASE)
        if article_match:
            return article_match.group(0).strip()
        
        # 3. 如果都没找到，返回原始response（让比较器处理）
        return response.strip()

    async def _compare_pair_async(self, prompt: str, response1: str, response2: str, i: int, j: int, debug_dir: str = None):
        """
        异步比较一对回答
        
        Args:
            prompt: 输入问题
            response1: 第一个回答
            response2: 第二个回答  
            i: 第一个回答的索引
            j: 第二个回答的索引
            debug_dir: 调试文件保存目录
            
        Returns:
            tuple: (i, j, winner, comparison_info)
        """
        # 提取HTML代码用于比较
        html1 = self._extract_html_from_response(response1)
        html2 = self._extract_html_from_response(response2)
        
        # 执行异步比较（使用提取的HTML代码）
        result = await self.comparator.compare_responses_with_details_async(prompt, html1, html2)
        winner = result["winner"]
        details = {
            "ai_response": result.get("ai_response", ""),
            "prompt_used": result.get("prompt", ""),
            "response1_html": result.get("response1", ""),
            "response2_html": result.get("response2", ""),
            "error_msg": result.get("error_msg", "")
        }
        
        # 构建比较信息
        comparison_info = {
            "comparison_id": f"{i}_vs_{j}",
            "response_idx_1": i,
            "response_idx_2": j,
            "winner": winner,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        # 保存单个对比文件
        if debug_dir:
            comparison_file = os.path.join(debug_dir, f"comparison_{i}_vs_{j}.json")
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_info, f, ensure_ascii=False, indent=2)
        
        return i, j, winner, comparison_info

    async def _process_group_async(self, data, uid, indices):
        """
        异步处理单个group的tournament比较
        
        Args:
            data: 完整的数据batch
            uid: 当前group的uid
            indices: 该group中所有response的索引
            
        Returns:
            tuple: (uid, indices, scores, rewards, win_info)
                  如果只有单个response，scores、rewards和win_info为None
        """
        async with self.group_semaphore:  # 控制group级别并发数
            if len(indices) < 2:
                # 如果只有一个response，返回None表示跳过tournament
                return uid, indices, None, None, None
            
            # 获取prompt和所有responses
            first_item = data[indices[0]]
            
            # 从 extra_info 中提取原始问题作为 prompt
            try:
                original_question = first_item.non_tensor_batch.get('extra_info', {}).get('question', '')
                
                if original_question:
                    prompt_str = original_question
                    print(f"🔍 Using original question as prompt: '{prompt_str[:200]}...'")
                else:
                    # 如果没有原始问题，记录错误
                    error_msg = "❌ ERROR: Cannot extract prompt - 'extra_info.question' not found in data"
                    print(error_msg)
                    prompt_str = "ERROR_NO_QUESTION_IN_DATA"
                
            except Exception as e:
                error_msg = f"❌ ERROR: Failed to extract prompt from extra_info: {e}"
                print(error_msg)
                prompt_str = f"PROMPT_EXTRACTION_ERROR: {str(e)}"
            
            print(f"🔍 Final extracted prompt: '{prompt_str[:200]}...'" if len(prompt_str) > 200 else f"🔍 Final extracted prompt: '{prompt_str}'")
            
            # 提取responses - 每个item有自己的response
            responses = []
            try:
                for idx in indices:
                    data_item = data[idx]
                    response_ids = data_item.batch['responses']
                    attention_mask = data_item.batch['attention_mask']
                    
                    # 动态获取prompt和response的长度（从tensor形状）
                    prompt_length = data_item.batch['prompts'].shape[0]
                    response_length = response_ids.shape[0]
                    
                    # 使用attention mask找到response的有效部分
                    response_mask = attention_mask[prompt_length:prompt_length + response_length]
                    valid_response_length = response_mask.sum().item()
                    
                    if valid_response_length > 0:
                        valid_response_ids = response_ids[:valid_response_length]
                        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    else:
                        response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    
                    responses.append(response_str)
                    print(f"🔍 Response {len(responses)-1}: '{response_str[:100]}...'")
                    
            except Exception as e:
                print(f"🔍 Response extraction error: {e}")
                responses = [f"RESPONSE_ERROR_{i}" for i in range(len(indices))]
            
            # 提取ground truth
            try:
                reward_model_data = first_item.non_tensor_batch['reward_model']
                ground_truth = reward_model_data['ground_truth']
                print(f"✅ Ground truth: '{ground_truth}'")
                
            except Exception as e:
                print(f"🔍 Ground truth extraction error: {e}")
                ground_truth = ""
            
            # 计算胜率（使用OpenAI comparator与ground truth比较）
            win_info = await self._compute_win_rate_async(prompt_str, responses, ground_truth)
            
            # 进行异步tournament比较
            scores = await self._conduct_tournament(prompt_str, responses, group_id=str(uid))
            
            # 分配奖励
            rewards = self._assign_rewards(scores, len(responses))
            
            return uid, indices, scores, rewards, win_info
    
    def _assign_rewards(self, scores: dict[int, float], n_responses: int) -> list[float]:
        """
        基于tournament结果分配奖励
        
        Args:
            scores: 每个response的得分
            n_responses: 总response数量
            
        Returns:
            list: 每个response的最终奖励
        """
        # 按得分排序
        sorted_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 计算优势组阈值（前50%）
        advantage_threshold = n_responses // 2
        
        # 分配奖励：优势组1.0分，劣势组0.0分
        rewards = [0.0] * n_responses
        for rank, (response_idx, score) in enumerate(sorted_by_score):
            if rank < advantage_threshold:
                rewards[response_idx] = 1.0  # 优势组
            else:
                rewards[response_idx] = 0.0  # 劣势组
        
        return rewards

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        主要調用函數，處理整個batch的數據
        """
        # 如果已有rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        
        # 檢查是否為 validation 階段
        is_validate = data.meta_info.get("validate", False)
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # Validation 階段：跳過 pairwise comparison，給予簡單獎勵並計算正確率
        if is_validate:
            print(f"🔍 Validation mode detected - skipping pairwise comparison tournament")
            prompt_ids = data.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            attention_mask = data.batch["attention_mask"]
            valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
            
            # 計算 validation 正確率
            validation_correct = 0
            validation_total = 0
            
            for i in range(len(data)):
                length = valid_response_lengths[i].item()
                if length > 0:
                    # 統一獎勵 1.0（在最後一個有效 token 位置）
                    reward_tensor[i, length - 1] = 1.0
                    
                    # 提取並檢查答案正確性
                    try:
                        # 解碼 response
                        response_ids = data[i].batch['responses'][:length]
                        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                        
                        # 獲取 ground truth
                        ground_truth = data[i].non_tensor_batch.get('reward_model', {}).get('ground_truth', '')
                        
                        if ground_truth:
                            validation_total += 1
                            # 使用統一的方法檢查正確性
                            is_correct, extracted_answer, _ = self._check_answer_correctness(response_text, ground_truth)
                            
                            if is_correct:
                                validation_correct += 1
                            
                            # 記錄到 extra_info
                            reward_extra_info["answer_correct"].append(is_correct)
                            reward_extra_info["extracted_answer"].append(extracted_answer)
                    except Exception as e:
                        print(f"⚠️ Validation check error for sample {i}: {e}")
            
            # 計算並輸出正確率
            if validation_total > 0:
                validation_accuracy = validation_correct / validation_total
                print(f"📊 Validation Accuracy: {validation_accuracy:.3f} ({validation_correct}/{validation_total})")
                
                # 添加到 extra_info 供 metrics 系統使用
                # 使用 'acc' 会显示在 val-core 区域，更显眼
                reward_extra_info["acc"] = [validation_accuracy] * len(data)
                reward_extra_info["validation_accuracy"] = [validation_accuracy] * len(data)
                reward_extra_info["validation_correct"] = [validation_correct] * len(data)
                reward_extra_info["validation_total"] = [validation_total] * len(data)
            else:
                print(f"⚠️ Validation: No ground truth available for win rate calculation")
            
            print(f"✅ Validation: Assigned uniform rewards to {len(data)} responses")
            
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                return reward_tensor
        
        # 胜率统计
        all_win_rates = []
        all_win_counts = []
        all_total_counts = []
        
        # 按照uid分组处理（同一个prompt的多个responses）
        uid_to_indices = defaultdict(list)
        for i in range(len(data)):
            uid = data[i].non_tensor_batch["uid"]
            uid_to_indices[uid].append(i)
        
        # 🚀 并行处理所有groups
        print(f"🏁 Processing {len(uid_to_indices)} groups in parallel...")
        # breakpoint()
        # 创建异步任务列表
        group_tasks = []
        for uid, indices in uid_to_indices.items():
            task = self._process_group_async(data, uid, indices)
            group_tasks.append(task)
        
        # 并行执行所有group处理
        try:
            # 获取当前事件循环
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 如果没有运行中的事件循环，创建一个新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        group_results = loop.run_until_complete(asyncio.gather(*group_tasks))
        print(f"✅ Completed {len(group_results)} groups in parallel!")
        
        # 处理并行结果，更新reward_tensor
        for uid, indices, scores, rewards, win_info in group_results:
            if scores is None:  # 单个response的情况
                # 单个response也需要基于胜率评分
                for idx in indices:
                    # 计算有效response长度
                    data_item = data[idx]
                    prompt_length = data_item.batch['prompts'].shape[0]
                    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
                    # 只在最后一个有效token位置赋值（符合PPO标准）
                    if valid_response_length > 0:
                        reward_tensor[idx, valid_response_length - 1] = 1.0
            else:
                # 更新reward tensor（只在最后一个有效token位置赋值）
                for i, idx in enumerate(indices):
                    # 计算有效response长度
                    data_item = data[idx]
                    prompt_length = data_item.batch['prompts'].shape[0]
                    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
                    # 只在最后一个有效token位置赋值（符合PPO标准）
                    if valid_response_length > 0:
                        reward_tensor[idx, valid_response_length - 1] = rewards[i]
                    
                    # 记录额外信息
                    reward_extra_info["tournament_score"].append(scores[i])
                    reward_extra_info["reward_value"].append(rewards[i])
                
                # 收集胜率信息
                if win_info: 
                    all_win_rates.append(win_info["win_rate"])
                    all_win_counts.append(win_info["win_count"])
                    all_total_counts.append(win_info["total_count"])
                    
                    # 记录胜率相关信息
                    for i, idx in enumerate(indices):
                        is_win = i in win_info["win_indices"]
                        reward_extra_info["answer_correct"].append(is_win)
                        reward_extra_info["extracted_answer"].append(
                            self._extract_answer_from_response(
                                self.tokenizer.decode(data[idx].batch["responses"][:data[idx].batch["attention_mask"][data[idx].batch["prompts"].shape[-1]:].sum()], skip_special_tokens=True)
                            )
                        )
        
        # 计算整体胜率统计
        if all_win_rates:
            overall_win_rate = sum(all_win_counts) / sum(all_total_counts) if sum(all_total_counts) > 0 else 0.0
            avg_group_win_rate = sum(all_win_rates) / len(all_win_rates)
            
            # 添加胜率统计到extra info（这些会被传递给metrics系统）
            reward_extra_info["win_rate"] = [overall_win_rate] * len(reward_tensor)
            reward_extra_info["group_win_rate"] = [avg_group_win_rate] * len(reward_tensor)
            reward_extra_info["win_total"] = [sum(all_win_counts)] * len(reward_tensor)
            reward_extra_info["total_samples"] = [sum(all_total_counts)] * len(reward_tensor)
            
            print(f"📊 Task Win Rate Metrics:")
            print(f"   - Overall Win Rate: {overall_win_rate:.3f} ({sum(all_win_counts)}/{sum(all_total_counts)})")
            print(f"   - Average Group Win Rate: {avg_group_win_rate:.3f}")
            print(f"   - Processed {len(all_win_rates)} groups with ground truth")
        
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
