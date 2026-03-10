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

import inspect
import logging
import os

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score
from rdkit import Chem
from verl.workers.rollout.vllm_rollout import VLLMBeamSearchManager
import re
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


def validate_smiles(pred_product, org_product):
    org_product = Chem.MolToSmiles(Chem.MolFromSmiles(org_product), canonical=True)
    try:
        pred_product = Chem.MolToSmiles(Chem.MolFromSmiles(pred_product), canonical=True)
        return True, pred_product == org_product
    except:
        return False, False

def create_input_batch(args):
    """Extract input_ids, tokenizer, and product from args tuple."""
    input_ids, tokenizer, product = args
    sequence = tokenizer.decode(input_ids, skip_special_tokens=True)
    # logger.warning(f"sequence: {sequence}")
    # pattern = r"<answer>(.*?)</answer>"
    # matches = re.findall(pattern, sequence)
    # if not matches or ("REACTANT" in matches[-1]):
    #     return False, None
    # match_content = matches[-1]    
    match_content = product.split("assistant")[-1].strip().replace("\n", "").replace("<think>", "").replace("</think>", "").replace("<|im_end|>", "")
    try:
        match_content = Chem.MolToSmiles(Chem.MolFromSmiles(match_content), canonical=True)
    except:
        return False, None

    messages = [
        {
            "role": "system",
            "content": "You are an expert chemist specializing in chemical reaction product prediction. Given the SMILES representations of the reactants, your task is to predict the SMILES representation of the product. The input consists of the reactant SMILES, and you should output only the product SMILES."
        },
        {
            "role": "user",
            "content": match_content
        }
    ]
    
    pyload = {
        "messages": messages,
        "product": product,
    }
    return True, pyload



@register("forward_rdkit_min")
class ForwardRDKitMinRewardManager(RewardManagerBase):
    """The reward manager."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
        placement_group=None,
        start_bundle_index=None,
        num_gpus=None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = True
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.tokenizer = tokenizer

        # Use forward_model config instead of reward_model to avoid duplicate initialization
        # This ensures only one VLLMBeamSearchManager is created per worker
        forward_model_config = config.reward.get("forward_model", {})
        model_path = forward_model_config.get("model_path", "")
        num_gpus = num_gpus if num_gpus is not None else forward_model_config.get("num_gpus", 1)

        # Initialize beam search manager; use placement_group to share global_pool when provided
        self.vllm_beamsearch_manager = VLLMBeamSearchManager(
            model_path,
            config.reward,
            num_gpus,
            placement_group=placement_group,
            start_bundle_index=start_bundle_index,
        )
        
        # self.vllm_beamsearch_manager.sleep()
    
    async def run_batch_forward(self, data: DataProto) -> DataProto:
        """Async batch forward to avoid blocking the event loop in Ray async actors."""
        await self.vllm_beamsearch_manager.wake_up_async()
        input_ids = data.batch["input_ids"]
        
        length = len(data)
        # Initialize is_valid for all items (use tensor for TensorDict compatibility)
        import torch
        is_valid_flags = torch.zeros(length, dtype=torch.bool)
        
        # Process sequentially to avoid Ray + multiprocessing conflicts
        # The original multiprocessing.Pool caused port conflicts with Ray actors
        results = []
        for i, data_item in enumerate(data):
            # Each data_item.non_tensor_batch["reward_model"]["ground_truth"] is the per-sample ground truth
            # Do NOT index batch-level numpy arrays with string keys (would cause IndexError)
            reward_model_info = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model_info["ground_truth"]
            args = (input_ids[i], self.tokenizer, ground_truth)
            result = create_input_batch(args)
            results.append(result)
        
        invalid_items = []
        input_batch = []
        logger.info(f"results: {results}")
        for i, (valid, pyload) in enumerate(results):
            if not valid:
                invalid_items.append(i)
                # Mark as invalid in tensor
                is_valid_flags[i] = False
                continue
            
            input_batch.append((i, pyload["messages"]))
        # logger.warning(f"input_batch: {len(input_batch)}")
        # logger.warning(f"input_batch: {input_batch[0]}")
        # Generate answers using beam search (async to avoid blocking event loop)
        answers = await self.vllm_beamsearch_manager.generate_async(input_batch)
        await self.vllm_beamsearch_manager.sleep_async()
        
        # Process answers and extract SMILES from model output
        # Initialize responses_forward as numpy array with dtype=object
        import numpy as np
        if "responses_forward" not in data.non_tensor_batch:
            data.non_tensor_batch["responses_forward"] = np.empty(length, dtype=object)
            for i in range(length):
                data.non_tensor_batch["responses_forward"][i] = None
        
        for idx, sequences in answers:
            # sequences is a list of beam search results
            # Extract SMILES by removing <think></think> tags
            processed_sequences = []
            for seq in sequences:
                # Remove <think>...</think> and extract SMILES
                # Pattern: <think>\n\n</think>\n\n + SMILES
                cleaned = re.split("</think>", seq)[-1].replace("<think>", "").replace("</think>", "").replace("\n", "").replace(" ","")
                cleaned = cleaned.strip()
                processed_sequences.append(cleaned)
            
            # Store responses_forward in non_tensor_batch (numpy array with dtype=object)
            data.non_tensor_batch["responses_forward"][idx] = processed_sequences
            is_valid_flags[idx] = True
        
        # Add is_valid to batch as a tensor
        data.batch["is_valid"] = is_valid_flags
        
        return data
    

    
    async def _compute_forward_score(self, data_source, solution_strs, ground_truth, extra_info, **extra_reward_kwargs):
        solution_strs = [solution_str.strip().split("<|im_start|>assistant")[-1].\
            replace("\n", "").replace("<think>", "").replace("</think>", "").replace("<|im_end|>", "") for solution_str in solution_strs]
        gt = Chem.MolToSmiles(Chem.MolFromSmiles(ground_truth), canonical=True)
        target_strs = []
        fpgen = AllChem.GetRDKitFPGenerator()
        for i,solution_str in enumerate(solution_strs):
            try:
                ans = Chem.MolToSmiles(Chem.MolFromSmiles(solution_str), canonical=True)
                ans_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(ans))
                target_strs.append(ans_fp)
            except:
                continue
        if not target_strs:
            # logger.warning(f"target_strs error: {solution_strs}")
            return {"score": 1000, "acc": 0.0, "tanimoto": 0.0}  # 1000 as tag
        
        ground_truth_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(gt))
        similarities = []
        for target_fp in target_strs:
            similarity = DataStructs.TanimotoSimilarity(ground_truth_fp, target_fp)
            similarities.append(similarity)

        best_tanimoto = min(similarities)
        logger.info(
            f"sample data_source={data_source} Tanimoto score (min): {best_tanimoto:.4f}, "
            f"per_beam: {[round(s, 4) for s in similarities]}"
        )

        if min(similarities) == 1.0:
            return {"score": 1.0, "acc": 1.0, "tanimoto": best_tanimoto}
        elif min(similarities) > 0.85:
            return {"score": min(similarities)-1.0, "acc": 0.0, "tanimoto": best_tanimoto}
        else:
            return {"score": -1.0, "acc": 0.0, "tanimoto": best_tanimoto}
        
        
    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        
        # Check if_valid from batch tensor (with fallback if key doesn't exist)
        if "is_valid" in data_item.batch.keys():
            if_valid = data_item.batch["is_valid"].item() if hasattr(data_item.batch["is_valid"], 'item') else data_item.batch["is_valid"]
            if not if_valid:
                return {"reward_score": -2.0, "reward_extra_info": {"score": -2.0, "acc": 0.0, "tanimoto": -1.0}}
        else:
            # logger.warning("is_valid key not found in batch, assuming valid=True (run_batch_forward may not have been called)")
            if_valid = True

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        # Get responses_forward from non_tensor_batch (stored as list)
        if "responses_forward" not in data_item.non_tensor_batch:
            # logger.error("responses_forward not found in non_tensor_batch, run_batch_forward was not called properly")
            return {"reward_score": -2.0, "reward_extra_info": {"score": -2.0, "acc": 0.0, "tanimoto": -1.0}}
        
        response_strs = data_item.non_tensor_batch["responses_forward"]
        
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        result = await self._compute_forward_score(
            data_source=data_source,
            solution_strs=response_strs,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **extra_reward_kwargs,
        )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["score"] = score
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
