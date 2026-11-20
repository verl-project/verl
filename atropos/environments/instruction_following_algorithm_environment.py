import json
import random
import re
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import Dataset, load_dataset
from langdetect import LangDetectException, detect
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# System prompt can be reused or adapted for instruction following tasks
system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


class IFConfig(BaseEnvConfig):
    dataset_name: str = Field("allenai/RLVR-IFeval", description="Default dataset name")
    dataset_config_name: Optional[str] = Field(
        None, description="Dataset config name, if any"
    )
    test_set_ratio: float = Field(
        0.05, description="The ratio of the selected dataset for testing"
    )


class InstructionFollowingEnv(BaseEnv):
    env_config_cls = IFConfig

    def __init__(
        self,
        config: IFConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []

    @classmethod
    def config_init(
        self,
    ) -> Tuple[IFConfig, List[APIServerConfig]]:
        # Configuration for the Instruction Following Environment
        env_config = IFConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=32,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=500,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 15,
            inference_weight=1.0,
            wandb_name="instruction_following_rlvr_ifeval",  # Specific WandB project name
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            dataset_name="allenai/RLVR-IFeval",  # Default dataset
            dataset_config_name=None,  # RLVR-IFeval doesn't have a specific config name, uses 'default'
            test_set_ratio=0.05,  # The ratio of the selelcted dataset in %
        )
        # Server configurations can be similar to SingleToolCallingEnv or adjusted
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]
        return env_config, server_configs

    async def create_rollout_table(self, wandb_metrics):
        # Logs rollouts to a WandB table for visualization
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score", "constraint_details"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    # item[0] is model output, item[1] is score, item[2] is constraint info
                    table.add_data(item[0], item[1], json.dumps(item[2]))
            wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        # Logs metrics to WandB
        if wandb_metrics is None:
            wandb_metrics = dict()

        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass  # Buffer might be empty

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """
        Load and preprocess the dataset for instruction following.
        This method is specifically tailored to process 'allenai/RLVR-IFeval' dataset structure.
        Each item from RLVR-IFeval is expected to have:
        - 'messages': A list of dictionaries, e.g., [{'role': 'user', 'content': 'instruction...'}]
        - 'ground_truth': A JSON string containing 'func_name' and arguments for the verifier.

        The method will parse these to produce items for the environment with:
        - 'prompt': The user's instruction string.
        - 'func_name': The string name of the verifier function.
        - 'args': A dictionary of arguments for that verifier function.
        """  # noqa: E501
        dataset_name = getattr(self.config, "dataset_name", "allenai/RLVR-IFeval")
        dataset_config_name = getattr(
            self.config, "dataset_config_name", None
        )  # Default is None, RLVR-IFeval has no sub-config

        processed_items = []
        try:
            print(
                f"Attempting to load dataset: {dataset_name}, "
                f"config: {dataset_config_name if dataset_config_name else 'default'}"
            )
            if dataset_config_name:
                full_dataset_raw = load_dataset(
                    dataset_name,
                    dataset_config_name,
                    split="train",
                    trust_remote_code=True,
                )
            else:
                full_dataset_raw = load_dataset(
                    dataset_name, split="train", trust_remote_code=True
                )
            print(
                f"Successfully loaded raw dataset. Number of items: {len(full_dataset_raw)}"
            )

            for i, item in enumerate(full_dataset_raw):
                # Extract prompt from 'messages' field
                item_messages = item.get("messages")
                if (
                    not item_messages
                    or not isinstance(item_messages, list)
                    or len(item_messages) == 0
                ):
                    print(
                        f"Warning: Item {i} has invalid or empty 'messages' field. Skipping. Item: {item}"
                    )
                    continue
                # Assuming the relevant prompt is the content of the first message in the list
                # (or last, if multiple user messages were possible, but IFEval is typically single user instruction)
                prompt_text = item_messages[0].get("content")
                if not prompt_text:
                    print(
                        f"Warning: Item {i} '{item_messages[0]}' has no content. Skipping."
                    )
                    continue

                # Get the ground_truth JSON string
                ground_truth_json_str = item.get("ground_truth")
                if not ground_truth_json_str or not isinstance(
                    ground_truth_json_str, str
                ):
                    print(
                        f"Warning: Item {i} missing or has invalid 'ground_truth' string. Skipping. "
                        f"Prompt: {prompt_text[:50]}..."
                    )
                    continue

                try:
                    parsed_gt = json.loads(ground_truth_json_str)
                    if not isinstance(parsed_gt, dict):
                        raise ValueError("Parsed ground_truth is not a dictionary.")
                except (json.JSONDecodeError, ValueError) as e:
                    print(
                        f"Warning: Could not parse 'ground_truth' JSON for item {i}. Error: {e}. "
                        f"GT String: '{ground_truth_json_str}'. Prompt: {prompt_text[:50]}... Skipping."
                    )
                    continue

                func_name_from_gt = parsed_gt.get("func_name")
                if not func_name_from_gt:
                    print(
                        f"Warning: Item {i} parsed 'ground_truth' has no 'func_name'. GT: {parsed_gt}. "
                        f"Prompt: {prompt_text[:50]}... Skipping."
                    )
                    continue

                if func_name_from_gt not in IF_FUNCTIONS_MAP:
                    print(
                        f"Warning: func_name '{func_name_from_gt}' in item {i} not in IF_FUNCTIONS_MAP. "
                        f"Prompt: {prompt_text[:50]}... Skipping."
                    )
                    continue

                # Prepare args for the verifier function: remove func_name and keep others.
                # Verifier functions will only use args they expect.
                args_dict = {
                    k: v
                    for k, v in parsed_gt.items()
                    if k != "func_name" and v is not None
                }

                processed_items.append(
                    {
                        "prompt": prompt_text,
                        "func_name": func_name_from_gt,
                        "args": args_dict,
                        "original_constraints_for_logging": str(
                            item.get("constraint", "")
                        ),  # For logging, from RLVR-IFeval structure
                        "expected_response_for_logging": "",
                    }
                )

            if not processed_items:
                print(
                    "Warning: No items successfully processed from the dataset. "
                    "Check dataset format/content or parsing logic."
                )
                raise ValueError(
                    "Dataset processing resulted in no valid items for RLVR-IFeval. Cannot proceed without data."
                )

            full_dataset = Dataset.from_list(processed_items)
            print(
                f"Successfully processed {len(full_dataset)} items from dataset '{dataset_name}'."
            )

        except Exception as e:
            # This block is a fallback if the primary dataset loading/processing fails catastrophically.
            # For RLVR-IFeval, a failure here suggests issues with Hugging Face access,
            # dataset integrity, or fundamental code errors.
            print(
                f"CRITICAL: Failed to load or process primary dataset '{dataset_name}': {e}. "
                f"Using a DUMMY dataset as fallback."
            )
            dummy_data_for_fallback = [
                {
                    "prompt": "Dummy Instruction 1: Ensure your response contains the word 'example'.",
                    "func_name": "verify_keywords",
                    "args": {"keyword_list": ["example"]},
                    "original_constraints_for_logging": "Contains 'example'",
                    "expected_response_for_logging": "This is an example response.",
                },
                {
                    "prompt": "Dummy Instruction 2: Output a valid JSON with key 'data' and value 'test'.",
                    "func_name": "validate_json_format",
                    "args": {},
                    "original_constraints_for_logging": "Output valid JSON.",
                    "expected_response_for_logging": '{\\"data\\": \\"test\\"}',
                },
            ]
            full_dataset = Dataset.from_list(dummy_data_for_fallback)
            print(
                f"Initialized with DUMMY dataset of {len(full_dataset)} items "
                f"due to previous errors."
            )

        full_dataset = full_dataset.shuffle(seed=42)

        actual_test_size = self.config.test_set_ratio  # Read from config
        num_items = len(full_dataset)

        if num_items == 0:
            print("ERROR: Dataset is empty. Cannot create train/test split.")
            self.train = Dataset.from_list([])
            self.test = Dataset.from_list([])
        elif num_items == 1:
            print("Warning: Dataset has only 1 item. Using it for both train and test.")
            self.train = full_dataset
            self.test = full_dataset
        else:  # num_items > 1
            # Ensure test_size results in at least 1 item for test set if possible, but not more than train set
            if num_items < 5:  # For 2,3,4 items, make test size 1
                min_test_items = 1
            else:  # For 5+ items, 20% is fine
                min_test_items = max(1, int(num_items * actual_test_size))

            # Ensure test split is not too large, e.g. not more than 50% unless dataset is very small
            # And ensure train always has at least one item if num_items > 1
            calculated_test_size = min_test_items / num_items
            if (
                calculated_test_size >= 0.5 and num_items > 2
            ):  # If test is 50% or more and we have 3+ items
                calculated_test_size = (
                    num_items - 1
                ) / num_items  # Make train have at least 1

            split_dataset = full_dataset.train_test_split(
                test_size=calculated_test_size, seed=42
            )
            self.train = split_dataset["train"]
            self.test = split_dataset["test"]
            # Final check for empty train/test after split, should not happen with logic above if num_items > 0
            if len(self.train) == 0 and len(self.test) > 0:
                print(
                    "Warning: Train set empty after split, test set has data. "
                    "This is unusual. Swapping."
                )
                self.train = self.test  # Fallback, though indicates issue
            elif len(self.test) == 0 and len(self.train) > 0:
                print(
                    "Warning: Test set empty after split, train set has data. "
                    "Using full train set for test as well."
                )
                self.test = self.train

        self.iter = 0
        print(
            f"Dataset setup complete. Train size: {len(self.train)}, Test size: {len(self.test)}"
        )

    async def _get_score_from_verifier(
        self, model_response_text: str, func_name: str, args: Dict
    ) -> float:
        """Helper to call verifier function and get a numerical score.
        Also enforces strict <think>...</think> formatting.
        """

        # 1. Count <think> and </think> tags
        num_think_open = len(re.findall(r"<think>", model_response_text, re.IGNORECASE))
        num_think_close = len(
            re.findall(r"</think>", model_response_text, re.IGNORECASE)
        )

        if not (num_think_open == 1 and num_think_close == 1):
            return 0.0

        # 3. Find the first occurrence of <think> and </think>
        try:
            think_open_match = re.search(r"<think>", model_response_text, re.IGNORECASE)
            think_close_match = re.search(
                r"</think>", model_response_text, re.IGNORECASE
            )

            # These should exist due to the count check, but access .start() and .end() safely
            idx_think_open = think_open_match.start()
            idx_think_close_start = think_close_match.start()
            idx_think_close_end = think_close_match.end()

        except AttributeError:
            return 0.0

        # 4. If <think> appears after </think>, malformed.
        if idx_think_open >= idx_think_close_start:
            # print(f"DEBUG: <think> tag appears at or after </think> tag. Response: '{model_response_text[:200]}...'")
            return 0.0

        # 5. Extract text_to_verify (content after the first </think>)
        text_to_verify = model_response_text[idx_think_close_end:].strip()

        # 6. Check if text_to_verify itself contains any further <think> or </think> tags.
        if re.search(r"<think>", text_to_verify, re.IGNORECASE) or re.search(
            r"</think>", text_to_verify, re.IGNORECASE
        ):
            return 0.0

        # If all checks pass, proceed with verification using text_to_verify
        if func_name not in IF_FUNCTIONS_MAP:
            print(
                f"Warning: Verifier function '{func_name}' not found in IF_FUNCTIONS_MAP."
            )
            return 0.0

        verifier_func = IF_FUNCTIONS_MAP[func_name]

        raw_score = None
        try:
            if func_name == "validate_placeholders":
                raw_score = verifier_func(text_to_verify, N=args.get("N"))
            elif func_name == "verify_bullet_points":
                raw_score = verifier_func(text_to_verify, N=args.get("N"))
            elif func_name == "validate_repeat_prompt":
                raw_score = verifier_func(
                    text_to_verify, args.get("original_prompt", "")
                )
            else:
                from inspect import signature

                sig = signature(verifier_func)
                valid_params = [p for p in sig.parameters if p != "text"]
                filtered_args = {
                    k: args[k]
                    for k in valid_params
                    if k in args and args[k] is not None
                }
                raw_score = verifier_func(text_to_verify, **filtered_args)

        except LangDetectException:
            print(
                f"Warning: langdetect failed for func_name '{func_name}'. Scoring as incorrect."
            )
            return 0.0
        except ImportError as e:
            print(
                f"Warning: ImportError during verifier function '{func_name}': {e}. Check dependencies."
            )
            return 0.0
        except TypeError as e:
            print(
                f"TypeError calling {func_name} with args {args}: {e}. Text: '{text_to_verify[:100]}...'"
            )
            return 0.0
        except Exception as e:
            print(
                f"Unexpected error in verifier function '{func_name}' with args {args}: {e}"
            )
            return 0.0

        if isinstance(raw_score, tuple):
            score_value = float(raw_score[0])
        elif isinstance(raw_score, bool):
            score_value = float(raw_score)
        else:
            print(
                f"Warning: Verifier '{func_name}' returned unexpected type: {type(raw_score)}. Expected bool or tuple."
            )
            score_value = 0.0

        return score_value

    async def rollout_and_score_eval(self, test_item: Dict):
        # test_item is a dictionary from the test set, processed by setup()
        # It should contain 'prompt', 'func_name', 'args'
        instruction_prompt_text = test_item["prompt"]
        func_name = test_item["func_name"]
        args_for_verifier = test_item["args"]

        print(
            f"DEBUG: Entering rollout_and_score_eval. Prompt: {instruction_prompt_text[:200]}..."
        )  # DEBUG

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": instruction_prompt_text})

        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        print(
            f"DEBUG: Calling self.server.completion in rollout_and_score_eval. Prompt: {prompt_str[:200]}..."
        )  # DEBUG
        completion = await self.server.completion(
            prompt=prompt_str,
            n=1,
            max_tokens=self.config.max_token_length,  # Use config for max_tokens
            temperature=0.2,  # Temperature for eval, can be 0 for deterministic
            split="eval",
        )
        print("DEBUG: Received completion in rollout_and_score_eval.")  # DEBUG

        model_response_text = completion.choices[0].text
        score_value = await self._get_score_from_verifier(
            model_response_text, func_name, args_for_verifier
        )

        return (
            score_value  # Returns 1.0 for correct, 0.0 for incorrect based on verifier
        )

    async def evaluate(self, *args, **kwargs):
        # Evaluates the model on the test set
        if not self.test or len(self.test) == 0:
            print("Warning: Test set is empty. Skipping evaluation.")
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        print(f"DEBUG: Starting evaluation. Test set size: {len(self.test)}")  # DEBUG
        eval_tasks = []
        for test_item_dict in self.test:  # self.test contains dicts after setup
            eval_tasks.append(self.rollout_and_score_eval(test_item_dict))

        scores = await tqdm_asyncio.gather(*eval_tasks)

        if not scores:  # If gather returns empty list
            percent_correct = 0.0
        else:
            percent_correct = sum(scores) / len(scores)

        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        print(f"Evaluation percent correct: {percent_correct}")

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        # item = (prompt_messages_tuple, answer_info_dict)
        # answer_info_dict = {"func_name": ..., "args": ...}
        print(f"DEBUG: Entering collect_trajectories. Item: {str(item)}")  # DEBUG
        prompt_messages_list = [dict(msg_fset) for msg_fset in item[0]]
        answer_info = item[1]

        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages_list, add_generation_prompt=True, tokenize=False
        )

        print(
            f"DEBUG: Calling self.server.completion in collect_trajectories. Prompt: {prompt_str[:200]}..."
        )  # DEBUG
        try:
            completions = await self.server.completion(
                prompt=prompt_str,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=0.8,  # Temperature for diverse responses during training rollouts
            )
            print(
                f"DEBUG: Received {len(completions.choices)} completions in collect_trajectories."
            )  # DEBUG
        except Exception as e:
            print(
                f"ERROR: Exception during self.server.completion in collect_trajectories: {e}"
            )  # DEBUG
            # Depending on the desired behavior, you might want to return None or raise the exception
            return None, []

        to_score_list = []
        for choice in completions.choices:
            trajectory_messages = [dict(msg_fset) for msg_fset in item[0]]  # Fresh copy
            trajectory_messages.append({"role": "assistant", "content": choice.text})
            to_score_list.append(
                (tuple(trajectory_messages), answer_info)
            )  # Pass answer_info

        if not to_score_list:
            return None, []

        print(
            f"DEBUG: Scoring {len(to_score_list)} trajectories in collect_trajectories."
        )  # DEBUG
        scored_data = await self.score(to_score_list)
        to_backlog = []  # Backlog not currently used but part of signature

        print(
            f"DEBUG: Exiting collect_trajectories. Scored data: {bool(scored_data)}"
        )  # DEBUG
        return scored_data, to_backlog

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def score(
        self, rollout_group_data: List[Tuple[tuple, Dict]]
    ) -> Optional[ScoredDataGroup]:
        # rollout_group_data is a list of (trajectory_messages_tuple, answer_info_dict)
        # answer_info_dict = {"func_name": ..., "args": ...}

        scores_container = ScoredDataGroup()
        scores_container["tokens"] = list()
        scores_container["masks"] = list()
        scores_container["scores"] = list()

        if not rollout_group_data:
            return None

        # The 'answer_info' (func_name, args) is consistent for all items in this group,
        # as it comes from the same initial prompt.
        # We can extract it once if needed, but it's passed per item.

        random.shuffle(rollout_group_data)  # Shuffle to avoid bias

        for trajectory_item in rollout_group_data:
            full_trajectory_messages = trajectory_item[0]
            answer_info = trajectory_item[1]  # {"func_name": ..., "args": ...}

            model_response_text = full_trajectory_messages[-1]["content"]
            func_name = answer_info["func_name"]
            args_for_verifier = answer_info["args"]

            # Get score (1.0 for correct, 0.0 for incorrect from verifier)
            score_value = await self._get_score_from_verifier(
                model_response_text, func_name, args_for_verifier
            )

            # Map to reward: 1.0 for correct, 0 for incorrect
            reward = 1.0 if score_value == 1.0 else 0

            # Tokenize the conversation for PPO training
            # Ensure full_trajectory_messages is a list of dicts
            list_of_dicts_trajectory = [dict(msg) for msg in full_trajectory_messages]
            out_dict = tokenize_for_trainer(self.tokenizer, list_of_dicts_trajectory)
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Filter out examples with insufficient context (too short)
            if (
                sum(1 for m_val in masks if m_val != -100) < 10
            ):  # At least 10 non-masked tokens
                continue

            scores_container["tokens"].append(tokens)
            scores_container["masks"].append(masks)
            scores_container["scores"].append(reward)

            # Stop if we have enough examples for the group
            if len(scores_container["tokens"]) >= self.config.group_size:
                break

        if not scores_container["tokens"]:  # No valid items collected
            return None

        # Record success rate for logging (based on positive rewards)
        for rwd in scores_container["scores"]:
            self.percent_correct_buffer.append(
                max(0, rwd)
            )  # If reward is 1.0, it's a success

        # Optional: Apply length penalty if all responses are correct (reward 1.0)
        # This logic is from SingleToolCallingEnv, may need adjustment for IF
        if all(s == 1.0 for s in scores_container["scores"]):
            token_lengths = [len(t) for t in scores_container["tokens"]]
            if not token_lengths or max(token_lengths) == 0:
                return scores_container  # Avoid division by zero, or if all empty

            max_allowed_length = self.config.max_token_length
            # Threshold can be adjusted, e.g., 75% of max_token_length
            length_threshold = max_allowed_length * 0.75

            penalized_scores = []
            for i, length in enumerate(token_lengths):
                original_score = scores_container["scores"][i]  # Should be 1.0 here
                if length <= length_threshold:
                    penalized_scores.append(original_score)
                else:
                    # Linear penalty for exceeding threshold
                    penalty_factor = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    penalty_factor = min(penalty_factor, 1.0)  # Cap penalty factor at 1
                    # Penalized score scales from original_score down to original_score * (1-1) = 0
                    penalized_scores.append(original_score * (1.0 - penalty_factor))
            scores_container["scores"] = penalized_scores

        # If all scores are identical after potential penalties, no learning signal
        if (
            len(set(scores_container["scores"])) <= 1
            and len(scores_container["scores"]) > 1
        ):
            return None  # Avoid sending data with no variance

        return scores_container

    async def get_next_item(self) -> Item:
        # Fetches the next preprocessed item from the training set
        if not self.train or len(self.train) == 0:
            # This case should be handled by setup, but as a safeguard:
            print("Error: Training data is empty in get_next_item.")
            # Return a dummy item to prevent crashes, though this indicates a setup issue
            dummy_prompt_messages = (
                frozenset({"role": "system", "content": system_prompt}.items()),
                frozenset(
                    {"role": "user", "content": "Dummy instruction: say hello."}.items()
                ),
            )
            dummy_answer_info = {
                "func_name": "verify_keywords",
                "args": {"keyword_list": ["hello"]},
            }
            return (dummy_prompt_messages, dummy_answer_info)

        raw_item = self.train[self.iter % len(self.train)]  # raw_item is a dict
        self.iter += 1

        instruction_prompt_text = raw_item["prompt"]

        # Construct messages for the LLM (prompt tuple part of Item)
        # Using frozenset as required by BaseEnv's Item type hint
        prompt_messages_tuple = (
            frozenset({"role": "system", "content": system_prompt}.items()),
            frozenset({"role": "user", "content": instruction_prompt_text}.items()),
        )

        # The "answer" part for scoring purposes (answer_info dict part of Item)
        answer_info = {
            "func_name": raw_item["func_name"],
            "args": raw_item["args"],
            # Optionally include other info for logging/debugging if needed from raw_item
            "original_constraints_for_logging": raw_item.get(
                "original_constraints", ""
            ),
            "expected_response_for_logging": raw_item.get(
                "expected_response_for_logging", ""
            ),
        }

        return (prompt_messages_tuple, answer_info)

    async def add_rollouts_for_wandb(
        self,
        scored_data: ScoredDataGroup,  # Assuming single ScoredDataGroup here
        item: Item = None,  # item = (prompt_messages_tuple, answer_info_dict)
    ):
        # Saves rollouts for WandB logging
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:  # Log all rollouts in the group
            num_keep = len(scored_data["tokens"])

        # item[1] is the answer_info_dict containing func_name and args
        constraint_details_for_log = item[1] if item else {}

        rollout_batch = []
        for i in range(min(num_keep, len(scored_data["tokens"]))):
            decoded_text = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=False
            )
            score = scored_data["scores"][i]
            rollout_batch.append((decoded_text, score, constraint_details_for_log))

        self.rollouts_for_wandb.append(rollout_batch)

        # Limit the number of rollout groups stored
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)


# ----- IFEval Verifier Functions and Map -----
# adapted from https://github.com/allenai/open-instruct/blob/main/scripts/eval_constraints/if_functions.py


# Helper function for verify_keyword_frequency, moved import re to top level
def _extract_words(text: str) -> List[str]:
    return re.findall(r"\\b\\w+\\b", text.lower())


# include keywords: Include keywords {keyword1}, {keyword2} in your response
def verify_keywords(text: str, keyword_list: List[str]) -> bool:
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


# Keyword Frequency: In your response, the word {word} should appear {N} times.
def verify_keyword_frequency(text: str, word: str, N: int) -> bool:
    text_lower = text.lower()
    keyword_lower = word.lower()
    words = _extract_words(text_lower)
    actual_count = sum(1 for w in words if w == keyword_lower)
    return actual_count == N


# Forbidden Words: Do not include keywords {forbidden words} in the response.
def validate_forbidden_words(text: str, forbidden_words: List[str]) -> bool:
    text_lower = text.lower()
    return not any(word.lower() in text_lower for word in forbidden_words)


# Letter Frequency : In your response, the letter {letter} should appear {N} times.
def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    if len(letter) != 1:
        # This should ideally raise ValueError, but for RL reward, return False
        return False
    actual_count = text.count(letter)
    return actual_count == N


# Response Language: Your ENTIRE response should be in {language}, no other language is allowed.
def validate_response_language(text: str, language: str) -> bool:
    try:
        detected_language = detect(text)
        return detected_language == language
    except LangDetectException:  # Catching specific exception from detect()
        print(
            f"Warning: langdetect failed to detect language for text: '{text[:50]}...'"
        )
        return False


# Number Paragraphs: Your response should contain {N} paragraphs. You separate paragraphs using the markdown divider:
# * * *
def verify_paragraph_count(text: str, N: int) -> bool:
    def clean_text(txt: str) -> str:
        return "\\n".join(line.strip() for line in txt.splitlines()).strip()

    cleaned_text = clean_text(text)
    # Paragraphs are separated by '* * *'. N dividers mean N+1 paragraphs.
    # If the text IS paragraphs, then N paragraphs will have N-1 dividers.
    # The prompt implies N paragraphs are expected.
    # If N=1, 0 dividers. If N=2, 1 divider. So, count of parts = N.
    paragraphs = cleaned_text.split("* * *")
    actual_count = len(paragraphs)
    # Verify each split resulted in non-empty content, if text itself is not empty
    if not cleaned_text and N == 0:
        return True  # 0 paragraphs, empty text
    if not cleaned_text and N > 0:
        return False

    # This check might be too strict if empty paragraphs are allowed by the constraint definition
    # If "paragraph" implies non-empty content:
    # return len(valid_paragraphs) == N and actual_count == N
    # If constraint just means N segments separated by dividers:
    return actual_count == N


# Number Words: Answer with at least / around / at most {N} words
def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)  # For 'around'

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    return False


# Number Sentences: Answer with at least / around / at most {N} sentences.
def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    # Basic sentence splitting, might need more robust NLP for complex cases
    sentences = re.split(
        r"(?<![a-zA-Z0-9_]\.[a-zA-Z0-9_]\.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s",
        text.strip(),
    )
    # Filter out empty strings that might result from splitting
    sentences = [s for s in sentences if s.strip()]
    actual_count = len(sentences)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        # "around" for sentences usually means exact or +/-1
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    return False


# Number Paragraphs + First Word in i-th Paragraph
def validate_paragraphs(text: str, N: int, first_word: str, i: int) -> bool:
    # Paragraphs separated by double line breaks
    paragraphs = text.split("\\n\\n")
    if len(paragraphs) != N:
        return False
    # i is 1-indexed for paragraph number
    if not (1 <= i <= len(paragraphs)):
        return False
    # Check first word of the i-th paragraph
    # .strip() to handle leading/trailing whitespace in paragraph
    # .split()[0] to get the first word
    try:
        actual_first_word = paragraphs[i - 1].strip().split()[0]
        # Case-insensitive comparison for first_word might be more robust
        return actual_first_word.lower() == first_word.lower()
    except IndexError:  # Handles empty paragraph or paragraph without words
        return False


# Postscript: At the end of your response, please explicitly add a postscript starting with {postscript marker}
def verify_postscript(text: str, postscript_marker: str) -> bool:
    marker_index = text.rfind(postscript_marker)  # Find last occurrence
    if marker_index == -1:
        return False
    # Check if it's truly a postscript (i.e., near the end, and has content after marker)
    # This interpretation: marker exists, and something follows it OR it's at the very end.
    # The original IFEval might have a stricter definition (e.g. specific distance from end)
    # A simple check: marker is present and the text from marker to end is mostly the postscript.
    # For RL, simpler: marker is present and is not just prefix of a word.
    # Test if the marker is at a word boundary if it's not the start of the string
    if (
        marker_index > 0
        and text[marker_index - 1].isalnum()
        and postscript_marker[0].isalnum()
    ):
        # Avoid matching mid-word, e.g. "script" in "postscript" if marker is "script"
        # This check is heuristic. A regex with word boundaries might be better.
        pass  # Heuristic, might need refinement

    # Check if content exists after marker, or if marker itself is the end
    remaining_text = text[marker_index:].strip()
    return len(remaining_text) >= len(postscript_marker.strip())


# Number Placeholder: The response must contain at least {N} placeholders ... [address].
def validate_placeholders(text: str, N: int) -> Tuple[bool, List[str]]:
    placeholders_found = re.findall(r"\\[(.*?)\\]", text)  # Matches [content]
    return len(placeholders_found) >= N, placeholders_found


# Number Bullets: Your answer must contain exactly {N} bullet points. * This is a point.
def verify_bullet_points(
    text: str, N: int
) -> bool:  # Original had tuple[bool,str] in doc, bool in code
    lines = text.splitlines()
    # Markdown bullets usually start with '*', '-', or '+' followed by a space.
    bullet_points = [
        line.strip()
        for line in lines
        if re.match(r"^(\\s*)[\\*\\-\\+]\\s+", line.strip())
    ]
    return len(bullet_points) == N


# Title: Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.
def validate_title(text: str) -> bool:
    return bool(re.search(r"<<(.*?)>>", text))


# Choose: From Answer with one of the following options: {options}
def validate_choice(text: str, options: List[str]) -> bool:
    # Assuming 'text' should be one of the 'options' exactly, or contain one of them.
    # The original prompt "Answer with one of..." implies the response *is* one of the options.
    # Case-insensitive comparison for robustness.
    text_cleaned = text.strip().lower()
    return any(text_cleaned == opt.strip().lower() for opt in options)


# Minimum Number Highlighted Section: Highlight at least {N} sections ... *highlighted section*
def validate_highlighted_sections(text: str, N: int) -> bool:
    # Markdown italics/bold *highlight* or **highlight**
    # This regex looks for single asterisks: *content*
    matches = re.findall(
        r"\*(.*?)(?<!\\)\*", text  # Ensure the closing * is not escaped
    )
    # Filter out empty matches or those that are just whitespace if needed.
    # matches = [m for m in matches if m.strip()]
    return len(matches) >= N


# Multiple Sections: Your response must have {N} sections. Mark ... with {section splitter} X.
def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    # Example: section_splitter = "Section" -> "Section 1", "Section 2"
    # This implies the splitter itself might include a number or be just the prefix.
    # If splitter is "---", then text.split("---").
    # If splitter is "Topic X:", this is more complex.
    # Assuming a simple string split is intended by the original IFEval function.
    # The prompt phrasing "Mark the beginning of each section with {section splitter} X"
    # suggests counting occurrences of the splitter pattern.

    # If section_splitter is like "SECTION", we'd look for "SECTION 1", "SECTION 2", ...
    # This is hard to generalize perfectly without knowing how IFEval defines 'X'.
    # Simplest: count occurrences of the base splitter string.
    # sections = text.split(section_splitter)
    # num_sections = len(sections) -1 if sections[0].strip() == "" else len(sections)
    # A slightly more robust way for "Splitter X":
    # Count how many times "splitter" followed by something (like a number) appears.
    # Example: if splitter is "Chapter", we look for "Chapter 1", "Chapter ...".
    # This regex is a placeholder for more specific logic IFEval might use.

    # Let's use a simple count of the splitter string for now.
    # This might need to be adjusted based on IFEval's exact expectation for "X".
    # For "SECTION 1.", "SECTION 2.", if splitter is "SECTION ":
    actual_sections = len(
        re.findall(
            re.escape(section_splitter) + r"\\s*\\d*[:\\.\\s]", text, re.IGNORECASE
        )
    )

    # If N=0 and no splitters, it's true. If N>0 and no splitters, false.
    if N == 0:
        return actual_sections == 0
    return actual_sections == N


# JSON Format : Entire output should be wrapped in JSON format.
def validate_json_format(text: str) -> bool:
    try:
        json.loads(text.strip())  # .strip() to handle leading/trailing whitespace
        return True
    except json.JSONDecodeError:
        return False


# Repeat Prompt: First, repeat the request without change, then give your answer
def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    # Normalize whitespace for comparison robustness
    text_norm = " ".join(text.strip().split())
    original_prompt_norm = " ".join(original_prompt.strip().split())
    return text_norm.startswith(original_prompt_norm)


# Two Responses: Give two different responses. Separated by 6 asterisk symbols: ******.
def validate_two_responses(text: str) -> bool:
    if text.count("******") == 1:
        parts = text.split("******")
        if len(parts) == 2:
            # Check if parts are non-empty and different
            resp1 = parts[0].strip()
            resp2 = parts[1].strip()
            return bool(resp1 and resp2 and resp1 != resp2)
    return False


# All Uppercase: Your entire response should be in English, capital letters only.
def validate_uppercase(text: str) -> bool:
    # Check if it has letters and all letters are uppercase
    if not any(
        c.isalpha() for c in text
    ):  # No letters, technically not violating "all capital"
        return True  # Or False, depending on interpretation of "response"
    return text == text.upper()


# All Lowercase: Your entire response should be in English, and in all lowercase letters.
def validate_lowercase(text: str) -> bool:
    if not any(c.isalpha() for c in text):
        return True
    return text == text.lower()


# Frequency of All-capital Words
def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    # Words with all capital letters, e.g., "NASA", "AI". Min 2 chars to be a "word".
    capital_words = re.findall(r"\\b[A-Z]{2,}\\b", text)
    actual_count = len(capital_words)
    tolerance = max(round(N * 0.1), 1)  # For 'around'

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif (
        quantifier == "around"
    ):  # Using exact for 'around' with capital words unless specified
        return abs(actual_count - N) <= tolerance  # Or just actual_count == N
    return False


# End Checker: Finish your response with this exact phrase {end phrase}.
def validate_end(text: str, end_phrase: str) -> bool:
    # Normalize whitespace at the end of text for robustness
    return text.strip().endswith(end_phrase.strip())


# Quotation: Wrap your entire response with double quotation marks.
def validate_quotation(text: str) -> bool:
    stripped_text = text.strip()
    return stripped_text.startswith('"') and stripped_text.endswith('"')


# No Commas: In your entire response, refrain from the use of any commas.
def validate_no_commas(text: str) -> bool:
    return "," not in text


IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}

if __name__ == "__main__":
    InstructionFollowingEnv.cli()
