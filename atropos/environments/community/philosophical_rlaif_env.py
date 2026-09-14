import asyncio
import random
import re  # Added import for regular expressions
from typing import Dict, List, Optional, Tuple

import wandb  # type: ignore
from datasets import load_dataset  # type: ignore
from pydantic import Field

from atroposlib.envs.base import BaseEnvConfig  # Ensure BaseEnvConfig is imported
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Judge system prompt can remain somewhat generic for deep thinking
judge_system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)
judge_system_prompt += (
    "You are allocated a maximum of 4096 tokens for your thoughts and response, "
    "please strive to use less for thinking.\n\n"
    "You will then provide your preferred response like this: \\boxed{A or B}.\n\n"
    "It is important that you provide your answer in the correct format.\n"
    "If you do not, you will not receive credit for your answer.\n\n"
    "So please end your answer with \\boxed{A or B}"
)

# New preference string for philosophical depth
philosophical_preference_string = (
    "I prefer responses that demonstrate deep philosophical inquiry. The ideal response "
    "explores multiple perspectives, questions underlying assumptions, uses nuanced language, "
    "and encourages further critical thinking. Avoid superficial answers; aim for depth "
    "and intellectual stimulation. Responses that identify underlying principles, paradoxes, "
    "or ethical considerations are highly valued. Clarity and cogency in expressing complex "
    "ideas are also important."
)

RLAIF_user_prompt_format_str = (
    "[Conversation]\n{conversation}\n\n"
    "[Response A]\n{response_a}\n\n"
    "[Response B]\n{response_b}\n\n"
    "[END]\n\n"
    "Here's a conversation that has two responses. Please pick the one that is more aligned "
    "with the following desired qualities for a response:\n{rl_preference_string}\n\n"
    "The goal is to select the response that, if used for training an AI, would better steer "
    "it towards generating text with these qualities, even if neither response is a perfect example.\n"
    "Please do your best to evaluate which response better embodies or tends towards the "
    "described philosophical depth and nuance.\n\n"
    "Go ahead and think through it, then give me your answer with \\boxed{{A or B}}."
)


class PhilosophicalRLAIFConfig(
    BaseEnvConfig
):  # Custom config if needed, inherits BaseEnvConfig
    # Add any specific configurations here if needed in the future
    # For now, we rely on BaseEnvConfig defaults and overrides in config_init
    judge_model_name: str = Field(
        default="gpt-3.5-turbo", description="Model to use for judging preferences."
    )
    generator_model_name: str = Field(
        default="gpt-3.5-turbo",
        description="Model to use for generating initial responses.",
    )
    judge_max_tokens: int = Field(
        default=2048, description="Max tokens for judge response."
    )
    generator_max_tokens: int = Field(
        default=1024, description="Max tokens for generator response."
    )


class PhilosophicalRLAIFEnv(BaseEnv):
    name = "philosophical_rlaif"
    env_config_cls = PhilosophicalRLAIFConfig

    def __init__(
        self,
        config: PhilosophicalRLAIFConfig,  # Use the new config
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.eval_metrics = list()  # Kept for consistency, though eval is basic
        self.judgement_strings_buffer: List[Tuple[str, str, str]] = list()
        self.preference_scores_buffer: List[float] = list()
        self.train_dataset = None  # Initialize attribute

    @classmethod
    def config_init(cls) -> Tuple[PhilosophicalRLAIFConfig, List[APIServerConfig]]:
        env_config = PhilosophicalRLAIFConfig(  # Use the new config class
            tokenizer_name="cl100k_base",  # Changed from gpt2
            group_size=2,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=20,
            batch_size=4,
            steps_per_eval=10,
            max_token_length=4096,  # Increased from 3072
            score_buffer_size=4,
            wandb_name="philosophical_rlaif_shortgen",  # New wandb name for this attempt
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            judge_model_name="gpt-3.5-turbo",
            generator_model_name="gpt-3.5-turbo",
            judge_max_tokens=1024,  # Reduced as inputs will be shorter
            generator_max_tokens=768,  # Increased from 256
            data_path_to_save_groups="./philosophical_rlaif_rollouts.jsonl",
            ensure_scores_are_not_same=False,  # More lenient for ties
        )
        # We'll use one server config, assuming generator and judge models are on the same API endpoint
        # The actual model used for each call can be specified in the chat_completion call if needed,
        # or we assume the server config's model_name is used if not overridden.
        # For this example, we'll use the same model for both roles from the config.
        server_configs = [
            APIServerConfig(
                model_name=env_config.judge_model_name,  # Default model for the server
                base_url=None,  # Use OpenAI default
                api_key=None,  # Expect API key from environment (.env file)
                num_requests_for_eval=32,  # For potential eval calls
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.judgement_strings_buffer:
            table = wandb.Table(columns=["resp_a", "resp_b", "sample_judgement"])
            for item in self.judgement_strings_buffer:
                table.add_data(item[0], item[1], item[2])
            self.judgement_strings_buffer.clear()
            wandb_metrics["train/judgement_table"] = table
            print("Logged judgement table to W&B.")

        if self.preference_scores_buffer:
            avg_pref_score = sum(self.preference_scores_buffer) / len(
                self.preference_scores_buffer
            )
            wandb_metrics["train/avg_normalized_preference_score"] = avg_pref_score
            print(
                f"Average normalized preference score for batch: {avg_pref_score:.3f} "
                f"(over {len(self.preference_scores_buffer)} scores)"
            )
            self.preference_scores_buffer.clear()

        # Log other eval metrics if any
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = list()  # Clear after logging

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Using a small subset for faster loading during tests.
        # In a real scenario, use the full split or a larger subset.
        try:
            self.train_dataset = load_dataset(
                "allenai/WildChat", split="train[:1000]"
            )  # Smaller subset
            self.iter = 0
            print(
                f"PhilosophicalRLAIFEnv initialized with {len(self.train_dataset)} "
                "training examples from WildChat."
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.train_dataset = []  # Ensure it's an empty list on failure
            self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval_item(self, eval_item) -> float:
        # Simplified eval: generate two responses, have judge pick, score 1 if first is picked, 0 otherwise.
        # This is a placeholder, proper RLAIF eval is more complex.
        original_chat = [
            dict(msg) for msg in eval_item if msg["role"] != "assistant"
        ]  # Get initial prompt
        if not original_chat:
            return 0.0

        # Generate two responses (A and B)
        completions = await self.server.chat_completion(
            messages=original_chat,
            n=2,
            max_tokens=self.config.generator_max_tokens,
            temperature=0.7,
            model=self.config.generator_model_name,  # Specify generator model
            split="eval",
        )
        if len(completions.choices) < 2:
            return 0.0  # Not enough responses to compare

        response_a_content = completions.choices[0].message.content
        response_b_content = completions.choices[1].message.content

        conversation_str = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in original_chat]
        )

        judge_prompt_content = RLAIF_user_prompt_format_str.format(
            conversation=conversation_str,
            response_a=response_a_content,
            response_b=response_b_content,
            rl_preference_string=philosophical_preference_string,
        )

        judge_response = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": judge_prompt_content},
            ],
            n=1,
            max_tokens=self.config.judge_max_tokens,
            temperature=0.0,  # Judge should be deterministic
            model=self.config.judge_model_name,  # Specify judge model
            split="eval",
        )

        chosen_val_match = re.search(
            r"\boxed{(A|B)}", judge_response.choices[0].message.content
        )
        if chosen_val_match:
            return (
                1.0 if chosen_val_match.group(1) == "A" else 0.0
            )  # Arbitrary: score 1 if A is chosen
        return 0.0  # No clear choice or format error

    async def evaluate(self, *args, **kwargs):
        if (
            not self.train_dataset or len(self.train_dataset) == 0
        ):  # Use train_dataset for eval examples for simplicity
            print("No evaluation data available (using subset of train_dataset).")
            return

        num_eval_samples = min(
            10, len(self.train_dataset)
        )  # Evaluate on a small sample
        eval_samples = random.sample(list(self.train_dataset), num_eval_samples)

        print(
            f"Evaluating on {num_eval_samples} samples from WildChat for philosophical preference..."
        )
        total_score = 0

        for i, sample in enumerate(eval_samples):
            print(f"Eval sample {i+1}/{num_eval_samples}")
            # The 'conversation' field in WildChat is a list of dicts
            eval_item_chat = sample.get("conversation", [])
            if not eval_item_chat:
                continue
            score = await self.rollout_and_score_eval_item(eval_item_chat)
            total_score += score

        if num_eval_samples > 0:
            avg_score = total_score / num_eval_samples
            self.eval_metrics.append(
                ("eval/preference_consistency_A", avg_score)
            )  # Example metric name
            print(f"Evaluation finished. Average 'A' preference score: {avg_score:.2f}")
        else:
            print("Evaluation completed with no samples processed.")

    async def collect_trajectories(
        self, item_tuple: Tuple
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        # item_tuple is expected to contain one element: the conversation history (list of dicts)
        raw_chat_history = item_tuple[0]
        chat_for_generation = []
        added_system_prompt_for_rl = False

        # Optional: Inject RL preference string into system prompt with some probability
        if random.random() < 0.05:  # Small chance to directly prime the generator
            chat_for_generation.append(
                {
                    "role": "system",
                    "content": "Please respond in a way that aligns with this preference: "
                    + philosophical_preference_string,
                }
            )
            added_system_prompt_for_rl = True

        for msg_fset in raw_chat_history:  # msg_fset is a frozenset of items
            chat_for_generation.append(dict(msg_fset))

        # Ensure last message isn't assistant, or remove it to get a prompt
        if chat_for_generation and chat_for_generation[-1]["role"] == "assistant":
            chat_for_generation.pop()
        if (
            not chat_for_generation or chat_for_generation[-1]["role"] == "assistant"
        ):  # Still ends with assistant or empty
            print(
                "Skipping trajectory collection: prompt ends with assistant or is empty after processing."
            )
            return None, []

        # Check token length before generation
        # Note: This tokenizer length check is approximate for the prompt only.
        prompt_tokens = self.tokenizer.apply_chat_template(
            chat_for_generation, tokenize=True, add_generation_prompt=False
        )
        # Max length for prompt should ensure (prompt + generated_response) fits
        # self.config.max_token_length for tokenize_for_trainer.
        if len(prompt_tokens) > (
            self.config.max_token_length - self.config.generator_max_tokens
        ):
            print(
                f"Skipping trajectory collection: prompt too long ({len(prompt_tokens)} tokens) "
                f"for max_token_length budget ({self.config.max_token_length} - "
                f"{self.config.generator_max_tokens})."
            )
            return None, []

        # The previous check for effective_generator_context_window was for the API call itself,
        # this new one is for downstream compatibility with tokenize_for_trainer.
        # We should also respect the generator's own context window limit.
        effective_generator_context_window = (
            3500  # Assuming gpt-3.5-turbo, give some buffer from 4096
        )
        if len(prompt_tokens) > (
            effective_generator_context_window - self.config.generator_max_tokens
        ):
            print(
                f"Skipping trajectory collection: prompt too long ({len(prompt_tokens)} tokens) "
                f"for generator's own context window budget ({effective_generator_context_window} - "
                f"{self.config.generator_max_tokens})."
            )
            return None, []

        # Generate two responses (A and B)
        # If we added a system prompt for RL, one response with it, one without, for variety
        if added_system_prompt_for_rl:
            resp1_future = self.server.chat_completion(
                messages=chat_for_generation,  # With RL system prompt
                n=1,
                max_tokens=self.config.generator_max_tokens,
                temperature=0.7,  # Allow some creativity
                model=self.config.generator_model_name,
            )
            # Create a version of the chat without the injected RL system prompt for the second response
            chat_for_generation_no_rl_prompt = [
                m
                for m in chat_for_generation
                if not (
                    m["role"] == "system"
                    and philosophical_preference_string in m["content"]
                )
            ]
            if (
                not chat_for_generation_no_rl_prompt and chat_for_generation
            ):  # if all was system prompt
                chat_for_generation_no_rl_prompt = (
                    chat_for_generation[1:]
                    if len(chat_for_generation) > 1
                    else chat_for_generation
                )

            resp2_future = self.server.chat_completion(
                messages=chat_for_generation_no_rl_prompt,  # Without RL system prompt
                n=1,
                max_tokens=self.config.generator_max_tokens,
                temperature=0.7,
                model=self.config.generator_model_name,
            )
            resp1, resp2 = await asyncio.gather(resp1_future, resp2_future)
            # Combine choices:
            # Need to ensure the structure matches what chat_completion would return for n=2
            # This is a bit manual; ideally, the server handles n=2 better with mixed prompts
            if resp1.choices and resp2.choices:
                # Create a dummy completions object to hold both
                class DummyChoice:
                    def __init__(self, message, finish_reason):
                        self.message = message
                        self.finish_reason = finish_reason

                class DummyMessage:
                    def __init__(self, content):
                        self.content = content

                # Ensure choices are valid before proceeding
                if not resp1.choices[0].message or not resp2.choices[0].message:
                    print("Skipping due to invalid choices from generator.")
                    return None, []

                chat_completions_choices = [
                    DummyChoice(
                        DummyMessage(resp1.choices[0].message.content),
                        resp1.choices[0].finish_reason,
                    ),
                    DummyChoice(
                        DummyMessage(resp2.choices[0].message.content),
                        resp2.choices[0].finish_reason,
                    ),
                ]
            else:  # Not enough responses
                print(
                    "Skipping trajectory collection: not enough responses from generator."
                )
                return None, []

        else:  # Standard generation of two diverse responses
            completions_obj = await self.server.chat_completion(
                messages=chat_for_generation,
                n=2,  # Generate two responses
                max_tokens=self.config.generator_max_tokens,
                temperature=0.7,
                model=self.config.generator_model_name,
            )
            if not completions_obj or len(completions_obj.choices) < 2:
                print(
                    "Skipping trajectory collection: not enough responses from generator (n=2 path)."
                )
                return None, []
            chat_completions_choices = completions_obj.choices

        # Prepare data for the judge
        # The original prompt is `chat_for_generation`
        # (or `chat_for_generation_no_rl_prompt` if that was used for B)
        # For simplicity, use the prompt that led to respA as the "base" conversation for judging.

        # This needs to be a list of ( (full_chat_A, finish_reason_A), (full_chat_B, finish_reason_B) )
        # to pass to self.score
        rollout_pair_for_scoring = []

        response_a_content = chat_completions_choices[0].message.content
        response_a_finish = chat_completions_choices[0].finish_reason
        chat_A = chat_for_generation + [
            {"role": "assistant", "content": response_a_content}
        ]
        rollout_pair_for_scoring.append((chat_A, response_a_finish))

        response_b_content = chat_completions_choices[1].message.content
        response_b_finish = chat_completions_choices[1].finish_reason
        chat_B = chat_for_generation + [
            {"role": "assistant", "content": response_b_content}
        ]
        rollout_pair_for_scoring.append((chat_B, response_b_finish))

        # Call score to get the scored data. `score` expects a list of two items.
        scored_data_group = await self.score(rollout_pair_for_scoring)  # Pass the pair

        return scored_data_group, []  # No backlog items for now

    async def score(
        self, rollout_pair_data: List[Tuple[List[Dict[str, str]], str]]
    ) -> Optional[ScoredDataGroup]:
        # rollout_pair_data is [(chat_A, finish_A), (chat_B, finish_B)]
        if len(rollout_pair_data) < 2:
            print("Score function received less than 2 rollouts to compare.")
            return None

        chat_A_full, finish_A = rollout_pair_data[0]
        chat_B_full, finish_B = rollout_pair_data[1]

        # Handle cases where one or both responses were cut off by length
        # If both are length-limited, it's hard to judge preference, could skip or penalize both.
        # If one is length-limited, it's likely worse.
        if finish_A == "length" and finish_B == "length":
            # Penalize both if we want to discourage long, incomplete answers
            # For now, let's try to judge them anyway, but this could be a spot for different logic.
            print("Both responses A and B hit length limit.")
        elif finish_A == "length" or finish_B == "length":
            print(
                f"One response hit length limit: A_len_limit={finish_A == 'length'}, "
                f"B_len_limit={finish_B == 'length'}"
            )
        # We could assign a strong negative score to the length-limited one here,
        # or let the judge decide. For now, let judge decide.

        # Prepare for the judge LLM
        # The conversation context is the prompt part of chat_A (or chat_B, should be same up to 'assistant')
        conversation_context_list = chat_A_full[
            :-1
        ]  # All but the last assistant message
        conversation_str = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_context_list]
        )

        response_a_content = chat_A_full[-1]["content"]
        response_b_content = chat_B_full[-1]["content"]

        # Create prompts for forward (A vs B) and reverse (B vs A) judging for robustness
        fwd_judge_prompt_content = RLAIF_user_prompt_format_str.format(
            conversation=conversation_str,
            response_a=response_a_content,
            response_b=response_b_content,
            rl_preference_string=philosophical_preference_string,
        )
        # For reverse, swap A and B in the prompt
        rvs_judge_prompt_content = RLAIF_user_prompt_format_str.format(
            conversation=conversation_str,
            response_a=response_b_content,  # Swapped
            response_b=response_a_content,  # Swapped
            rl_preference_string=philosophical_preference_string,
        )

        num_judgements_per_pair = 1  # Can increase for more robust scoring, e.g., 3

        fwd_judge_future = self.server.chat_completion(
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": fwd_judge_prompt_content},
            ],
            n=num_judgements_per_pair,
            max_tokens=self.config.judge_max_tokens,
            temperature=0.0,  # Judge should be as deterministic as possible
            model=self.config.judge_model_name,
        )
        rvs_judge_future = self.server.chat_completion(
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": rvs_judge_prompt_content},
            ],
            n=num_judgements_per_pair,
            max_tokens=self.config.judge_max_tokens,
            temperature=0.0,
            model=self.config.judge_model_name,
        )

        fwd_judge_responses, rvs_judge_responses = await asyncio.gather(
            fwd_judge_future, rvs_judge_future
        )

        # Store one example judgement for wandb logging
        if fwd_judge_responses.choices:
            self.judgement_strings_buffer.append(
                (
                    response_a_content,
                    response_b_content,
                    fwd_judge_responses.choices[0].message.content,
                )
            )

        # Calculate scores from forward and reverse judgements
        score_for_A = 0.0
        score_for_B = 0.0

        # Process forward judgements (Judge chose between A and B)
        for choice in fwd_judge_responses.choices:
            judgement_text = choice.message.content
            # Extract \boxed{A} or \boxed{B}
            chosen_val_match = re.search(r"\boxed{(A|B)}", judgement_text)
            if chosen_val_match:
                chosen = chosen_val_match.group(1)
                if chosen == "A":
                    score_for_A += 1.0
                elif chosen == "B":
                    score_for_B += 1.0

        # Process reverse judgements (Judge chose between B (as A') and A (as B'))
        for choice in rvs_judge_responses.choices:
            judgement_text = choice.message.content
            chosen_val_match = re.search(
                r"\boxed{(A|B)}", judgement_text
            )  # A here means original B, B means original A
            if chosen_val_match:
                chosen = chosen_val_match.group(1)
                if chosen == "A":  # Judge chose B (presented as A')
                    score_for_B += 1.0
                elif chosen == "B":  # Judge chose A (presented as B')
                    score_for_A += 1.0

        total_judgements = (
            2 * num_judgements_per_pair
        )  # Each pair judged forward and reverse

        # Normalize scores: can be simple (preferred_score - non_preferred_score) or Bradley-Terry, etc.
        # Here, let's use a simple difference normalized by total judgements, then mean-center.
        # Effective score for A is (times A preferred) / total_judgements
        # Effective score for B is (times B preferred) / total_judgements
        # We want to assign these as rewards.
        # For DPO, we often need one score for (chosen - rejected).
        # Here, we have two rollouts (A and B). We give A `score_for_A` and B `score_for_B`.
        # Let's normalize them so they sum to 0 for the pair to represent preference.

        # If total_judgements is 0 (e.g. API error), or no clear preference.
        if total_judgements == 0 or score_for_A + score_for_B == 0:
            # No basis for preference, or judge failed. Could assign 0 or skip.
            print(
                "Judge failed to provide preference or API error. Assigning neutral scores."
            )
            final_score_A = 0.0
            final_score_B = 0.0
        else:
            # Normalize scores to represent preference strength, e.g., ranging roughly -1 to 1
            # A simple way: (score_A - score_B) / total_judgements can be one reward, and its negative for the other.
            # Or, score A as (score_A / total_judgements) and B as (score_B / total_judgements)
            # then normalize these (e.g., subtract mean).
            # For PPO-style RL, each gets its own reward.
            # Let's try: A_reward = score_for_A - score_for_B; B_reward = score_for_B - score_for_A
            # Scaled by total_judgements

            # We want a score for A and a score for B.
            # Let's make them centered around 0 for the pair.
            # Paired scores: (score_A - score_B) / total_judgements and (score_B - score_A) / total_judgements
            if score_for_A > score_for_B:
                final_score_A = 1.0
                final_score_B = -1.0
            elif score_for_B > score_for_A:
                final_score_A = -1.0
                final_score_B = 1.0
            else:  # Tie or no preference
                final_score_A = 0.0
                final_score_B = 0.0

        # Handle length penalties explicitly if desired (could override judge scores)
        if (
            finish_A == "length" and final_score_A > -0.9
        ):  # If it was good but cut off, penalize
            final_score_A = -1.0
        if finish_B == "length" and final_score_B > -0.9:
            final_score_B = -1.0

        self.preference_scores_buffer.append(final_score_A)
        self.preference_scores_buffer.append(final_score_B)

        # Prepare ScoredDataGroup
        scores_container = ScoredDataGroup()
        scores_container["tokens"] = list()
        scores_container["masks"] = list()
        scores_container["scores"] = list()

        for i, (full_chat, finish_reason) in enumerate(
            [rollout_pair_data[0], rollout_pair_data[1]]
        ):
            tokenized_output = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=full_chat,  # full conversation including assistant's response
                finish_reason=finish_reason,
                include_messages=self.config.include_messages,
            )
            if (
                len(
                    [
                        mask_val
                        for mask_val in tokenized_output["masks"]
                        if mask_val != -100
                    ]
                )
                < 1
            ):
                continue  # Skip if no assistant tokens to learn from

            scores_container["tokens"].append(tokenized_output["tokens"])
            scores_container["masks"].append(tokenized_output["masks"])
            scores_container["scores"].append(
                final_score_A if i == 0 else final_score_B
            )

        if not scores_container["tokens"]:
            print("No valid tokens found for ScoredDataGroup after processing pair.")
            return None

        # Ensure scores are not the same if configured (for DPO-style data)
        if (
            self.config.ensure_scores_are_not_same
            and len(scores_container["scores"]) >= 2
            and scores_container["scores"][0] == scores_container["scores"][1]
        ):
            print(
                f"Scores are the same ({scores_container['scores'][0]}) but "
                "ensure_scores_are_not_same is True. Skipping pair."
            )
            # This can happen if judge gives no preference or if logic results in tie.
            # For RLAIF leading to PPO, it's okay. For DPO, distinct preferred/rejected is needed.
            # The current scoring final_score_A/B aims for -1/1, so this check is important.
            # If they are same (e.g. both 0.0), it means no preference.
            if (
                scores_container["scores"][0] == 0.0
            ):  # If tie, this is a valid case of no preference.
                pass  # Allow ties if they are both zero (no preference)
            else:  # if scores are identical and non-zero, implies an issue or specific setup
                return None

        return scores_container

    async def get_next_item(
        self,
    ) -> Tuple[List[frozenset], Dict, Dict]:  # Matches BaseEnv signature more closely
        if not self.train_dataset or len(self.train_dataset) == 0:
            raise StopAsyncIteration("Dataset is empty or not loaded.")

        next_raw_item = self.train_dataset[self.iter % len(self.train_dataset)]
        self.iter += 1

        # 'conversation' in WildChat is a list of dicts: [{'role': ..., 'content': ...}, ...]
        conversation_history = next_raw_item.get("conversation", [])
        if not conversation_history:  # Should not happen with WildChat but good check
            # Return an empty prompt or handle as error
            return (
                [],
                {},
                {
                    "id": next_raw_item.get("id", self.iter - 1),
                    "error": "empty_conversation",
                },
            )

        # Convert to the frozenset format if BaseEnv expects it (original rlaif_server used this)
        # My BaseEnv.get_next_item returns messages, metadata, correct_answer_optional
        # Let's simplify what this get_next_item returns for now for collect_trajectories
        # collect_trajectories was defined as item_tuple: Tuple
        # The original rlaif_server.py in get_next_item returned (prompt_tuple,)
        # where prompt_tuple was tuple of frozensets.
        # Let's return List[Dict[str,str]] directly for the conversation

        prompt_messages = [
            dict(msg) for msg in conversation_history
        ]  # Ensure mutable dicts

        # Return just the messages, collect_trajectories will handle it
        # The tuple structure for item in collect_trajectories needs to be consistent.
        # If BaseEnv.process expects get_next_item to return (messages, metadata, correct_answer)
        # then we need to adhere. For now, let's assume collect_trajectories takes (messages_list,)
        # This is a deviation from the stricter BaseEnv typing; might need adjustment if `process` complains.
        # The original `rlaif_server.py` had get_next_item returning (prompt_frozenset_tuple, )
        # and collect_trajectories took item[0] which was that tuple.
        # Let's match that for now.
        prompt_frozenset_tuple = tuple(
            frozenset(msg.items()) for msg in prompt_messages
        )
        return (
            prompt_frozenset_tuple,
            {},
            {},
        )  # (messages_frozenset_tuple, metadata_dict, correct_answer_dict)


if __name__ == "__main__":
    PhilosophicalRLAIFEnv.cli()
