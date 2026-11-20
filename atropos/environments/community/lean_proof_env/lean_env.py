import asyncio  # For async operations if PyPantograph is async
import random
from typing import Dict, List, Optional, Tuple, TypedDict

import wandb  # For wandb.Table
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio  # For progress bars in evaluate

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import (  # Item might not be directly used if LeanProblemRow is self-contained
    Item,
    number,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# from pypantograph import PyPantograph # Assuming it\'s a package
# For now, mock PyPantograph:


# Mock PyPantograph - Replace with your actual PyPantograph integration
class PyPantograph:
    @staticmethod
    async def check_lean_code(lean_code: str) -> Tuple[bool, Optional[str]]:
        """
        Checks if the Lean code compiles.
        Returns a tuple (compiles_ok, error_message).
        Error_message is None if compiles_ok is True.
        Simulates an async check.
        """
        await asyncio.sleep(0.01)  # Simulate I/O or computation time

        # Basic checks
        if "sorry" in lean_code.lower():
            return False, "Proof contains 'sorry'."
        if not lean_code.strip() or len(lean_code.strip()) < 10:
            return False, "Proof is empty or too short."
        if "begin end" in lean_code or lean_code.strip().endswith("begin"):
            return False, "Incomplete or empty proof structure."

        # Simulate compilation success/failure based on some keywords or randomness
        if "simple_theorem_correct" in lean_code:
            return True, None
        if "simple_theorem_error" in lean_code:
            return False, "Mock Lean: Type mismatch on 'simple_theorem_error'."

        # Default random outcome
        if random.random() < 0.6:  # 60% chance of mock compilation
            return True, None
        else:
            errors = [
                "Mock Lean: Unknown identifier 'xyz'.",
                "Mock Lean: Type mismatch, expected 'nat', got 'Prop'.",
                "Mock Lean: Tactic failed.",
                "Mock Lean: Universe level constraint violation.",
            ]
            return False, random.choice(errors)


# System prompt for Lean
lean_system_prompt = (
    "You are an expert Lean mathematician. Your mission is to complete the given Lean proof.\\n"
    "You will be provided with a Lean theorem statement, often prefixed by necessary import and open commands. "
    "The theorem itself (e.g., starting with 'theorem ...') will typically end with ':= sorry'.\\n"
    "Your task is to replace 'sorry' (and its surrounding ' := ' if necessary) with the correct proof steps "
    "within the provided theorem structure.\\n"
    'Provide only the completed Lean theorem block (e.g., starting from "theorem ..." '
    'or "def ..." up to its final "end" or conclusion), '
    "including the statement and the proof, as a single Lean code block.\\n"
    "Do not repeat the import or open commands that were part of the input.\\n"
    "Do not include any other explanatory text, comments, "
    "or markdown code fences (```lean ... ```) around your response.\\n"
    "Ensure your proof is self-contained (assuming the provided imports) and syntactically correct Lean code.\\n\\n"
    "Example of input you will receive from the user (header + formal statement):\\n"
    "import Mathlib.Data.Nat.Basic\\n"
    "open Nat\\n\\n"
    "theorem add_comm (a b : nat) : a + b = b + a := sorry\\n\\n"
    "Example of a correct full response from you (just the completed theorem block):\\n"
    "theorem add_comm (a b : nat) : a + b = b + a :=\\\\n"
    "begin\\\\n"
    "  rw nat.add_comm,\\\\n"
    "end"
)


class LeanProblemRow(TypedDict):
    id: str  # Unique identifier for the problem
    header: str  # The import and open statements
    problem_statement: (
        str  # e.g., "theorem add_comm (a b : nat) : a + b = b + a := sorry"
    )
    # ground_truth_proof: Optional[str] # For reference or more advanced evaluation


class LeanEnv(BaseEnv):
    name = "lean_proof"  # Used for wandb naming unless overridden

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_compiled_buffer = []  # For training batches
        self.eval_metrics = []  # For storing (metric_name, value) tuples for wandb
        # self.rollouts_for_wandb is inherited. We\'ll populate it with tuples:
        # (problem_statement, generated_proof, score, status_message)

        # self.train_data and self.test_data will be populated in setup()
        self.train_data: List[LeanProblemRow] = []
        self.test_data: List[LeanProblemRow] = []
        self.testing_mode = testing  # Store testing flag for setup

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="Qwen/Qwen3-235B-A22B",  # Choose appropriate tokenizer for your LLM
            group_size=4,  # Number of proofs to generate per problem for scoring diversity
            use_wandb=True,
            rollout_server_url="http://localhost:8000",  # Default Atropos server URL
            total_steps=1000,  # Total training steps
            batch_size=8,  # Number of scored groups to send to trainer at once
            steps_per_eval=50,  # How often to run evaluation
            max_token_length=1024,  # Max token length for generated Lean proofs
            wandb_name="lean_proof_rl_eval",  # Custom WandB project/group name prefix
            ensure_scores_are_not_same=False,  # For compile/no-compile, allowing same scores (e.g., all fail) is useful
            num_rollouts_to_keep=32,  # Number of recent groups to show in WandB table
            num_rollouts_per_group_for_logging=1,  # Log 1 example from each group (-1 for all in group)
            include_messages=False,  # We\'ll format messages for wandb table ourselves
        )
        # Configuration for the LLM API server (e.g., TGI, vLLM)
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen3-235B-A22B",  # Ensure this matches the deployed LLM and tokenizer
                base_url="http://localhost:9001/v1",  # Your LLM server endpoint (OpenAI compatible)
                api_key="EMPTY",  # API key if required by your server
                num_requests_for_eval=128,  # Max concurrent requests for evaluation pass
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """Load Lean dataset from Hugging Face."""
        hf_dataset_path = "brando/minif2f-lean4"  # Correctly keep this as a string

        try:
            print(
                f"Loading training data from Hugging Face: {hf_dataset_path} (split=train)"
            )
            # The dataset "brando/minif2f-lean4" doesn't have a canonical "train" split from its viewer.
            # It has "test" and "validation". Let's assume for now you want to use "validation" for training
            # and "test" for evaluation, or vice-versa.
            # For this example, I'll use "validation" for training and "test" for eval.
            # Please adjust if your intention is different.
            print(
                "Attempting to use 'validation' split for training and 'test' split for evaluation."
            )
            raw_train_data = load_dataset(
                hf_dataset_path, split="validation"
            )  # Using validation for training
            print(
                f"Loading test data for evaluation from Hugging Face: {hf_dataset_path} (split=test)"
            )
            raw_eval_data = load_dataset(
                hf_dataset_path, split="test"
            )  # Using test for evaluation
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            print(
                "Please ensure the dataset path is correct, splits exist, "
                "and you have internet access / the dataset is cached."
            )
            print(
                "Falling back to empty data. The environment will likely fail to run without data."
            )
            self.train_data = []
            self.test_data = []  # This holds the data for evaluation
            self.train_iter = 0
            return

        # Adapt this mapping based on your dataset's column names
        # Using "formal_statement" for the problem statement based on provided features.
        self.train_data = [
            LeanProblemRow(
                id=str(item.get("id", f"train_idx_{i}")),
                header=item["header"],
                problem_statement=item["formal_statement"],
            )
            for i, item in enumerate(raw_train_data)
        ]
        self.test_data = [
            LeanProblemRow(
                id=str(item.get("id", f"eval_idx_{i}")),
                header=item["header"],
                problem_statement=item["formal_statement"],
            )
            for i, item in enumerate(raw_eval_data)
        ]

        if self.testing_mode:  # If in testing mode, use a small subset
            print("Testing mode active: Using a small subset of the loaded data.")
            self.train_data = self.train_data[:5]
            self.test_data = self.test_data[:3]
            if not self.train_data:
                print("Warning: Test subset for training data is empty.")
            if not self.test_data:
                print("Warning: Test subset for validation data is empty.")

        random.shuffle(self.train_data)  # Optionally shuffle

        self.train_iter = 0
        print(
            f"LeanEnv setup: Loaded {len(self.train_data)} train problems "
            "(from validation split) and {len(self.test_data)} eval problems (from test split)."
        )
        if not self.train_data and self.config.total_steps > 0:
            print(
                "CRITICAL WARNING: No training data loaded. Environment will "
                "not be able to produce training trajectories."
            )
        if not self.test_data and self.config.steps_per_eval > 0:
            print(
                "WARNING: No eval data loaded. Evaluation steps will have nothing to evaluate."
            )

    async def get_next_item(self) -> LeanProblemRow:
        """Get the next Lean problem from the training set."""
        if not self.train_data:
            # This should ideally not happen if total_steps is managed correctly
            # Or, you might want to loop indefinitely over the training data
            print(
                "Warning: Ran out of unique training data. Restarting from beginning."
            )
            self.train_iter = 0
        item = self.train_data[self.train_iter % len(self.train_data)]
        self.train_iter += 1
        return item

    async def _get_llm_proof_attempt(
        self, item: LeanProblemRow, split: str, temperature: float = 0.7
    ) -> str:
        """Generates a proof attempt using the LLM server for a given LeanProblemRow item."""
        # LLM sees header + formal_statement
        full_problem_context = item["header"] + "\n\n" + item["problem_statement"]
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": lean_system_prompt},
                {"role": "user", "content": full_problem_context},
            ],
            n=1,  # Single completion for this helper
            max_tokens=self.config.max_token_length,
            temperature=temperature,
            split=split,  # 'train' or 'eval' for server-side request tracking
        )
        return completion.choices[
            0
        ].message.content.strip()  # This should be the completed theorem block

    async def rollout_and_score_eval(self, problem_item: LeanProblemRow) -> number:
        """Rollout a single problem for evaluation and score it based on compilation."""
        # For evaluation, use a lower temperature for more deterministic outputs
        llm_generated_theorem_block = await self._get_llm_proof_attempt(
            problem_item, split="eval", temperature=0.1
        )

        # Code to check with PyPantograph = original header + LLM's completed theorem block
        code_to_check = problem_item["header"] + "\n\n" + llm_generated_theorem_block
        compiles, _error_msg = await PyPantograph.check_lean_code(code_to_check)
        return 1 if compiles else 0

    async def evaluate(self, *args, **kwargs):
        """Evaluate the LLM on the test set of Lean problems."""
        print(f"Starting evaluation on {len(self.test_data)} Lean problems...")
        if not self.test_data:
            print("No test data to evaluate.")
            self.eval_metrics.append(("eval/percent_compiled", 0.0))
            return

        eval_tasks = [self.rollout_and_score_eval(item) for item in self.test_data]
        scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating Lean Proofs")

        percent_compiled = (
            sum(s for s in scores if s > 0) / len(scores) if scores else 0.0
        )
        self.eval_metrics.append(("eval/percent_compiled", percent_compiled))
        print(f"Evaluation finished. Percent compiled: {percent_compiled:.2%}")

    async def collect_trajectories(
        self, item: LeanProblemRow  # Item is LeanProblemRow
    ) -> Tuple[
        Optional[ScoredDataGroup], list[Item]
    ]:  # Return type List[Item] for backlog
        """
        Collect a group of proof attempts for a single Lean problem and score them.
        """
        # LLM sees header + formal_statement (problem_statement in LeanProblemRow is the formal_statement)
        full_problem_context = item["header"] + "\n\n" + item["problem_statement"]

        chat_completions = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": lean_system_prompt},
                {"role": "user", "content": full_problem_context},
            ],
            n=self.config.group_size,  # Generate n attempts
            max_tokens=self.config.max_token_length,
            temperature=0.7,  # Higher temperature for diversity in training
            split="train",
        )

        # Data to be scored by `score_lean_attempts`
        attempts_to_score_data = []
        for choice in chat_completions.choices:
            # Construct message history for tokenization
            # The user message here reflects what the LLM actually saw
            messages_history = (
                {"role": "system", "content": lean_system_prompt},
                {"role": "user", "content": full_problem_context},
                # LLM's output is its attempt at completing the theorem block
                {"role": "assistant", "content": choice.message.content},
            )
            attempts_to_score_data.append(
                {
                    "messages_history": messages_history,
                    "llm_generated_theorem_block": choice.message.content.strip(),
                    "finish_reason": choice.finish_reason,
                    "problem_item": item,
                }
            )

        scored_data_group = await self.score_lean_attempts(attempts_to_score_data)

        return scored_data_group, []  # No backlog items generated in this simple setup

    async def score_lean_attempts(
        self, attempts_data: List[Dict]
    ) -> Optional[ScoredDataGroup]:
        """
        Scores a group of Lean proof attempts.
        `attempts_data` contains 'llm_generated_theorem_block', 'messages_history', 'finish_reason', 'problem_item'.
        The 'problem_item' is the original LeanProblemRow, containing the header and the formal_statement.
        """
        scored_group = ScoredDataGroup(tokens=[], masks=[], scores=[])
        self._current_group_rollout_details_log: List[Tuple[str, str, float, str]] = []

        for attempt_data in attempts_data:
            llm_generated_theorem_block = attempt_data["llm_generated_theorem_block"]
            messages_history = attempt_data["messages_history"]
            finish_reason = attempt_data["finish_reason"]
            problem_item: LeanProblemRow = attempt_data["problem_item"]

            # Code to check with PyPantograph = original header + LLM's completed theorem block
            code_to_check = (
                problem_item["header"] + "\n\n" + llm_generated_theorem_block
            )
            compiles, error_msg = await PyPantograph.check_lean_code(code_to_check)

            reward = 1.0 if compiles else -1.0
            status_message = (
                "Compiled Successfully"
                if compiles
                else f"Compilation Failed: {error_msg or 'Unknown error'}"
            )

            # For WandB logging, problem_item["problem_statement"] is the formal_statement (original theorem with sorry)
            # llm_generated_theorem_block is what the LLM produced to complete it.
            self._current_group_rollout_details_log.append(
                (
                    problem_item["problem_statement"],
                    llm_generated_theorem_block,
                    reward,
                    status_message,
                )
            )

            out_dict = tokenize_for_trainer(
                self.tokenizer, list(messages_history), finish_reason
            )

            # Basic filter for very short/empty generations not caught by PyPantograph mock
            # Typically, the prompt and problem statement are part of the input tokens.
            # We check the number of generated (assistant) tokens.
            assistant_token_count = sum(
                1 for m_idx in out_dict["masks"] if m_idx != -100
            )
            if (
                assistant_token_count < 3
            ):  # Arbitrary small number of tokens for a proof part
                # This attempt might be too short to be a valid proof attempt.
                # In a real scenario, you might still want to penalize it.
                # For now, we are not explicitly filtering but PyPantograph mock handles some.
                pass  # Or `continue` if you want to filter these out from training batch

            scored_group["tokens"].append(out_dict["tokens"])
            scored_group["masks"].append(out_dict["masks"])
            scored_group["scores"].append(reward)

            # Update buffer for average batch compilation rate
            self.percent_compiled_buffer.append(1.0 if compiles else 0.0)

        if not scored_group["tokens"]:  # If all attempts were filtered or no attempts
            self._current_group_rollout_details_log = []  # Clear if no valid data
            return None

        return scored_group

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Custom WandB logging for Lean environment."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_compiled_buffer:
            wandb_metrics["train/batch_avg_percent_compiled"] = sum(
                self.percent_compiled_buffer
            ) / len(self.percent_compiled_buffer)
            self.percent_compiled_buffer = []

        for metric_name, value in self.eval_metrics:
            wandb_metrics[metric_name] = value
        self.eval_metrics = []

        # The parent super().wandb_log will call self.create_rollout_table()
        await super().wandb_log(wandb_metrics)

    async def add_rollouts_for_wandb(
        self,
        scored_data: ScoredDataGroup,  # This is the ScoredDataGroup returned by score_lean_attempts
        item: Optional[
            LeanProblemRow
        ] = None,  # `item` is the original LeanProblemRow from get_next_item
    ):
        """
        Called by BaseEnv\'s handle_send_to_api. Prepares detailed rollouts for our custom WandB table.
        """
        # `_current_group_rollout_details_log` was populated in `score_lean_attempts`.
        # It contains: (problem_statement, generated_proof, score, status_message)
        if (
            hasattr(self, "_current_group_rollout_details_log")
            and self._current_group_rollout_details_log
        ):
            num_to_log = self.config.num_rollouts_per_group_for_logging
            if num_to_log == -1:  # Log all from the group
                rollouts_to_add = self._current_group_rollout_details_log
            else:
                rollouts_to_add = self._current_group_rollout_details_log[:num_to_log]

            # self.rollouts_for_wandb is a list of lists/groups of these tuples
            if rollouts_to_add:  # Ensure there\'s something to add
                self.rollouts_for_wandb.append(rollouts_to_add)

            # Important: Clean up the temporary instance variable after use
            if hasattr(self, "_current_group_rollout_details_log"):
                del self._current_group_rollout_details_log

        # Keep only the configured number of recent groups for the WandB table
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Overrides BaseEnv method to create a custom WandB table for Lean proofs."""
        if self.rollouts_for_wandb:
            table_name_key = "train/lean_proof_attempts"
            # BaseEnv might prepend a name like "env_0_train/lean_proof_attempts"
            # If self.wandb_prepend is set (by BaseEnv.register_env), use it.
            # Otherwise, default to just "train/..."
            # The super().wandb_log() handles prepending to other metrics,
            # but for tables we might need to handle it here or ensure consistency.
            # For now, let's use a simple name and see how BaseEnv handles it.
            # If BaseEnv's wandb_log prepends, this name will also be prepended.

            if self.wandb_prepend:
                table_name_key = f"{self.wandb_prepend}_{table_name_key.split('/')[-1]}"

            table = wandb.Table(
                columns=[
                    "Problem Statement",
                    "Generated Proof",
                    "Score",
                    "Compilation Status",
                ]
            )

            # self.rollouts_for_wandb is a list of groups, and each group is a list of rollout tuples
            for group_of_rollouts in self.rollouts_for_wandb:
                for rollout_detail_tuple in group_of_rollouts:
                    # Each tuple: (problem_statement, generated_proof, score, status_message)
                    problem_stmt, gen_proof, score, status_msg = rollout_detail_tuple

                    # Truncate for display, WandB tables can be slow with very long text
                    problem_display = (
                        (problem_stmt[:250] + "...")
                        if len(problem_stmt) > 250
                        else problem_stmt
                    )
                    proof_display = (
                        (gen_proof[:400] + "...") if len(gen_proof) > 400 else gen_proof
                    )

                    table.add_data(
                        problem_display, proof_display, float(score), status_msg
                    )

            wandb_metrics[table_name_key] = table
        return wandb_metrics


if __name__ == "__main__":
    # This allows running:
    # python environments/lean_env.py serve (to connect to Atropos trainer)
    # python environments/lean_env.py process (for local data generation/testing)
    LeanEnv.cli()
