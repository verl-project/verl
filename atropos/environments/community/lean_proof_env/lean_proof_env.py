import asyncio
import os
import random
import re
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

# import wandb # Conditionally import later
# from datasets import load_dataset # Conditionally import later
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig as AtroposAPIServerConfig,  # Renamed to avoid conflict if needed later
)
from atroposlib.envs.base import BaseEnv as AtroposBaseEnv
from atroposlib.envs.base import BaseEnvConfig as AtroposBaseEnvConfig
from atroposlib.envs.base import EvalHandlingEnum, ScoredDataGroup

# Global variable to hold wandb if imported
wandb = None
load_dataset = None  # Placeholder for conditional import
python_dotenv_available = False
try:
    from dotenv import load_dotenv

    python_dotenv_available = True
except ImportError:
    pass


class LeanProofEnvConfig(AtroposBaseEnvConfig):  # Inherit from actual Atropos config
    tokenizer_name: str = Field("Salesforce/codegen-350M-mono")
    group_size: int = Field(8)
    use_wandb: bool = Field(False)
    total_steps: int = Field(
        10
    )  # For process mode, this might be interpreted as number of items to process
    batch_size: int = Field(2)
    steps_per_eval: int = Field(1)
    max_token_length: int = Field(
        1536
    )  # Max length for tokenizer input, not necessarily LLM generation
    wandb_name: str = Field("lean_proof_rl_minif2f")
    eval_handling: EvalHandlingEnum = Field(EvalHandlingEnum.LIMIT_TRAIN)
    eval_limit_ratio: float = Field(0.1)
    lean_executable_path: str = Field("lean")
    lean_problem_dataset_name: Optional[str] = Field("internal_simple_test")
    lean_problem_dataset_split: str = Field("train")
    num_rollouts_to_keep: int = Field(5)  # For WandB table logging
    num_cpus_maxtasksperchild: int = Field(1)
    max_proof_generation_tokens: int = Field(
        512, description="Maximum tokens for the LLM to generate for a proof attempt."
    )
    proof_verification_timeout_seconds: int = Field(
        60, description="Timeout for Lean proof verification."
    )
    # Add any other config fields specific to LeanProofEnv or expected by AtroposBaseEnvConfig


async def verify_lean_proof(
    lean_executable_path: str, proof_content: str, timeout_seconds: int = 60
) -> Tuple[bool, str]:
    """
    Verifies a Lean proof by writing it to a temporary file and running Lean.
    Returns (True, "success") if proof is valid, (False, error_message) otherwise.
    """
    common_imports = ""  # Empty for basic examples
    full_content_for_lean = common_imports + proof_content

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(full_content_for_lean)
        tmp_file_name = tmp_file.name

    process = None  # Ensure process is defined for finally block
    try:
        process = await asyncio.create_subprocess_exec(
            lean_executable_path,
            tmp_file_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Wait for communicate with a timeout
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )

        if process.returncode == 0:
            return True, "success"
        else:
            error_output = (
                stderr.decode("utf-8", errors="ignore").strip()
                if stderr
                else stdout.decode("utf-8", errors="ignore").strip()
            )
            error_output = error_output.replace(
                tmp_file_name + ":", ""
            )  # Remove file path from error
            return False, (
                error_output
                if error_output
                else "Lean verification failed with non-zero exit code and no error message."
            )
    except FileNotFoundError:
        error_msg = (
            f"Lean executable not found at {lean_executable_path}. "
            "Please ensure Lean is installed and in PATH, or configure lean_executable_path."
        )
        return False, error_msg
    except asyncio.TimeoutError:
        if process and process.returncode is None:  # Check if process is still running
            try:
                process.kill()
                await process.wait()  # Ensure process is cleaned up
            except ProcessLookupError:
                pass  # Process already terminated
            except Exception as e_kill:
                print(f"Error killing timed-out Lean process: {e_kill}")
        return False, f"Lean verification timed out after {timeout_seconds} seconds."
    except Exception as e:
        return False, f"Error during Lean verification: {str(e)}"
    finally:
        if (
            process and process.returncode is None
        ):  # Ensure process is terminated if loop exited early
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass  # Process already terminated
            except Exception as e_kill_finally:
                print(f"Error killing Lean process in finally: {e_kill_finally}")
        if os.path.exists(tmp_file_name):
            try:
                os.remove(tmp_file_name)
            except Exception as e_remove:
                print(
                    f"Warning: Could not remove temporary file {tmp_file_name}: {e_remove}"
                )


class LeanProofEnv(AtroposBaseEnv):  # Inherit from actual Atropos BaseEnv
    name = "lean_proof"
    env_config_cls = LeanProofEnvConfig

    def __init__(
        self,
        config: LeanProofEnvConfig,
        server_configs: List[
            AtroposAPIServerConfig
        ],  # Use renamed AtroposAPIServerConfig
        slurm=True,  # Default slurm to True as in original BaseEnv, can be overridden
        testing=False,  # Default testing to False
    ):
        global wandb
        self.wandb_available = False
        if hasattr(config, "use_wandb") and config.use_wandb:
            try:
                import wandb as wb

                wandb = wb
                self.wandb_available = True
            except ImportError:
                print(
                    "Warning: wandb could not be imported. wandb logging will be disabled."
                )
                config.use_wandb = False

        print("Initializing LeanProofEnv with Atropos...")
        super().__init__(config, server_configs, slurm=slurm, testing=testing)

        try:
            self.mp_executor = ProcessPoolExecutor(config.num_cpus_maxtasksperchild)
        except (AttributeError, TypeError):
            print(
                "Warning: could not create ProcessPoolExecutor with "
                "config.num_cpus_maxtasksperchild. Using default."
            )
            self.mp_executor = ProcessPoolExecutor(
                max_workers=(
                    config.num_cpus_maxtasksperchild
                    if hasattr(config, "num_cpus_maxtasksperchild")
                    else 1
                )
            )

        self.eval_metrics = list()
        self.pass_at_groupsize = list()
        self.successful_proofs_rollouts = list()
        self.failed_proofs_rollouts = list()
        self.iter = 0
        self.problems = []
        self.rollouts_table = None

    async def chat_completion(self, *args, **kwargs):
        if not self.server:
            raise RuntimeError(
                "Server not initialized. Ensure AtroposBaseEnv sets up self.server."
            )
        return await self.server.chat_completion(*args, **kwargs)

    @classmethod
    def config_init(cls) -> Tuple[LeanProofEnvConfig, List[AtroposAPIServerConfig]]:
        env_config = LeanProofEnvConfig(
            tokenizer_name="Salesforce/codegen-350M-mono",
            group_size=8,
            use_wandb=False,
            total_steps=10,
            batch_size=2,
            steps_per_eval=1,
            max_token_length=1536,
            wandb_name="lean_proof_rl_new_env",  # Changed wandb_name for clarity
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            lean_executable_path="lean",
            lean_problem_dataset_name="internal_simple_test",
            lean_problem_dataset_split="train",
            num_rollouts_to_keep=5,
            num_cpus_maxtasksperchild=1,
            max_proof_generation_tokens=512,
        )

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key and python_dotenv_available:
            print(
                "OPENAI_API_KEY not found in environment, attempting to load from .env file..."
            )
            load_dotenv()  # Load .env file from current directory
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables or .env file. "
                "Please set it to run this environment. You can create a .env file "
                "in the execution directory with OPENAI_API_KEY='your_key'."
            )

        server_configs = [
            AtroposAPIServerConfig(
                model_name="gpt-4o",  # Default model, can be overridden by CLI
                base_url="https://api.openai.com/v1",
                api_key=openai_api_key,
            ),
        ]
        print(
            "INFO: LeanProofEnv will use OpenAI model (default/from CLI) with API key from env/dotenv."
        )
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if self.wandb_available and wandb and wandb_metrics:
            print(f"[WandB] Metrics: {wandb_metrics}")
        elif wandb_metrics:
            print(f"[Metrics (wandb disabled)] {wandb_metrics}")

    async def setup(self):
        global load_dataset
        if self.config.lean_problem_dataset_name == "internal_simple_test":
            print("Using internal_simple_test: Loading hardcoded simple Lean problems.")
            self.problems = [
                {
                    "name": "trivial_true",
                    "formal_statement": "theorem trivial_true : True :=\n  trivial",
                    "theorem_header": "theorem trivial_true : True :=",
                    "proof_prefix": "theorem trivial_true : True :=\n",
                    "statement_to_prove": "True",
                },
                {
                    "name": "rfl_nat_add_zero",
                    "formal_statement": "theorem rfl_nat_add_zero (n : Nat) : n + 0 = n :=\n  rfl",
                    "theorem_header": "theorem rfl_nat_add_zero (n : Nat) : n + 0 = n :=",
                    "proof_prefix": "theorem rfl_nat_add_zero (n : Nat) : n + 0 = n :=\n",
                    "statement_to_prove": "(n : Nat) : n + 0 = n",
                },
                {
                    "name": "exact_refl",
                    "formal_statement": "theorem exact_refl (P : Prop) (h : P) : P :=\n  exact h",
                    "theorem_header": "theorem exact_refl (P : Prop) (h : P) : P :=",
                    "proof_prefix": "theorem exact_refl (P : Prop) (h : P) : P :=\n",
                    "statement_to_prove": "(P : Prop) (h : P) : P",
                },
                {
                    "name": "id_apply",
                    "formal_statement": "theorem id_apply (P : Prop) (h : P) : P :=\n  h",
                    "theorem_header": "theorem id_apply (P : Prop) (h : P) : P :=",
                    "proof_prefix": "theorem id_apply (P : Prop) (h : P) : P :=\n",
                    "statement_to_prove": "(P : Prop) (h : P) : P",
                },
                {
                    "name": "nat_add_comm",
                    "formal_statement": "theorem nat_add_comm (n m : Nat) : n + m = m + n :=\n  Nat.add_comm n m",
                    "theorem_header": "theorem nat_add_comm (n m : Nat) : n + m = m + n :=",
                    "proof_prefix": "theorem nat_add_comm (n m : Nat) : n + m = m + n :=\n",
                    "statement_to_prove": "(n m : Nat) : n + m = m + n",
                },
                {
                    "name": "true_intro_example",
                    "formal_statement": "theorem true_intro_example : True :=\n  True.intro",
                    "theorem_header": "theorem true_intro_example : True :=",
                    "proof_prefix": "theorem true_intro_example : True :=\n",
                    "statement_to_prove": "True",
                },
                {
                    "name": "and_intro_example",
                    "formal_statement": (
                        "theorem and_intro_example (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q :=\n  And.intro hp hq"
                    ),
                    "theorem_header": "theorem and_intro_example (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q :=",
                    "proof_prefix": "theorem and_intro_example (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q :=\n",
                    "statement_to_prove": "(P Q : Prop) (hp : P) (hq : Q) : P ∧ Q",
                },
                {
                    "name": "list_nil_is_empty_example",
                    "formal_statement": (
                        "theorem list_nil_is_empty_example {α : Type} : "
                        "List.isEmpty ([] : List α) :=\n  rfl"
                    ),
                    "theorem_header": (
                        "theorem list_nil_is_empty_example {α : Type} : "
                        "List.isEmpty ([] : List α) :="
                    ),
                    "proof_prefix": (
                        "theorem list_nil_is_empty_example {α : Type} : "
                        "List.isEmpty ([] : List α) :=\n"
                    ),
                    "statement_to_prove": "{α : Type} : List.isEmpty ([] : List α)",
                },
            ]
            print(f"Loaded {len(self.problems)} simple hardcoded problems.")
            return
        if load_dataset is None:
            try:
                from datasets import load_dataset as ld

                load_dataset = ld
                print("Successfully imported load_dataset from datasets library.")
            except ImportError:
                print(
                    "Error: The 'datasets' library is not installed. "
                    "Please install it with 'pip install datasets' to use the MiniF2F benchmark."
                )
                self.problems = [
                    {
                        "name": "dummy_add_zero_no_dataset_lib",
                        "formal_statement": "theorem dummy_add_zero (n : Nat) : n + 0 = n :=\n  sorry",
                        "theorem_header": "theorem dummy_add_zero (n : Nat) : n + 0 = n :=",
                        "proof_prefix": "theorem dummy_add_zero (n : Nat) : n + 0 = n :=\n",
                        "statement_to_prove": "(n : Nat) : n + 0 = n",
                    }
                ]
                print(
                    f"Using {len(self.problems)} hardcoded problem due to missing 'datasets' library."
                )
                return
        if self.config.lean_problem_dataset_name:
            print(
                f"Attempting to load dataset: {self.config.lean_problem_dataset_name} "
                f"split: {self.config.lean_problem_dataset_split}"
            )
            try:
                dataset = load_dataset(
                    self.config.lean_problem_dataset_name,
                    split=self.config.lean_problem_dataset_split,
                    trust_remote_code=True,
                )
                processed_problems = []
                for i, item in enumerate(dataset):
                    formal_statement = item.get("formal_statement")
                    if not formal_statement or not isinstance(formal_statement, str):
                        print(
                            f"Skipping item {i} due to missing or invalid formal_statement: {item}"
                        )
                        continue
                    name_match = re.search(r"theorem\s+([\w_]+)", formal_statement)
                    problem_name = (
                        name_match.group(1) if name_match else f"minif2f_problem_{i}"
                    )
                    proof_start_marker = ":="
                    if proof_start_marker in formal_statement:
                        header_part, _ = formal_statement.split(proof_start_marker, 1)
                        theorem_header = header_part.strip() + f" {proof_start_marker}"
                        proof_prefix = theorem_header + "\n"
                        statement_to_prove_match = re.search(
                            r"theorem\s+[\w_]+\s*(.*)\s*:=", theorem_header
                        )
                        statement_to_prove = (
                            statement_to_prove_match.group(1).strip()
                            if statement_to_prove_match
                            else ""
                        )
                    else:
                        print(
                            f"Warning: Could not find ':=' in formal_statement for {problem_name}. "
                            "Using full statement as header."
                        )
                        theorem_header = formal_statement.strip()
                        proof_prefix = theorem_header + "\n"
                        statement_to_prove = formal_statement
                    processed_problems.append(
                        {
                            "name": problem_name,
                            "formal_statement": formal_statement,
                            "theorem_header": theorem_header,
                            "proof_prefix": proof_prefix,
                            "statement_to_prove": statement_to_prove,
                        }
                    )
                self.problems = processed_problems
                print(
                    f"Loaded and processed {len(self.problems)} problems from "
                    f"{self.config.lean_problem_dataset_name}."
                )
            except Exception as e:
                print(
                    f"Failed to load or process dataset {self.config.lean_problem_dataset_name}: {e}. "
                    "Using hardcoded examples."
                )
                self.problems = []
        if not self.problems:
            self.problems = [
                {
                    "name": "dummy_add_zero",
                    "formal_statement": "theorem dummy_add_zero (n : Nat) : n + 0 = n :=\n  sorry",
                    "theorem_header": "theorem dummy_add_zero (n : Nat) : n + 0 = n :=",
                    "proof_prefix": "theorem dummy_add_zero (n : Nat) : n + 0 = n :=\n",
                    "statement_to_prove": "(n : Nat) : n + 0 = n",
                },
                {
                    "name": "dummy_true",
                    "formal_statement": "theorem dummy_true : True :=\n  trivial",
                    "theorem_header": "theorem dummy_true : True :=",
                    "proof_prefix": "theorem dummy_true : True :=\n",
                    "statement_to_prove": "True",
                },
            ]
            print(
                f"Using {len(self.problems)} hardcoded problems due to failure in "
                "dataset loading or processing."
            )

    async def get_next_item(self) -> Dict[str, Any]:
        if not self.problems:
            print(
                "Error: No problems loaded. Cannot get next item. "
                "Ensure dataset is configured and loaded correctly."
            )
            return {
                "history": [
                    {
                        "role": "system",
                        "content": "You are a Lean theorem prover. Error: No problems available.",
                    }
                ],
                "model_name": (
                    self.server.servers[0].config.model_name
                    if self.server.servers and hasattr(self.server.servers[0], "config")
                    else "error_model"
                ),
                "item_uuid": str(uuid.uuid4()),
                "env_specific_info": {"problem_name": "dummy_no_problems_loaded"},
                "metadata": {},
                "max_tokens": self.config.max_proof_generation_tokens,
            }
        problem = random.choice(self.problems)
        history = [
            {
                "role": "system",
                "content": (
                    "You are an expert Lean theorem prover. Complete the given Lean proof. "
                    "Only output the proof steps after the `:=` a single newline. "
                    "Do not repeat the theorem statement."
                ),
            },
            {"role": "user", "content": problem["proof_prefix"]},
        ]
        return {
            "history": history,
            "model_name": (
                self.server.servers[0].config.model_name
                if self.server.servers and hasattr(self.server.servers[0], "config")
                else "default_model"
            ),
            "item_uuid": str(uuid.uuid4()),
            "env_specific_info": problem,
            "metadata": {"problem_name": problem["name"]},
            "max_tokens": self.config.max_proof_generation_tokens,
        }

    async def evaluate(self, *args, **kwargs):
        print(f"Evaluate called with args: {args}, kwargs: {kwargs}")
        eval_metrics = {"placeholder_eval_metric": random.random()}
        if self.wandb_available and wandb:
            await self.wandb_log({"eval": eval_metrics})
        else:
            print(f"[Metrics (wandb disabled) - Eval] {eval_metrics}")
        self.eval_metrics.append(eval_metrics)

    async def verify_lean_proof(
        self, theorem_header: str, proof_completion: str, timeout_seconds: int = 60
    ) -> Tuple[bool, str]:
        full_proof = theorem_header + "\n" + proof_completion
        print(
            f"LeanProofEnv.verify_lean_proof attempting to verify: {repr(full_proof)}"
        )
        return await verify_lean_proof(
            lean_executable_path=self.config.lean_executable_path,
            proof_content=full_proof,
            timeout_seconds=timeout_seconds,
        )

    async def collect_trajectories(
        self, item: Dict[str, Any]
    ) -> Tuple[Optional[ScoredDataGroup], List[Dict[str, Any]]]:
        problem_data = item["env_specific_info"]
        prompt_messages = item["history"]
        prompt_text_for_tokenizer = "\n".join(
            [msg["content"] for msg in prompt_messages if "content" in msg]
        )
        llm_raw_outputs = []
        try:
            llm_output_response = await self.chat_completion(
                messages=prompt_messages,
                model=item["model_name"],
                n=self.config.group_size,
                max_tokens=item.get(
                    "max_tokens", self.config.max_proof_generation_tokens
                ),
                temperature=0.7,
            )
            if llm_output_response.choices:
                llm_raw_outputs = [
                    choice.message.content
                    for choice in llm_output_response.choices
                    if choice.message.content is not None
                ]
            else:
                print(
                    f"Warning: LLM output for item {item.get('item_uuid')} has no choices."
                )
                llm_raw_outputs = [""] * self.config.group_size
        except Exception as e:
            print(f"Error during LLM call for item {item.get('item_uuid')}: {e}")
            llm_raw_outputs = [f"LLM_ERROR: {e}"] * self.config.group_size
        scores = []
        completions_data = []
        processed_messages_for_html = []
        for i, raw_llm_output in enumerate(llm_raw_outputs):
            cleaned_proof_steps = raw_llm_output.strip()
            if cleaned_proof_steps.startswith("```lean"):
                cleaned_proof_steps = cleaned_proof_steps[len("```lean") :].strip()
            elif cleaned_proof_steps.startswith("```"):
                cleaned_proof_steps = cleaned_proof_steps[len("```") :].strip()
            if cleaned_proof_steps.endswith("```"):
                cleaned_proof_steps = cleaned_proof_steps[: -len("```")].strip()
            theorem_header = problem_data["theorem_header"]
            is_valid, error_message = await self.verify_lean_proof(
                theorem_header=theorem_header,
                proof_completion=cleaned_proof_steps,
                timeout_seconds=self.config.proof_verification_timeout_seconds,
            )
            score = 1.0 if is_valid else 0.0
            scores.append(score)
            completions_data.append(
                {
                    "completion": cleaned_proof_steps,
                    "raw_llm_output": raw_llm_output,
                    "score": score,
                    "error_message": error_message if not is_valid else "",
                }
            )
            current_messages = [
                f"{msg_dict['role'].capitalize()}:\n\n{msg_dict['content']}"
                for msg_dict in prompt_messages
            ]
            current_messages.append(f"Assistant:\n\n{cleaned_proof_steps}")
            processed_messages_for_html.append("\n\n---\n\n".join(current_messages))
        try:
            tokens = (
                self.tokenizer.encode(prompt_text_for_tokenizer)
                if hasattr(self, "tokenizer") and self.tokenizer
                else []
            )
            masks = [1] * len(tokens) if tokens else []
        except Exception as e:
            print(f"Warning: Tokenization failed - {e}. Using empty tokens/masks.")
            tokens, masks = [], []
        scored_data = ScoredDataGroup(
            item_uuid=item["item_uuid"],
            messages=processed_messages_for_html,
            scores=scores,
            tokens=[tokens] * len(llm_raw_outputs),
            masks=[masks] * len(llm_raw_outputs),
            metadata=[
                {
                    "problem_name": problem_data["name"],
                    "attempt": i,
                    "error": comp_data["error_message"],
                }
                for i, comp_data in enumerate(completions_data)
            ],
        )
        step_details = []
        for i, comp_data in enumerate(completions_data):
            step_details.append(
                {
                    "problem_name": problem_data["name"],
                    "prompt": prompt_messages,
                    "raw_llm_output": comp_data["raw_llm_output"],
                    "cleaned_proof": comp_data["completion"],
                    "score": comp_data["score"],
                    "error_message": comp_data["error_message"],
                    "is_valid": comp_data["score"] == 1.0,
                }
            )
        if self.wandb_available and wandb:
            if (
                self.rollouts_table
                and hasattr(self.config, "num_rollouts_to_keep")
                and self.config.num_rollouts_to_keep > 0
            ):
                for detail in step_details:
                    self.rollouts_table.add_data(
                        item["item_uuid"],
                        problem_data["name"],
                        str(detail["prompt"]),
                        detail["cleaned_proof"],
                        detail["score"],
                        detail["error_message"],
                    )
        else:
            for detail in step_details:
                print(
                    f"Problem: {detail['problem_name']}, Valid: {detail['is_valid']}, "
                    f"Score: {detail['score']}, Proof: {repr(detail['cleaned_proof'])}, "
                    f"Error: {detail['error_message']}"
                )
        return scored_data, step_details


if __name__ == "__main__":
    LeanProofEnv.cli()
