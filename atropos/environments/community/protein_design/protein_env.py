import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import wandb
import yaml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    Item,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from .prompts import SYSTEM_PROMPT, construct_user_prompt
from .tool_definitions import ALL_TOOLS_LIST
from .tool_executor import ToolExecutor

logger = logging.getLogger(__name__)
load_dotenv()


def load_target_binder_pairs(
    dataset_name: str, target_col: str, binder_col: str, split: str = "train"
) -> Dataset:
    """
    Loads and transforms a Hugging Face dataset to contain only 'target' and 'binder' columns.

    Args:
        dataset_name (str): Hugging Face dataset identifier.
        target_col (str): Name of the column containing target protein sequences.
        binder_col (str): Name of the column containing binder sequences.
        split (str): Dataset split to load.

    Returns:
        Dataset: Hugging Face Dataset object with columns ['target', 'binder'].
    """

    ds = load_dataset(dataset_name, split=split)
    logger.info(f"Loaded dataset with columns: {ds.column_names}")
    actual_target_col = "receptor"
    actual_binder_col = "peptide"

    try:
        ds = ds.rename_columns(
            {actual_target_col: "target", actual_binder_col: "binder"}
        )
        ds = ds.remove_columns(
            [col for col in ds.column_names if col not in {"target", "binder"}]
        )
    except ValueError as e:
        logger.error(f"Error renaming columns: {e}")
        logger.error(f"Available columns: {ds.column_names}")
        if (
            actual_target_col in ds.column_names
            and actual_binder_col in ds.column_names
        ):
            ds = ds.select_columns([actual_target_col, actual_binder_col])
            ds = ds.rename_columns(
                {actual_target_col: "target", actual_binder_col: "binder"}
            )
        else:
            logger.error(
                f"Could not find expected columns in dataset. Available columns: {ds.column_names}"
            )
            raise ValueError(
                f"Dataset {dataset_name} doesn't have the expected columns. Please check your dataset configuration."
            )

    return ds


class BinderRow(TypedDict):
    target: str
    binder: str


class BinderBenchConfig(BaseEnvConfig):
    nim_api_key: Optional[str] = Field(None, description="NVIDIA NIM API key")
    nim_api_base_url: str = Field(
        "https://health.api.nvidia.com/v1", description="NIM API base URL"
    )
    api_timeout: int = Field(1800, description="Timeout for NIM API calls")
    polling_interval: int = Field(30, description="Polling interval for NIM jobs")
    output_dir: str = Field(
        default=str(Path(__file__).parent / "outputs"),
        description="Directory to save PDBs, etc.",
    )
    debug_protein_design_calls: bool = Field(
        False,
        description="Enable debug mode for NIM protein API calls, returning mock data.",
    )
    max_retries_per_internal_step: int = Field(
        100,
        description="Max retries for a failed tool call within a workflow step (0 means no retries).",
    )
    dataset_name: str = Field(
        "ronig/protein_binding_sequences", description="Dataset for target sequences"
    )
    target_col: str = Field(
        "receptor", description="Target column name (actual column in the dataset)"
    )
    binder_col: str = Field(
        "peptide", description="Binder column name (actual column in the dataset)"
    )


class BinderBenchEnv(BaseEnv):
    name = "binderbench"
    env_config_cls = BinderBenchConfig

    def __init__(
        self,
        config: BinderBenchConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: BinderBenchConfig
        self.process_mode = False
        self.tools = ALL_TOOLS_LIST
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_state = {}
        self.completed_episode_metrics: List[Dict] = []
        self.rollouts_for_wandb = []
        self.tool_executor = ToolExecutor(
            nim_api_key=self.config.nim_api_key,
            api_timeout=self.config.api_timeout,
            polling_interval=self.config.polling_interval,
            output_dir=self.output_dir,
            debug_protein_design_calls=self.config.debug_protein_design_calls,
        )

    async def _execute_tool(
        self, tool_name: str, args: Dict, workflow_state: Dict
    ) -> Dict:
        """Delegates tool execution and then updates workflow_state based on the result."""

        execution_result_package = await self.tool_executor.dispatch_tool_call(
            tool_name, args, workflow_state
        )
        tool_output = execution_result_package.get("tool_output", {})
        state_updates = execution_result_package.get("state_updates", {})
        if state_updates:
            workflow_state.update(state_updates)
            logger.debug(
                f"Workflow {workflow_state['item_id']}: State updated with keys: {list(state_updates.keys())}"
            )

        return tool_output

    @classmethod
    def config_init(cls) -> Tuple[BinderBenchConfig, List[APIServerConfig]]:
        default_yaml_path = (
            Path(__file__).parent / "configs" / "binderbench_default.yaml"
        )
        yaml_config_values = {}
        if default_yaml_path.exists():
            with open(default_yaml_path, "r") as f:
                yaml_config_values = yaml.safe_load(f) or {}

        env_config = BinderBenchConfig(
            use_wandb=True,
            wandb_name=cls.name,
            nim_api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
            debug_protein_design_calls=yaml_config_values.get(
                "debug_protein_design_calls",
                bool(os.environ.get("DEBUG_PROTEIN_DESIGN_CALLS", False)),
            ),
        )

        llm_api_key = os.environ.get("OPENAI_API_KEY")
        llm_base_url = os.environ.get("OPENAI_API_BASE")

        server_configs = [
            APIServerConfig(
                model_name=os.environ.get("DEFAULT_LLM_MODEL", "gpt-4-turbo"),
                api_key=llm_api_key,
                base_url=llm_base_url,
            )
        ]
        return env_config, server_configs

    async def setup(self):
        self.iter = 0
        self.train = load_target_binder_pairs(
            dataset_name=self.config.dataset_name,
            target_col=self.config.target_col,
            binder_col=self.config.binder_col,
        )
        logger.info(f"Loaded {len(self.train)} target-binder pairs for {self.name}.")

        if not self.config.nim_api_key:
            self.config.nim_api_key = os.environ.get("NVIDIA_NIM_API_KEY")
            if not self.config.nim_api_key:
                logger.warning(
                    "NVIDIA NIM API key not set. Protein design functions may not work properly."
                )

    def _initialize_workflow_state(
        self, item_id: str, target_sequence: str, ground_truth_binder: Optional[str]
    ) -> Dict:
        """Initializes or resets the state for a new workflow."""
        return {
            "item_id": item_id,
            "current_internal_step": 0,
            "target_sequence": target_sequence,
            "ground_truth_binder_sequence": ground_truth_binder,
            "target_pdb_content": None,
            "target_chain_details": None,
            "binder_backbone_pdb_content": None,
            "designed_binder_sequence": None,
            "complex_pdb_content_path": None,
            "af2_multimer_plddt": 0.0,
            "target_structure_predicted": False,
            "binder_backbone_designed": False,
            "binder_sequence_designed": False,
            "complex_evaluated": False,
            "workflow_complete_flag": False,
            "last_tool_success": True,
            "cumulative_reward": 0.0,
            "turn_messages_history": [],
            "retry_count_this_internal_step": 0,
            "previous_tool_error_message": None,
        }

    async def get_next_item(self) -> Item:
        """
        Provides the initial information for a new protein design workflow.
        Returns an Item tuple: (item_id, initial_target_sequence_info)
        """
        raw_item: BinderRow = self.train[self.iter % len(self.train)]
        self.iter += 1

        item_id = str(uuid.uuid4())
        target_sequence = raw_item["target"]
        ground_truth_binder = raw_item.get("binder")

        self.episodes_state[item_id] = self._initialize_workflow_state(
            item_id, target_sequence, ground_truth_binder
        )

        return item_id

    def reset_state(self, item_id: str) -> dict:
        """Retrieves the workflow state for the given item_id."""
        if item_id in self.episodes_state:
            return self.episodes_state[item_id]
        else:
            logger.error(
                f"No state found for item_id {item_id}. Creating a default state."
            )
            return self._initialize_workflow_state(item_id, "", None)

    async def collect_trajectories(
        self, item_id: str
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        workflow_state = self.episodes_state.get(item_id)
        if not workflow_state:
            logger.error(f"Workflow state for item_id {item_id} not found. Skipping.")
            return None, []

        if workflow_state.get("workflow_complete_flag"):
            logger.info(f"Workflow for {item_id} already marked complete. Skipping.")
            return None, []

        is_processing_mode = getattr(self, "process_mode", False)  # Check the flag

        if is_processing_mode:
            all_turns_data_for_jsonl = []
            MAX_INTERNAL_STEPS = 4

            while workflow_state[
                "current_internal_step"
            ] < MAX_INTERNAL_STEPS and not workflow_state.get("workflow_complete_flag"):

                current_turn_messages: List[Message] = []
                user_prompt_str = construct_user_prompt(workflow_state)
                current_turn_messages.append(
                    Message(role="system", content=SYSTEM_PROMPT)
                )
                current_turn_messages.append(
                    Message(role="user", content=user_prompt_str)
                )

                llm_response = await self.server.chat_completion(
                    messages=current_turn_messages,
                    tools=self.tools,
                    tool_choice="auto",
                    n=1,
                    max_tokens=self.config.max_token_length,
                    temperature=0.5,
                )
                assistant_message_obj = llm_response.choices[0].message
                assistant_content = assistant_message_obj.content or ""
                assistant_tool_calls = []
                if (
                    hasattr(assistant_message_obj, "tool_calls")
                    and assistant_message_obj.tool_calls
                ):
                    assistant_tool_calls = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_message_obj.tool_calls
                    ]
                current_turn_messages.append(
                    Message(
                        role="assistant",
                        content=assistant_content,
                        tool_calls=(
                            assistant_tool_calls if assistant_tool_calls else None
                        ),
                    )
                )

                tool_error_for_retry_prompt = None
                if assistant_tool_calls:
                    tool_call_request = assistant_tool_calls[0]
                    tool_name = tool_call_request["function"]["name"]
                    try:
                        tool_args = json.loads(
                            tool_call_request["function"]["arguments"]
                        )
                        tool_result = await self._execute_tool(
                            tool_name, tool_args, workflow_state
                        )
                        current_turn_messages.append(
                            Message(
                                role="tool",
                                tool_call_id=tool_call_request["id"],
                                name=tool_name,
                                content=json.dumps(tool_result),
                            )
                        )
                        workflow_state["last_tool_success"] = tool_result.get(
                            "success", False
                        )
                        if not workflow_state["last_tool_success"]:
                            tool_error_for_retry_prompt = tool_result.get(
                                "error", "Tool execution failed."
                            )
                    except Exception as e:
                        error_msg = f"Error processing tool {tool_name}: {str(e)}"
                        current_turn_messages.append(
                            Message(
                                role="tool",
                                tool_call_id=tool_call_request["id"],
                                name=tool_name,
                                content=error_msg,
                            )
                        )
                        workflow_state["last_tool_success"] = False
                        tool_error_for_retry_prompt = error_msg
                else:
                    workflow_state["last_tool_success"] = False
                    expected_tool_name = {
                        0: "AF2",
                        1: "RFD",
                        2: "PMPNN",
                        3: "AF2M",
                    }.get(workflow_state["current_internal_step"], "a tool")
                    tool_error_for_retry_prompt = (
                        f"No tool was called, but {expected_tool_name} was expected."
                    )

                workflow_state["previous_tool_error_message"] = (
                    tool_error_for_retry_prompt
                )

                turn_score_details = self._score_trajectory(
                    current_turn_messages, workflow_state
                )
                current_turn_reward = turn_score_details.get("overall_reward", 0.0)
                workflow_state["cumulative_reward"] += current_turn_reward

                tokenization_result = tokenize_for_trainer(
                    self.tokenizer, current_turn_messages, include_messages=False
                )
                all_turns_data_for_jsonl.append(
                    {
                        "tokens_this_turn": tokenization_result["tokens"],
                        "masks_this_turn": tokenization_result["masks"],
                        "score_this_turn": current_turn_reward,
                        "messages_this_turn": current_turn_messages.copy(),
                        "overrides_this_turn": turn_score_details.copy(),
                    }
                )

                if workflow_state["last_tool_success"]:
                    workflow_state["current_internal_step"] += 1
                    workflow_state["retry_count_this_internal_step"] = 0
                    workflow_state["previous_tool_error_message"] = None
                else:
                    if workflow_state["current_internal_step"] <= 3:
                        workflow_state["retry_count_this_internal_step"] += 1
                        if (
                            workflow_state["retry_count_this_internal_step"]
                            > self.config.max_retries_per_internal_step
                        ):
                            logger.warning(
                                f"Workflow {item_id}, Step {workflow_state['current_internal_step']}: "
                                f"Max retries ({self.config.max_retries_per_internal_step}) reached. "
                                f"Terminating workflow for this item."
                            )
                            workflow_state["workflow_complete_flag"] = True
                            break
                        else:
                            logger.info(
                                f"Workflow {item_id}, Step {workflow_state['current_internal_step']}: "
                                f"Failed, attempt {workflow_state['retry_count_this_internal_step']}. "
                                f"Retrying same step."
                            )

                    else:
                        logger.warning(
                            f"Workflow {item_id}, Step {workflow_state['current_internal_step']}: "
                            f"Failure at non-retryable step. Terminating workflow."
                        )
                        workflow_state["workflow_complete_flag"] = True
                        break

                if workflow_state["current_internal_step"] >= MAX_INTERNAL_STEPS:
                    workflow_state["workflow_complete_flag"] = True
                    logger.info(
                        f"Workflow {item_id}: All internal steps completed successfully."
                    )

            if not all_turns_data_for_jsonl:
                logger.warning(
                    f"Workflow {item_id} in process mode: No turn data collected."
                )
                return None, []

            html_compatible_messages: List[str] = []
            html_compatible_scores: List[float] = []
            overrides_for_jsonl: List[Dict[str, Any]] = []

            for turn_idx, turn_data in enumerate(all_turns_data_for_jsonl):
                turn_str_parts = [f"--- Workflow {item_id} - Turn {turn_idx + 1} ---"]
                if turn_data.get("messages_this_turn"):
                    for msg_obj in turn_data["messages_this_turn"]:
                        content_str = str(msg_obj.get("content", "[No Content]"))
                        if msg_obj.get("tool_calls"):
                            try:
                                tool_calls_str = json.dumps(
                                    msg_obj.get("tool_calls"), indent=2
                                )
                                content_str += f"\nTool Calls:\n{tool_calls_str}"
                            except TypeError:  # Handle non-serializable content if any
                                content_str += (
                                    "\nTool Calls: [Error serializing tool_calls]"
                                )
                        turn_str_parts.append(
                            f"**{msg_obj.get('role', 'unknown').upper()}**: {content_str}"
                        )
                else:
                    turn_str_parts.append("No messages recorded for this turn.")

                html_compatible_messages.append("\n\n".join(turn_str_parts))

                turn_score = turn_data.get("overrides_this_turn", {}).get(
                    "overall_reward", 0.0
                )
                html_compatible_scores.append(turn_score)

                overrides_for_jsonl.append(turn_data.get("overrides_this_turn", {}))

            final_workflow_reward = workflow_state.get("cumulative_reward", 0.0)
            if workflow_state.get("complex_evaluated") and workflow_state.get(
                "last_tool_success"
            ):
                final_workflow_reward = (
                    all_turns_data_for_jsonl[-1]
                    .get("overrides_this_turn", {})
                    .get("overall_reward", 0.0)
                )

            all_tokens_per_turn = [
                turn_data["tokens_this_turn"]
                for turn_data in all_turns_data_for_jsonl
                if turn_data.get("tokens_this_turn")
            ]
            all_masks_per_turn = [
                turn_data["masks_this_turn"]
                for turn_data in all_turns_data_for_jsonl
                if turn_data.get("masks_this_turn")
            ]

            if len(all_tokens_per_turn) != len(html_compatible_messages):
                logger.error(
                    f"CRITICAL: Mismatch between tokenized turns ({len(all_tokens_per_turn)}) "
                    f"and HTML messages ({len(html_compatible_messages)}). JSONL will be problematic."
                )
                if all_turns_data_for_jsonl and all_tokens_per_turn:
                    last_tokens = all_tokens_per_turn[-1]
                    last_masks = all_masks_per_turn[-1]
                    all_tokens_per_turn = [last_tokens] * len(html_compatible_messages)
                    all_masks_per_turn = [last_masks] * len(html_compatible_messages)
                else:
                    all_tokens_per_turn = [[] for _ in html_compatible_messages]
                    all_masks_per_turn = [[] for _ in html_compatible_messages]

            process_mode_scored_data = ScoredDataGroup(
                tokens=all_tokens_per_turn,
                masks=all_masks_per_turn,
                messages=html_compatible_messages,
                scores=html_compatible_scores,
                overrides=overrides_for_jsonl,
                group_overrides={
                    "group_size": len(html_compatible_messages),
                    "item_id": item_id,
                    "is_process_mode_full_workflow": True,
                    "final_score_for_workflow": final_workflow_reward,
                    "target_sequence": workflow_state.get("target_sequence", "N/A"),
                    "designed_binder_sequence": workflow_state.get(
                        "designed_binder_sequence", "N/A"
                    ),
                    "final_plddt": workflow_state.get("af2_multimer_plddt", 0.0),
                },
            )

            await self.add_rollouts_for_wandb(data_for_log=workflow_state.copy())

            self.completed_episode_metrics.append(workflow_state.copy())
            if item_id in self.episodes_state:
                del self.episodes_state[item_id]
            return process_mode_scored_data, []

        else:
            current_turn_messages_serve: List[Message] = []
            user_prompt_str_serve = construct_user_prompt(workflow_state)
            current_turn_messages_serve.append(
                Message(role="system", content=SYSTEM_PROMPT)
            )
            current_turn_messages_serve.append(
                Message(role="user", content=user_prompt_str_serve)
            )

            llm_response_serve = await self.server.chat_completion(
                messages=current_turn_messages_serve,
                tools=self.tools,
                tool_choice="auto",
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.5,
            )
            assistant_message_obj_serve = llm_response_serve.choices[0].message
            assistant_content_serve = assistant_message_obj_serve.content or ""
            assistant_tool_calls_serve = []
            if (
                hasattr(assistant_message_obj_serve, "tool_calls")
                and assistant_message_obj_serve.tool_calls
            ):
                assistant_tool_calls_serve = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message_obj_serve.tool_calls
                ]
            current_turn_messages_serve.append(
                Message(
                    role="assistant",
                    content=assistant_content_serve,
                    tool_calls=(
                        assistant_tool_calls_serve
                        if assistant_tool_calls_serve
                        else None
                    ),
                )
            )

            tool_error_for_retry_prompt_serve = None
            if assistant_tool_calls_serve:
                tool_call_request_serve = assistant_tool_calls_serve[0]
                tool_name_serve = tool_call_request_serve["function"]["name"]
                try:
                    tool_args_json_str = tool_call_request_serve["function"][
                        "arguments"
                    ]
                    tool_args_serve = json.loads(tool_args_json_str)
                    tool_result_serve = await self._execute_tool(
                        tool_name_serve, tool_args_serve, workflow_state
                    )
                    current_turn_messages_serve.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_call_request_serve["id"],
                            name=tool_name_serve,
                            content=json.dumps(tool_result_serve),
                        )
                    )
                    workflow_state["last_tool_success"] = tool_result_serve.get(
                        "success", False
                    )
                    if not workflow_state["last_tool_success"]:
                        tool_error_for_retry_prompt_serve = tool_result_serve.get(
                            "error", "Tool execution failed."
                        )
                except Exception as e:
                    error_msg_serve = (
                        f"Error processing tool {tool_name_serve}: {str(e)}"
                    )
                    current_turn_messages_serve.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_call_request_serve["id"],
                            name=tool_name_serve,
                            content=error_msg_serve,
                        )
                    )
                    workflow_state["last_tool_success"] = False
                    tool_error_for_retry_prompt_serve = error_msg_serve
            else:
                workflow_state["last_tool_success"] = False
                expected_tool_name_serve = {
                    0: "AF2",
                    1: "RFD",
                    2: "PMPNN",
                    3: "AF2M",
                }.get(workflow_state["current_internal_step"], "a tool")
                tool_error_for_retry_prompt_serve = (
                    f"No tool was called, but {expected_tool_name_serve} was expected."
                )

            workflow_state["previous_tool_error_message"] = (
                tool_error_for_retry_prompt_serve
            )

            turn_score_details_serve = self._score_trajectory(
                current_turn_messages_serve, workflow_state
            )
            current_turn_reward_serve = turn_score_details_serve.get(
                "overall_reward", 0.0
            )
            workflow_state["cumulative_reward"] += current_turn_reward_serve
            workflow_state["turn_messages_history"].append(
                current_turn_messages_serve.copy()
            )

            tokenization_result_serve = tokenize_for_trainer(
                self.tokenizer,
                current_turn_messages_serve,
                include_messages=self.config.include_messages,
            )
            scored_data_serve = ScoredDataGroup(
                tokens=[tokenization_result_serve["tokens"]],
                masks=[tokenization_result_serve["masks"]],
                scores=[current_turn_reward_serve],
                messages=(
                    [current_turn_messages_serve]
                    if self.config.include_messages
                    else None
                ),
                overrides=[turn_score_details_serve],
                group_overrides={"group_size": 1},
            )

            backlog_items_serve = []
            if workflow_state["last_tool_success"]:
                workflow_state["current_internal_step"] += 1
                workflow_state["retry_count_this_internal_step"] = 0
                workflow_state["previous_tool_error_message"] = None
            else:
                if workflow_state["current_internal_step"] <= 3:
                    workflow_state["retry_count_this_internal_step"] += 1
                    if (
                        workflow_state["retry_count_this_internal_step"]
                        > self.config.max_retries_per_internal_step
                    ):
                        logger.warning(
                            f"Workflow {item_id}, Step {workflow_state['current_internal_step']} "
                            f"(Serve Mode): Max retries reached. Terminating."
                        )
                        workflow_state["workflow_complete_flag"] = True
                else:
                    logger.warning(
                        f"Workflow {item_id}, Step {workflow_state['current_internal_step']} "
                        f"(Serve Mode): Failure at non-retryable step. Terminating."
                    )
                    workflow_state["workflow_complete_flag"] = True

            if workflow_state["current_internal_step"] < 4 and not workflow_state.get(
                "workflow_complete_flag"
            ):
                should_add_to_backlog = False
                if workflow_state["last_tool_success"]:
                    should_add_to_backlog = True
                elif (
                    workflow_state["current_internal_step"] <= 3
                    and workflow_state["retry_count_this_internal_step"]
                    <= self.config.max_retries_per_internal_step
                ):
                    should_add_to_backlog = True

                if should_add_to_backlog:
                    backlog_items_serve.append(item_id)
                else:
                    workflow_state["workflow_complete_flag"] = True
                    logger.info(
                        f"Workflow for {item_id} (Serve Mode) not added to backlog and marked complete. "
                        f"Internal step: {workflow_state['current_internal_step']}"
                    )

            if workflow_state.get("workflow_complete_flag"):
                if item_id in self.episodes_state:
                    await self.add_rollouts_for_wandb(
                        data_for_log=self.episodes_state[item_id].copy()
                    )
                    self.completed_episode_metrics.append(
                        self.episodes_state[item_id].copy()
                    )
                    del self.episodes_state[item_id]

            return scored_data_serve, backlog_items_serve

    def _score_trajectory(
        self, turn_messages: List[Message], workflow_state: Dict
    ) -> Dict[str, float]:
        """
        Scores a single turn's trajectory based on the specified reward logic.
        - Steps 0-2: Format reward (0.2 for correct & successful tool call, 0 otherwise).
        - Step 3 (AF2-Multimer): Reward based on pLDDT.
        """
        detailed_scores = {
            "overall_reward": 0.0,
            "raw_plddt": 0.0,
        }

        internal_step = workflow_state.get("current_internal_step")
        last_tool_success = workflow_state.get("last_tool_success", False)
        assistant_msg_dict = next(
            (m for m in reversed(turn_messages) if m.get("role") == "assistant"), None
        )

        expected_tool_for_step = {
            0: "predict_target_structure_alphafold2",
            1: "design_binder_backbone_rfdiffusion",
            2: "design_binder_sequence_proteinmpnn",
            3: "evaluate_binder_complex_alphafold2_multimer",
        }.get(internal_step)

        called_tool_name = None
        if assistant_msg_dict and assistant_msg_dict.get("tool_calls"):
            tool_calls_list = assistant_msg_dict.get("tool_calls")
            if (
                tool_calls_list
                and isinstance(tool_calls_list, list)
                and len(tool_calls_list) > 0
            ):
                function_call_dict = tool_calls_list[0].get("function")
                if function_call_dict and isinstance(function_call_dict, dict):
                    called_tool_name = function_call_dict.get("name")

        if internal_step < 3:
            if last_tool_success and called_tool_name == expected_tool_for_step:
                detailed_scores["overall_reward"] = 0.2
                logger.info(
                    f"Workflow {workflow_state['item_id']}, Step {internal_step}: "
                    f"Correct tool '{called_tool_name}' used successfully. Reward: 0.2"
                )
            else:
                detailed_scores["overall_reward"] = 0.0
                if not last_tool_success and called_tool_name:
                    logger.warning(
                        f"Workflow {workflow_state['item_id']}, Step {internal_step}: "
                        f"Tool '{called_tool_name}' execution failed. Reward: 0.0"
                    )
                elif called_tool_name != expected_tool_for_step:
                    logger.warning(
                        f"Workflow {workflow_state['item_id']}, Step {internal_step}: "
                        f"Incorrect tool '{called_tool_name}' used (expected '{expected_tool_for_step}'). "
                        f"Reward: 0.0"
                    )
                elif not called_tool_name and expected_tool_for_step:
                    logger.warning(
                        f"Workflow {workflow_state['item_id']}, Step {internal_step}: "
                        f"No tool called, but expected '{expected_tool_for_step}'. Reward: 0.0"
                    )

        elif internal_step == 3:
            if (
                workflow_state.get("complex_evaluated")
                and last_tool_success
                and called_tool_name == expected_tool_for_step
            ):
                plddt = workflow_state.get("af2_multimer_plddt", 0.0)
                detailed_scores["raw_plddt"] = plddt

                if plddt > 90.0:
                    detailed_scores["overall_reward"] = 1.0
                elif plddt > 50.0:
                    detailed_scores["overall_reward"] = 0.0 + (plddt - 50.0) * (
                        1.0 - 0.0
                    ) / (90.0 - 50.0)
                    detailed_scores["overall_reward"] = max(
                        0.0, min(detailed_scores["overall_reward"], 1.0)
                    )
                else:
                    detailed_scores["overall_reward"] = 0.0

                logger.info(
                    f"Workflow {workflow_state['item_id']}, Step {internal_step} (AF2-Multimer): "
                    f"pLDDT={plddt:.2f}. Reward: {detailed_scores['overall_reward']:.2f}"
                )
            else:
                detailed_scores["overall_reward"] = 0.0
                logger.warning(
                    f"Workflow {workflow_state['item_id']}, Step {internal_step} (AF2-Multimer): "
                    f"Evaluation failed or wrong tool. Reward: 0.0. Last tool success: {last_tool_success}, "
                    f"Called: {called_tool_name}"
                )

        else:
            logger.error(
                f"Workflow {workflow_state['item_id']}: "
                f"Invalid internal_step {internal_step} in scoring."
            )
            detailed_scores["overall_reward"] = -1.0

        return detailed_scores

    async def postprocess_histories(
        self, trajectories: Optional[ScoredDataGroup]
    ) -> Optional[ScoredDataGroup]:
        """
        Post-processes a ScoredDataGroup for a single turn.
        Can be used for final adjustments or filtering if needed.
        """
        return trajectories

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment's performance.
        This method is called periodically by the BaseEnv.env_manager.
        For BinderBenchEnv, it will aggregate metrics from completed workflows.
        """
        logger.info(f"Running evaluation for {self.name}...")
        if not self.completed_episode_metrics:
            logger.info("No completed episodes to evaluate since last evaluation.")
            self.eval_metrics = (
                []
            )  # Ensure eval_metrics is an empty list if no new data
            if self.config.use_wandb:
                await self.wandb_log({})  # Log that no eval data was present this cycle
            return

        plddts, cumulative_rewards, workflow_successes = [], [], []
        current_eval_episodes = self.completed_episode_metrics.copy()
        for ep_state in current_eval_episodes:
            if ep_state.get("complex_evaluated") and ep_state.get("last_tool_success"):
                plddts.append(ep_state.get("af2_multimer_plddt", 0.0))
                workflow_successes.append(1.0)
            else:
                workflow_successes.append(0.0)
            cumulative_rewards.append(ep_state.get("cumulative_reward", 0.0))

        self.eval_metrics = []  # Reset class member for current evaluation results
        if plddts:
            self.eval_metrics.append(("eval/avg_plddt", sum(plddts) / len(plddts)))
        if cumulative_rewards:
            self.eval_metrics.append(
                (
                    "eval/avg_cumulative_reward",
                    sum(cumulative_rewards) / len(cumulative_rewards),
                )
            )
        if workflow_successes:
            self.eval_metrics.append(
                (
                    "eval/workflow_success_rate",
                    sum(workflow_successes) / len(workflow_successes),
                )
            )

        logger.info(f"Evaluation complete. Calculated metrics: {self.eval_metrics}")

        if self.config.use_wandb:
            await self.wandb_log({})

        self.completed_episode_metrics.clear()

    async def add_rollouts_for_wandb(
        self,
        scored_data_group: ScoredDataGroup = None,
        item_id: Item = None,
        data_for_log: Dict = None,
    ):
        """Adds a workflow summary to the wandb rollout buffer.

        This method has two modes of operation:
        1. Direct logging with workflow_state (preferred for detailed logging):
           - Called from within collect_trajectories with data_for_log=workflow_state.copy()
           - This provides maximum detail for logging

        2. BaseEnv compatibility mode:
           - Called from BaseEnv.handle_send_to_api with scored_data_group and item_id
           - Used automatically by the framework
           - May have less detail if workflow_state was already deleted

        Args:
            scored_data_group: The ScoredDataGroup containing token, mask, and score data (from BaseEnv)
            item_id: The item identifier, which is the key to our episodes_state (from BaseEnv)
            data_for_log: Direct workflow state to log (our custom parameter for direct logging)
        """
        if not self.config.use_wandb or not hasattr(self, "rollouts_for_wandb"):
            if not hasattr(self, "rollouts_for_wandb"):
                self.rollouts_for_wandb = []

        workflow_state = None

        if data_for_log is not None and isinstance(data_for_log, dict):
            workflow_state = data_for_log
            if item_id is None and "item_id" in workflow_state:
                item_id = workflow_state["item_id"]

        elif item_id is not None and item_id in self.episodes_state:
            workflow_state = self.episodes_state[item_id]

        if workflow_state is None:
            logger.debug(
                f"No workflow_state available for WandB logging (item_id={item_id})"
            )
            return

        target_seq = workflow_state.get("target_sequence", "N/A")

        plddt = workflow_state.get("af2_multimer_plddt", 0.0)
        cumulative_reward = workflow_state.get("cumulative_reward", 0.0)

        last_turn_messages_str = "No messages."
        try:
            if (
                workflow_state.get("turn_messages_history")
                and len(workflow_state["turn_messages_history"]) > 0
            ):
                last_turn_convo = workflow_state["turn_messages_history"][-1]
                last_turn_messages_str = "\n---\n".join(
                    [
                        f"{msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:200]}..."
                        for msg in last_turn_convo
                    ]
                )
        except Exception as e:
            logger.error(f"Error processing messages for WandB: {e}")
            last_turn_messages_str = "Error processing messages"

        target_preview = (
            target_seq[:30] + "..."
            if isinstance(target_seq, str) and len(target_seq) > 30
            else target_seq
        )
        designed_binder_data = workflow_state.get("designed_binder_sequence", "N/A")

        binder_preview = "N/A"
        if isinstance(designed_binder_data, list) and designed_binder_data:
            first_chain_seq = str(designed_binder_data[0])
            preview_text = (
                first_chain_seq[:30] + "..."
                if len(first_chain_seq) > 30
                else first_chain_seq
            )
            if len(designed_binder_data) > 1:
                binder_preview = f"{len(designed_binder_data)} chains: {preview_text}"
            else:
                binder_preview = preview_text
        elif isinstance(designed_binder_data, str) and designed_binder_data != "N/A":
            binder_preview = (
                designed_binder_data[:30] + "..."
                if len(designed_binder_data) > 30
                else designed_binder_data
            )

        if item_id is None:
            item_id = workflow_state.get("item_id", "unknown-id")

        self.rollouts_for_wandb.append(
            (
                str(item_id),
                target_preview,
                binder_preview,
                f"{plddt:.2f}",
                f"{cumulative_reward:.3f}",
                last_turn_messages_str,
            )
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Creates a wandb.Table from the buffered rollouts."""
        if hasattr(self, "rollouts_for_wandb") and self.rollouts_for_wandb:
            columns = [
                "Item ID",
                "Target (Preview)",
                "Designed Binder (Preview)",
                "Final pLDDT",
                "Cumulative Reward",
                "Last Turn Messages",
            ]
            table = wandb.Table(columns=columns)
            for rollout_tuple in self.rollouts_for_wandb:
                table.add_data(*rollout_tuple)

            table_key = f"env_rollouts/{self.wandb_prepend}/completed_workflows"
            if self.wandb_prepend is None and hasattr(self, "name"):
                table_key = f"env_rollouts/{self.name}/completed_workflows"

            wandb_metrics[table_key] = table
            self.rollouts_for_wandb.clear()
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if hasattr(self, "rollouts_for_wandb") and self.rollouts_for_wandb:
            wandb_metrics = await self.create_rollout_table(wandb_metrics)

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    BinderBenchEnv.cli()
