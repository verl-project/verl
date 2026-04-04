from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import wandb
from pydantic import Field

from atroposlib.envs.base import (
    APIServer,
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)


class MetRLConfig(BaseEnvConfig):
    tokenizer_name: str = Field(default="Qwen/Qwen3-8B")
    group_size: int = Field(default=2)
    use_wandb: bool = Field(default=True)
    max_num_workers: int = Field(default=64)
    rollout_server_url: str = Field(default="http://localhost:8000")
    total_steps: int = Field(default=2000)
    batch_size: int = Field(default=-1)
    steps_per_eval: int = Field(default=100)
    max_token_length: int = Field(default=2048)
    inference_weight: float = Field(default=1.0)
    wandb_name: Optional[str] = Field(default=None)
    data_path_to_save_groups: Optional[str] = Field(
        default="data/MeteorologyForecastRL.jsonl"
    )
    eval_handling: EvalHandlingEnum = Field(default=EvalHandlingEnum.STOP_TRAIN)
    eval_limit_ratio: float = Field(default=0.5)
    num_eval_samples: int = Field(default=20)
    num_rollouts_to_log: int = Field(default=10)
    min_items_sent_before_logging: int = Field(default=2)
    include_messages: bool = Field(default=True)
    num_rollouts_to_keep: int = Field(default=32)
    num_rollouts_per_group_for_logging: int = Field(default=1)
    ensure_scores_are_not_same: bool = Field(default=False)
    max_eval_workers: int = Field(default=16)
    max_num_workers_per_node: int = Field(default=8)
    max_batches_offpolicy: int = Field(default=3)

    sounding_data_root: str = Field(
        default="environments/community/meteorology_forecast/data/",
        description="Root directory for all sounding and AFD data.",
    )
    target_date: str = Field(
        default="20250314",
        description="The specific date to load data for (YYYYMMDD format).",
    )
    judge_model_name: str = Field(
        default="google/gemini-2.5-flash-preview",
        description="Identifier for the Judge model on OpenRouter.",
    )
    judge_api_key_env_var: str = Field(
        default="OPENROUTER_API_KEY",
        description="Environment variable name for OpenRouter API key for the Judge.",
    )
    judge_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the OpenRouter API (for Judge).",
    )
    nwp_models_to_use: List[str] = Field(
        default=["RAP"], description="List of NWP models to use (e.g., RAP, HRRR)."
    )
    forecast_hours_to_sample: List[int] = Field(
        default=[6, 9, 12, 15, 18],
        description="Which forecast hours (UTC) from the model run to provide to the LLM.",
    )
    target_forecast_hour_offset: int = Field(
        default=1,
        description="Offset from the latest provided sounding hour to set the target forecast time.",
    )
    max_afds_for_judge: int = Field(
        default=3,
        description="Maximum number of AFD files to provide to the judge model.",
    )
    max_reasoning_tokens_llm: int = Field(
        default=3000, description="Max tokens for the agent LLM's generation."
    )
    max_tokens_judge: int = Field(
        default=2000, description="Max tokens for the judge model's generation."
    )


AGENT_SYSTEM_PROMPT = """You are a highly skilled AI meteorologist. Your task is to analyze
numerical weather prediction (NWP) model sounding data for a specific location and time period.
Based on your analysis, you must:
1.  Provide a detailed step-by-step reasoning process. This should include identifying trends,
    interpreting meteorological parameters, and connecting them to potential weather phenomena.
2.  If you determine that additional real-time observational data is crucial for a more accurate
    assessment, specify the tools you would use. For each tool, output a line in the exact format:
    TOOL_CALL: {{"tool_name": "tool_name_here", "arguments": {{"param1": "value1", ...}}}}
    Available conceptual tools: get_surface_observations, get_latest_radar_imagery,
    get_satellite_imagery, get_upper_air_sounding.
3.  Conclude with a concise forecast summary for the specified target time. Start this summary
    with "FORECAST_SUMMARY: ".

Analyze the provided data thoroughly. Your reasoning should be comprehensive."""

AGENT_USER_PROMPT_TEMPLATE = """Please analyze the following NWP model sounding data for station {location_id}.
The soundings provided are from the {model_name} model, run on {run_date_full_z}, valid at the
following UTC times: {sounding_times_str}.
Your goal is to make a preliminary forecast assessment focusing on severe weather potential for
{location_id} around {target_forecast_time_utc}.

Sounding Data:
{soundings_json_blob}

Remember to include your reasoning, any TOOL_CALL: {{"tool_name": "tool_name_here",
"arguments": {{"param1": "value1", ...}}}} lines, and a final FORECAST_SUMMARY: statement."""

JUDGE_SYSTEM_PROMPT = """You are an expert meteorologist acting as a judge. You will evaluate
an AI assistant's analysis of model sounding data.
The AI was asked to provide reasoning, call tools if necessary, and give a forecast summary.
You will be given the AI's output and relevant Area Forecast Discussions (AFDs) from human forecasters for context.

Your evaluation should focus on:
1.  **Meteorological Soundness of Reasoning (0-5 points):**
    *   Correct interpretation of sounding parameters and trends.
    *   Logical connections between data and potential weather.
    *   Avoidance of meteorological fallacies or hallucinations.
    *   Depth and detail of the thought process.
2.  **Tool Call Relevance & Justification (0-3 points):**
    *   Were the tools called (if any) appropriate given the AI's reasoning and the model data?
    *   Would these tools genuinely help a meteorologist in this situation?
    *   Were critical tool calls missed?
3.  **Forecast Summary Quality (0-2 points):**
    *   Clarity and conciseness.
    *   Alignment with the AI's own reasoning and the provided AFDs (or sensible deviation if model
        data strongly suggested it).

Provide a numerical score for each category and a total score (max 10.0). Also, provide a brief
overall justification for your scores.
Your output MUST be in the following exact format:
REASONING_SCORE: {{{{0-5 score}}}}
TOOL_CALL_SCORE: {{{{0-3 score}}}}
FORECAST_SUMMARY_SCORE: {{{{0-2 score}}}}
TOTAL_SCORE: {{{{sum of scores, e.g., 7.5}}}}
JUSTIFICATION: {{{{Your brief textual justification here.}}}}"""

JUDGE_USER_PROMPT_TEMPLATE = """AI Assistant's Output:
---
{llm_full_output}
---

Contextual Area Forecast Discussions (AFDs):
---
{afds_blob}
---

Please evaluate the AI assistant's output based on the criteria and provide your scores and
justification in the specified format."""


@dataclass
class CaseData:
    case_id: str
    location_id: str
    model_name: str
    run_date_full_z: str
    target_forecast_time_utc: str
    model_soundings_data: List[Any]
    sounding_times_str: str
    afd_texts: List[str]


class MeteorologyForecastRLEnv(BaseEnv):
    env_config_cls = MetRLConfig
    name = "MeteorologyForecastRL"

    def __init__(
        self,
        config: MetRLConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ) -> None:
        super().__init__(config, server_configs, slurm, testing)
        self.agent_llm_server: Optional[APIServer] = None
        self.judge_server: Optional[APIServer] = None
        if self.server.servers:
            self.agent_llm_server = self.server.servers[0]
            self.judge_server = self.server.servers[-1]
        self.cases: List[CaseData] = []
        self.current_idx = 0
        self.iter = 0
        self.judge_scores_buffer: List[float] = []
        self.eval_scores_buffer: List[Dict[str, float]] = []
        self.rollouts_for_wandb: List[Tuple[str, str, str, str, float, str]] = []

    async def add_rollouts_for_wandb(
        self,
        scored_data: ScoredDataGroup | List[ScoredDataGroup],
        item: CaseData | None = None,
    ) -> None:
        """Override BaseEnv behavior to avoid adding default rollout entries."""
        return

    @classmethod
    def config_init(cls) -> Tuple[MetRLConfig, List[APIServerConfig]]:
        env_config = MetRLConfig()
        agent_model_name = os.environ.get(
            "AGENT_LLM_MODEL_NAME", env_config.tokenizer_name
        )
        agent_api_key = os.environ.get("AGENT_LLM_API_KEY", "EMPTY_KEY_IF_LOCAL_VLLM")
        agent_base_url = os.environ.get(
            "AGENT_LLM_BASE_URL", "http://localhost:8080/v1"
        )
        judge_api_key = os.environ.get(env_config.judge_api_key_env_var)
        if not judge_api_key:
            logging.warning(
                f"Environment variable {env_config.judge_api_key_env_var} not set for Judge API."
            )
        server_configs = [
            APIServerConfig(
                model_name=agent_model_name,
                base_url=agent_base_url,
                api_key=agent_api_key,
            ),
            APIServerConfig(
                model_name=env_config.judge_model_name,
                base_url=env_config.judge_base_url,
                api_key=judge_api_key,
            ),
        ]
        return env_config, server_configs

    async def setup(self) -> None:
        data_root = Path(self.config.sounding_data_root)
        date_path = data_root / self.config.target_date
        if not date_path.is_dir():
            logger.error("Target date directory not found: %s", date_path)
            return
        for loc in sorted(date_path.iterdir()):
            if not loc.is_dir():
                continue
            soundings: List[Any] = []
            sounding_times: List[str] = []
            model_name = self.config.nwp_models_to_use[0]
            found_hours: set[int] = set()
            pattern = f"{loc.name}_{model_name}_{self.config.target_date}*Z.buf_default_llm_optimized.jsonl"
            for path in sorted(loc.glob(pattern)):
                with open(path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        tm = data.get("tm")
                        if not tm:
                            continue
                        line_hour = int(tm.split("/")[1][:2])
                        if line_hour not in self.config.forecast_hours_to_sample:
                            continue
                        if line_hour in found_hours:
                            continue
                        soundings.append(data)
                        sounding_times.append(f"{line_hour:02d}00Z")
                        found_hours.add(line_hour)
                        if len(found_hours) == len(
                            self.config.forecast_hours_to_sample
                        ):
                            break
                if len(found_hours) == len(self.config.forecast_hours_to_sample):
                    break

            if not soundings:
                continue
            pairs = sorted(zip(sounding_times, soundings), key=lambda x: int(x[0][:2]))
            sounding_times = [p[0] for p in pairs]
            soundings = [p[1] for p in pairs]
            afd_texts = []
            for afd_path in sorted(loc.glob("AFD_*.txt"))[
                : self.config.max_afds_for_judge
            ]:
                with open(afd_path, encoding="utf-8", errors="replace") as f:
                    afd_texts.append(
                        "".join(c for c in f.read() if c.isprintable() or c.isspace())
                    )
            latest_hour = int(sounding_times[-1][:2])
            target_hour = latest_hour + self.config.target_forecast_hour_offset
            target_time = (
                f"{target_hour:02d}00Z on {self.config.target_date[4:6]}/"
                f"{self.config.target_date[6:8]}/{self.config.target_date[0:4]}"
            )
            run_time = soundings[0].get("tm", "00/00Z").split("/")[1][:2] + "Z"
            run_date_full_z = f"{self.config.target_date} at {run_time}"
            case = CaseData(
                case_id=f"{self.config.target_date}_{loc.name}",
                location_id=loc.name,
                model_name=model_name,
                run_date_full_z=run_date_full_z,
                target_forecast_time_utc=target_time,
                model_soundings_data=soundings,
                sounding_times_str=", ".join(sounding_times),
                afd_texts=afd_texts,
            )
            self.cases.append(case)
        random.shuffle(self.cases)
        self.iter = 0

    async def get_next_item(self) -> Optional[CaseData]:
        if not self.cases:
            return None
        if self.current_idx >= len(self.cases):
            random.shuffle(self.cases)
            self.current_idx = 0
        case = self.cases[self.current_idx]
        self.current_idx += 1
        self.iter += 1
        return case

    @staticmethod
    def _parse_llm_output(text: str) -> Dict[str, Any]:
        think_match = re.search(
            r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE
        )
        think_content = think_match.group(1).strip() if think_match else ""
        forecast_summary = ""
        tool_calls: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line_upper = line.strip().upper()
            if line_upper.startswith("TOOL_CALL:"):
                try:
                    tool_json = json.loads(line[len("TOOL_CALL:") :].strip())
                    tool_calls.append(tool_json)
                except json.JSONDecodeError:
                    logger.warning("Malformed TOOL_CALL line skipped: %s", line)
            elif line_upper.startswith("FORECAST_SUMMARY:"):
                forecast_summary = line[len("FORECAST_SUMMARY:") :].strip()
        return {
            "think_content": think_content,
            "tool_calls": tool_calls,
            "forecast_summary": forecast_summary,
        }

    @staticmethod
    def _parse_judge_output(text: str) -> Tuple[float, Dict[str, float], str]:
        scores = {
            "reasoning": 0.0,
            "tool_call": 0.0,
            "forecast_summary": 0.0,
            "total": 0.0,
        }
        for key in scores:
            match = re.search(rf"{key.upper()}_SCORE:\s*([0-9.]+)", text)
            if match:
                scores[key] = float(match.group(1))
        just_match = re.search(r"JUSTIFICATION:\s*(.*)", text, re.DOTALL)
        justification = just_match.group(1).strip() if just_match else ""
        total = (
            scores.get("reasoning", 0.0)
            + scores.get("tool_call", 0.0)
            + scores.get("forecast_summary", 0.0)
        )
        total = round(total, 2)
        if abs(total - scores.get("total", total)) > 0.1:
            scores["total"] = total
        return total, scores, justification

    async def _call_agent(self, messages: List[Dict[str, str]]) -> List[str]:
        assert self.agent_llm_server is not None
        resp = await self.agent_llm_server.chat_completion(
            messages=messages,
            model=self.agent_llm_server.config.model_name,
            n=self.config.group_size,
            max_tokens=self.config.max_reasoning_tokens_llm,
            temperature=0.7,
            stop=["<|im_end|>", "<|endoftext|>", "<|eot_id|>"],
        )
        return [c.message.content for c in resp.choices]

    async def _call_judge(self, messages: List[Dict[str, str]]) -> str:
        assert self.judge_server is not None
        resp = await self.judge_server.chat_completion(
            messages=messages,
            model=self.judge_server.config.model_name,
            max_tokens=self.config.max_tokens_judge,
            temperature=0.2,
            n=1,
        )
        return resp.choices[0].message.content

    async def collect_trajectories(
        self, item: CaseData
    ) -> Tuple[Optional[ScoredDataGroup], List[Any]]:
        soundings_blob = json.dumps(item.model_soundings_data, indent=2)
        agent_user_prompt = AGENT_USER_PROMPT_TEMPLATE.format(
            location_id=item.location_id,
            model_name=item.model_name,
            run_date_full_z=item.run_date_full_z,
            sounding_times_str=item.sounding_times_str,
            target_forecast_time_utc=item.target_forecast_time_utc,
            soundings_json_blob=soundings_blob,
        )
        agent_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": agent_user_prompt},
        ]
        outputs = await self._call_agent(agent_messages)

        group: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "overrides": [],
        }

        for llm_output in outputs:
            parsed = self._parse_llm_output(llm_output)

            judge_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
                llm_full_output=llm_output,
                afds_blob=(
                    "\n\n---\n\n".join(item.afd_texts)
                    if item.afd_texts
                    else "No AFDs provided."
                ),
            )
            judge_messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ]
            judge_out = await self._call_judge(judge_messages)
            final_score, judge_scores, justification = self._parse_judge_output(
                judge_out
            )
            self.judge_scores_buffer.append(final_score)

            tokenized = tokenize_for_trainer(
                self.tokenizer,
                agent_messages + [{"role": "assistant", "content": llm_output}],
                self.config.max_token_length,
            )

            group["tokens"].append(tokenized["tokens"])
            group["masks"].append(tokenized["masks"])
            group["scores"].append(final_score)
            group["overrides"].append(
                {
                    "case_id": item.case_id,
                    "llm_think": parsed["think_content"],
                    "llm_tools": str(parsed["tool_calls"]),
                    "llm_summary": parsed["forecast_summary"],
                    "judge_justification": justification,
                    "judge_score_reasoning": judge_scores.get("reasoning", 0.0),
                    "judge_score_tool": judge_scores.get("tool_call", 0.0),
                    "judge_score_forecast": judge_scores.get("forecast_summary", 0.0),
                }
            )

            self.rollouts_for_wandb.append(
                (
                    agent_user_prompt[:300] + "...",
                    parsed["think_content"][:500] + "...",
                    str(parsed["tool_calls"])[:300] + "...",
                    parsed["forecast_summary"][:300] + "...",
                    final_score,
                    justification[:500] + "...",
                )
            )

        return group, []

    async def evaluate(self, *args, **kwargs) -> None:
        if not self.cases:
            return
        sample_cases = random.sample(
            self.cases, k=min(len(self.cases), self.config.num_eval_samples)
        )
        for case in sample_cases:
            soundings_blob = json.dumps(case.model_soundings_data, indent=2)
            user_prompt = AGENT_USER_PROMPT_TEMPLATE.format(
                location_id=case.location_id,
                model_name=case.model_name,
                run_date_full_z=case.run_date_full_z,
                sounding_times_str=case.sounding_times_str,
                target_forecast_time_utc=case.target_forecast_time_utc,
                soundings_json_blob=soundings_blob,
            )
            agent_messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            llm_output = await self._call_agent(agent_messages)
            judge_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
                llm_full_output=llm_output,
                afds_blob=(
                    "\n\n---\n\n".join(case.afd_texts) if case.afd_texts else "No AFDs."
                ),
            )
            judge_messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ]
            judge_out = await self._call_judge(judge_messages)
            _, scores, _ = self._parse_judge_output(judge_out)
            self.eval_scores_buffer.append(scores)

    async def wandb_log(self, metrics: Optional[Dict] = None) -> None:
        if metrics is None:
            metrics = {}
        if self.judge_scores_buffer:
            metrics["train/avg_judge_total_score"] = sum(
                self.judge_scores_buffer
            ) / len(self.judge_scores_buffer)
            self.judge_scores_buffer.clear()
        if self.eval_scores_buffer:
            avg_total = sum(x["total"] for x in self.eval_scores_buffer) / len(
                self.eval_scores_buffer
            )
            metrics["eval/avg_judge_total_score"] = avg_total
            self.eval_scores_buffer.clear()
        if self.rollouts_for_wandb and wandb.run:
            table = wandb.Table(
                columns=[
                    "Prompt Hint",
                    "LLM Think",
                    "LLM Tools",
                    "LLM Summary",
                    "Judge Score",
                    "Judge Justification",
                ]
            )
            for row in self.rollouts_for_wandb[: self.config.num_rollouts_to_log]:
                table.add_data(*row)
            metrics["train/detailed_rollouts"] = table
            self.rollouts_for_wandb.clear()
        if metrics:
            await super().wandb_log(metrics)


if __name__ == "__main__":
    MeteorologyForecastRLEnv.cli()
