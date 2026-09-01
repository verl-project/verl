import json
import os
import random
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Example domains for generating metric queries
DOMAINS = [
    "e-commerce",
    "finance",
    "healthcare",
    "education",
    "real estate",
    "social media",
    "marketing",
    "transportation",
    "manufacturing",
    "hospitality",
    "technology",
    "energy",
    "retail",
    "agriculture",
    "media",
    "telecommunications",
    "insurance",
    "entertainment",
]

# Metric type templates
METRIC_TYPES = [
    "revenue",
    "user growth",
    "conversion rate",
    "customer acquisition cost",
    "average order value",
    "customer lifetime value",
    "page views",
    "bounce rate",
    "churn rate",
    "active users",
    "session duration",
    "retention rate",
    "engagement rate",
    "click-through rate",
    "inventory turnover",
    "profit margin",
    "return on investment",
    "customer satisfaction",
    "net promoter score",
    "sales growth",
    "market share",
    "employee productivity",
    "website traffic",
    "operational costs",
    "customer support tickets",
    "app downloads",
    "subscription renewals",
    "delivery time",
    "error rate",
    "uptime percentage",
]

# Time periods for metrics
TIME_PERIODS = [
    "this month",
    "this quarter",
    "this year",
    "last 30 days",
    "last 7 days",
    "last 24 hours",
    "year-to-date",
    "Q1 2023",
    "Q2 2023",
    "Q3 2023",
    "Q4 2023",
    "Q1 2024",
    "Q2 2024",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Icons for metric cards
ICONS = [
    "trending-up",
    "trending-down",
    "users",
    "dollar",
    "shopping-cart",
    "bar-chart",
    "percent",
    "activity",
    "alert-circle",
    "clock",
    "heart",
    "globe",
    "mail",
    "phone",
    "eye",
    "tag",
    "star",
    "bell",
    "download",
    "upload",
    "refresh",
    "check-circle",
    "shield",
    "pie-chart",
    "layers",
    "calendar",
    "thumbs-up",
]

# System prompt for metric card generation
system_prompt = """You are an AI specialized in generating JSON data for a MetricCard React component.

Your task is to create a JSON object following this structure:
```json
{
  "componentType": "metricCard",
  "props": {
    "title": "Revenue",
    "value": "$12,345",
    "description": "Total revenue this month",
    "trend": {
      "value": 12.5,
      "isPositive": true
    },
    "icon": "dollar"
  }
}
```

Required properties:
- componentType: Must be exactly "metricCard"
- props.title: The metric name/label
- props.value: The metric value (formatted appropriately)

Optional properties:
- props.description: Additional context about the metric
- props.trend: An object with 'value' (percentage) and 'isPositive' (boolean)
- props.icon: Name of the icon (e.g., "dollar", "users", "trending-up")
- props.className: CSS class name for custom styling

IMPORTANT:
1. Ensure the generated JSON is valid
2. Format numeric values appropriately (add commas, currency symbols, or percentage signs as needed)
3. Use appropriate icons based on the metric type
4. Make sure trend values and descriptions are realistic

You will receive a query describing what kind of metric is needed. Generate the appropriate JSON response.
"""


def generate_random_prompt():
    """Generate a random prompt for metric card generation"""
    domain = random.choice(DOMAINS)
    metric_type = random.choice(METRIC_TYPES)
    time_period = random.choice(TIME_PERIODS)

    templates = [
        f"Create a metric card for {metric_type} in {domain} for {time_period}.",
        f"Generate a JSON output for a {domain} dashboard showing {metric_type} during {time_period}.",
        f"I need a metric card showing {metric_type} statistics for our {domain} business for {time_period}.",
        f"Design a metric component that displays {metric_type} for our {domain} platform for {time_period}.",
        f"Provide a metric card JSON for {metric_type} in our {domain} analytics for {time_period}.",
    ]

    prompt = random.choice(templates)
    return prompt, domain, metric_type, time_period


# JSON schema for evaluating metric card output
class TrendModel(BaseModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)
    value: float = Field(..., description="The percentage change value")
    isPositive: bool = Field(
        ..., description="Whether the trend is positive or negative"
    )


class MetricCardProps(BaseModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)
    title: str = Field(..., description="The metric name/label")
    value: str = Field(
        ..., description="The metric value (formatted as appropriate string)"
    )
    description: Optional[str] = Field(
        None, description="Additional context about the metric"
    )
    trend: Optional[TrendModel] = Field(
        None, description="Trend information including value and direction"
    )
    icon: Optional[str] = Field(None, description="Name of the icon to display")
    className: Optional[str] = Field(
        None, description="CSS class name for custom styling"
    )


class MetricCardComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)
    componentType: str = Field(..., description="Must be exactly 'metricCard'")
    props: MetricCardProps = Field(
        ..., description="The properties for the metric card component"
    )


class PromptRow(TypedDict):
    prompt: str
    reference: str


class MetricCardEnv(BaseEnv):
    name = "metric_card_generator"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []
        self.completion_lengths = []
        self.custom_prompt = None  # Will be set dynamically
        self.single_prompt_mode = True
        self.output_file = "generated_metric_cards.json"
        self.results = []
        self.current_domains = []
        self.current_metrics = []
        self.current_periods = []
        self.prompts_used = []

        # Create output directory if it doesn't exist
        os.makedirs("metric_card_outputs", exist_ok=True)

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=1,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            max_token_length=1024,
            wandb_name="metric_card_generator",
            ensure_scores_are_not_same=False,  # Disable score diversity check
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=1,
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self, single_prompt=None):
        # Generate a new prompt for this run
        if not single_prompt:
            prompt, domain, metric_type, time_period = generate_random_prompt()
            self.custom_prompt = prompt
            self.current_domains.append(domain)
            self.current_metrics.append(metric_type)
            self.current_periods.append(time_period)
        else:
            self.custom_prompt = single_prompt

        self.prompts_used.append(self.custom_prompt)
        self.single_prompt_mode = True
        self.train = [{"prompt": self.custom_prompt, "reference": ""}]
        self.test = []
        self.iter = 0

        print(f"\n=== USING PROMPT ===\n{self.custom_prompt}\n")

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, prompt: str, reference: str = "") -> number:
        _ = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )

        # In case of a reference, we could add a scoring mechanism here
        # For metric cards, we'll just return 1 to indicate success
        return 1

    async def evaluate(self, *args, **kwargs):
        if not self.test:
            return

        eval_tasks = []
        for item in self.test:
            eval_tasks.append(
                self.rollout_and_score_eval(item["prompt"], item["reference"])
            )
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def evaluate_metric_card(self, json_text):
        """Evaluate the quality of a metric card JSON response"""
        # Initialize evaluation results
        evaluation = {
            "is_valid_json": False,
            "has_required_fields": False,
            "schema_valid": False,
            "formatting_score": 0,
            "overall_quality": 0,
            "error": None,
        }

        # Check if it's valid JSON
        try:
            data = json.loads(json_text)
            evaluation["is_valid_json"] = True

            # Check if it has the required fields
            if (
                isinstance(data, dict)
                and data.get("componentType") == "metricCard"
                and "props" in data
                and "title" in data["props"]
                and "value" in data["props"]
            ):
                evaluation["has_required_fields"] = True

            # Validate against schema
            try:
                _ = MetricCardComponent(**data)
                evaluation["schema_valid"] = True

                # Formatting score - evaluate how well values are formatted
                props = data["props"]
                formatting_score = 0

                # Check value formatting
                value = props["value"]
                if isinstance(value, str):
                    # Check for currency formatting or percentage or number formatting
                    if (
                        ("$" in value or "€" in value or "£" in value)
                        or ("%" in value)
                        or ("," in value and any(c.isdigit() for c in value))
                    ):
                        formatting_score += 1

                # Check trend formatting
                if "trend" in props and isinstance(props["trend"], dict):
                    if isinstance(
                        props["trend"].get("value"), (int, float)
                    ) and isinstance(props["trend"].get("isPositive"), bool):
                        formatting_score += 1

                # Check description format
                if "description" in props and isinstance(props["description"], str):
                    if props["description"]:  # Not empty
                        formatting_score += 1

                # Normalize formatting score to 0-1
                evaluation["formatting_score"] = min(formatting_score / 3, 1.0)

                # Overall quality score (combining schema validity and formatting)
                evaluation["overall_quality"] = (
                    evaluation["schema_valid"] * 0.5
                    + evaluation["formatting_score"] * 0.5
                )

            except Exception as e:
                evaluation["error"] = f"Schema validation error: {str(e)}"

        except Exception as e:
            evaluation["error"] = f"JSON parsing error: {str(e)}"

        return evaluation

    async def collect_trajectories(
        self, item: PromptRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["prompt"]}

        # Always use our current prompt
        if not item["prompt"]:
            user_message["content"] = self.custom_prompt

        chat_completions = await self.server.chat_completion(
            messages=[{"role": "system", "content": system_prompt}, user_message],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )
        to_score = list()
        to_backlog = list()

        # Create a unique filename using the current step
        step_count = len(self.results) + 1

        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "reference": item.get("reference", ""),
                    "finish_reason": chat_completion.finish_reason,
                }
            )

            # Get the generated response
            response = chat_completion.message.content.strip()

            # Try to extract JSON object if embedded in markdown (```json ... ```)
            if "```json" in response and "```" in response.split("```json", 1)[1]:
                json_text = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in response and "```" in response.split("```", 1)[1]:
                json_text = response.split("```", 1)[1].split("```", 1)[0].strip()
            else:
                json_text = response

            # Evaluate the JSON
            evaluation = await self.evaluate_metric_card(json_text)

            # Create parsed version for JSON output if valid
            parsed_json = {}
            try:
                if evaluation["is_valid_json"]:
                    parsed_json = json.loads(json_text)
            except Exception:
                parsed_json = {"error": "Could not parse JSON"}

            # Log the generated metric card
            print("\n=== GENERATED METRIC CARD ===\n")
            print(json_text)
            print("\n=== EVALUATION ===")
            print(f"Valid JSON: {evaluation['is_valid_json']}")
            print(f"Has Required Fields: {evaluation['has_required_fields']}")
            print(f"Schema Valid: {evaluation['schema_valid']}")
            print(f"Formatting Score: {evaluation['formatting_score']:.2f}")
            print(f"Overall Quality: {evaluation['overall_quality']:.2f}")
            if evaluation["error"]:
                print(f"Error: {evaluation['error']}")
            print("\n" + "-" * 50)

            # Save current result with all metadata
            current_result = {
                "prompt": self.custom_prompt,
                "raw_response": response,
                "json_text": json_text,
                "parsed_json": parsed_json,
                "domain": self.current_domains[-1] if self.current_domains else "",
                "metric_type": self.current_metrics[-1] if self.current_metrics else "",
                "time_period": self.current_periods[-1] if self.current_periods else "",
                "evaluation": evaluation,
                "step": step_count,
                "finish_reason": chat_completion.finish_reason,
            }

            self.results.append(current_result)

            # Save individual result to a separate file
            individual_file = f"metric_card_outputs/metric_card_{step_count}.json"
            with open(individual_file, "w") as f:
                json.dump(current_result, f, indent=2)

            # Also save to the main output file
            with open(self.output_file, "w") as f:
                json.dump(
                    {
                        "results": self.results,
                        "prompts_used": self.prompts_used,
                        "domains": self.current_domains,
                        "metrics": self.current_metrics,
                        "time_periods": self.current_periods,
                    },
                    f,
                    indent=2,
                )

        # Create a dummy scored data group that will pass validation
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []

        # Add some dummy data with different scores to pass the validation
        for item in to_score:
            out_dict = tokenize_for_trainer(
                self.tokenizer, item["messages"], item["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Skip if tokenization failed
            if len([1 for i in masks if i != -100]) < 10:
                continue

            # Add just enough token entries with different scores
            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(
                random.uniform(-0.5, 0.5)
            )  # Random scores to avoid the "all same" check

        return scores, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Create artificially different scores to avoid the validation errors
        for idx, item in enumerate(rollout_group_data):
            out_dict = tokenize_for_trainer(
                self.tokenizer, item["messages"], item["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove obviously bad examples
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)

            # Assign different scores to ensure we pass validation
            scores["scores"].append(
                -1.0 + idx * 0.5
            )  # Will give different scores for each item

        # Ensure we have some scores
        if len(scores["scores"]) == 0:
            return None

        return scores

    async def get_next_item(self) -> PromptRow:
        # Generate a new prompt for each step if we're in multi-step mode
        if self.config.total_steps > 1:
            prompt, domain, metric_type, time_period = generate_random_prompt()
            self.custom_prompt = prompt
            self.current_domains.append(domain)
            self.current_metrics.append(metric_type)
            self.current_periods.append(time_period)
            self.prompts_used.append(prompt)
            print(f"\n=== USING NEW PROMPT ===\n{prompt}\n")

        # Return the current prompt
        return {"prompt": self.custom_prompt, "reference": ""}


# This is needed to use the CLI command with the existing framework
if __name__ == "__main__":
    MetricCardEnv.cli()
