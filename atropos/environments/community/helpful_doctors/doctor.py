import os
import random
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

from datasets import load_dataset
from openai import OpenAI
from patient import patient_profiles

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item

DatasetItem = TypedDict(
    "DatasetItem",
    {
        "question": str,
        "answer": str,
        "options": Dict[str, str],
        "meta_info": str,
        "answer_idx": str,
        "diagnosis": str,
        "metamap_sequence": Sequence[str],
    },
)

client = OpenAI(
    api_key=os.environ.get("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

final_message = "The diagnosis is:"

doctor_system_prompt = """You are a doctor. You are interacting with a patient.
You need to diagnose the patient based on the symptoms.
You will need to ask the patient follow up questions to diagnose them.
Ask up to 10 follow up questions. After that make your diagnosis.
Once you are confident in your diagnosis, provide it in the format:

The diagnosis is: {possible_illness}
"""
# ## For example,

# user: I have a headache.
# assistant: What is the severity of your headache?
# user: It's a 3/10.
# assistant: What is the location of your headache?
# user: It's in the front of my head.
# assistant: What is the duration of your headache?
# user: It's been going on for 2 days.
# assistant: The patient is diagnosed with <diagnosis>headache</diagnosis>
# """


doctor_model = "NousResearch/DeepHermes-3-Llama-3-8B-Preview"

USER_TAG = "user"
ASSISTANT_TAG = "assistant"


class DoctorEnv(BaseEnv):

    name = "doctor"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []
        self.print_this_env = False

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name=doctor_model,
            group_size=32,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            wandb_name="doctor",
            max_num_workers=128,
            total_steps=100,
            batch_size=1024,
            steps_per_eval=1,
            max_token_length=1024 * 15,
            inference_weight=1.0,
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            debug_mode=True,
        )
        server_configs = [
            APIServerConfig(
                model_name=doctor_model,
                base_url="http://localhost:9001/v1",
                api_key="EMPTY",
                num_requests_for_eval=256,
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

    async def setup(self):
        """
        Set up the environment by loading and preparing the dataset.
        """
        # Load the full dataset
        full_dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

        full_dataset = full_dataset.shuffle(seed=42)

        # Keep the splits as is - no need to reformat
        self.train = full_dataset["train"]
        # Limit test set size to prevent evaluation from taking too long
        self.test = full_dataset["test"].select(
            range(min(128, len(full_dataset["test"])))
        )

        # Print some dataset statistics
        print(
            f"Loaded dataset with {len(self.train)} training examples and {len(self.test)} test examples"
        )
        print(f"Example item format: {self.train[0]}")

        # Initialize iteration counter
        self.iter = 0

    async def evaluate(self, *args, **kwargs):
        pass

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        # Grab a dedicated llm server to take advantage of caching
        async with self.server.dedicated_server() as server:

            scores = ScoredDataGroup()
            scores["scores"] = list()

            patient_messages = []
            doctor_messages = [{"role": "system", "content": doctor_system_prompt}]

            patient_profile = random.choice(patient_profiles)
            symptoms = item["question"]
            patient_system_prompt = patient_profile.format(symptoms=symptoms)

            patient_messages = [{"role": "system", "content": patient_system_prompt}]
            # print("before xai message")
            completion = client.chat.completions.create(
                model="grok-3-latest",
                messages=patient_messages,
            )

            patient_msg = completion.choices[0].message.content
            # print("patient message", patient_msg)

            print("patient message", patient_msg)

            doctor_messages.append({"role": USER_TAG, "content": patient_msg})
            patient_messages.append({"role": ASSISTANT_TAG, "content": patient_msg})
            # print("after  xai message")
            score = -1
            while True:
                if (
                    len(self.tokenizer.apply_chat_template(doctor_messages))
                    > self.config.max_token_length - 10
                ):
                    score = 0
                    break
                max_tokens = self.config.max_token_length - len(
                    self.tokenizer.apply_chat_template(
                        doctor_messages, add_generation_prompt=True
                    )
                )
                # print("before doctor response")
                # print("messages", doctor_messages)
                doctor_completions = await server.chat_completion(
                    messages=doctor_messages,
                    n=1,
                    max_tokens=max_tokens,
                )

                doctor_msg = doctor_completions.choices[0].message.content
                print("doctor message", doctor_msg)

                # print("doctor message", doctor_msg)

                doctor_messages.append({"role": ASSISTANT_TAG, "content": doctor_msg})
                patient_messages.append({"role": USER_TAG, "content": doctor_msg})
                # print("after doctor response")
                # check output
                if doctor_msg.startswith(final_message):
                    diagnosis = doctor_msg.strip(final_message)
                    diagnosis = diagnosis.strip()

                    if item["answer"] in diagnosis:
                        score = 1
                    else:
                        score = 0
                    break

                completion = client.chat.completions.create(
                    model="grok-3-latest",
                    messages=patient_messages,
                )

                patient_msg = completion.choices[0].message.content

                doctor_messages.append({"role": USER_TAG, "content": patient_msg})
                patient_messages.append({"role": ASSISTANT_TAG, "content": patient_msg})

            self.percent_correct_buffer.append(max(score, 0))
            tokens = self.tokenizer.apply_chat_template(doctor_messages)

            masks = []
            for i, msg in enumerate(doctor_messages):
                if i == len(doctor_messages) - 1:
                    masks.extend(tokens[len(masks) :])
                else:
                    curr_tokens = self.tokenizer.apply_chat_template(
                        doctor_messages[: i + 1],
                        add_generation_prompt=doctor_messages[i + 1]["role"]
                        == ASSISTANT_TAG,
                    )
                    if doctor_messages[i]["role"] == USER_TAG:
                        masks.extend([-100] * (len(curr_tokens) - len(masks)))
                    else:
                        masks.extend(curr_tokens[len(masks) :])

            scores["scores"].append(1.0 if score else -1.0)

        scored_data_item = ScoredDataItem(
            messages=doctor_messages,
            finish_reason=score,
            tokens=tokens,
            masks=masks,
            scores=score,
        )

        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        return scored_data_item, []

    async def get_next_item(self):
        """
        Get the next training item from the dataset.

        Returns:
            A tuple containing prompt and expected answer
        """
        next_item: DatasetItem = self.train[self.iter % len(self.train)]
        self.iter += 1

        return next_item


if __name__ == "__main__":
    DoctorEnv.cli()
