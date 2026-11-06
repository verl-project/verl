import asyncio
import os
import random
from typing import List, Optional, Tuple

import docker
import httpx
import regex as re
from datasets import load_dataset

from atroposlib.envs.base import BaseEnv, ScoredDataGroup
from atroposlib.type_definitions import GameHistory, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)


async def submit_code(client, code, test_input, language="python"):
    url = "http://localhost:5002/execute"
    payload = {"code": code, "input": test_input, "language": language}
    response = await client.post(url, json=payload)
    response_json = response.json()
    return response_json["output"]


async def get_results(code, answer):
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(len(answer)):
            tasks.append(submit_code(client, code, answer[i]))

        results = await asyncio.gather(*tasks)
    return [result for result in results]


def init_docker():
    client = docker.from_env()

    def build_docker_image():
        try:
            # Build the Docker image
            print("Building Docker image...")
            current_dir = os.path.dirname(
                os.path.abspath(__file__)
            )  # Get the current directory of the script
            image, logs = client.images.build(path=current_dir, tag="code-executor")

            # Print the build logs
            for log in logs:
                print(log.get("stream", "").strip())

            print("Docker image built successfully.")
            return image
        except docker.errors.BuildError as e:
            print(f"Error during Docker image build: {e}")

    def run_docker_container():
        try:
            # Run the Docker container
            print("Running Docker container...")
            container = client.containers.run(
                "code-executor", ports={"5002/tcp": 5002}, detach=True
            )  # Runs in detached mode (in the background)

            print(f"Docker container is running with ID: {container.id}")
            return container
        except docker.errors.ContainerError as e:
            print(f"Error during Docker container run: {e}")

    build_docker_image()
    container = run_docker_container()
    return container


class CodingEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[GameHistory | None, List[Item]]:
        chat_completions = await self.server.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You must submit your answer with ```python\n{code}```",
                },
                dict(item[0][0]),
            ],
            n=self.config.group_size,
            max_tokens=1024 * 4,
        )
        to_score = list()
        to_backlog = list()
        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                dict(item[0][0]),
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                (
                    messages,
                    item[1],
                )
            )

        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment, this is called every steps_per_eval steps

        Included here is an example on how to use eval workers to run a task.

        You may however do whatever you want in this method.

        :param args:
        :param kwargs:
        :return: None.
        """
        return

    async def setup(self):
        """Setup the environment"""
        self.container = init_docker()
        self.train = load_dataset("deepmind/code_contests", split="train")
        self.iter = 0

    async def get_next_item(self) -> Item:
        """
        Get the next items to be rolled out
        """
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        prompt = tuple(
            [frozenset({"role": "user", "content": next_item["description"]}.items())]
        )
        answer = (
            tuple(next_item["private_tests"]["input"]),
            tuple(next_item["private_tests"]["output"]),
            tuple(next_item["generated_tests"]["input"]),
            tuple(next_item["generated_tests"]["output"]),
        )
        return (prompt, answer)

    def extract_python_code_blocks(self, text):
        # Regex specifically looks for ```python\n...code...\n```
        pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
        result = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        python_blocks = [r for r in result]
        return python_blocks

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        # print("Rollout group data", rollout_group_data)
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]
            """
            CALCULATE REWARD NOW
            """
            code = self.extract_python_code_blocks(item[0][-1]["content"])[0]
            test_cases = list(item[1][0]) + list(item[1][2])
            x = await get_results(code, test_cases)
            output_cases = list(item[1][1]) + list(item[1][3])
            assert len(x) == len(output_cases)
            reward = True
            for k in range(len(x)):
                if x[k] != output_cases[k]:
                    reward = False
                    break
            # remove obviously bad examples
            if len([1 for i in masks if i != -100]) < 10:
                continue
            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)
            if len(scores["tokens"]) >= self.config.group_size:
                break
        # check if all the same
        # print(scores['scores'])
        # if all([scores["scores"][0] == score for score in scores["scores"]]):
        #     return None  # If all the same, we return None
        return scores


if __name__ == "__main__":
    CodingEnv.cli()
