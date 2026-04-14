from typing import Dict, List, Optional, Tuple

import gymnasium as gym

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item

start_msg = """### Description
There are four designated locations in the grid world indicated by R(ed),
G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off
at a random square and the passenger is at a random location. The taxi
drives to the passenger's location, picks up the passenger, drives to the
passenger's destination (another one of the four specified locations), and
then drops off the passenger. Once the passenger is dropped off, the episode ends.

Map:

    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+

### Actions
There are 6 discrete deterministic actions:
- 0: move south (increases row index)
- 1: move north (decreases row index)
- 2: move east (increases column index)
- 3: move west (decreases column index)
- 4: pickup passenger (IF on a letter location, AND passenger is located at the same location, pickup passenger)
- 5: drop off passenger

### Observations

Passenger locations:
- 0: R(ed)
- 1: G(reen)
- 2: Y(ellow)
- 3: B(lue)
- 4: in taxi

Destinations:
- 0: R(ed) (Row 0, Col 0)
- 1: G(reen) (Row 4, Col 4)
- 2: Y(ellow) (Row 0, Col 4)
- 3: B(lue) (Row 3, Col 3)

### Instructions
Please perform the actions that will let you pick up and/or drop off the passenger.
Please respond with the action number only.
You cannot move the taxi into walls, which are displayed as | in the map. : means you are free to move through that column.


For an example, if the passenger is at R, and the destination is G, and the taxi is at (2, 2), then here are the following actions to solve this in the correct order:

3 (move west)
3 (move west)
1 (move north)
1 (move north)
4 (pickup passenger)
0 (move south)
0 (move south)
2 (move east)
2 (move east)
2 (move east)
2 (move east)
0 (move south)
0 (move south)
5 (drop off passenger)

If you are stuck, try moving to row idx 2, as there are no walls there.

Submit your response as a number between 0 and 5 only to perform the discrete action.
Each turn we will give you the current state of the environment, and you will need to respond with the action number only from the available actions."""  # noqa: E501


def decode(i):
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    x = reversed(out)
    # Making it explicit so I don't have to look into gym code
    taxi_row, taxi_col, pass_idx, dest_idx = x
    return taxi_row, taxi_col, pass_idx, dest_idx


# Note: Works for both the passenger and the destination
TO_LOC_MAP = {
    0: "R(Row 0, Col 0)",
    1: "G (Row 4, Col 4)",
    2: "Y (Row 0, Col 4)",
    3: "B (Row 3, Col 3)",
    4: "in taxi",
}
MAP_LOC = {0: (0, 0), 1: (4, 4), 2: (0, 4), 3: (3, 3)}
TO_ACTION_MAP = {
    0: "south",
    1: "north",
    2: "east",
    3: "west",
    4: "pickup",
    5: "dropoff",
}


def state_render_to_user_msg(last_state, state, action_mask, render):
    taxi_row, taxi_col, pass_idx, dest_idx = decode(state)
    if last_state is not None:
        last_taxi_row, last_taxi_col, last_pass_idx, last_dest_idx = decode(last_state)
    available_actions = "\n".join(
        [
            f"- {i}: {TO_ACTION_MAP[i]}"
            for i in range(6)
            if (action_mask[i] == 1)
            and (
                (i != 5)
                or (
                    (i == 5)
                    and (taxi_row == MAP_LOC[dest_idx][0])
                    and (taxi_col == MAP_LOC[dest_idx][1])
                )
            )
        ]
    )
    if last_state is not None:
        ret_str = (
            f"Previous Taxi Location: Row: {last_taxi_row}, Col: {last_taxi_col}\n"
        )
    else:
        ret_str = ""
    ret_str += (
        f"Current state:\nTaxi: Row: {taxi_row}, Col: {taxi_col}\nPassenger: {TO_LOC_MAP[pass_idx]}\n"
        f"Destination: {TO_LOC_MAP[dest_idx]}\n\n"
        f"Map:\n{render}\n\n"
        f"Available actions:\n{available_actions}"
    )
    if (
        (pass_idx == 4)
        and (taxi_row == MAP_LOC[dest_idx][0])
        and (taxi_col == MAP_LOC[dest_idx][1])
    ):
        ret_str += "\n\nPlease drop off the passenger."
    elif pass_idx == 4:
        ret_str += f"\n\nPlease move the taxi to {TO_LOC_MAP[dest_idx]} to drop off the passenger."
    elif (taxi_row == MAP_LOC[pass_idx][0]) and (taxi_col == MAP_LOC[pass_idx][1]):
        ret_str += "\n\nPlease pick up the passenger."
    else:
        ret_str += f"\n\nPlease move the taxi to {TO_LOC_MAP[pass_idx]} to pick up the passenger."
    return ret_str


class GymTaxiEnv(BaseEnv):

    name = "gym_taxi"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.percent_picked_up_passenger_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []
        self.print_this_env = False

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=32,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=8192,
            wandb_name="gym_taxi",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
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
        try:
            wandb_metrics["train/percent_picked_up_passenger"] = sum(
                self.percent_picked_up_passenger_buffer
            ) / len(self.percent_picked_up_passenger_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        self.percent_picked_up_passenger_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.iter = 0

    async def evaluate(self, *args, **kwargs):
        pass

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        # Grab a dedicated llm server to take advantage of caching
        async with self.server.dedicated_server() as server:
            env = gym.make("Taxi-v3", render_mode="ansi")
            state, info = env.reset(seed=item["seed"])
            last_state = None
            taxi_row, taxi_col, pass_idx, dest_idx = decode(state)
            init_msg = f"{start_msg}\n\n" + state_render_to_user_msg(
                last_state, state, info["action_mask"], env.render()
            )
            messages = [{"role": "user", "content": init_msg}]
            score = -1
            while True:
                if (
                    len(self.tokenizer.apply_chat_template(messages))
                    > self.config.max_token_length - 10
                ):
                    break
                max_tokens = self.config.max_token_length - len(
                    self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                )
                chat_completions = await server.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=max_tokens,
                )
                choice = (
                    chat_completions.choices[0]
                    .message.content.strip()
                    .replace(".", "")[-1]
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": chat_completions.choices[0].message.content,
                    }
                )
                if choice.isdigit() and 0 <= int(choice) <= 5:
                    action = int(choice)
                else:
                    break
                if info["action_mask"][action] == 0:
                    break
                if action == 3:
                    # picked up passenger
                    score = 0
                next_state, reward, terminated, truncated, info = env.step(action)
                last_state = state
                state = next_state
                if terminated:
                    score = 1
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": state_render_to_user_msg(
                            last_state, state, info["action_mask"], env.render()
                        ),
                    }
                )
            self.percent_correct_buffer.append(max(score, 0))
            self.percent_picked_up_passenger_buffer.append(1 if score >= 0 else 0)
            tokens = self.tokenizer.apply_chat_template(messages)
            masks = []
            for i, msg in enumerate(messages):
                if i == len(messages) - 1:
                    masks.extend(tokens[len(masks) :])
                else:
                    curr_tokens = self.tokenizer.apply_chat_template(
                        messages[: i + 1],
                        add_generation_prompt=messages[i + 1]["role"] == "assistant",
                    )
                    if messages[i]["role"] == "user":
                        masks.extend([-100] * (len(curr_tokens) - len(masks)))
                    else:
                        masks.extend(curr_tokens[len(masks) :])
        scored_data_item = ScoredDataItem(
            messages=messages,
            finish_reason=score,
            tokens=tokens,
            masks=masks,
            scores=score,
        )
        return scored_data_item, []

    async def get_next_item(self):
        next_item = {"seed": self.iter}
        self.iter += 1
        return next_item


if __name__ == "__main__":
    GymTaxiEnv.cli()
