import json
from typing import Optional, Union

import openai
from poke_env import RandomPlayer
from poke_env.environment.battle import AbstractBattle, Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player import Player

client = openai.OpenAI()

# ANSI escape codes for colors
LIGHT_BLUE = "\033[94m"
LIGHT_RED = "\033[91m"
RESET_COLOR = "\033[0m"

# The RandomPlayer is a basic agent that makes decisions randomly,
# serving as a starting point for more complex agent development.
random_player = RandomPlayer()


# Here's one way to pretty print the results of the cross evaluation using `tabulate`:


# ## Building GPT Player


def log_pokemon(pokemon: Pokemon, is_opponent: bool = False):
    lines = [
        f"[{pokemon.species} ({pokemon.name}) {'[FAINTED]' if pokemon.fainted else ''}]",
        f"Types: {[t.name for t in pokemon.types]}",
    ]

    if is_opponent:
        lines.append(f"Possible Tera types {pokemon.tera_type}")

    lines.extend(
        [
            f"HP: {pokemon.current_hp}/{pokemon.max_hp} ({pokemon.current_hp_fraction * 100:.1f}%)",
            f"Base stats: {pokemon.base_stats}",
            f"Stats: {pokemon.stats}",
            f"{'Possible abililities' if is_opponent else 'Ability'}: {pokemon.ability}",
            f"{'Possible items' if is_opponent else 'Item'}: {pokemon.item}",
            f"Status: {pokemon.status}",
        ]
    )

    if pokemon.status:
        lines.append(f"Status turn count: {pokemon.status_counter}")

    lines.append("Moves:")
    lines.extend(
        [
            f"Move ID: `{move.id}` Base Power: {move.base_power} "
            f"Accuracy: {move.accuracy * 100}% PP: ({move.current_pp}/{move.max_pp}) "
            f"Priority: {move.priority}  "
            for move in pokemon.moves.values()
        ]
    )

    lines.extend([f"Stats: {pokemon.stats}", f"Boosts: {pokemon.boosts}"])

    return "\n".join(lines)


def log_player_info(battle: AbstractBattle):
    lines = [
        "== Player Info ==",
        "Active pokemon:",
        log_pokemon(battle.active_pokemon),
        f"Tera Type: {battle.can_tera}",
        "-" * 10,
        f"Team: {battle.team}",
    ]

    for _, mon in battle.team.items():
        if not mon.active:
            lines.append(log_pokemon(mon))
            lines.append("")

    return "\n".join(lines)


def log_opponent_info(battle: AbstractBattle):
    return "\n".join(
        [
            "== Opponent Info ==",
            "Opponent active pokemon:",
            log_pokemon(battle.opponent_active_pokemon, is_opponent=True),
            f"Opponent team: {battle.opponent_team}",
        ]
    )


def log_battle_info(battle: AbstractBattle):
    lines = ["== Battle Info ==", f"Turn: {battle.turn}"]

    # Field info
    if battle.weather:
        lines.append(f"Weather: {battle.weather}")
    if battle.fields:
        lines.append(f"Fields: {battle.fields}")
    if battle.side_conditions:
        lines.append(f"Player side conditions: {battle.side_conditions}")
    if battle.opponent_side_conditions:
        lines.append(f"Opponent side conditions: {battle.opponent_side_conditions}")
    if battle.trapped:
        lines.append(f"Trapped: {battle.trapped}")

    return "\n".join(lines)


def create_prompt(battle_info, player_info, opponent_info, available_moves) -> str:
    prompt = f"""
Here is the current state of the battle:

{battle_info}

Here is the current state of your team:

{player_info}

Here is the current state of the opponent's team:

{opponent_info}

Your goal is to win the battle. You can only choose one move to make.

Here is the list of available moves:

{available_moves}

Reason carefully about the best move to make. Consider things like the opponent's team,
the weather, the side conditions (i.e. stealth rock, spikes, sticky web, etc.).
Consider the effectiveness of the move against the opponent's team, but also consider
the power of the move, and the accuracy. You may also switch to a different pokemon
if you think it is a better option. Given the complexity of the game, you may also
sometimes choose to "sacrifice" your pokemon to put your team in a better position.

Finally, write a conclusion that includes the move you will make, and the reason you made that move.

"""
    return prompt


class GPTPlayer(Player):

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        # OpenAI client will use environment variables if api_key or base_url are None
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def choose_max_damage_move(self, battle: Battle):
        return max(battle.available_moves, key=lambda move: move.base_power)

    def choose_move(self, battle: AbstractBattle):

        def choose_order_from_id(
            move_id: str, battle: AbstractBattle
        ) -> Union[Move, Pokemon]:
            try:
                return list(
                    filter(lambda move: move.id == move_id, battle.available_moves)
                )[0]
            except Exception as e:
                print(f"Error picking move: {e}")
                return battle.available_moves[0]

        # Chooses a move with the highest base power when possible
        if battle.available_moves:
            # Define tool call dsl
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "choose_order_from_id",
                        "strict": True,
                        "description": "Choose a move from the list of available moves.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "move_id": {
                                    "type": "string",
                                    "description": "The id (name of move) of the move to choose. "
                                    "This must be one of the available move IDs.",
                                }
                            },
                            "required": ["move_id"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]

            # Pass state of game to the Agent
            system_prompt = create_prompt(
                log_battle_info(battle),
                log_player_info(battle),
                log_opponent_info(battle),
                battle.available_moves,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Select a move based on the move id (the name of the move) "
                    f"{battle.available_moves}. You must use the choose_order_from_id tool.",
                },
            ]

            # Single call to get the tool choice
            tool_selection_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                temperature=1.0,
                tool_choice={
                    "type": "function",
                    "function": {"name": "choose_order_from_id"},
                },
            )

            if (
                tool_selection_response.choices
                and tool_selection_response.choices[0].message.tool_calls
            ):
                tool_call = tool_selection_response.choices[0].message.tool_calls[0]
                if tool_call.function.name == "choose_order_from_id":
                    args = json.loads(tool_call.function.arguments)
                    # print(f"{self.color}Available moves: {battle.available_moves}{RESET_COLOR}")
                    # Optional: for debugging
                    chosen_order = choose_order_from_id(args["move_id"], battle)
                    # print(f"Chosen order by GPT: {chosen_order}")
                    return self.create_order(chosen_order)
                else:
                    print(
                        f"Error: Unexpected tool call {tool_call.function.name}. "
                        f"Choosing random move."
                    )
                    return self.choose_random_move(battle)
            else:
                # This case includes if choices is empty, message is None, or tool_calls is None/empty
                error_message = "Error: No tool call found in LLM response."
                if (
                    tool_selection_response.choices
                    and tool_selection_response.choices[0].finish_reason
                ):
                    error_message += f" Finish reason: {tool_selection_response.choices[0].finish_reason}."
                if (
                    tool_selection_response.usage
                ):  # Log usage for debugging if available
                    error_message += f" Usage: {tool_selection_response.usage}."

                # print(f"{error_message} Choosing random move.")
                return self.choose_random_move(battle)

        else:
            # print(f"No moves available calling random")
            # If no attacking move is available, perform a random switch
            # This involves choosing a random move, which could be a switch or another available action
            return self.choose_random_move(battle)


# Max damage player
class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


# Creating players
random_player = RandomPlayer()
max_damage_player = MaxDamagePlayer()
# gpt_player = GPTPlayer(model="gpt-4.1") # Old instantiation
# gpt_player_o4 = GPTPlayer(model="gpt-4o") # Old instantiation

# Example of new instantiation (actual instantiation will be in PokemonEnv)
# gpt_player = GPTPlayer(model_name="gpt-4.1-nano", api_key="YOUR_API_KEY", base_url="YOUR_BASE_URL_IF_NEEDED")
# gpt_player_o4 = GPTPlayer(model_name="gpt-4o", api_key="YOUR_API_KEY", base_url="YOUR_BASE_URL_IF_NEEDED")
