#!/usr/bin/env python3
"""
BlackjackEnv: Trainer environment for Gymnasium Blackjack

This wraps Gymnasium's Blackjack-v1 environment to train an LLM via a best-of-n pattern
using function-call style actions. Extends BaseEnv.

Sort of inspired by VinePPO, but uses a recursive exact value calculation
instead of Monte Carlo sampling (because the action space is so small and
the environment is deterministic, plus short episode lengths).
"""

import copy
import json
import logging
import random
import re
from typing import Dict, List, Optional, Tuple

import gymnasium
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.utils.best_of_n_selection import select_best_index
from atroposlib.utils.message_history_utils import (
    ensure_trajectory_token_limit,
    truncate_thinking,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)


class BlackjackEnvConfig(BaseEnvConfig):
    env_name: str = "Blackjack-v1"
    temperature: float = 0.7
    top_p: float = 0.9
    max_turns: Optional[int] = 5
    wandb_name: str = "blackjack"
    thinking_active: bool = True
    eval_episodes: int = 100
    max_think_chars_history: int = 3000
    max_trajectory_tokens: int = 24576  # seq_len of RL trainer
    debug_mode: bool = False
    group_size: int = 16
    tiebreak_token_factor: float = 0.01


class BlackjackScoredDataGroup(ScoredDataGroup):
    seed: int
    tokens: Optional[List[List[int]]] = None
    masks: Optional[List[List[int]]] = None
    scores: Optional[List[float]] = None
    messages: Optional[List[List[Dict]]] = None
    parsed_actions: Optional[List[int]] = None


class EpisodeState:
    def __init__(self, seed: int, env: gymnasium.Env):
        self.seed = seed
        self.env = env
        self.message_history: List[Dict] = []
        self.actions: List[int] = []
        self.step_rewards: List[float] = []
        self.total_reward: float = 0.0
        self.num_steps: int = 0
        self.num_correct_actions: int = 0
        self.num_total_actions: int = 0


class BlackjackEnv(BaseEnv):
    def __init__(
        self,
        config: BlackjackEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.episodes: Dict[int, EpisodeState] = {}
        self.debug_mode = config.debug_mode
        self.completed_episode_metrics_buffer: List[Dict[str, float]] = []
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            if logger.level == logging.NOTSET or logger.level > logging.WARNING:
                logger.setLevel(logging.WARNING)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": "Choose to 'hit' or 'stick' in Blackjack.",
                    "parameters": {
                        "action": {"type": "string", "enum": ["hit", "stick"]}
                    },
                },
            }
        ]

        tools_json = json.dumps(self.tools)
        self.system_prompt = (
            "You are an AI agent playing Blackjack who uses extreme long chains of thought "
            "to carefully consider the probabilities and optimal strategy. "
            "You need to decide whether to hit or stick based on your current hand and the dealer's showing card.\n\n"
            "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then "
            "provide your decision using the take_action function call. You may use extremely long chains "
            "of thought to carefully consider the probabilities and optimal strategy.\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            '<tool_call>\n{"arguments": {"action": "hit"}, "name": "take_action"}\n</tool_call>\n\n'
            "Your full answer format should be:\n"
            "<think>\n[Your detailed reasoning process about whether to hit or stick]\n</think>\n\n"
            '<tool_call>\n{"arguments": {"action": "stick"}, "name": "take_action"}\n</tool_call>\n\n'
            "Remember to carefully consider the probabilities and optimal strategy for Blackjack."
        )

    def _get_or_create_episode(self, seed: int) -> EpisodeState:
        if seed not in self.episodes:
            env = gymnasium.make(self.config.env_name)
            obs, _ = env.reset(seed=seed)
            ep = EpisodeState(seed, env)
            ep.message_history = [{"role": "system", "content": self.system_prompt}]
            ep.message_history.append(
                {"role": "environment", "content": self._format_observation(obs)}
            )
            self.episodes[seed] = ep
        return self.episodes[seed]

    def _format_observation(self, obs: Tuple[int, int, int]) -> str:
        player_sum, dealer_card, usable_ace = obs
        return f"Your hand sum is {player_sum}. Dealer showing: {dealer_card}. You have a usable ace: {usable_ace}."

    def _score_response(
        self,
        env_reward: float,
        response_text: str,
        parsed_action: int,
    ) -> float:
        """
        Calculates a score for a single agent response based purely on environment reward
        and a penalty for invalid action format.
        """
        current_env_reward = env_reward * 1.0
        # Action is good?
        if parsed_action == -1:
            current_env_reward -= 0.2
        else:
            current_env_reward += 0.2

        # Check the thinking tags exist
        match = re.search(r"<think>(.*?)</think>", response_text)
        if match:
            thinking_content = match.group(1)
            if thinking_content:
                current_env_reward += 0.2
            # Check there's actually valid content (not just whitespace)
            if not thinking_content.strip():
                current_env_reward -= 0.2
        else:
            current_env_reward -= 0.2

        # Calculate the number of tokens in the agent's response
        if response_text:
            num_tokens = len(self.tokenizer.encode(response_text))
        else:
            num_tokens = 0

        # tiebreak & small length penalty
        if self.config.max_token_length > 0:
            token_ratio = min(1.0, num_tokens / self.config.max_token_length)
            tiebreak_bonus = self.config.tiebreak_token_factor * (1.0 - token_ratio)
            current_env_reward += tiebreak_bonus
        return current_env_reward

    def _parse_tool_call(self, response: str) -> int:
        if not response:
            logger.warning(
                "Attempted to parse an empty response string. Returning invalid action (-1)."
            )
            return -1

        parsed_name, parsed_args, is_error = parse_tool_call(
            response, self.tools, ["tool_call"]
        )
        if is_error:
            error_detail = (
                parsed_name
                if isinstance(parsed_name, str) and parsed_name
                else "Parser indicated error, but no specific message was returned in the typical error slot."
            )
            logger.warning(
                f"Failed to parse tool call. Full response: '{response}'. Error detail: {error_detail}"
            )
            return -1

        action = parsed_args.get("action", "").lower()
        if action == "hit":
            return 1
        elif action == "stick":
            return 0
        else:
            logger.warning(
                f"Successfully parsed tool call, but action is invalid. Action: '{action}'. "
                f"Full response: '{response}'. Parsed args: {parsed_args}"
            )
            return -1

    async def _sample_response(self, messages: List[Dict], n: int = 1) -> List[str]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        try:
            completions = await self.server.completion(
                prompt=prompt,
                n=n,
                max_tokens=self.config.max_token_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            return [choice.text for choice in completions.choices]
        except Exception as e:
            logger.error(f"API error during completion: {e}")
            return []

    async def _estimate_value(
        self,
        episode_seed_for_sim: int,
        env_actions_to_replay: List[int],
    ) -> float:
        """Calculate exact state value V*(s)

        Args:
            episode_seed_for_sim: The seed of the original episode for deterministic env creation.
            env_actions_to_replay: List of environment actions (0 or 1) taken to reach current state s.
        """
        v_star_cache: Dict[Tuple[int, int, int], float] = {}

        def _get_v_star_recursive(
            obs_tuple: Tuple[int, int, int], current_env: gymnasium.Env
        ) -> float:
            player_sum, _, _ = obs_tuple

            # Base Case 1: Bust
            if player_sum > 21:
                return -1.0

            # Base Case 2: Check memoization cache
            if obs_tuple in v_star_cache:
                return v_star_cache[obs_tuple]

            env_for_stick = copy.deepcopy(current_env)
            _, reward_stick, _, _, _ = env_for_stick.step(0)
            # stick is terminal, so reward is final outcome
            q_star_stick = reward_stick

            # Q-value for HIT (action 1)
            env_for_hit = copy.deepcopy(current_env)
            obs_hit, reward_hit, term_hit, trunc_hit, _ = env_for_hit.step(1)

            if term_hit or trunc_hit:  # Game ended after hitting
                q_star_hit = reward_hit
            else:  # Game continues, recursively find V* of next state
                # reward_hit is typically 0 if the game didn't end
                q_star_hit = reward_hit + _get_v_star_recursive(obs_hit, env_for_hit)

            v_star = max(q_star_stick, q_star_hit)
            v_star_cache[obs_tuple] = v_star
            return v_star

        sim_env = None
        try:
            sim_env = gymnasium.make(self.config.env_name)
            current_obs, _ = sim_env.reset(seed=episode_seed_for_sim)

            # Replay actions to reach the current state s_t
            is_terminal_after_replay = False
            for action_idx, prev_action in enumerate(env_actions_to_replay):
                current_obs, _, term_replay, trunc_replay, _ = sim_env.step(prev_action)
                if term_replay or trunc_replay:
                    logger.debug(
                        f"[_estimate_value] State became terminal during action replay "
                        f"(action {action_idx+1}/{len(env_actions_to_replay)} of prev_actions). Value is 0."
                    )
                    is_terminal_after_replay = True
                    break

            if is_terminal_after_replay:
                return 0.0
            final_v_star = _get_v_star_recursive(current_obs, sim_env)
            return final_v_star

        except Exception as e:
            logger.error(
                f"[_estimate_value] Error during exact value"
                f" calculation for seed {episode_seed_for_sim}, "
                f"actions {env_actions_to_replay}: {e}",
                exc_info=True,
            )
            return 0.0
        finally:
            if sim_env is not None:
                sim_env.close()

    async def _next_step(
        self, ep: EpisodeState, current_turn: int, max_turns: int
    ) -> Tuple[Optional[BlackjackScoredDataGroup], bool]:
        """Process one step/turn of an episode.

        This involves estimating current state value, sampling multiple (G) responses from the LLM,
        evaluating each response by simulating its action, calculating advantages,
        selecting the best response/action, updating the episode state (message history, actions, rewards),
        and stepping the main environment.

        Args:
            ep: The current state of the episode.
            current_turn: The current turn number (0-indexed).
            max_turns: The maximum number of turns allowed in the episode.

        Returns:
            A tuple containing:
                - BlackjackScoredDataGroup: Data collected for this step (tokens, masks, advantages, etc.).
                                          None if a critical error occurred during the step.
                - bool: True if the episode terminated during this step, False otherwise.
        """
        G = self.config.group_size

        current_state_messages = ep.message_history.copy()
        logger.debug(
            f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}/{max_turns}] "
            f"Current state history length: {len(current_state_messages)}"
        )

        try:
            value_t = await self._estimate_value(
                episode_seed_for_sim=ep.seed, env_actions_to_replay=ep.actions
            )
            logger.debug(
                f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] Estimated V(s_t) = {value_t:.4f}"
            )
        except Exception as e_vt:
            logger.error(
                f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] Error estimating V(s_t): {e_vt}",
                exc_info=True,
            )
            return None, True  # Indicate error and episode termination

        messages_for_llm = current_state_messages.copy()
        agent_prompt_content = "<think>\n" if self.config.thinking_active else ""
        messages_for_llm.append({"role": "agent", "content": agent_prompt_content})

        try:
            responses = await self._sample_response(messages_for_llm, n=G)
            if not responses or len(responses) != G:
                logger.error(
                    f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] "
                    f"Expected {G} responses, got {len(responses) if responses else 0}. "
                    f"Aborting step."
                )
                return None, True  # Indicate error and episode termination
        except Exception as e_sample:
            logger.error(
                f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] Error sampling responses: {e_sample}",
                exc_info=True,
            )
            return None, True  # Indicate error and episode termination

        alt_full_responses: List[str] = []
        alt_parsed_actions: List[int] = []
        alt_env_actions: List[int] = []
        alt_raw_rewards: List[float] = []
        alt_combined_rewards: List[float] = []
        alt_next_state_msgs: List[List[Dict]] = []
        alt_is_terminal: List[bool] = []
        alt_tokens: List[List[int]] = []
        alt_masks: List[List[int]] = []
        alt_advantages: List[float] = []

        for i in range(G):
            llm_output_only = responses[i]
            full_agent_response = agent_prompt_content + llm_output_only
            alt_full_responses.append(full_agent_response)

            parsed_action = self._parse_tool_call(full_agent_response)
            alt_parsed_actions.append(parsed_action)

            env_action = (
                parsed_action if parsed_action != -1 else 0
            )  # Default to stick on parse error
            alt_env_actions.append(env_action)

            sim_env_i = None
            raw_env_reward_i = 0.0
            term_i, trunc_i = False, False
            next_state_msgs_i = []
            sim_obs_next_i = None

            try:
                sim_env_i = gymnasium.make(self.config.env_name)
                # replay env to same state as current episode
                _, _ = sim_env_i.reset(seed=ep.seed)
                for prev_action_idx, prev_action in enumerate(ep.actions):
                    _, _, term_replay, trunc_replay, _ = sim_env_i.step(prev_action)
                    if term_replay or trunc_replay:
                        logger.error(
                            f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1} Alt: {i}] "
                            f"Sim env for alternative {i} terminated prematurely during history replay "
                            f"(action {prev_action_idx+1}/{len(ep.actions)}). State mismatch or unexpected termination."
                        )
                        term_i, trunc_i = True, True
                        raw_env_reward_i = 0.0
                        break

                if not (term_i or trunc_i):
                    sim_obs_next_i, raw_env_reward_i, term_i, trunc_i, _ = (
                        sim_env_i.step(env_action)
                    )

                alt_raw_rewards.append(raw_env_reward_i)
                alt_is_terminal.append(term_i or trunc_i)

                combined_reward_i = self._score_response(
                    raw_env_reward_i, full_agent_response, parsed_action
                )
                alt_combined_rewards.append(combined_reward_i)

                current_state_plus_response_i = current_state_messages + [
                    {"role": "agent", "content": full_agent_response}
                ]
                if sim_obs_next_i is not None and not (term_i or trunc_i):
                    next_state_msgs_i = current_state_plus_response_i + [
                        {
                            "role": "environment",
                            "content": self._format_observation(sim_obs_next_i),
                        }
                    ]
                else:
                    next_state_msgs_i = current_state_plus_response_i
                alt_next_state_msgs.append(next_state_msgs_i)

                tokenized_i = tokenize_for_trainer(self.tokenizer, next_state_msgs_i)
                alt_tokens.append(tokenized_i["tokens"])
                alt_masks.append(tokenized_i["masks"])

            except Exception as e_sim:
                logger.error(
                    f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1} Alt: {i}] "
                    f"Error simulating action {env_action} for alternative: {e_sim}",
                    exc_info=True,
                )
                alt_raw_rewards.append(0.0)
                alt_combined_rewards.append(-1.0)
                alt_next_state_msgs.append(
                    current_state_messages
                    + [{"role": "agent", "content": full_agent_response}]
                )
                alt_is_terminal.append(True)
                alt_tokens.append([])
                alt_masks.append([])
            finally:
                if sim_env_i:
                    sim_env_i.close()

        alt_value_next: List[float] = []
        for i in range(G):
            if not alt_is_terminal[i]:
                try:
                    actions_to_reach_s_prime = ep.actions + [alt_env_actions[i]]
                    value_next_i = await self._estimate_value(
                        episode_seed_for_sim=ep.seed,
                        env_actions_to_replay=actions_to_reach_s_prime,
                    )
                    alt_value_next.append(value_next_i)
                except Exception as e_vn:
                    logger.error(
                        f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1} Alt: {i}] "
                        f"Error estimating V(s') for alternative: {e_vn}",
                        exc_info=True,
                    )
                    alt_value_next.append(0.0)
            else:
                alt_value_next.append(0.0)

        for i in range(G):
            if (
                i < len(alt_combined_rewards)
                and i < len(alt_value_next)
                and value_t is not None
            ):
                advantage_i = alt_combined_rewards[i] + alt_value_next[i] - value_t
                alt_advantages.append(advantage_i)
                logger.debug(
                    f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1} Alt: {i}] "
                    f"CombinedR={alt_combined_rewards[i]:.2f}, V_t={value_t:.2f}, "
                    f"V_t+1={alt_value_next[i]:.2f} => Advantage={advantage_i:.2f}"
                )
            else:
                logger.warning(
                    f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1} Alt: {i}] "
                    f"Skipping advantage calculation due to missing data or value_t. "
                    f"len(alt_combined_rewards)={len(alt_combined_rewards)}, len(alt_value_next)={len(alt_value_next)}"
                )
                alt_advantages.append(-float("inf"))

        if not (
            len(alt_tokens) == G
            and len(alt_masks) == G
            and len(alt_advantages) == G
            and len(alt_next_state_msgs) == G
            and len(alt_parsed_actions) == G
        ):
            logger.error(
                f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] "
                f"Mismatch in alternative list lengths before creating ScoredDataGroup. "
                f"Tokens:{len(alt_tokens)}, Masks:{len(alt_masks)}, Adv:{len(alt_advantages)}, "
                f"Msgs:{len(alt_next_state_msgs)}, ParsedAct:{len(alt_parsed_actions)}. Expected {G} for all. "
                f"Aborting step."
            )
            return None, True

        current_step_data = BlackjackScoredDataGroup(
            seed=ep.seed,
            tokens=alt_tokens,
            masks=alt_masks,
            scores=alt_advantages,
            messages=alt_next_state_msgs,
            parsed_actions=alt_parsed_actions,
        )

        alt_token_lengths = [len(tkns) if tkns else 0 for tkns in alt_tokens]

        best_advantage_idx = select_best_index(
            primary_scores=alt_advantages,
            secondary_scores=alt_token_lengths,
            primary_higher_is_better=True,
            secondary_lower_is_better=True,
        )

        chosen_advantage_for_log = (
            alt_advantages[best_advantage_idx]
            if best_advantage_idx < len(alt_advantages)
            else float("-inf")
        )
        chosen_token_length_for_log = (
            alt_token_lengths[best_advantage_idx]
            if best_advantage_idx < len(alt_token_lengths)
            else float("-inf")
        )
        logger.debug(
            f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] "
            f"Selected Alt {best_advantage_idx} "
            f"(Adv: {chosen_advantage_for_log}, "
            f"Tokens: {chosen_token_length_for_log}) "
            f"from {G} alternatives."
        )

        chosen_env_action = (
            alt_env_actions[best_advantage_idx]
            if best_advantage_idx < len(alt_env_actions)
            else 0
        )
        chosen_full_response = (
            alt_full_responses[best_advantage_idx]
            if best_advantage_idx < len(alt_full_responses)
            else ""
        )
        chosen_parsed_action = (
            alt_parsed_actions[best_advantage_idx]
            if best_advantage_idx < len(alt_parsed_actions)
            else -1
        )

        logger.info(
            f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] Chosen action to step env: "
            f"{chosen_env_action} (from Alt {best_advantage_idx} with "
            f"Adv {chosen_advantage_for_log})"
        )

        ep.num_total_actions += 1
        if chosen_parsed_action != -1:
            ep.num_correct_actions += 1

        response_for_history = truncate_thinking(
            chosen_full_response,
            self.tokenizer,
            self.config.max_think_chars_history,
        )
        ep.message_history.append({"role": "agent", "content": response_for_history})

        (
            main_obs_next,
            main_reward_this_step,
            main_term_this_step,
            main_trunc_this_step,
        ) = (None, 0.0, False, False)
        try:
            (
                main_obs_next,
                main_reward_this_step,
                main_term_this_step,
                main_trunc_this_step,
                _,
            ) = ep.env.step(chosen_env_action)

            ep.actions.append(chosen_env_action)
            ep.step_rewards.append(main_reward_this_step)
            ep.num_steps += 1

            if main_obs_next:
                ep.message_history.append(
                    {
                        "role": "environment",
                        "content": self._format_observation(main_obs_next),
                    }
                )
        except Exception as e_main_step:
            logger.error(
                f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] "
                f"Error stepping MAIN environment with chosen action {chosen_env_action}: {e_main_step}",
                exc_info=True,
            )
            main_term_this_step, main_trunc_this_step = True, True

        is_episode_terminal_this_step = main_term_this_step or main_trunc_this_step

        return current_step_data, is_episode_terminal_this_step

    async def score(
        self, rollout_group_data: List[BlackjackScoredDataGroup]
    ) -> List[Optional[BlackjackScoredDataGroup]]:
        """Pass through rollout data. The 'scores' field in BlackjackScoredDataGroup
        already contains the A*(s,a) advantages from the collection phase.

        If you wanted to play around with additional scoring metrics, you could do so here.
        Eg, bonuses for the specific winning action trajectory

        Args:
            rollout_group_data: List of BlackjackScoredDataGroup objects containing the collected rollout data.

        Returns:
            List of BlackjackScoredDataGroup objects with the scores field updated.
        """
        logger.info(f"[Score] Processing {len(rollout_group_data)} steps.")
        return rollout_group_data

    async def collect_trajectories(
        self, item: Tuple[int, int]
    ) -> Tuple[List[BlackjackScoredDataGroup], List[Tuple[int, int]]]:
        """Collect data for ONE FULL trajectory (episode) by repeatedly calling _next_step.

        This method initializes an episode, then iterates through turns, calling `_next_step`
        to get data for each turn. It accumulates this data and handles episode termination,
        final metric calculation, and cleanup.

        Args:
            item: Tuple containing the seed and the group index (group index currently unused here).

        Returns:
            Tuple of two lists:
            - List of BlackjackScoredDataGroup objects: Contains the collected data for each step of the trajectory.
            - List of Tuple[int, int]: Backlog items (always empty in this implementation).
        """
        seed, _ = item
        G_config = self.config.group_size
        max_turns = self.config.max_turns or 5

        trajectory_data_for_trainer: List[BlackjackScoredDataGroup] = []

        logger.info(
            f"[Collect Trajectories Seed: {seed}] Starting new trajectory. "
            f"Group size G={G_config}, Max turns={max_turns}."
        )

        try:
            ep = self._get_or_create_episode(seed)
        except Exception as e:
            logger.error(
                f"[Collect Trajectories Seed: {seed}] Fatal error creating/getting episode: {e}",
                exc_info=True,
            )
            return [], []

        for turn_idx in range(max_turns):
            logger.debug(
                f"[Collect Trajectories Seed: {seed}] Attempting turn {turn_idx + 1}/{max_turns}."
            )

            step_data, is_terminal_this_step = await self._next_step(
                ep, turn_idx, max_turns
            )

            if step_data:
                trajectory_data_for_trainer.append(step_data)
            else:
                logger.error(
                    f"[Collect Trajectories Seed: {seed}] Turn {turn_idx + 1} failed to produce data."
                    " Terminating episode."
                )
                is_terminal_this_step = True

            if is_terminal_this_step:
                final_reward_at_termination = (
                    sum(ep.step_rewards) if ep.step_rewards else 0.0
                )
                logger.info(
                    f"[Collect Trajectories Seed: {seed}] Episode ended at turn {turn_idx + 1}. "
                    f"Reason: step reported terminal. Total raw env reward: {final_reward_at_termination:.2f}"
                )
                break
        else:
            logger.info(
                f"[Collect Trajectories Seed: {seed}] Episode reached max_turns ({max_turns})."
            )

        final_raw_reward = sum(ep.step_rewards) if ep.step_rewards else 0.0
        logger.info(
            f"[Collect Trajectories Seed: {seed}] Finished collecting trajectory. "
            f"Total steps in trajectory: {len(trajectory_data_for_trainer)}, "
            f"Actual turns in episode: {ep.num_steps}, "
            f"Final raw reward: {final_raw_reward:.2f}"
        )

        if ep:
            game_outcome = 0
            if final_raw_reward > 0:
                game_outcome = 1
            elif final_raw_reward < 0:
                game_outcome = -1

            episode_summary_metrics = {
                "seed": ep.seed,
                "total_reward": final_raw_reward,
                "num_steps": ep.num_steps,
                "num_correct_actions": ep.num_correct_actions,
                "num_total_actions": ep.num_total_actions,
                "game_outcome": game_outcome,
            }
            self.completed_episode_metrics_buffer.append(episode_summary_metrics)
            logger.debug(
                f"[Collect Trajectories Seed: {seed}] Added episode summary to buffer: {episode_summary_metrics}"
            )

        if seed in self.episodes:
            try:
                if (
                    hasattr(self.episodes[seed], "env")
                    and self.episodes[seed].env is not None
                ):
                    self.episodes[seed].env.close()
            except Exception as e_close:
                logger.warning(
                    f"[Collect Trajectories Seed: {seed}] Exception closing environment for episode: {e_close}",
                    exc_info=True,
                )
            del self.episodes[seed]

        if not trajectory_data_for_trainer:
            logger.warning(
                f"[Collect Trajectories Seed: {seed}] Collected an empty trajectory (no valid steps)."
            )
            return [], []

        limited_trajectory_data = ensure_trajectory_token_limit(
            trajectory_data_for_trainer,
            self.tokenizer,
            self.config.max_trajectory_tokens,
        )

        if not limited_trajectory_data:
            logger.warning(
                f"[Collect Trajectories Seed: {seed}] Trajectory became empty after token limiting."
            )
            return [], []

        return limited_trajectory_data, []

    async def setup(self):
        pass

    async def get_next_item(self) -> Tuple[int, int]:
        return (random.randint(0, 1000000), 0)

    async def rollout_and_score_eval(self, seed: int) -> Dict[str, float]:
        """Run a single episode for evaluation with a single response per step."""
        ep = self._get_or_create_episode(seed)
        max_turns = self.config.max_turns or 5
        metrics = {
            "seed": seed,
            "total_reward": 0.0,
            "num_turns": 0,
            "num_correct_actions": 0,
            "num_invalid_actions": 0,
            "game_outcome": 0,
        }

        for turn in range(max_turns):
            messages = ep.message_history.copy()
            agent_prompt_content = "<think>\n" if self.config.thinking_active else ""
            messages.append({"role": "agent", "content": agent_prompt_content})

            responses = await self._sample_response(messages, n=1)
            if not responses:
                logger.error(
                    f"[Eval Seed: {seed}, Turn: {turn+1}] No response. Aborting."
                )
                break

            llm_output_only = responses[0]
            full_agent_response = agent_prompt_content + llm_output_only

            action = self._parse_tool_call(full_agent_response)
            if action == -1:
                metrics["num_invalid_actions"] += 1
                action = 0
            else:
                metrics["num_correct_actions"] += 1

            try:
                obs, reward, term, trunc, _ = ep.env.step(action)
            except Exception as e:
                logger.error(f"[Eval Seed: {seed}, Turn: {turn+1}] Env error: {e}")
                term = True
                reward = -1.0
                obs = None

            metrics["total_reward"] += reward
            metrics["num_turns"] = turn + 1

            response_for_history = truncate_thinking(
                full_agent_response, self.tokenizer, self.config.max_think_chars_history
            )

            ep.message_history.append(
                {"role": "agent", "content": response_for_history}
            )

            if obs:
                ep.message_history.append(
                    {"role": "environment", "content": self._format_observation(obs)}
                )

            if term or trunc:
                metrics["game_outcome"] = int(reward)
                logger.info(f"[Eval Seed: {seed}] Episode ended. Outcome: {reward}")
                break

        ep.env.close()
        del self.episodes[seed]
        return metrics

    async def evaluate(self, *args, **kwargs):
        if not self.config.use_wandb:
            logger.info("Skipping evaluation as wandb is not enabled.")
            return
        num_eval_episodes = self.config.eval_episodes
        logger.info(f"Starting evaluation for {num_eval_episodes} episodes.")
        eval_tasks = [
            self.rollout_and_score_eval(random.randint(1000001, 2000000))
            for _ in range(num_eval_episodes)
        ]
        all_metrics = await tqdm_asyncio.gather(*eval_tasks)
        valid_metrics = [m for m in all_metrics if m]
        if not valid_metrics:
            logger.warning("No valid evaluation metrics.")
            return

        num_completed = len(valid_metrics)
        avg_total_reward = sum(m["total_reward"] for m in valid_metrics) / num_completed
        avg_num_turns = sum(m["num_turns"] for m in valid_metrics) / num_completed
        total_correct = sum(m["num_correct_actions"] for m in valid_metrics)
        total_invalid = sum(m["num_invalid_actions"] for m in valid_metrics)
        total_actions = total_correct + total_invalid
        action_accuracy = total_correct / total_actions if total_actions > 0 else 0
        win_rate = (
            sum(1 for m in valid_metrics if m["game_outcome"] == 1) / num_completed
        )
        loss_rate = (
            sum(1 for m in valid_metrics if m["game_outcome"] == -1) / num_completed
        )
        draw_rate = (
            sum(1 for m in valid_metrics if m["game_outcome"] == 0) / num_completed
        )

        self.eval_metrics = [
            ("eval/avg_total_reward", avg_total_reward),
            ("eval/avg_num_turns", avg_num_turns),
            ("eval/action_accuracy", action_accuracy),
            ("eval/win_rate", win_rate),
            ("eval/loss_rate", loss_rate),
            ("eval/draw_rate", draw_rate),
            ("eval/num_completed_episodes", num_completed),
        ]
        logger.info(f"Evaluation completed. Metrics: {self.eval_metrics}")

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.completed_episode_metrics_buffer:
            num_episodes = len(self.completed_episode_metrics_buffer)
            avg_reward = (
                sum(m["total_reward"] for m in self.completed_episode_metrics_buffer)
                / num_episodes
            )
            avg_steps = (
                sum(m["num_steps"] for m in self.completed_episode_metrics_buffer)
                / num_episodes
            )
            win_rate = (
                sum(
                    1
                    for m in self.completed_episode_metrics_buffer
                    if m["game_outcome"] == 1
                )
                / num_episodes
            )
            wandb_metrics[
                f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_reward"
            ] = avg_reward
            wandb_metrics[
                f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_steps"
            ] = avg_steps
            wandb_metrics[
                f"{self.wandb_prepend or 'blackjack'}_train/episode_win_rate"
            ] = win_rate
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/num_episodes"] = (
                num_episodes
            )
            self.completed_episode_metrics_buffer = []
        await super().wandb_log(wandb_metrics)

    @classmethod
    def config_init(cls) -> Tuple[BlackjackEnvConfig, List[APIServerConfig]]:
        env_config = BlackjackEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            max_num_workers=128,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="fundamental_metric_prediction",
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            env_name="Blackjack-v1",
            temperature=0.7,
            top_p=0.9,
            max_turns=5,
            thinking_active=True,
            eval_episodes=100,
            max_think_chars_history=3000,
            max_trajectory_tokens=24576,
            debug_mode=False,
            tiebreak_token_factor=0.01,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=256,
            )
        ]
        return env_config, server_configs

    @classmethod
    def cli(cls):
        super().cli()


if __name__ == "__main__":
    BlackjackEnv.cli()
