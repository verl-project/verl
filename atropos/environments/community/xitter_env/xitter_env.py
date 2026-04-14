import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# --- Mocked Trending Topics ---
# In a real implementation, this would come from an API
MOCK_TRENDING_TOPICS = [
    "AI in Art",
    "Climate Change Solutions",
    "New Space Discoveries",
]

# --- Mocked Social Media State (Simplified) ---
# This would be more complex in a real environment, likely managed by dedicated classes
# and updated dynamically.
MOCK_SOCIAL_FEED = [
    {
        "id": "post1",
        "agent_id": "agent_alpha",
        "content": "Just enjoyed a great virtual concert! #metaverse",
        "likes": 10,
        "comments": [],
    },
    {
        "id": "post2",
        "agent_id": "agent_beta",
        "content": "Excited about the upcoming Atropos hackathon!",
        "likes": 15,
        "comments": [],
    },
]
MOCK_AGENT_PROFILES = {
    "agent_gamma": {"posts": [], "score": 0, "notifications": []},
    "agent_delta": {"posts": [], "score": 0, "notifications": []},
}


# We'll need to define a system prompt
SYSTEM_PROMPT_TEMPLATE = """You are '{agent_id}', an agent on Xitter, a simulated social media platform.
Your goal is to maximize engagement by posting interesting content (Xits), liking relevant Xits,
and making insightful comments. You can also choose to DO_NOTHING.
Current trending topics: {trending_topics}

Recent Xits in your feed (newest first):
{feed_preview}

Your recent notifications:
{notifications_preview}

Choose one action:
1. POST <your_xits_content_here> (max 140 chars)
2. LIKE <post_id_to_like> (e.g., LIKE post_3)
3. COMMENT <post_id_to_comment_on> <your_comment_content_here> (e.g., COMMENT post_2 Great point!)
4. DO_NOTHING

Your response should be ONLY the action string (e.g., "POST This is my new Xit! #awesome").
"""


@dataclass
class XitterEnvConfig(BaseEnvConfig):
    """Configuration for the Xitter (Social Media) Environment."""

    # Reward weights
    like_reward_weight: float = 0.1
    comment_reward_weight: float = 0.5
    trending_topic_bonus: float = 0.2
    perform_like_reward: float = 0.05  # Small reward for the act of liking
    perform_comment_reward: float = 0.1  # Small reward for the act of commenting
    do_nothing_reward: float = 0.0  # Reward for doing nothing
    invalid_action_penalty: float = -0.5
    action_cost: float = -0.01  # Small cost for any action

    # Environment parameters
    num_agents: int = 2
    max_feed_size: int = 20
    max_notifications_display: int = 5  # How many notifications to show in prompt
    initial_trending_topics: List[str] = field(
        default_factory=lambda: ["AI in Art", "Climate Solutions", "Space Exploration"]
    )

    # For wandb logging of agent-specific scores
    track_individual_agent_scores: bool = True


class XitterEnv(BaseEnv):
    # Assuming XitterEnvConfig is defined elsewhere and includes:
    # group_size, max_token_length, tokenizer_name, various reward_weights, etc.
    env_config_cls = XitterEnvConfig  # Assign your custom config

    def __init__(
        self,
        config: BaseEnvConfig,  # Use BaseEnvConfig or your specific XitterEnvConfig
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        # Initialize social media state
        self.social_feed = MOCK_SOCIAL_FEED  # list of posts
        self.agent_profiles = MOCK_AGENT_PROFILES  # dict of agent_id -> profile_data
        self.trending_topics = MOCK_TRENDING_TOPICS
        self.current_agent_turn = 0  # Simple round-robin for turns
        self.agent_ids = list(self.agent_profiles.keys())

        # Example reward function instances (you'd define these)
        # self.engagement_reward_fn = EngagementReward(...)
        # self.relevance_reward_fn = RelevanceReward(...)

    async def setup(self):
        # Load tokenizer, initialize agents, fetch initial trends, etc.
        # self.tokenizer is already initialized in BaseEnv
        logging.info(f"{self.name or 'XitterEnv'} setup complete.")
        # Potentially fetch initial trending topics here
        # self.trending_topics = await self.fetch_trending_topics()

    async def get_next_item(self) -> Item:
        # Determine which agent's turn it is and prepare their observation
        agent_id_for_turn = self.agent_ids[
            self.current_agent_turn % len(self.agent_ids)
        ]
        self.current_agent_turn += 1

        # Construct observation for the agent
        # For simplicity, we'll just pass the agent_id and let collect_trajectories build the full prompt
        # In a more complex setup, you'd build a richer observation here.
        # The Item can be any structure your collect_trajectories method expects.
        # Here, a tuple: (agent_id_acting, current_feed_snapshot, current_trends, agent_notifications)
        # For now, let's simplify and pass agent_id and have collect_trajectories build the prompt.
        return (
            agent_id_for_turn,
            {
                "trending_topics": self.trending_topics,
                "feed_preview": self.social_feed[:5],
            },
        )

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Generates a group of potential actions for an agent and gathers data for scoring.
        The `item` from `get_next_item` should provide context for the current agent's turn.
        """
        agent_id_acting, observation_context = item
        # trending_topics_str = ", ".join(observation_context.get("trending_topics", []))

        # Construct the prompt for the LLM agent based on its observation
        # This would include a view of the feed, notifications, and trending topics.
        # For this example, we'll use a simplified prompt.
        # The actual prompt engineering is a key part of designing the environment.
        recent_posts = str(observation_context.get("feed_preview", []))
        prompt_content = f"It's your turn, {agent_id_acting}. Recent posts: {recent_posts}. What do you do?"

        feed_preview_text = ", ".join(
            [post["content"] for post in observation_context.get("feed_preview", [])]
        )
        notifications_text = ", ".join(
            self.agent_profiles[agent_id_acting].get("notifications", [])
        )

        messages_for_llm = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(
                    agent_id=agent_id_acting,
                    trending_topics=", ".join(self.trending_topics),
                    feed_preview=feed_preview_text,
                    notifications_preview=notifications_text,
                ),
            },
            {"role": "user", "content": prompt_content},
        ]

        # Apply chat template for the LLM
        # The prefill is not used here as the action choice is part of the LLM's generation
        prompt_str_for_llm = self.tokenizer.apply_chat_template(
            messages_for_llm,
            tokenize=False,
            add_generation_prompt=True,  # Important for instruct/chat models
        )

        # Get self.config.group_size completions from the LLM
        # Each completion is a potential action (post, like, comment)
        try:
            completions_obj = await self.server.completion(  # Using completion for free-form action generation
                prompt=prompt_str_for_llm,  # BaseEnv.server.completion expects a string prompt
                n=self.config.group_size,
                max_tokens=self.config.max_token_length
                // 4,  # Max tokens for the action itself
                temperature=0.7,  # Allow some diversity in actions
            )
        except Exception as e:
            logging.error(
                f"Error getting completions from LLM for agent {agent_id_acting}: {e}"
            )
            return None, []

        # This list will hold tuples of (full_chat_history_for_action, action_details_for_scoring)
        # where action_details_for_scoring will be passed to the `score` method.
        trajectories_for_scoring: List[Tuple[List[Dict[str, str]], Dict[str, Any]]] = []

        for choice in completions_obj.choices:
            llm_generated_action_text = choice.text.strip()

            # Simulate the action and its effect on the environment state
            # This is a placeholder for your actual simulation logic.
            # It needs to parse llm_generated_action_text (e.g., "POST My new cat video #cats")
            # and update self.social_feed, self.agent_profiles, etc.
            action_type, action_params, action_valid = self._parse_and_simulate_action(
                agent_id_acting, llm_generated_action_text
            )

            if not action_valid:
                # Potentially penalize invalid actions in the score function or give a default low score
                # For now, we'll still include it to be scored (and likely penalized)
                logging.warning(
                    f"Agent {agent_id_acting} performed an invalid action: {llm_generated_action_text}"
                )

            # Create the full message history for this trajectory (system, user, assistant_action)
            # The `messages_for_llm` already contains system and user turns.
            current_trajectory_messages = messages_for_llm + [
                {"role": "assistant", "content": llm_generated_action_text}
            ]

            # Context needed by the score function for this specific action
            # This will depend heavily on your reward components.
            scoring_context = {
                "agent_id": agent_id_acting,
                "action_type": action_type,  # "post", "like", "comment", "invalid", "do_nothing"
                "action_params": action_params,  # e.g., post_id for like, content for post
                "was_valid_action": action_valid,
                # For scoring a "post" action, you'd later fill these after observing next turn's interactions:
                # "likes_received_on_post": X,
                # "comments_received_on_post": Y,
                "trending_topics": self.trending_topics,  # Pass current trends for relevance scoring
            }
            # If the action was a 'post', we store the new post_id in scoring_context
            # so that the score function can later attribute likes/comments to it.
            # This implies that rewards for posts might be delayed by one or more turns.
            if action_type == "post" and "new_post_id" in action_params:
                scoring_context["post_id_created"] = action_params["new_post_id"]

            trajectories_for_scoring.append(
                (current_trajectory_messages, scoring_context)
            )

        # The `score` method will take this list and produce the ScoredDataGroup
        # The actual rewards might be assigned in `score` based on the outcome of these actions,
        # potentially looking at the state *after* all agents in a round have acted,
        # or even after a delay (e.g. likes/comments on a post arrive in future turns).
        # For simplicity, this example implies immediate scoring, but delayed rewards are common.

        # Pass the collected trajectories and their scoring contexts to the score method
        scored_data_group = await self.score(trajectories_for_scoring)

        # No backlog items in this simple version
        return scored_data_group, []

    def _parse_and_simulate_action(
        self, agent_id: str, action_text: str
    ) -> Tuple[str, Dict, bool]:
        """
        Parses the LLM's action string and simulates its effect on the environment.
        Returns: (action_type, action_params, was_valid)
        This is a placeholder and needs to be implemented based on your defined action space.
        """
        action_text_lower = action_text.lower()
        new_post_id_counter = len(self.social_feed)

        if action_text_lower.startswith("post "):
            content = action_text[5:].strip()
            if content:
                new_post_id_counter += 1
                post_id = f"post{new_post_id_counter}"
                new_post = {
                    "id": post_id,
                    "agent_id": agent_id,
                    "content": content,
                    "likes": 0,
                    "comments": [],
                }
                self.social_feed.insert(0, new_post)  # Add to top of feed
                self.agent_profiles[agent_id]["posts"].append(post_id)
                logging.info(f"Agent {agent_id} POSTED: {content}")
                return (
                    "post",
                    {"content": content, "new_post_id": post_id},
                    True,
                )
        elif action_text_lower.startswith("like "):
            try:
                post_id_to_like = action_text.split(" ")[1]
                for post in self.social_feed:
                    if post["id"] == post_id_to_like:
                        post["likes"] += 1
                        # Notify original poster (simplified)
                        original_poster_id = post["agent_id"]
                        if (
                            original_poster_id != agent_id
                            and original_poster_id in self.agent_profiles
                        ):
                            self.agent_profiles[original_poster_id][
                                "notifications"
                            ].append(f"{agent_id} liked your post {post_id_to_like}")
                        logging.info(f"Agent {agent_id} LIKED post: {post_id_to_like}")
                        return "like", {"post_id": post_id_to_like}, True
            except IndexError:
                return "invalid_like_format", {}, False
            return (
                "like_post_not_found",
                {"post_id": action_text.split(" ")[1]},
                False,
            )  # Post not found
        elif action_text_lower.startswith("comment "):
            parts = action_text.split(" ", 2)
            if len(parts) == 3:
                post_id_to_comment_on = parts[1]
                comment_content = parts[2].strip()
                if comment_content:
                    for post in self.social_feed:
                        if post["id"] == post_id_to_comment_on:
                            comment_id = f"comment{len(post['comments']) + 1}_on_{post_id_to_comment_on}"
                            post["comments"].append(
                                {
                                    "id": comment_id,
                                    "agent_id": agent_id,
                                    "content": comment_content,
                                }
                            )
                            # Notify original poster (simplified)
                            original_poster_id = post["agent_id"]
                            if (
                                original_poster_id != agent_id
                                and original_poster_id in self.agent_profiles
                            ):
                                self.agent_profiles[original_poster_id][
                                    "notifications"
                                ].append(
                                    f"{agent_id} commented on your post {post_id_to_comment_on}"
                                )
                            logging.info(
                                f"Agent {agent_id} COMMENTED on {post_id_to_comment_on}: {comment_content}"
                            )
                            return (
                                "comment",
                                {
                                    "post_id": post_id_to_comment_on,
                                    "content": comment_content,
                                },
                                True,
                            )
                return (
                    "invalid_comment_format",
                    {},
                    False,
                )  # Invalid comment content
            return (
                "invalid_comment_format",
                {},
                False,
            )  # Invalid command format
        elif action_text_lower == "do_nothing":
            logging.info(f"Agent {agent_id} DID NOTHING.")
            return "do_nothing", {}, True

        return "unknown_action", {"raw_action": action_text}, False

    async def score(
        self,
        trajectories_with_context: List[Tuple[List[Dict[str, str]], Dict[str, Any]]],
    ) -> Optional[ScoredDataGroup]:
        """
        Scores a group of trajectories.
        Each item in `trajectories_with_context` is a tuple:
        (full_message_history, scoring_context_for_this_action)
        """
        final_scores_group = ScoredDataGroup(tokens=[], masks=[], scores=[])
        if self.config.include_messages:  # From BaseEnvConfig
            final_scores_group["messages"] = []

        for (
            full_trajectory_messages,
            scoring_context,
        ) in trajectories_with_context:
            agent_id = scoring_context["agent_id"]
            action_type = scoring_context["action_type"]
            action_params = scoring_context["action_params"]
            was_valid_action = scoring_context["was_valid_action"]

            current_reward = 0.0

            if not was_valid_action:
                current_reward -= 0.5  # Penalty for invalid action
            else:
                # --- Engagement Rewards ---
                if action_type == "post":
                    # These rewards might be delayed. For now, let's assume we can get some immediate proxy
                    # or that `scoring_context` is populated with likes/comments that occurred *since* this post.
                    # This is a simplification; a more realistic model would update these over subsequent turns.
                    current_reward += (
                        scoring_context.get("likes_received_on_post_this_turn", 0)
                        * self.config.like_reward_weight
                    )
                    current_reward += (
                        scoring_context.get("comments_received_on_post_this_turn", 0)
                        * self.config.comment_reward_weight
                    )

                    # --- Content Quality & Relevance Rewards for Posts ---
                    post_content = action_params.get("content", "")
                    # Pseudo-code for relevance to trending topics
                    # relevance_score = calculate_relevance(post_content, self.trending_topics)
                    # current_reward += relevance_score * self.config.relevance_weight
                    # Example: check if any trending topic keyword is in the post
                    for trend in self.trending_topics:
                        if trend.lower() in post_content.lower():
                            current_reward += (
                                self.config.trending_topic_bonus
                            )  # Add this to your config
                            break  # Add bonus once per post if it hits any trend

                elif action_type == "like":
                    current_reward += (
                        self.config.perform_like_reward
                    )  # Small reward for liking
                elif action_type == "comment":
                    current_reward += (
                        self.config.perform_comment_reward
                    )  # Small reward for commenting
                    # comment_content = action_params.get("content", "")
                    # relevance_to_op_score = calculate_relevance(comment_content, original_post_content_for_comment)
                    # current_reward += relevance_to_op_score * self.config.comment_relevance_weight

                elif action_type == "do_nothing":
                    current_reward += (
                        self.config.do_nothing_reward
                    )  # Could be small positive, zero, or small negative

            # --- Action Cost ---
            # current_reward -= self.config.action_cost

            # Tokenize the full trajectory (system, user, assistant_action)
            # The `tokenize_for_trainer` utility handles creating tokens and appropriate masks.
            # `train_on_all_assistant_turns=True` ensures only assistant messages are unmasked for loss calculation.
            try:
                tokenized_output = tokenize_for_trainer(
                    self.tokenizer,
                    full_trajectory_messages,
                    train_on_all_assistant_turns=True,  # Or False if you want only the last turn
                    include_messages=self.config.include_messages,
                )
            except Exception as e:
                logging.error(
                    f"Tokenization error for agent {agent_id}, action {action_type}: {e}. Skipping trajectory."
                )
                logging.error(f"Problematic messages: {full_trajectory_messages}")
                continue

            final_scores_group["tokens"].append(tokenized_output["tokens"])
            final_scores_group["masks"].append(tokenized_output["masks"])
            final_scores_group["scores"].append(current_reward)
            if self.config.include_messages:
                final_scores_group["messages"].append(tokenized_output["messages"])

        if not final_scores_group["tokens"]:  # If all trajectories failed tokenization
            return None

        # Ensure scores are not all the same if configured
        if (
            self.config.ensure_scores_are_not_same
            and len(set(final_scores_group["scores"])) <= 1
            and len(final_scores_group["scores"]) > 1
        ):
            logging.info("All scores in the group are identical, returning None.")
            return None

        return final_scores_group

    # ... (wandb_log, create_rollout_table, etc. can be inherited or customized)
    # ... (evaluate method would simulate multiple turns and aggregate scores)
