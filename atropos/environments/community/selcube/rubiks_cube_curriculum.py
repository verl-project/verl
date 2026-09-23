#!/usr/bin/env python3
"""
RubiksCubeCurriculum: Curriculum learning utilities for Rubik's Cube environment

This module provides classes and functions to implement curriculum learning for
the Rubik's cube environment, where the difficulty gradually increases as the
model improves in solving simpler challenges.
"""

import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CurriculumLevel:
    """Represents a curriculum learning level for Rubik's cube solving"""

    def __init__(
        self,
        level: int,
        min_scramble_moves: int,
        max_scramble_moves: int,
        max_steps: int,
        reward_per_correctly_placed_cubie: float,
        example_patterns: List[List[str]] = None,
        description: str = None,
    ):
        """
        Initialize a curriculum level

        Args:
            level: Level number (higher is more difficult)
            min_scramble_moves: Minimum number of scramble moves
            max_scramble_moves: Maximum number of scramble moves
            max_steps: Maximum allowed steps to solve at this level
            reward_per_correctly_placed_cubie: Reward multiplier for correctly placed cubies
            example_patterns: Optional list of move sequences to learn at this level
            description: Human-readable description of this level
        """
        self.level = level
        self.min_scramble_moves = min_scramble_moves
        self.max_scramble_moves = max_scramble_moves
        self.max_steps = max_steps
        self.reward_per_correctly_placed_cubie = reward_per_correctly_placed_cubie
        self.example_patterns = example_patterns or []
        self.description = (
            description
            or f"Level {level}: {min_scramble_moves}-{max_scramble_moves} scramble moves"
        )

    def get_scramble_moves(self) -> int:
        """Get a random number of scramble moves within the level's range"""
        return random.randint(self.min_scramble_moves, self.max_scramble_moves)

    def __repr__(self) -> str:
        return (
            f"CurriculumLevel(level={self.level}, "
            f"scramble_moves={self.min_scramble_moves}-{self.max_scramble_moves})"
        )


class RubiksCubeCurriculum:
    """Manages curriculum progression for Rubik's cube solver training"""

    def __init__(
        self,
        starting_level: int = 1,
        max_level: int = 5,
        auto_progress: bool = True,
        success_threshold: float = 0.7,
        advancement_window_size: int = 50,
        min_solved_at_level: int = 25,
    ):
        """
        Initialize the curriculum manager

        Args:
            starting_level: Initial curriculum level
            max_level: Maximum curriculum level
            auto_progress: Whether to automatically progress through levels
            success_threshold: Success rate threshold to advance to next level
            advancement_window_size: Number of episodes to consider for advancement
            min_solved_at_level: Minimum number of episodes that must be solved at a level
                                 before considering advancement
        """
        self.current_level = starting_level
        self.max_level = max_level
        self.auto_progress = auto_progress
        self.success_threshold = success_threshold
        self.advancement_window_size = advancement_window_size
        self.min_solved_at_level = min_solved_at_level

        # Track episode results for potential advancement
        self.episode_results = []  # List of (level, is_solved, num_steps) tuples

        # Define curriculum levels
        self.levels = self._create_default_curriculum()

    def _create_default_curriculum(self) -> Dict[int, CurriculumLevel]:
        """Create the default curriculum progression"""
        levels = {}

        # Level 1: Very simple scrambles (1-3 moves)
        levels[1] = CurriculumLevel(
            level=1,
            min_scramble_moves=1,
            max_scramble_moves=3,
            max_steps=15,
            reward_per_correctly_placed_cubie=0.1,
            description="Beginner level - Single move to Triple moves scrambles",
        )

        # Level 2: Simple scrambles (4-7 moves)
        levels[2] = CurriculumLevel(
            level=2,
            min_scramble_moves=4,
            max_scramble_moves=7,
            max_steps=20,
            reward_per_correctly_placed_cubie=0.075,
            description="Easy level - Learn basic patterns and simple sequences",
        )

        # Level 3: Moderate scrambles (8-12 moves)
        levels[3] = CurriculumLevel(
            level=3,
            min_scramble_moves=8,
            max_scramble_moves=12,
            max_steps=25,
            reward_per_correctly_placed_cubie=0.05,
            description="Intermediate level - More complex patterns and sequences",
        )

        # Level 4: Challenging scrambles (13-17 moves)
        levels[4] = CurriculumLevel(
            level=4,
            min_scramble_moves=13,
            max_scramble_moves=17,
            max_steps=30,
            reward_per_correctly_placed_cubie=0.025,
            description="Advanced level - Complex scrambles requiring deep planning",
        )

        # Level 5: Expert scrambles (18-22 moves)
        levels[5] = CurriculumLevel(
            level=5,
            min_scramble_moves=18,
            max_scramble_moves=22,
            max_steps=40,
            reward_per_correctly_placed_cubie=0.01,
            description="Expert level - Near optimal scrambles approaching God's number",
        )

        return levels

    def get_current_level(self) -> CurriculumLevel:
        """Get the current curriculum level"""
        return self.levels[self.current_level]

    def record_episode_result(
        self, level: int, is_solved: bool, num_steps: int
    ) -> None:
        """
        Record the result of an episode

        Args:
            level: The curriculum level of the episode
            is_solved: Whether the cube was solved successfully
            num_steps: Number of steps taken in the episode
        """
        self.episode_results.append((level, is_solved, num_steps))

        # Keep only the most recent window of results
        if len(self.episode_results) > self.advancement_window_size:
            self.episode_results = self.episode_results[-self.advancement_window_size :]

        # Check if we should advance to the next level
        if self.auto_progress:
            self._check_advancement()

    def _check_advancement(self) -> None:
        """Check if we should advance to the next level based on recent performance"""
        # Only consider episodes at the current level
        current_level_results = [
            r for r in self.episode_results if r[0] == self.current_level
        ]

        # Need enough data to make a decision
        if len(current_level_results) < self.min_solved_at_level:
            return

        # Calculate success rate at current level
        success_count = sum(1 for _, is_solved, _ in current_level_results if is_solved)
        success_rate = success_count / len(current_level_results)

        # Log the current performance
        logger.info(
            f"Curriculum performance: Level {self.current_level}, "
            f"Success rate: {success_rate:.2f} ({success_count}/{len(current_level_results)})"
        )

        # Check if we should advance
        if (
            success_rate >= self.success_threshold
            and success_count >= self.min_solved_at_level
            and self.current_level < self.max_level
        ):

            self.current_level += 1
            logger.info(
                f"Advancing to curriculum level {self.current_level}: "
                f"{self.levels[self.current_level].description}"
            )

            # Reset episode results after advancing
            self.episode_results = []

    def set_level(self, level: int) -> None:
        """
        Manually set the curriculum level

        Args:
            level: The new curriculum level (must be between 1 and max_level)
        """
        if level < 1 or level > self.max_level:
            logger.warning(
                f"Invalid curriculum level {level}. Must be between 1 and {self.max_level}. "
                f"Keeping current level {self.current_level}."
            )
            return

        self.current_level = level
        logger.info(
            f"Manually set curriculum to level {level}: {self.levels[level].description}"
        )

        # Reset episode results after manual level change
        self.episode_results = []

    def get_level_metrics(self) -> Dict[str, Any]:
        """Get metrics for the current curriculum level"""
        current_level_results = [
            r for r in self.episode_results if r[0] == self.current_level
        ]

        if not current_level_results:
            return {
                "curriculum_level": self.current_level,
                "curriculum_description": self.levels[self.current_level].description,
                "level_success_rate": 0.0,
                "level_episodes": 0,
                "level_solved_count": 0,
                "level_avg_steps": 0.0,
                "progress_to_next_level": 0.0,
            }

        success_count = sum(1 for _, is_solved, _ in current_level_results if is_solved)
        success_rate = success_count / len(current_level_results)

        # Calculate average steps for solved episodes
        solved_episodes = [
            (level, solved, steps)
            for level, solved, steps in current_level_results
            if solved
        ]
        avg_steps = sum(steps for _, _, steps in solved_episodes) / max(
            1, len(solved_episodes)
        )

        # Calculate progress to next level (0.0 to 1.0)
        if self.current_level >= self.max_level:
            progress_to_next = 1.0
        else:
            progress_threshold = self.success_threshold * self.min_solved_at_level
            current_progress = success_rate * len(current_level_results)
            progress_to_next = min(1.0, current_progress / progress_threshold)

        return {
            "curriculum_level": self.current_level,
            "curriculum_description": self.levels[self.current_level].description,
            "level_success_rate": success_rate,
            "level_episodes": len(current_level_results),
            "level_solved_count": success_count,
            "level_avg_steps": avg_steps,
            "progress_to_next_level": progress_to_next,
        }


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create curriculum manager
    curriculum = RubiksCubeCurriculum(
        starting_level=1,
        max_level=5,
        auto_progress=True,
        success_threshold=0.7,
        advancement_window_size=50,
        min_solved_at_level=25,
    )

    # Simulate some episodes
    # In a real setup, these results would come from actual cube-solving episodes
    for _ in range(40):
        # Simulate success with 80% probability for level 1
        is_solved = random.random() < 0.8
        steps = random.randint(5, 15)
        curriculum.record_episode_result(1, is_solved, steps)

    # Print metrics
    print(curriculum.get_level_metrics())

    # Current level should now be 2 if enough episodes were solved
    print(f"Current level: {curriculum.current_level}")

    # Manually set to level 3
    curriculum.set_level(3)
    print(f"After manual set, current level: {curriculum.current_level}")
