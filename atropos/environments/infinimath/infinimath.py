#!/usr/bin/env python3
"""
Infinite Math - A reinforcement learning environment for math practice
using the mathgenerator library with curriculum-based advancement.
"""

from typing import Any, Dict, Optional

from .curriculum import MathCurriculum


class InfiniteMath:
    """
    A reinforcement learning environment for practicing math skills with
    curriculum-based advancement.

    This class uses the MathCurriculum to generate appropriate math problems
    and track performance, advancing difficulty as the learner improves.
    """

    def __init__(
        self,
        starting_level: int = 1,
        progress_threshold: float = 0.8,
        min_evaluations: int = 5,
        max_attempts_per_problem: int = 3,
    ):
        """
        Initialize the InfiniteMath environment.

        Args:
            starting_level: Initial difficulty level (default: 1)
            progress_threshold: Success rate needed to advance levels (default: 0.8)
            min_evaluations: Minimum evaluations before considering advancement (default: 5)
            max_attempts_per_problem: Maximum attempts allowed per problem (default: 3)
        """
        self.curriculum = MathCurriculum(
            starting_level=starting_level,
            progress_threshold=progress_threshold,
            min_evaluations=min_evaluations,
        )

        self.max_attempts = max_attempts_per_problem
        self.current_problem = None
        self.current_solution = None
        self.current_generator_id = None
        self.attempts_remaining = 0
        self.total_problems = 0
        self.correct_problems = 0

        # Generate the first problem
        self._generate_problem()

    def _generate_problem(self) -> None:
        """Generate a new problem from the curriculum."""
        self.current_problem, self.current_solution, self.current_generator_id = (
            self.curriculum.get_problem()
        )
        self.attempts_remaining = self.max_attempts

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.

        Returns:
            Dictionary with current state information
        """
        return {
            "problem": self.current_problem,
            "attempts_remaining": self.attempts_remaining,
            "current_level": self.curriculum.get_current_level(),
            "total_levels": self.curriculum.get_num_levels(),
            "level_description": self.curriculum.get_level_description(),
            "total_problems": self.total_problems,
            "correct_problems": self.correct_problems,
            "accuracy": self.correct_problems / max(1, self.total_problems),
        }

    def submit_answer(self, answer: str) -> Dict[str, Any]:
        """
        Submit an answer to the current problem.

        Args:
            answer: The learner's answer to the current problem

        Returns:
            Dictionary with the result of the submission and updated state
        """
        if self.current_problem is None:
            return {"error": "No active problem. Call reset() to start a new session."}

        # Clean up the answer for comparison (strip whitespace, convert to lowercase)
        cleaned_answer = str(answer).strip().lower()
        cleaned_solution = str(self.current_solution).strip().lower()

        # Check if the answer is correct
        is_correct = cleaned_answer == cleaned_solution

        # Update attempts
        self.attempts_remaining -= 1

        result = {
            "is_correct": is_correct,
            "correct_answer": (
                self.current_solution
                if self.attempts_remaining == 0 or is_correct
                else None
            ),
            "attempts_remaining": self.attempts_remaining,
        }

        # If correct or out of attempts, record performance and generate a new problem
        if is_correct or self.attempts_remaining == 0:
            self.total_problems += 1
            if is_correct:
                self.correct_problems += 1

            # Record performance in the curriculum
            self.curriculum.record_performance(self.current_generator_id, is_correct)

            # Check if we should advance to the next level
            did_advance = self.curriculum.advance_difficulty()
            result["did_advance_level"] = did_advance

            if did_advance:
                result["new_level"] = self.curriculum.get_current_level()
                result["level_description"] = self.curriculum.get_level_description()

            # Generate a new problem
            self._generate_problem()
            result["new_problem"] = self.current_problem

        return result

    def reset(self, level: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the environment, optionally to a specific difficulty level.

        Args:
            level: The difficulty level to reset to (default: current level)

        Returns:
            Dictionary with the new state
        """
        if level is not None:
            self.curriculum.reset(level)

        self.total_problems = 0
        self.correct_problems = 0

        # Generate a new problem
        self._generate_problem()

        return self.get_state()

    def get_difficulty_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get performance statistics for each difficulty level.

        Returns:
            Dictionary with statistics for each level
        """
        stats = {}

        for level in self.curriculum.DIFFICULTY_LEVELS.keys():
            success_rate = self.curriculum.get_success_rate(level)
            history = self.curriculum.performance_history[level]

            stats[level] = {
                "description": self.curriculum.get_level_description(level),
                "problems_attempted": len(history),
                "success_rate": (
                    success_rate if success_rate is not None else float("nan")
                ),
                "is_current_level": level == self.curriculum.get_current_level(),
            }

        return stats


def main():
    """Example usage of the InfiniteMath environment."""
    # Create the environment
    env = InfiniteMath(starting_level=1)

    print("Welcome to InfiniteMath!")
    print(
        f"Starting at level {env.get_state()['current_level']}: {env.get_state()['level_description']}"
    )

    playing = True
    while playing:
        # Display current problem
        state = env.get_state()
        print("\n" + "=" * 50)
        print(f"Level {state['current_level']}/{state['total_levels']}")
        print(f"Problem: {state['problem']}")
        print(f"Attempts remaining: {state['attempts_remaining']}")

        # Get user input
        answer = input("Your answer (or 'q' to quit, 'r' to reset): ")

        if answer.lower() == "q":
            playing = False
            continue
        elif answer.lower() == "r":
            level = input("Reset to level (1-7, or press Enter for current level): ")
            if level and level.isdigit() and 1 <= int(level) <= 7:
                env.reset(int(level))
            else:
                env.reset()
            continue

        # Submit the answer
        result = env.submit_answer(answer)

        # Display the result
        if result.get("is_correct", False):
            print("Correct!")
        else:
            print("Incorrect.")

            if result.get("correct_answer") is not None:
                print(f"The correct answer is: {result['correct_answer']}")

        # Check if we advanced to a new level
        if result.get("did_advance_level", False):
            print(f"\nCongratulations! You've advanced to level {result['new_level']}!")
            print(f"New level: {result['level_description']}")

        # If we have a new problem, show it
        if result.get("new_problem") is not None:
            print("\nNext problem is ready.")

    # Show final statistics
    print("\nFinal Statistics:")
    print(f"Total problems attempted: {env.total_problems}")
    print(f"Correct answers: {env.correct_problems}")
    if env.total_problems > 0:
        print(f"Overall accuracy: {env.correct_problems / env.total_problems:.2%}")

    level_stats = env.get_difficulty_stats()
    print("\nPerformance by level:")
    for level, stats in level_stats.items():
        if stats["problems_attempted"] > 0:
            print(
                f"Level {level}: {stats['success_rate']:.2%} success rate ({stats['problems_attempted']} problems)"
            )


if __name__ == "__main__":
    main()
