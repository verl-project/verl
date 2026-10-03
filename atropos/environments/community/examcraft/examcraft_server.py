#!/usr/bin/env python3
"""
ExamCraft: Adaptive LLM Teacher Training Environment for Atropos

This environment trains language models to become better teachers by generating
adaptive questions, providing explanations, and creating personalized lesson plans.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Try different Atropos import patterns
try:
    from atroposlib.api import APIServerConfig, BaseEnvConfig
    from atroposlib.envs import BaseLanguageEnv
    from atroposlib.utils import parse_args_and_command

    ATROPOS_AVAILABLE = True
    print("✅ Successfully imported Atropos classes")
except ImportError as e1:
    try:
        from atroposlib.api.config import APIServerConfig, BaseEnvConfig
        from atroposlib.envs.base import BaseLanguageEnv
        from atroposlib.utils import parse_args_and_command

        ATROPOS_AVAILABLE = True
        print("✅ Successfully imported Atropos classes (alternative path)")
    except ImportError as e2:
        try:
            # Check what's actually available
            import atroposlib.api
            import atroposlib.envs

            print("Available in atroposlib.envs:", dir(atroposlib.envs))
            print("Available in atroposlib.api:", dir(atroposlib.api))
            raise ImportError("Could not find correct imports")
        except ImportError as e3:
            print(
                f"Warning: Could not import Atropos classes. Errors: {e1}, {e2}, {e3}"
            )
            print("Running in standalone mode...")
            ATROPOS_AVAILABLE = False

            # Create mock classes for testing
            class BaseLanguageEnv:
                def __init__(self, config):
                    self.config = config

            class BaseEnvConfig:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            class APIServerConfig:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)


@dataclass
class ExamCraftConfig(BaseEnvConfig):
    """Configuration for ExamCraft environment."""

    profile_path: str = "example_profile.json"
    max_questions_per_episode: int = 8
    student_learning_rate: float = 0.03
    enable_lesson_plans: bool = True
    difficulty_adaptation_rate: float = 0.1


class ExamCraftEnv(BaseLanguageEnv):
    """
    ExamCraft: Trains LLMs to be adaptive teachers.

    Features:
    - Adaptive question generation based on student proficiency
    - Multi-topic curriculum support
    - Real-time difficulty adjustment
    - Comprehensive reward system for teaching effectiveness
    - Lesson plan generation and evaluation
    """

    def __init__(self, config: ExamCraftConfig):
        super().__init__(config)
        self.config = config

        # Load student profile
        if os.path.exists(config.profile_path):
            with open(config.profile_path, "r") as file:
                self.profile = json.load(file)
        else:
            # Default profile if file not found
            self.profile = self._create_default_profile()

        # Initialize metrics
        self.reset_metrics()

        # Track teaching session
        self.session_history = []
        self.current_episode = 0

    def _create_default_profile(self) -> Dict[str, Any]:
        """Create a default student profile if none exists."""
        return {
            "student_id": "default_001",
            "target_grade": "11th grade",
            "learning_goal": "Master foundational mathematics",
            "current_avg_score": 65,
            "topics": [
                {"name": "algebra", "proficiency": 0.4},
                {"name": "geometry", "proficiency": 0.6},
                {"name": "statistics", "proficiency": 0.3},
                {"name": "calculus", "proficiency": 0.2},
            ],
            "preferred_learning_style": "visual",
        }

    def reset_metrics(self):
        """Reset student metrics for a new episode."""
        self.student_metrics = {
            "overall_accuracy": self.profile.get("current_avg_score", 65) / 100,
            "topic_accuracies": {},
            "difficulty_preferences": {},
        }

        # Initialize from profile
        for topic in self.profile.get("topics", []):
            topic_name = topic.get("name")
            proficiency = topic.get("proficiency", 0.5)
            self.student_metrics["topic_accuracies"][topic_name] = proficiency
            self.student_metrics["difficulty_preferences"][topic_name] = "medium"

        # Reset session tracking
        self.question_count = 0
        self.correct_count = 0
        self.session_history = []

    def get_system_message(self) -> str:
        """System message defining the teacher's role and capabilities."""
        topics_str = "\n".join(
            [
                f"- {topic['name']}: {topic['proficiency']:.1%} proficiency"
                for topic in self.profile.get("topics", [])
            ]
        )

        target_grade = self.profile.get("target_grade", "high school")
        return f"""You are ExamCraft, an adaptive AI teacher specializing in {target_grade} education.

STUDENT PROFILE:
{topics_str}
Learning Goal: {self.profile.get('learning_goal', 'Academic improvement')}
Preferred Style: {self.profile.get('preferred_learning_style', 'mixed')}
Current Average: {self.profile.get('current_avg_score', 65)}%

TEACHING CAPABILITIES:
1. QUESTION: Generate adaptive multiple-choice questions
2. EXPLANATION: Provide detailed explanations for concepts
3. LESSON_PLAN: Create personalized study plans

RESPONSE FORMAT - Return valid JSON only:
{{
    "action_type": "QUESTION|EXPLANATION|LESSON_PLAN",
    "topic": "topic_name",
    "difficulty": "easy|medium|hard",
    "content": {{
        "question": "Clear, engaging question text",
        "options": {{
            "A": "First option",
            "B": "Second option",
            "C": "Third option",
            "D": "Fourth option"
        }},
        "correct_answer": "A|B|C|D",
        "explanation": "Detailed explanation for the correct answer and why others are wrong",
        "learning_objective": "What this question teaches"
    }}
}}

TEACHING STRATEGY:
- Prioritize topics with low proficiency scores
- Adapt difficulty based on recent performance
- Provide detailed explanations that build understanding
- Focus on the student's learning goal and preferred style"""

    def get_user_message(self) -> str:
        """Generate context-aware prompt for the teacher."""
        if not self.session_history:
            learning_goal = self.profile.get("learning_goal", "general concepts")
            challenge_text = (
                "Please begin with an appropriate question for this student. "
                "Target their weakest areas while maintaining appropriate challenge level."
            )
            prompt = f"""NEW TEACHING SESSION STARTED

Student needs help with: {learning_goal}

Current topic proficiencies:
{json.dumps(self.student_metrics['topic_accuracies'], indent=2)}

{challenge_text}"""
            return prompt

        # Analyze recent performance
        recent_correct = sum(
            1 for item in self.session_history[-3:] if item.get("was_correct", False)
        )
        recent_total = len(self.session_history[-3:])

        last_interaction = self.session_history[-1]
        performance_trend = (
            "improving" if recent_correct > recent_total / 2 else "struggling"
        )

        # Identify struggling topics
        weak_topics = [
            topic
            for topic, acc in self.student_metrics["topic_accuracies"].items()
            if acc < 0.5
        ]

        topic_info = last_interaction.get("topic", "unknown")
        diff_info = last_interaction.get("difficulty", "unknown")
        correct_mark = "✓" if last_interaction.get("was_correct") else "✗"

        return f"""TEACHING SESSION UPDATE

Recent Performance: {recent_correct}/{recent_total} correct - student is {performance_trend}
Last Question: {topic_info} ({diff_info}) - {correct_mark}

Current Status:
- Questions asked: {self.question_count}
- Overall accuracy: {self.student_metrics['overall_accuracy']:.1%}
- Topics needing attention: {', '.join(weak_topics) if weak_topics else 'None'}

Updated proficiencies:
{json.dumps(self.student_metrics['topic_accuracies'], indent=2)}

Please select your next teaching action based on this student's progress."""

    async def step(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Process teacher's response and simulate student interaction."""
        try:
            # Clean and parse JSON response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            teacher_action = json.loads(response.strip())

            # Validate required fields
            if not all(key in teacher_action for key in ["action_type", "content"]):
                return await self._handle_error("Missing required fields in response")

            action_type = teacher_action["action_type"].upper()

            if action_type == "QUESTION":
                return await self._handle_question(teacher_action)
            elif action_type == "EXPLANATION":
                return await self._handle_explanation(teacher_action)
            elif action_type == "LESSON_PLAN":
                return await self._handle_lesson_plan(teacher_action)
            else:
                return await self._handle_error(f"Invalid action_type: {action_type}")

        except json.JSONDecodeError as e:
            return await self._handle_error(f"JSON parsing failed: {str(e)}")
        except Exception as e:
            return await self._handle_error(f"Unexpected error: {str(e)}")

    async def _handle_question(
        self, action: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Process a question from the teacher."""
        content = action["content"]
        topic = action.get("topic", "general")
        difficulty = action.get("difficulty", "medium")

        # Validate question format
        required_fields = ["question", "options", "correct_answer"]
        if not all(field in content for field in required_fields):
            return await self._handle_error("Question missing required fields")

        if len(content["options"]) != 4:
            return await self._handle_error("Question must have exactly 4 options")

        # Simulate student response
        student_answer, is_correct = self._simulate_student_response(
            topic, difficulty, content["correct_answer"]
        )

        # Update metrics
        self.question_count += 1
        if is_correct:
            self.correct_count += 1

        # Apply learning effect
        self._update_student_learning(topic, difficulty, is_correct)

        # Calculate reward
        score = self._calculate_question_reward(topic, difficulty, is_correct, content)

        # Record interaction
        interaction = {
            "type": "question",
            "topic": topic,
            "difficulty": difficulty,
            "question": content["question"],
            "student_answer": student_answer,
            "correct_answer": content["correct_answer"],
            "was_correct": is_correct,
            "score": score,
            "timestamp": self.question_count,
        }
        self.session_history.append(interaction)

        # Check episode completion
        episode_done = self.question_count >= self.config.max_questions_per_episode

        info = {
            "score": score,
            "episode_done": episode_done,
            "info": interaction,
            "student_metrics": dict(self.student_metrics),
        }

        return self.get_user_message(), info

    async def _handle_explanation(
        self, action: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Process an explanation from the teacher."""
        content = action["content"]
        explanation = content.get("explanation", "")
        topic = action.get("topic", "general")

        # Score explanation quality
        score = self._score_explanation(explanation, topic)

        # Apply learning boost from good explanations
        if len(self.session_history) > 0:
            last_topic = self.session_history[-1].get("topic")
            if last_topic and score > 0.7:
                current_prof = self.student_metrics["topic_accuracies"].get(
                    last_topic, 0.5
                )
                boost = min(
                    0.1, score * 0.05
                )  # Max 0.1 boost from excellent explanation
                self.student_metrics["topic_accuracies"][last_topic] = min(
                    1.0, current_prof + boost
                )

        # Record interaction
        interaction = {
            "type": "explanation",
            "topic": topic,
            "explanation_length": len(explanation.split()),
            "score": score,
        }
        self.session_history.append(interaction)

        info = {"score": score, "episode_done": False, "info": interaction}

        return self.get_user_message(), info

    async def _handle_lesson_plan(
        self, action: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Process a lesson plan from the teacher."""
        content = action["content"]

        # Score lesson plan quality
        score = self._score_lesson_plan(content)

        # Record final interaction
        interaction = {
            "type": "lesson_plan",
            "score": score,
            "final_performance": dict(self.student_metrics),
            "total_questions": self.question_count,
            "overall_accuracy": self.correct_count / max(1, self.question_count),
        }
        self.session_history.append(interaction)

        info = {
            "score": score,
            "episode_done": True,  # Always end after lesson plan
            "info": interaction,
            "session_summary": self._generate_session_summary(),
        }

        return "", info  # Empty string indicates episode completion

    async def _handle_error(self, error_msg: str) -> Tuple[str, Dict[str, Any]]:
        """Handle errors in teacher responses."""
        score = -1.0
        info = {
            "score": score,
            "episode_done": False,
            "info": {"error": error_msg, "type": "error"},
        }
        return self.get_user_message(), info

    def _simulate_student_response(
        self, topic: str, difficulty: str, correct_answer: str
    ) -> Tuple[str, bool]:
        """Simulate student's answer based on proficiency and difficulty."""
        import random

        # Get base proficiency
        base_proficiency = self.student_metrics["topic_accuracies"].get(topic, 0.5)

        # Adjust for difficulty
        difficulty_modifiers = {"easy": 0.25, "medium": 0.0, "hard": -0.25}
        modifier = difficulty_modifiers.get(difficulty, 0.0)

        # Calculate success probability
        success_prob = max(0.1, min(0.9, base_proficiency + modifier))

        # Add some session momentum (recent success increases confidence)
        if len(self.session_history) >= 2:
            recent_correct = sum(
                1
                for item in self.session_history[-2:]
                if item.get("was_correct", False)
            )
            momentum = (recent_correct - 1) * 0.1  # ±0.1 based on recent performance
            success_prob = max(0.1, min(0.9, success_prob + momentum))

        # Determine correct/incorrect
        is_correct = random.random() < success_prob

        if is_correct:
            return correct_answer, True
        else:
            # Choose random incorrect answer
            options = ["A", "B", "C", "D"]
            if correct_answer in options:
                options.remove(correct_answer)
            return random.choice(options), False

    def _update_student_learning(self, topic: str, difficulty: str, is_correct: bool):
        """Update student metrics based on teaching interaction."""
        # Update overall accuracy
        self.student_metrics["overall_accuracy"] = (
            self.correct_count / self.question_count
        )

        # Update topic-specific learning
        current_prof = self.student_metrics["topic_accuracies"].get(topic, 0.5)

        if is_correct:
            # Small improvement from correct answers
            improvement = self.config.student_learning_rate * 0.5
        else:
            # Larger improvement from learning from mistakes (assumes good explanations follow)
            improvement = self.config.student_learning_rate

        # Difficulty affects learning rate
        difficulty_multipliers = {"easy": 0.5, "medium": 1.0, "hard": 1.5}
        multiplier = difficulty_multipliers.get(difficulty, 1.0)

        new_prof = current_prof + (improvement * multiplier)
        self.student_metrics["topic_accuracies"][topic] = max(0.0, min(1.0, new_prof))

    def _calculate_question_reward(
        self, topic: str, difficulty: str, is_correct: bool, content: Dict[str, Any]
    ) -> float:
        """Calculate reward for a question based on multiple factors."""
        # Base reward for correctness
        base_reward = 1.5 if is_correct else -0.3

        # Topic targeting bonus (higher reward for focusing on weak areas)
        topic_prof = self.student_metrics["topic_accuracies"].get(topic, 0.5)
        targeting_bonus = (1.0 - topic_prof) * 0.8  # Up to 0.8 bonus for weakest topics

        # Difficulty appropriateness (reward for matching difficulty to ability)
        difficulty_values = {"easy": 0.3, "medium": 0.6, "hard": 0.9}
        target_difficulty = difficulty_values.get(difficulty, 0.6)
        difficulty_appropriateness = 1.0 - abs(target_difficulty - topic_prof) * 2
        difficulty_bonus = max(0.0, difficulty_appropriateness) * 0.5

        # Question quality bonus
        question_text = content.get("question", "")
        explanation = content.get("explanation", "")
        quality_bonus = min(
            0.3, len(question_text.split()) / 50 + len(explanation.split()) / 100
        )

        # Learning objective bonus
        if "learning_objective" in content and len(content["learning_objective"]) > 10:
            objective_bonus = 0.2
        else:
            objective_bonus = 0.0

        total_reward = (
            base_reward
            + targeting_bonus
            + difficulty_bonus
            + quality_bonus
            + objective_bonus
        )
        return round(total_reward, 3)

    def _score_explanation(self, explanation: str, topic: str) -> float:
        """Score the quality of an explanation."""
        if not explanation:
            return 0.0

        words = explanation.split()

        # Length scoring (optimal around 50-150 words)
        word_count = len(words)
        if word_count < 20:
            length_score = word_count / 20 * 0.5
        elif word_count <= 150:
            length_score = 0.5 + (word_count - 20) / 130 * 0.5
        else:
            length_score = max(0.3, 1.0 - (word_count - 150) / 200)

        # Content quality indicators
        quality_indicators = [
            "because",
            "therefore",
            "however",
            "for example",
            "this means",
            "in other words",
            "specifically",
            "notice that",
            "remember",
        ]
        quality_score = min(
            0.5,
            sum(
                0.1
                for indicator in quality_indicators
                if indicator in explanation.lower()
            ),
        )

        return min(1.0, length_score + quality_score)

    def _score_lesson_plan(self, content: Dict[str, Any]) -> float:
        """Score the quality of a lesson plan."""
        base_score = 1.0

        # Check for key lesson plan elements
        if "prioritized_topics" in content:
            priorities = content["prioritized_topics"]
            if isinstance(priorities, list) and len(priorities) > 0:
                # Bonus for prioritizing weak topics
                weak_topics = [
                    topic
                    for topic, acc in self.student_metrics["topic_accuracies"].items()
                    if acc < 0.5
                ]
                weak_coverage = sum(1 for topic in priorities if topic in weak_topics)
                base_score += weak_coverage * 0.3

        if "study_activities" in content:
            activities = content["study_activities"]
            if isinstance(activities, (list, dict)) and len(activities) > 0:
                base_score += 0.4

        if "timeline" in content:
            base_score += 0.3

        if "learning_objectives" in content:
            objectives = content["learning_objectives"]
            if isinstance(objectives, list) and len(objectives) > 0:
                base_score += 0.5

        return min(2.5, base_score)

    def _generate_session_summary(self) -> Dict[str, Any]:
        """Generate a summary of the teaching session."""
        topics_covered = {}
        for interaction in self.session_history:
            if interaction.get("type") == "question":
                topic = interaction.get("topic")
                if topic:
                    if topic not in topics_covered:
                        topics_covered[topic] = {"total": 0, "correct": 0}
                    topics_covered[topic]["total"] += 1
                    if interaction.get("was_correct"):
                        topics_covered[topic]["correct"] += 1

        # Calculate topic accuracies for this session
        topic_accuracies = {}
        for topic, stats in topics_covered.items():
            topic_accuracies[topic] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            )

        return {
            "total_questions": self.question_count,
            "overall_accuracy": self.correct_count / max(1, self.question_count),
            "topics_covered": list(topics_covered.keys()),
            "topic_accuracies": topic_accuracies,
            "improvement": {
                topic: self.student_metrics["topic_accuracies"][topic]
                - next(
                    (
                        t["proficiency"]
                        for t in self.profile["topics"]
                        if t["name"] == topic
                    ),
                    0.5,
                )
                for topic in topics_covered.keys()
            },
            "final_proficiencies": dict(self.student_metrics["topic_accuracies"]),
        }

    def run_standalone_demo(self, output_file: str = "demo_output.jsonl"):
        """Run a standalone demo without full Atropos integration."""
        print("🎓 ExamCraft: Standalone Demo Mode")
        print("Simulating teacher-student interactions...")

        # Reset environment
        self.reset_metrics()

        # Simulate a teaching session
        outputs = []

        for episode in range(3):  # Run 3 episodes
            print(f"\n--- Episode {episode + 1} ---")
            episode_data = {
                "episode": episode + 1,
                "initial_state": dict(self.student_metrics),
                "interactions": [],
            }

            # Simulate questions
            for question_num in range(self.config.max_questions_per_episode):
                # Create a mock teacher response (since we don't have an LLM)
                weakest_topic = min(
                    self.student_metrics["topic_accuracies"].items(), key=lambda x: x[1]
                )[0]

                mock_response = {
                    "action_type": "QUESTION",
                    "topic": weakest_topic,
                    "difficulty": "medium",
                    "content": {
                        "question": f"Sample question about {weakest_topic}",
                        "options": {
                            "A": "Option A",
                            "B": "Option B",
                            "C": "Option C",
                            "D": "Option D",
                        },
                        "correct_answer": "B",
                        "explanation": f"This is a detailed explanation about {weakest_topic}.",
                        "learning_objective": f"Understand key concepts in {weakest_topic}",
                    },
                }

                # Process the mock response
                try:
                    result = asyncio.run(self.step(json.dumps(mock_response)))
                    next_prompt, info = result
                    episode_data["interactions"].append(
                        {
                            "question_num": question_num + 1,
                            "teacher_action": mock_response,
                            "result": info,
                        }
                    )
                    question_info = info["info"].get("topic")
                    is_correct = "✓" if info["info"].get("was_correct") else "✗"
                    score = info["score"]
                    print(
                        f"  Q{question_num + 1}: {question_info} - {is_correct} "
                        f"(Score: {score:.2f})"
                    )
                except Exception as e:
                    print(f"  Error processing question {question_num + 1}: {e}")
                    break

                if info.get("episode_done", False):
                    break

            episode_data["final_state"] = dict(self.student_metrics)
            outputs.append(episode_data)

        # Save outputs
        with open(output_file, "w") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")

        print(f"\n✅ Demo completed! Results saved to {output_file}")
        print("Final student proficiencies:")
        for topic, prof in self.student_metrics["topic_accuracies"].items():
            print(f"  {topic}: {prof:.1%}")

    @classmethod
    def config_init(cls) -> Tuple[ExamCraftConfig, List[APIServerConfig]]:
        """Initialize configuration for ExamCraft environment."""
        env_config = ExamCraftConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=6,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=150,
            batch_size=12,
            steps_per_eval=30,
            max_token_length=3072,
            wandb_name="examcraft-adaptive-teacher",
            profile_path="example_profile.json",
            max_questions_per_episode=8,
            student_learning_rate=0.03,
            enable_lesson_plans=True,
        )

        # API server configuration
        server_configs = [
            APIServerConfig(
                model_name="hermes-3-405b-instruct",
                base_url="https://api.nousresearch.com/v1",
                api_key=os.environ.get("NOUS_API_KEY"),
                num_requests_for_eval=64,
            ),
        ]

        return env_config, server_configs


def main():
    """Main entry point for ExamCraft environment."""
    parser = argparse.ArgumentParser(
        description="ExamCraft: Adaptive LLM Teacher Training Environment"
    )
    parser.add_argument(
        "command",
        choices=["process", "serve", "demo"],
        help="Command to run (demo for standalone mode)",
    )
    parser.add_argument(
        "--env.data_path_to_save_groups",
        type=str,
        default="demo_output.jsonl",
        help="Path to save demo output",
    )

    args = parser.parse_args()

    if args.command == "demo":
        # Run standalone demo
        config = ExamCraftConfig()
        env = ExamCraftEnv(config)
        output_file = getattr(args, "env.data_path_to_save_groups", "demo_output.jsonl")
        env.run_standalone_demo(output_file)
    elif ATROPOS_AVAILABLE:
        # Run with Atropos
        parse_args_and_command(ExamCraftEnv)
    else:
        print(
            "Atropos not available. Use 'python examcraft_server.py demo' for standalone mode."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
