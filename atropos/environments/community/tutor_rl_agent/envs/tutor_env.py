import json
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Import the TeacherAgent class
from teacher_agent import TeacherAgent


class TutorEnv(gym.Env):
    """
    TutorEnv for the LLM-Based Interactive Teacher-Student Tutor Environment.

    This environment follows the Atropos LanguageEnv pattern and is responsible for:
    1. Managing the interaction between TeacherAgent and StudentAgent
    2. Computing rewards based on student learning
    3. Tracking the state of the tutoring session
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, profile_path: str, render_mode: Optional[str] = None):
        """
        Initialize the TutorEnv with a student profile.

        Args:
            profile_path: Path to the JSON file containing student profile
            render_mode: Optional rendering mode
        """
        # Load student profile
        with open(profile_path, "r") as file:
            self.profile = json.load(file)

        # Initialize TeacherAgent
        self.teacher_agent = TeacherAgent(profile_path)

        # Initialize student metrics
        self.init_student_metrics()

        # Set up action and observation spaces
        # For simplicity, we use a discrete action space with 4 actions:
        # 0: Ask easy question
        # 1: Ask medium question
        # 2: Ask hard question
        # 3: Generate adaptive lesson plan
        self.action_space = spaces.Discrete(4)

        # Observation space will be a Dict of:
        # - student_performance: Box with performance metrics
        # - question_history: MultiBinary with question difficulty and correctness
        self.observation_space = spaces.Dict(
            {
                "student_performance": spaces.Box(
                    low=np.array(
                        [0, 0, 0, 0]
                    ),  # [overall_accuracy, vectors_acc, matrices_acc, linear_systems_acc]
                    high=np.array([1, 1, 1, 1]),
                    dtype=np.float32,
                ),
                "question_history": spaces.MultiBinary(
                    10
                ),  # Track last 10 questions (1=correct, 0=incorrect)
            }
        )

        # Track conversation state
        self.state = {
            "student_performance": np.zeros(4, dtype=np.float32),
            "question_history": np.zeros(10, dtype=np.int8),
        }

        # Track additional metrics for reward calculation
        self.current_question = None
        self.question_count = 0
        self.correct_count = 0
        self.episode_reward = 0.0
        self.last_action = None
        self.history = []

        # Set render mode
        self.render_mode = render_mode

    def init_student_metrics(self):
        """Initialize student metrics from profile."""
        self.student_metrics = {
            "overall_accuracy": self.profile.get("current_avg_score", 0) / 100,
            "topic_accuracies": {},
        }

        # Set initial proficiency values from profile
        for topic in self.profile.get("topics", []):
            topic_name = topic.get("name")
            proficiency = topic.get("proficiency", 0.5)
            self.student_metrics["topic_accuracies"][topic_name] = proficiency

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset metrics
        self.init_student_metrics()
        self.question_count = 0
        self.correct_count = 0
        self.episode_reward = 0.0
        self.last_action = None
        self.history = []

        # Reset state
        self.state = {
            "student_performance": np.array(
                [
                    self.student_metrics["overall_accuracy"],
                    self.student_metrics["topic_accuracies"].get("vectors", 0.5),
                    self.student_metrics["topic_accuracies"].get("matrices", 0.5),
                    self.student_metrics["topic_accuracies"].get("linear_systems", 0.5),
                ],
                dtype=np.float32,
            ),
            "question_history": np.zeros(10, dtype=np.int8),
        }

        return self.state, {}

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action: Integer action from the action space

        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        self.last_action = action

        # Process action
        if action == 3:  # Generate adaptive lesson plan
            lesson_plan = self.teacher_agent.generate_adaptive_lesson_plan()
            reward = self._compute_lesson_plan_reward(lesson_plan)
            info = {"lesson_plan": lesson_plan}
            done = True  # End episode after generating lesson plan
        else:
            # Map action to difficulty
            difficulty_map = {0: "easy", 1: "medium", 2: "hard"}
            difficulty = difficulty_map[action]

            # Generate question
            question = self.teacher_agent.generate_question(difficulty=difficulty)
            self.current_question = question

            # Simulate student answer (in a real implementation, this would come from StudentAgent)
            student_answer, is_correct = self._simulate_student_answer(question)

            # Evaluate response
            evaluation = self.teacher_agent.evaluate_response(
                len(self.teacher_agent.history) - 2,  # Index of the question in history
                student_answer,
            )

            # Update metrics
            self.question_count += 1
            if is_correct:
                self.correct_count += 1

            # Update student metrics based on the result
            self._update_student_metrics(question["topic"], is_correct)

            # Compute reward
            reward = self._compute_question_reward(question, is_correct)

            # Update state
            self._update_state(is_correct)

            # Add to history
            self.history.append(
                {
                    "action": action,
                    "question": question,
                    "student_answer": student_answer,
                    "is_correct": is_correct,
                    "reward": reward,
                }
            )

            info = {
                "question": question,
                "student_answer": student_answer,
                "evaluation": evaluation,
                "is_correct": is_correct,
            }

            # Episode is done after 10 questions
            done = self.question_count >= 10

        # Accumulate episode reward
        self.episode_reward += reward

        # Return results
        truncated = False
        return self.state, reward, done, truncated, info

    def _simulate_student_answer(self, question: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Simulate a student answering the question based on their profile.

        In a real implementation, this would be replaced with StudentAgent.
        """
        # Get topic and difficulty
        topic = question.get("topic")
        difficulty = question.get("difficulty")

        # Get student proficiency for this topic
        proficiency = self.student_metrics["topic_accuracies"].get(topic, 0.5)

        # Adjust probability of correct answer based on difficulty
        difficulty_factor = {
            "easy": 0.3,  # +30% chance of getting it right
            "medium": 0.0,  # no adjustment
            "hard": -0.2,  # -20% chance of getting it right
        }

        # Calculate probability of correct answer
        correct_prob = proficiency + difficulty_factor.get(difficulty, 0.0)
        correct_prob = max(0.1, min(0.9, correct_prob))  # Clamp between 0.1 and 0.9

        # Determine if answer is correct
        import random

        is_correct = random.random() < correct_prob

        if is_correct:
            # Return correct answer
            return question["correct_answer"], True
        else:
            # Return random incorrect answer
            options = list(question["options"].keys())
            options.remove(question["correct_answer"])
            return random.choice(options), False

    def _update_student_metrics(self, topic: str, is_correct: bool):
        """Update student metrics based on question result."""
        # Update overall accuracy
        self.student_metrics["overall_accuracy"] = (
            self.correct_count / self.question_count
        )

        # Update topic accuracy with learning effect
        current_accuracy = self.student_metrics["topic_accuracies"].get(topic, 0.5)

        if is_correct:
            # Small improvement when answering correctly
            new_accuracy = current_accuracy + 0.02
        else:
            # Larger improvement when learning from mistakes (assumes good explanations)
            new_accuracy = current_accuracy + 0.01

        # Clamp accuracy between 0 and 1
        new_accuracy = max(0.0, min(1.0, new_accuracy))
        self.student_metrics["topic_accuracies"][topic] = new_accuracy

    def _update_state(self, is_correct: bool):
        """Update the environment state."""
        # Update performance metrics
        self.state["student_performance"] = np.array(
            [
                self.student_metrics["overall_accuracy"],
                self.student_metrics["topic_accuracies"].get("vectors", 0.5),
                self.student_metrics["topic_accuracies"].get("matrices", 0.5),
                self.student_metrics["topic_accuracies"].get("linear_systems", 0.5),
            ],
            dtype=np.float32,
        )

        # Update question history (shift left and add new result)
        self.state["question_history"] = np.roll(self.state["question_history"], -1)
        self.state["question_history"][-1] = 1 if is_correct else 0

    def _compute_question_reward(
        self, question: Dict[str, Any], is_correct: bool
    ) -> float:
        """
        Compute reward for asking a question.

        Reward factors:
        1. Base reward for correct answer
        2. Bonus for appropriate difficulty (challenging but achievable)
        3. Bonus for targeting weak topics
        """
        # Base reward
        reward = 1.0 if is_correct else -0.5

        # Get topic and difficulty
        topic = question.get("topic")
        difficulty = question.get("difficulty")

        # Get student proficiency for this topic
        proficiency = self.student_metrics["topic_accuracies"].get(topic, 0.5)

        # Compute difficulty appropriateness (reward for matching difficulty to proficiency)
        difficulty_values = {"easy": 0.3, "medium": 0.6, "hard": 0.9}
        difficulty_value = difficulty_values.get(difficulty, 0.6)

        # Reward is highest when difficulty matches proficiency (within 0.2)
        difficulty_match = 1.0 - abs(difficulty_value - proficiency) * 2
        difficulty_bonus = max(0.0, difficulty_match) * 0.5

        # Bonus for targeting weak topics (inverse of proficiency)
        weakness_bonus = (1.0 - proficiency) * 0.5

        # Combine rewards
        total_reward = reward + difficulty_bonus + weakness_bonus

        return total_reward

    def _compute_lesson_plan_reward(self, lesson_plan: Dict[str, Any]) -> float:
        """
        Compute reward for generating an adaptive lesson plan.

        Reward factors:
        1. Base reward for generating a plan
        2. Quality of topic prioritization (focus on weak areas)
        3. Diversity of recommended activities
        """
        # Base reward
        reward = 1.0

        # Check if there was an error generating the plan
        if "error" in lesson_plan:
            return -1.0

        # Get prioritized topics
        prioritized_topics = lesson_plan.get("prioritized_topics", [])

        # Check if weak topics are prioritized
        topic_accuracies = [
            (topic, acc)
            for topic, acc in self.student_metrics["topic_accuracies"].items()
        ]
        topic_accuracies.sort(key=lambda x: x[1])  # Sort by accuracy (ascending)

        weakest_topics = [topic for topic, _ in topic_accuracies[:2]]

        # Count how many of the weakest topics are prioritized
        weak_topic_coverage = sum(
            1 for topic in prioritized_topics if topic in weakest_topics
        )
        weak_topic_bonus = weak_topic_coverage * 0.5

        # Check diversity of activities
        activity_count = sum(
            len(activities)
            for activities in lesson_plan.get("recommended_activities", {}).values()
        )
        activity_bonus = min(activity_count / 5, 1.0) * 0.5

        # Count example questions
        example_count = len(lesson_plan.get("example_questions", []))
        example_bonus = min(example_count / 3, 1.0) * 0.5

        # Total reward
        total_reward = reward + weak_topic_bonus + activity_bonus + example_bonus

        return total_reward

    def render(self):
        """Render the environment."""
        if self.render_mode != "human":
            return

        if self.last_action is None:
            print("\n=== TutorEnv: New session started ===")
            return

        # Render based on last action
        if self.last_action == 3:  # Lesson plan
            last_info = self.history[-1] if self.history else {}
            lesson_plan = last_info.get("lesson_plan", {})

            print("\n=== Generated Adaptive Lesson Plan ===")
            print(
                f"Prioritized topics: {', '.join(lesson_plan.get('prioritized_topics', []))}"
            )
            print("Recommended activities:")
            for topic, activities in lesson_plan.get(
                "recommended_activities", {}
            ).items():
                print(f"  - {topic}: {', '.join(activities)}")
            print(f"Overall strategy: {lesson_plan.get('overall_strategy', 'N/A')}")
        else:
            # Render question and answer
            last_item = self.history[-1] if self.history else {}
            question = last_item.get("question", {})
            student_answer = last_item.get("student_answer", "")
            is_correct = last_item.get("is_correct", False)
            reward = last_item.get("reward", 0.0)

            print("\n=== Question and Answer ===")
            print(f"Topic: {question.get('topic', 'N/A')}")
            print(f"Difficulty: {question.get('difficulty', 'N/A')}")
            print(f"Question: {question.get('question', 'N/A')}")
            print("Options:")
            for key, value in question.get("options", {}).items():
                print(f"  {key}: {value}")
            print(f"Student answer: {student_answer}")
            print(f"Correct answer: {question.get('correct_answer', 'N/A')}")
            print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
            print(f"Reward: {reward:.2f}")

        # Print current metrics
        print("\n=== Current Metrics ===")
        print(f"Questions asked: {self.question_count}")
        print(f"Correct answers: {self.correct_count}")
        print(f"Overall accuracy: {self.student_metrics['overall_accuracy']:.2f}")
        print("Topic accuracies:")
        for topic, accuracy in self.student_metrics["topic_accuracies"].items():
            print(f"  - {topic}: {accuracy:.2f}")
        print(f"Episode reward so far: {self.episode_reward:.2f}")

    def close(self):
        """Clean up resources."""
        pass
