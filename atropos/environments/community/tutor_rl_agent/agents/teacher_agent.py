import json
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class TeacherAgent:
    """
    TeacherAgent for the LLM-Based Interactive Teacher-Student Tutor Environment.

    This agent is responsible for:
    1. Generating appropriate questions based on student profile
    2. Evaluating student responses
    3. Providing explanations for incorrect answers
    4. Adapting teaching strategy based on student performance
    """

    def __init__(self, profile_path: str, api_key: Optional[str] = None):
        """
        Initialize the TeacherAgent with a student profile.

        Args:
            profile_path: Path to the JSON file containing student profile
            api_key: Optional API key for LLM (defaults to environment variable)
        """
        # Load student profile
        self.profile = self._load_profile(profile_path)

        # Set up LLM client
        self.api_key = api_key or os.getenv("NOUS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set NOUS_API_KEY in .env file or pass as argument."
            )

        # Nous API endpoint
        self.api_endpoint = "https://api.nousresearch.com/v1/chat/completions"
        self.history = []

        # Track student performance metrics
        self.student_metrics = {
            "questions_asked": 0,
            "correct_answers": 0,
            "topic_performance": {},
            "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
        }

        # Initialize the topics from profile
        for topic in self.profile.get("topics", []):
            self.student_metrics["topic_performance"][topic["name"]] = {
                "questions": 0,
                "correct": 0,
                "accuracy": 0.0,
            }

    def _load_profile(self, profile_path: str) -> Dict[str, Any]:
        """Load student profile from JSON file."""
        try:
            with open(profile_path, "r") as file:
                profile = json.load(file)
            return profile
        except Exception as e:
            raise ValueError(f"Failed to load profile from {profile_path}: {e}")

    def _call_llm(self, prompt: str) -> str:
        """Make a call to the Nous Research API."""
        try:
            response = requests.post(
                self.api_endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "hermes-3-405b-instruct",  # Using Hermes-3-405B model
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert teacher assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return "I couldn't generate a response at this time."

        except requests.exceptions.RequestException as e:
            print(f"Network error calling Nous API: {e}")
            return "I couldn't generate a response due to a network error."
        except Exception as e:
            print(f"Error calling Nous API: {e}")
            return "I couldn't generate a response at this time."

    def generate_question(
        self, topic: Optional[str] = None, difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a multiple-choice question based on student profile and history.

        Args:
            topic: Optional topic override
            difficulty: Optional difficulty override ('easy', 'medium', 'hard')

        Returns:
            Dict containing question, options, correct answer, and explanation
        """
        # Use provided topic/difficulty or select based on student performance
        selected_topic = topic or self._select_topic()
        selected_difficulty = difficulty or self._select_difficulty(selected_topic)

        # Craft prompt for LLM to generate a question
        prompt = self._craft_question_prompt(selected_topic, selected_difficulty)

        # Get response from LLM
        response = self._call_llm(prompt)

        # Parse the response to extract question components
        try:
            question_data = self._parse_question_response(response)

            # Update metrics
            self.student_metrics["questions_asked"] += 1
            self.student_metrics["difficulty_distribution"][selected_difficulty] += 1
            self.student_metrics["topic_performance"][selected_topic]["questions"] += 1

            # Add metadata
            question_data["topic"] = selected_topic
            question_data["difficulty"] = selected_difficulty

            # Add to history
            self.history.append({"type": "question", "data": question_data})

            return question_data

        except Exception as e:
            print(f"Failed to parse question response: {e}")
            return {
                "error": "Failed to generate valid question",
                "raw_response": response,
            }

    def _craft_question_prompt(self, topic: str, difficulty: str) -> str:
        """Craft a prompt for the LLM to generate a multiple-choice question."""
        grade_level = self.profile.get("target_grade", "high school")

        prompt = f"""
        Please create a {difficulty} level multiple-choice question about {topic}
        appropriate for a {grade_level} student.

        The question should have:
        1. A clear, concise question statement
        2. Four possible answer options (A, B, C, D)
        3. The correct answer (just the letter)
        4. A detailed explanation of why the correct answer is right and why the others are wrong

        Format your response as a JSON object with the following structure:
        {{
            "question": "...",
            "options": {{
                "A": "...",
                "B": "...",
                "C": "...",
                "D": "..."
            }},
            "correct_answer": "A/B/C/D",
            "explanation": "..."
        }}
        """
        return prompt

    def _parse_question_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract question data."""
        # Try to find JSON content within the response
        try:
            # Extract just the JSON part if there's additional text
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return json.loads(response)  # Try parsing the whole response
        except json.JSONDecodeError:
            raise ValueError("Could not parse JSON from LLM response")

    def _select_topic(self) -> str:
        """
        Select a topic to focus on based on student performance.
        Prioritizes topics where student is struggling.
        """
        if not self.history:
            # For first question, select a random topic from profile
            return self.profile["topics"][0]["name"]

        # Find topics with lowest accuracy
        topic_accuracies = {}
        for topic, data in self.student_metrics["topic_performance"].items():
            if data["questions"] > 0:
                accuracy = data["correct"] / data["questions"]
                topic_accuracies[topic] = accuracy
            else:
                # Prioritize untested topics
                topic_accuracies[topic] = 0.0

        # Get topic with lowest accuracy (or random if tied)
        import random

        min_accuracy = min(topic_accuracies.values())
        weakest_topics = [t for t, a in topic_accuracies.items() if a == min_accuracy]
        return random.choice(weakest_topics)

    def _select_difficulty(self, topic: str) -> str:
        """
        Select appropriate difficulty based on student performance in the topic.
        """
        topic_data = self.student_metrics["topic_performance"].get(topic, {})

        # If no questions asked yet, start with medium difficulty
        if topic_data.get("questions", 0) == 0:
            return "medium"

        # Calculate accuracy for the topic
        accuracy = topic_data.get("correct", 0) / topic_data.get("questions", 1)

        # Adjust difficulty based on accuracy
        if accuracy < 0.4:
            return "easy"  # Student is struggling, make it easier
        elif accuracy > 0.8:
            return "hard"  # Student is doing well, make it harder
        else:
            return "medium"  # Keep at medium difficulty

    def evaluate_response(
        self, question_id: int, student_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate student's answer to a question.

        Args:
            question_id: Index of the question in history
            student_answer: Student's selected answer (A/B/C/D)

        Returns:
            Dictionary with evaluation results
        """
        # Retrieve the question from history
        if (
            question_id >= len(self.history)
            or self.history[question_id]["type"] != "question"
        ):
            return {"error": "Invalid question ID"}

        question_data = self.history[question_id]["data"]

        # Check if answer is correct
        is_correct = student_answer.upper() == question_data["correct_answer"].upper()

        # Update metrics
        if is_correct:
            self.student_metrics["correct_answers"] += 1
            topic = question_data["topic"]
            self.student_metrics["topic_performance"][topic]["correct"] += 1

            # Update accuracy
            questions = self.student_metrics["topic_performance"][topic]["questions"]
            correct = self.student_metrics["topic_performance"][topic]["correct"]
            self.student_metrics["topic_performance"][topic]["accuracy"] = (
                correct / questions
            )

        # Prepare evaluation response
        evaluation = {
            "is_correct": is_correct,
            "correct_answer": question_data["correct_answer"],
            "explanation": question_data["explanation"],
        }

        # If incorrect, generate a tailored explanation
        if not is_correct:
            selected_option = student_answer.upper()
            if selected_option in question_data["options"]:
                prompt = self._craft_explanation_prompt(
                    question_data["question"],
                    question_data["options"],
                    question_data["correct_answer"],
                    selected_option,
                )
                tailored_explanation = self._call_llm(prompt)
                evaluation["tailored_explanation"] = tailored_explanation

        # Add to history
        self.history.append(
            {
                "type": "evaluation",
                "data": {
                    "question_id": question_id,
                    "student_answer": student_answer,
                    "evaluation": evaluation,
                },
            }
        )

        return evaluation

    def _craft_explanation_prompt(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        selected_answer: str,
    ) -> str:
        """Craft a prompt for the LLM to generate a tailored explanation."""
        prompt = f"""
        A student answered the following multiple-choice question incorrectly:

        Question: {question}

        Options:
        A: {options.get('A', 'N/A')}
        B: {options.get('B', 'N/A')}
        C: {options.get('C', 'N/A')}
        D: {options.get('D', 'N/A')}

        The correct answer is {correct_answer}: {options.get(correct_answer, 'N/A')}

        The student selected {selected_answer}: {options.get(selected_answer, 'N/A')}

        Please provide a detailed, supportive explanation that:
        1. Explains why their answer is incorrect
        2. Identifies the misconception that might have led them to this answer
        3. Clearly explains why the correct answer is right
        4. Provides an additional example or analogy to reinforce the concept
        """
        return prompt

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of student performance across topics and difficulties.

        Returns:
            Dictionary with performance metrics
        """
        total_questions = self.student_metrics["questions_asked"]
        total_correct = self.student_metrics["correct_answers"]

        summary = {
            "total_questions": total_questions,
            "total_correct": total_correct,
            "overall_accuracy": (
                total_correct / total_questions if total_questions > 0 else 0
            ),
            "topic_performance": self.student_metrics["topic_performance"],
            "difficulty_distribution": self.student_metrics["difficulty_distribution"],
        }

        return summary

    def generate_adaptive_lesson_plan(self) -> Dict[str, Any]:
        """
        Generate an adaptive lesson plan based on student performance.

        Returns:
            Dictionary with recommended topics and strategies
        """
        # Get performance summary
        performance = self.get_performance_summary()

        # Identify weakest topics (below 60% accuracy)
        weak_topics = []
        for topic, data in performance["topic_performance"].items():
            if data["questions"] > 0 and data["accuracy"] < 0.6:
                weak_topics.append(
                    {
                        "topic": topic,
                        "accuracy": data["accuracy"],
                        "questions": data["questions"],
                    }
                )

        # Sort weak topics by accuracy (ascending)
        weak_topics.sort(key=lambda x: x["accuracy"])

        # Craft prompt for lesson plan
        prompt = f"""
        Based on a student's performance data, generate an adaptive lesson plan.

        Overall accuracy: {performance["overall_accuracy"]:.2f}

        Topic performance:
        {json.dumps(performance["topic_performance"], indent=2)}

        Please create a focused lesson plan that:
        1. Prioritizes the weakest topics (if any)
        2. Recommends specific learning activities for each weak topic
        3. Suggests a balanced approach to reinforce strong topics while improving weak ones
        4. Includes 2-3 specific example questions/exercises for the weakest topic

        Format your response as a JSON object with the following structure:
        {{
            "prioritized_topics": ["topic1", "topic2", ...],
            "recommended_activities": {{
                "topic1": ["activity1", "activity2", ...],
                ...
            }},
            "example_questions": [
                {{
                    "topic": "topic1",
                    "question": "...",
                    "answer": "..."
                }},
                ...
            ],
            "overall_strategy": "..."
        }}
        """

        # Get response from LLM
        response = self._call_llm(prompt)

        # Parse the response
        try:
            lesson_plan = self._parse_question_response(response)
            return lesson_plan
        except Exception as e:
            print(f"Failed to parse lesson plan response: {e}")
            return {
                "error": "Failed to generate valid lesson plan",
                "raw_response": response,
            }


# Example usage:
if __name__ == "__main__":
    # Path to student profile
    profile_path = "config/example_profile.json"

    # Initialize teacher agent
    teacher = TeacherAgent(profile_path)

    # Generate a question
    question = teacher.generate_question()
    print("Generated question:", question)

    # Simulate student response (typically this would come from StudentAgent)
    student_answer = "A"  # Placeholder

    # Evaluate response
    evaluation = teacher.evaluate_response(0, student_answer)
    print("Evaluation:", evaluation)

    # Get performance summary
    summary = teacher.get_performance_summary()
    print("Performance summary:", summary)

    # Generate adaptive lesson plan
    lesson_plan = teacher.generate_adaptive_lesson_plan()
    print("Adaptive lesson plan:", lesson_plan)
