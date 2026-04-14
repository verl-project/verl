import os

import openai


# StudentAgent is responsible for answering teacher questions, evaluating teacher feedback,
# tracking weak areas, and generating performance summaries. It interacts with a TeacherAgent
# in an Atropos-compatible reinforcement learning loop.
class StudentAgent:
    def __init__(self, profile):
        self.profile = profile
        self.weak_areas = {}
        self.log = []
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_answer(self, question):
        """
        Generates a student-like answer to the teacher's question using the OpenAI LLM.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a student learning {self.profile['subject']} "
                    f"at {self.profile['difficulty']} level. "
                    f"Your goal is: {self.profile['goal']}."
                ),
            },
            {"role": "user", "content": f"Teacher asks: {question}"},
        ]

        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        answer = response.choices[0].message.content.strip()
        self.log.append({"question": question, "student_answer": answer})
        return answer

    def evaluate_teacher_effectiveness(self, question, explanation, student_answer):
        """
        Evaluates how effective the teacher's explanation was based on the student's response and learning goal.
        Returns a score between 0.0 and 1.0.
        """
        messages = [
            {
                "role": "user",
                "content": f"""
Teacher asked: {question}
Teacher explained: {explanation}
Student answered: {student_answer}
Student's learning goal: {self.profile['goal']}


Rate the teacher's effectiveness in helping the student learn the concept.
Respond with a number between 0.0 (not helpful) and 1.0 (very effective). Only output the number.
""",
            }
        ]

        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        return float(response.choices[0].message.content.strip())

    def update_weak_areas(self, question, student_answer):
        """
        Tracks weak topic areas by comparing the question and student's answer.
        """
        for topic in ["loops", "recursion", "lists", "variables", "functions"]:
            if topic in question.lower() and topic not in student_answer.lower():
                self.weak_areas[topic] = self.weak_areas.get(topic, 0) + 1
        return self.weak_areas

    def compare_answers(self, original, revised):
        """
        Compares original and revised answers. Returns 'improved' or 'no change'.
        """
        if len(revised) > len(original) and revised != original:
            return "improved"
        return "no change"

    def summarize_performance(self):
        """
        Summarizes teacher performance based on all interactions.
        Returns average score, total questions, and weak areas.
        """
        total = len(self.log)
        scores = [entry.get("score", 0.0) for entry in self.log if "score" in entry]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "total_questions": total,
            "avg_teacher_score": round(avg_score, 2),
            "weak_areas": self.weak_areas,
        }

    def revise_answer_based_on_explanation(self, question, explanation):
        """
        Generates a revised answer based on the teacher's explanation.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a student learning {self.profile['subject']} "
                    f"at {self.profile['difficulty']} level. "
                    f"Your goal is: {self.profile['goal']}."
                ),
            },
            {
                "role": "user",
                "content": f"""
Teacher asked: {question}
Teacher explained: {explanation}


Using the teacher's explanation, try to answer the original question again.
""",
            },
        ]
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        revised = response.choices[0].message.content.strip()
        self.log.append({"question": question, "revised_answer": revised})
        return revised

    def reset_log(self):
        """
        Resets the internal log and weak areas for a new episode.
        """
        self.log = []
        self.weak_areas = {}

    def log_score(self, score):
        """
        Logs the teacher score for the most recent interaction.
        """
        if self.log:
            self.log[-1]["score"] = score

    def get_last_answer(self):
        """
        Returns the most recent student answer, if available.
        """
        if self.log and "student_answer" in self.log[-1]:
            return self.log[-1]["student_answer"]
        return None

    def track_revision_success(self, question, revised_answer):
        """
        Checks if revised answer addressed previously missed weak topics.
        Removes resolved topics from weak_areas.
        """
        resolved_topics = []
        for topic in list(self.weak_areas.keys()):
            if topic in question.lower() and topic in revised_answer.lower():
                resolved_topics.append(topic)
                del self.weak_areas[topic]
        return resolved_topics
