#!/usr/bin/env python3
"""
Article Evaluator for Wikipedia Article Creator Environment

This module provides functionality to evaluate the factual accuracy of AI-generated
Wikipedia articles by comparing them against reference Wikipedia articles.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.warning(
        "python-dotenv not installed, environment variables must be set manually"
    )

# Import OpenAI client
try:
    from openai import OpenAI
except ImportError:
    logging.error("OpenAI package not installed. Install with 'pip install openai'")
    raise

# Get logger for this module only, without affecting the global configuration
logger = logging.getLogger(__name__)
# Set the level for just this module
logger.setLevel(logging.WARNING)  # Changed from INFO to WARNING to reduce verbosity


class ArticleEvaluator:
    """
    A class to evaluate the factual accuracy of AI-generated Wikipedia articles
    against reference articles using OpenAI models.
    """

    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the ArticleEvaluator with API credentials and model settings.

        Args:
            openai_api_key: API key for OpenAI (falls back to OPENAI_API_KEY env var)
            model: The OpenAI model to use for evaluation (default: gpt-4o)
        """
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        logger.info(f"ArticleEvaluator initialized with model: {model}")

    def get_reference_article(self, json_data: Dict, title: str) -> Optional[str]:
        """
        Retrieve reference article text from the JSON data.

        Args:
            json_data: The loaded JSON data with Wikipedia articles
            title: The title of the article to retrieve

        Returns:
            The plain text content of the reference article, or None if not found
        """
        # Search for the article by title
        for article in json_data:
            if article.get("title", "").lower() == title.lower():
                # Get the plain text content
                return article.get("plain_text", "")

        # If exact match not found, try partial match
        for article in json_data:
            if title.lower() in article.get("title", "").lower():
                logger.info(
                    f"Found partial title match: '{article.get('title')}' for query '{title}'"
                )
                return article.get("plain_text", "")

        logger.warning(f"No reference article found for title: {title}")
        return None

    def prepare_article_for_evaluation(
        self, article_content: str
    ) -> Tuple[str, List[str]]:
        """
        Prepare an AI-generated article for evaluation by numbering its lines.

        Args:
            article_content: The content of the AI-generated article

        Returns:
            A tuple containing:
            - Numbered article text suitable for the prompt
            - List of the original lines for further processing
        """
        # Clean up the article content
        article_content = article_content.strip()

        # Split the article into paragraphs
        paragraphs = [p for p in article_content.split("\n\n") if p.strip()]

        # Process each paragraph into a numbered format
        numbered_lines = []
        original_lines = []

        line_number = 1
        for paragraph in paragraphs:
            # Skip very short lines or separators
            if len(paragraph.strip()) < 3:
                continue

            # Add the line to our collections
            numbered_lines.append(f"{line_number}: {paragraph}")
            original_lines.append(paragraph)
            line_number += 1

        # Join the numbered lines with double newlines for readability
        numbered_text = "\n\n".join(numbered_lines)

        logger.info(f"Prepared article with {len(original_lines)} numbered lines")
        return numbered_text, original_lines

    def evaluate_article_accuracy(
        self, reference_content: str, generated_article: str, temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Evaluate the factual accuracy of an AI-generated article against a reference.

        Args:
            reference_content: The text of the reference Wikipedia article
            generated_article: The text of the AI-generated article
            temperature: The sampling temperature for the OpenAI API call

        Returns:
            Dictionary containing the evaluation results
        """
        # Prepare the AI-generated article with line numbers
        numbered_article, original_lines = self.prepare_article_for_evaluation(
            generated_article
        )

        # Format the prompt for the OpenAI model
        prompt = f"""
You are an expert fact-checker comparing an AI-generated article with a reference Wikipedia article.

# Classification Criteria
- CORRECT: The statement is accurate and verifiable in the reference article
- INCORRECT: The statement contradicts information in the reference article
- UNKNOWN: The reference doesn't mention this information or provides insufficient details to verify

# Output Format
You must produce valid YAML with this exact structure for each numbered line:
1:
  analysis: "Brief analysis of line 1"
  accuracy: "CORRECT|INCORRECT|UNKNOWN"
2:
  analysis: "Brief analysis of line 2"
  accuracy: "CORRECT|INCORRECT|UNKNOWN"
...

# REFERENCE ARTICLE:
{reference_content}

# AI-GENERATED ARTICLE (NUMBERED LINES):
{numbered_article}
"""

        # Call the OpenAI API
        try:
            logger.warning("Evaluating article factual accuracy...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precision fact-checker that produces only valid YAML.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )

            # Extract the response content
            yaml_content = response.choices[0].message.content

            # Clean up YAML content - remove backticks and yaml indicators
            yaml_pattern = r"```(?:yaml)?\s*([\s\S]*?)\s*```"
            yaml_match = re.search(yaml_pattern, yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)

            # Parse the YAML response
            try:
                logger.warning("Parsing evaluation results...")
                evaluation_data = yaml.safe_load(yaml_content)

                # Validate the evaluation data
                if not isinstance(evaluation_data, dict):
                    logger.error(
                        f"Evaluation did not return a dictionary: {evaluation_data}"
                    )
                    return {
                        "error": "Invalid evaluation format",
                        "raw_response": yaml_content,
                    }

                # Calculate statistics
                stats = self.calculate_accuracy_statistics(evaluation_data)

                # We're now skipping the detailed line-by-line output to reduce verbosity
                # Just keeping the summary statistics that will be shown later

                return {
                    "evaluation": evaluation_data,
                    "statistics": stats,
                    "lines_count": len(original_lines),
                    "evaluated_lines_count": len(evaluation_data),
                }

            except yaml.YAMLError as e:
                logger.error(f"Failed to parse YAML response: {e}")
                return {
                    "error": f"Failed to parse YAML response: {e}",
                    "raw_response": yaml_content,
                }

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {"error": f"API call failed: {e}"}

    def calculate_accuracy_score(self, evaluation_data: Dict) -> float:
        """
        Calculate a normalized accuracy score from evaluation data.

        Args:
            evaluation_data: The evaluation data from evaluate_article_accuracy

        Returns:
            A score between -1 and 1 for compatibility with existing scoring
        """
        if not evaluation_data or "evaluation" not in evaluation_data:
            return 0.0

        evaluation = evaluation_data["evaluation"]
        total_lines = len(evaluation)

        if total_lines == 0:
            return 0.0

        # Count the number of statements in each category
        correct_count = sum(
            1 for item in evaluation.values() if item.get("accuracy", "") == "CORRECT"
        )
        incorrect_count = sum(
            1 for item in evaluation.values() if item.get("accuracy", "") == "INCORRECT"
        )

        # Calculate percentages
        pct_correct = correct_count / total_lines if total_lines > 0 else 0
        pct_incorrect = incorrect_count / total_lines if total_lines > 0 else 0

        # Convert to score between -1 and 1
        # Formula: correct% * 2 - 1 with adjustment for incorrect%
        score = pct_correct * 2 - 1 - (pct_incorrect * 0.5)

        # Ensure score is within [-1, 1] range
        return max(-1, min(1, score))

    def calculate_accuracy_statistics(self, evaluation_data: Dict) -> Dict:
        """
        Calculate statistics from the evaluation data.

        Args:
            evaluation_data: The line-by-line evaluation dictionary

        Returns:
            Dictionary with accuracy statistics
        """
        if not evaluation_data:
            return {
                "correct_count": 0,
                "incorrect_count": 0,
                "unknown_count": 0,
                "total_count": 0,
                "pct_correct": 0,
                "pct_incorrect": 0,
                "pct_unknown": 0,
            }

        total_count = len(evaluation_data)

        # Count the number of statements in each category
        correct_count = sum(
            1
            for item in evaluation_data.values()
            if item.get("accuracy", "") == "CORRECT"
        )
        incorrect_count = sum(
            1
            for item in evaluation_data.values()
            if item.get("accuracy", "") == "INCORRECT"
        )
        unknown_count = sum(
            1
            for item in evaluation_data.values()
            if item.get("accuracy", "") == "UNKNOWN"
        )

        # Calculate percentages
        pct_correct = (correct_count / total_count) * 100 if total_count > 0 else 0
        pct_incorrect = (incorrect_count / total_count) * 100 if total_count > 0 else 0
        pct_unknown = (unknown_count / total_count) * 100 if total_count > 0 else 0

        return {
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "unknown_count": unknown_count,
            "total_count": total_count,
            "pct_correct": pct_correct,
            "pct_incorrect": pct_incorrect,
            "pct_unknown": pct_unknown,
        }

    def evaluation_to_dataframe(self, evaluation_data: Dict) -> pd.DataFrame:
        """
        Convert evaluation data to a pandas DataFrame for easier analysis.

        Args:
            evaluation_data: The evaluation data from evaluate_article_accuracy

        Returns:
            DataFrame with evaluation results
        """
        if not evaluation_data or "evaluation" not in evaluation_data:
            return pd.DataFrame()

        evaluation = evaluation_data["evaluation"]

        # Create a list of dictionaries for the DataFrame
        data = []
        for line_num, content in evaluation.items():
            data.append(
                {
                    "line_number": line_num,
                    "analysis": content.get("analysis", ""),
                    "accuracy": content.get("accuracy", "UNKNOWN"),
                }
            )

        # Create and return the DataFrame
        return pd.DataFrame(data)
