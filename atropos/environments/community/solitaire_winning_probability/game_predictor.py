import asyncio
import csv
import inspect
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from asteval import Interpreter
from games import (
    card_matching_game_2,
    card_matching_game_3,
    card_matching_game_4,
    easy_game_1,
    easy_game_2,
    easy_game_3,
    easy_game_4,
    odd_card_game,
)
from openai import AsyncOpenAI

GUIDELINES = """
Please provide your analysis using the exact format below, including all tags:

<reasoning>
[Your initial approach to solving this probability problem]
[List important observations about the game mechanics]
[Show your step-by-step mathematical derivation using probability theory]
[Include explanations of any combinations, permutations, or conditional probabilities used]
</reasoning>

<formula>
[IMPORTANT: Write ONLY the final, simplified mathematical formula for the probability of winning below.]
[CRITICAL: Do NOT include any text, explanations, comments, multiple formulas,
or intermediate calculation steps within this tag.]
[CRITICAL: If a precise mathematical formula cannot be determined, leave this section EMPTY.]
[Use C(n,r), P(n,r), factorial(n) and standard math operators: + - * / ^ ( ) ]
</formula>

Note: Use these notations ONLY in your formula:
- Factorial: factorial(n)
- Combinations: C(n,r)
- Permutations: P(n,r)
- Standard operators: *, /, +, -, ^, (, )
The formula must be in a format that can be directly evaluated.
Use parentheses liberally to ensure correct order of operations. For example,
write (A * B) / (C * D) instead of A * B / C * D if you intend the division
to apply to the result of (C * D). Be explicit!

What is the mathematical formula to calculate the exact probability of winning this game?
"""


@dataclass
class GameAnalysis:
    """Class to hold the analysis results of a game."""

    ai_analysis: str
    formula: Optional[str]
    calculated_probability: Optional[float]
    simulated_probability: float
    n_simulations: int
    probability_difference: Optional[float]


class GamePredictor:
    def __init__(
        self,
        openai_api_key: str,
        openai_api_base: str,
        model: str = "llama-4-maverick-17b-128e-instruct-fp8",
    ):
        """Initialize the GamePredictor with OpenAI API credentials."""
        self.client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model = model
        # Create a persistent asteval interpreter
        self.aeval = Interpreter()
        # Add math functions to the interpreter's symbol table
        self.aeval.symtable["factorial"] = self.factorial
        self.aeval.symtable["C"] = self.combination
        self.aeval.symtable["P"] = self.permutation
        # Add standard math functions if needed (optional, asteval includes many)
        # self.aeval.symtable['sqrt'] = math.sqrt
        # self.aeval.symtable['pow'] = math.pow

    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial."""
        return math.factorial(n)

    @staticmethod
    def combination(n: int, r: int) -> int:
        """Calculate combination nCr."""
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

    @staticmethod
    def permutation(n: int, r: int) -> int:
        """Calculate permutation nPr."""
        return math.factorial(n) // math.factorial(n - r)

    def _extract_formula(self, response_text: str) -> Optional[str]:
        """Extract formula from the AI response."""
        # Find all formula blocks
        formula_matches = re.findall(
            r"<formula>\n(.*?)\n</formula>", response_text, re.DOTALL
        )

        if not formula_matches:
            return None

        # Use the content of the last formula block found
        last_formula_content = formula_matches[-1].strip()

        if not last_formula_content:
            return None

        # Split the content into lines and filter out empty lines
        lines = [
            line.strip() for line in last_formula_content.split("\\n") if line.strip()
        ]

        if not lines:
            return None

        # Return the last non-empty line as the potential formula
        # This assumes the AI puts the final, clean formula last in the block
        return lines[-1]

    def _evaluate_formula(self, formula: str) -> float:
        """Evaluate the mathematical formula using asteval."""
        # No need for regex replacements if the AI uses C(), P(), factorial()
        try:
            # Evaluate the formula using the pre-configured interpreter
            result = self.aeval(formula)
            if isinstance(result, (int, float)):
                return float(result)
            else:
                # Explicitly handle non-numeric results
                raise ValueError(
                    f"Formula '{formula}' evaluated to non-numeric type: {type(result).__name__} ({result})"
                )
        except KeyError as e:
            # Handle cases where symbols are not found (e.g., undefined variables in formula)
            raise ValueError(
                f"Error evaluating formula '{formula}': Undefined symbol {e}"
            )
        except Exception as e:
            # Catch other potential evaluation errors from asteval
            raise ValueError(f"Error evaluating formula '{formula}' using asteval: {e}")

    def _create_prompt(self, game_func: Callable) -> str:
        """Create the prompt for the AI model."""
        # Get the function's source code and docstring
        source_code = inspect.getsource(game_func)
        description = game_func.__doc__ or "No description available."

        return f"""
                Analyze this game implemented in the following Python code:

                ```python
                {source_code}
                ```

                {description}
                """

    def simulate_game(self, game_func: Callable, n_simulations: int = 100000) -> float:
        """
        Simulate a game multiple times and return the win probability.

        Args:
            game_func: The game function to simulate
            n_simulations: Number of simulations to run

        Returns:
            float: The probability of winning the game based on simulation
        """
        wins = sum(1 for _ in range(n_simulations) if game_func())
        return wins / n_simulations

    def compare_probabilities(
        self, calculated: float, simulated: float
    ) -> Tuple[float, str]:
        """
        Compare calculated and simulated probabilities.

        Args:
            calculated: The probability calculated from the formula
            simulated: The probability obtained from simulation

        Returns:
            Tuple[float, str]: The absolute difference and a qualitative assessment
        """
        diff = abs(calculated - simulated)

        if diff < 0.01:
            assessment = "Excellent match between theory and simulation"
        elif diff < 0.05:
            assessment = "Good match between theory and simulation"
        elif diff < 0.1:
            assessment = "Fair match between theory and simulation"
        else:
            assessment = "Poor match between theory and simulation"

        return diff, assessment

    async def predict_game(
        self, game_func: Callable, n_simulations: int = 100000
    ) -> GameAnalysis:
        """
        Predict the probability of winning a game using both AI analysis and simulation.

        Args:
            game_func: Function that implements the game
            n_simulations: Number of simulations to run for verification

        Returns:
            GameAnalysis object containing all analysis results
        """
        # Create and send message to AI
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self._create_prompt(game_func) + "\n\n" + GUIDELINES,
                },
            ],
        }

        try:
            chat_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[message],
                temperature=0,  # Set temperature to 0 for deterministic output
            )
            response_text = chat_response.choices[0].message.content
        except Exception as e:
            # Handle potential API errors gracefully
            response_text = f"<error>API call failed: {e}</error>"
            # Consider logging the error here
            print(f"Warning: API call failed for a game: {e}")

        # Extract and evaluate formula
        formula = self._extract_formula(response_text)
        calculated_prob = None
        if formula:
            try:
                calculated_prob = self._evaluate_formula(formula)
            except ValueError as e:
                # Formula evaluation failed, set probability to None and maybe log/store the error
                calculated_prob = None
                # Optionally add error information to the analysis object if needed
                print(f"Warning: Could not evaluate formula '{formula}': {e}")
                # Update response_text or add a field to GameAnalysis if needed
                response_text += f"<error>Formula evaluation failed: {e}</error>"

        # Run simulation in a separate thread to avoid blocking the event loop
        simulated_prob = await asyncio.to_thread(
            self.simulate_game, game_func, n_simulations
        )

        # Calculate difference if both probabilities are available
        prob_diff = (
            abs(calculated_prob - simulated_prob)
            if calculated_prob is not None
            else None
        )

        return GameAnalysis(
            ai_analysis=response_text,
            formula=formula,
            calculated_probability=calculated_prob,
            simulated_probability=simulated_prob,
            n_simulations=n_simulations,
            probability_difference=prob_diff,
        )

    async def predict_games(
        self, games: Dict[str, Callable], n_simulations: int = 100000
    ) -> Dict[str, GameAnalysis]:
        """
        Predict probabilities for multiple games concurrently.

        Args:
            games: Dictionary mapping game names to game functions
            n_simulations: Number of simulations per game

        Returns:
            Dictionary mapping game names to their GameAnalysis results
        """
        # Create tasks for each game prediction
        tasks = {
            game_name: asyncio.create_task(self.predict_game(game_func, n_simulations))
            for game_name, game_func in games.items()
        }

        # Wait for all tasks to complete
        await asyncio.gather(*tasks.values())

        # Collect results
        results = {name: task.result() for name, task in tasks.items()}
        return results

    async def generate_qa_csv(
        self, games: Dict[str, Callable], n_simulations: int, csv_filepath: str
    ):
        """
        Generates a CSV file with questions (prompts) and answers (simulated probabilities).

        Args:
            games: Dictionary mapping game names to game functions.
            n_simulations: Number of simulations per game.
            csv_filepath: Path to save the CSV file.
        """
        qa_data = []
        # Create a list of tasks for simulation to run them concurrently if desired,
        # or simply iterate and await if sequential processing per game is fine.
        # For simplicity here, we'll process game simulations sequentially for prompt generation,
        # but the simulation itself runs in a thread.
        for game_name, game_func in games.items():
            print(f"Processing game for CSV: {game_name}")
            prompt = self._create_prompt(game_func)

            # simulate_game is synchronous, run it in a thread to avoid blocking
            simulated_prob = await asyncio.to_thread(
                self.simulate_game, game_func, n_simulations
            )
            answer = f"{simulated_prob:.6f}"  # Format probability as string

            qa_data.append({"question": prompt, "answer": answer})

        try:
            with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["question", "answer"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row_data in qa_data:
                    writer.writerow(row_data)
            print(f"Successfully generated Q&A CSV at {csv_filepath}")
        except IOError as e:
            print(f"Error writing CSV file {csv_filepath}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during CSV generation: {e}")


# Example usage:
async def main():
    # API credentials - Set these as environment variables or pass as parameters
    openai_api_key = "your_openai_api_key_here"
    openai_api_base = "https://api.lambda.ai/v1"

    # Create predictor instance
    predictor = GamePredictor(openai_api_key, openai_api_base)

    # Define games to analyze
    games = {
        "easy_game_1": easy_game_1,
        "easy_game_2": easy_game_2,
        "easy_game_3": easy_game_3,
        "easy_game_4": easy_game_4,
        "card_matching_2": card_matching_game_2,
        "card_matching_3": card_matching_game_3,
        "card_matching_4": card_matching_game_4,
        "odd_card": odd_card_game,
    }
    await predictor.generate_qa_csv(
        games,
        100000,
        "environments/community/solitaire_winning_probability/qa_data.csv",
    )
    # Get predictions for all games
    results = await predictor.predict_games(games)

    # Print results
    for game_name, analysis in results.items():
        print(f"\nResults for {game_name}:")
        # print("AI Analysis:")
        # print(analysis.ai_analysis)
        # print(f"\nFormula: {analysis.formula}")
        # # Handle potential None for calculated probability
        # if analysis.calculated_probability is not None:
        #     print(f"Calculated probability: {analysis.calculated_probability:.4f}")
        # else:
        #     print("Calculated probability: N/A (Formula missing or invalid)")
        print(f"Simulated probability: {analysis.simulated_probability:.4f}")
        # Compare only if calculated probability is available
        if analysis.calculated_probability is not None:
            diff, assessment = predictor.compare_probabilities(
                analysis.calculated_probability, analysis.simulated_probability
            )
            print(f"Probability difference: {diff:.4f}")
            print(f"Assessment: {assessment}")
        else:
            print("Probability difference: N/A")
            print("Assessment: N/A (Cannot compare without calculated probability)")


if __name__ == "__main__":
    asyncio.run(main())
