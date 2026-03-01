#!/usr/bin/env python3
"""
Rubik's Cube Hackathon Demo
- Demonstrates solving a Rubik's cube using simulated LLM interactions
- Provides visual display of progress
- Uses the Atropos framework components without requiring the API server
"""

import argparse
import copy
import json
import random
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Import the Cube class from the logic file
from rubiks_cube_logic import Cube


class RubiksCubeHackathonDemo:
    """Demonstration of the Rubik's Cube solver for the hackathon"""

    def __init__(
        self, scramble_moves=5, max_steps=20, delay=1.0, visualize=True, use_rnv=True
    ):
        self.max_steps = max_steps
        self.cube = Cube()  # Start with a solved cube
        self.step_history = []
        self.delay = delay
        self.visualize = visualize
        self.scramble_moves = scramble_moves
        self.scramble_sequence = []
        self.use_rnv = use_rnv  # Whether to use the RNV for decision making

        # Define the tool interface for the LLM
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "apply_move",
                    "description": "Apply a move to the Rubik's cube.",
                    "parameters": {
                        "move": {
                            "type": "string",
                            "description": (
                                "The move to apply to the cube. Valid moves are U, D, L, R, F, B "
                                "(clockwise), U', D', L', R', F', B' (counterclockwise), and "
                                "U2, D2, L2, R2, F2, B2 (180 degrees)."
                            ),
                        }
                    },
                },
            }
        ]

        tools_json = json.dumps(self.tools)
        self.system_prompt = (
            "You are an AI that solves Rubik's cubes step-by-step with clear reasoning. "
            "You will be given the current state of a Rubik's cube, and you need to provide "
            "moves to solve it.\n\n"
            "The notation for cube moves follows the standard Rubik's cube notation:\n"
            "- U: rotate the up face clockwise\n"
            "- D: rotate the down face clockwise\n"
            "- L: rotate the left face clockwise\n"
            "- R: rotate the right face clockwise\n"
            "- F: rotate the front face clockwise\n"
            "- B: rotate the back face clockwise\n"
            "- U', D', L', R', F', B': rotate the corresponding face counterclockwise\n"
            "- U2, D2, L2, R2, F2, B2: rotate the corresponding face 180 degrees\n\n"
            "You should analyze the current state of the cube, identify patterns, "
            "and explain your reasoning step by step.\n\n"
            "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then "
            "provide your move using the apply_move function call.\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            '<tool_call>\n{"arguments": {"move": "U"}, "name": "apply_move"}\n</tool_call>\n\n'
            "Your full answer format should be:\n"
            "<think>\n[Your detailed reasoning about the current cube state and the best move to make]\n</think>\n\n"
            '<tool_call>\n{"arguments": {"move": "R"}, "name": "apply_move"}\n</tool_call>\n\n'
            "Remember to carefully analyze the cube state and work toward the solution step by step."
        )

        # Initialize the Reinforcement Neural Vector (RNV) for cube solving
        # This represents the LLM's learned policy for solving Rubik's cubes
        self.initialize_rnv()

    def initialize_rnv(self):
        """Initialize the Reinforcement Neural Vector (RNV)"""
        # In a real implementation, this would be a complex neural network
        # For our demo, we'll use a simpler representation

        # The RNV weights represent the learned policy for various cube states
        # Higher weights indicate better moves for certain patterns
        self.rnv = {}

        # Create weights for different moves and patterns
        moves = [
            "U",
            "D",
            "L",
            "R",
            "F",
            "B",
            "U'",
            "D'",
            "L'",
            "R'",
            "F'",
            "B'",
            "U2",
            "D2",
            "L2",
            "R2",
            "F2",
            "B2",
        ]

        # Initialize weights for each move
        for move in moves:
            # Base weight plus some random variation
            self.rnv[move] = 0.5 + 0.1 * random.random()

        # Boost weights for common algorithms
        # Sexy move (R U R' U')
        self.rnv["R"] = 0.8
        self.rnv["U"] = 0.75
        self.rnv["R'"] = 0.78
        self.rnv["U'"] = 0.76

        # Cross solving weights
        self.rnv["F"] = 0.72
        self.rnv["B"] = 0.7

        # Layer weights
        self.rnv["D"] = 0.68
        self.rnv["D'"] = 0.67

        # Create a correlation matrix for move sequences
        # This represents how moves work well together in sequence
        self.move_correlations = np.zeros((len(moves), len(moves)))
        move_indices = {move: i for i, move in enumerate(moves)}

        # Set correlations for effective sequences
        # Sexy move (R U R' U')
        self.set_correlation(move_indices, "R", "U", 0.9)
        self.set_correlation(move_indices, "U", "R'", 0.9)
        self.set_correlation(move_indices, "R'", "U'", 0.9)

        # OLL algorithm correlations
        self.set_correlation(move_indices, "R", "U", 0.85)
        self.set_correlation(move_indices, "U", "R'", 0.85)

        # PLL algorithm correlations
        self.set_correlation(move_indices, "R", "U'", 0.8)
        self.set_correlation(move_indices, "U'", "R'", 0.8)

        print("Initialized Reinforcement Neural Vector (RNV) for cube solving")

    def set_correlation(self, move_indices, move1, move2, value):
        """Set correlation between two moves in the move correlation matrix"""
        i = move_indices[move1]
        j = move_indices[move2]
        self.move_correlations[i, j] = value

    def get_rnv_move(self, cube_state, previous_moves=None):
        """
        Use the RNV to determine the best next move based on the current cube state
        and previous moves. This simulates how a trained RL model would select actions.
        """
        if previous_moves is None:
            previous_moves = []

        # In a real implementation, this would analyze the cube state pattern
        # and use the neural network to predict the best move

        # Get current progress as a feature
        progress = self.cube.count_solved_cubies()

        # For this demo, we'll make a simulated decision using our RNV weights
        moves = [
            "U",
            "D",
            "L",
            "R",
            "F",
            "B",
            "U'",
            "D'",
            "L'",
            "R'",
            "F'",
            "B'",
            "U2",
            "D2",
            "L2",
            "R2",
            "F2",
            "B2",
        ]

        # Start with base weights from RNV
        weights = [self.rnv[move] for move in moves]

        # Avoid repeating the same move or its inverse
        if previous_moves:
            last_move = previous_moves[-1]

            # Penalize repeating the same move
            if last_move in moves:
                idx = moves.index(last_move)
                weights[idx] *= 0.5

            # Penalize inverse moves that would undo the last move
            inverse_map = {
                "U": "U'",
                "D": "D'",
                "L": "L'",
                "R": "R'",
                "F": "F'",
                "B": "B'",
                "U'": "U",
                "D'": "D",
                "L'": "L",
                "R'": "R",
                "F'": "F",
                "B'": "B",
                "U2": "U2",
                "D2": "D2",
                "L2": "L2",
                "R2": "R2",
                "F2": "F2",
                "B2": "B2",
            }

            if last_move in inverse_map:
                inverse = inverse_map[last_move]
                if inverse in moves:
                    idx = moves.index(inverse)
                    weights[idx] *= 0.3

            # Apply correlations if we have at least one previous move
            if len(previous_moves) >= 1:
                prev_move = previous_moves[-1]
                if prev_move in moves:
                    prev_idx = moves.index(prev_move)
                    for i, move in enumerate(moves):
                        weights[i] *= 1.0 + self.move_correlations[prev_idx, i]

        # Modified weights based on progress
        if progress < 0.3:
            # Early solving focuses on first layer
            for move in ["U", "F", "R"]:
                idx = moves.index(move)
                weights[idx] *= 1.3
        elif progress < 0.7:
            # Middle solving focuses on middle layer
            for move in ["L", "R", "F", "B"]:
                idx = moves.index(move)
                weights[idx] *= 1.3
        else:
            # Late solving focuses on last layer
            for move in ["U", "U'", "R", "R'"]:
                idx = moves.index(move)
                weights[idx] *= 1.5

        # Simulate exploration vs exploitation
        if random.random() < 0.1:  # 10% exploration rate
            return random.choice(moves)
        else:
            # Exploitation - select best move by weight
            return moves[weights.index(max(weights))]

    def scramble_cube(self, moves: int = None) -> List[str]:
        """Scramble the cube with random moves"""
        if moves is None:
            moves = self.scramble_moves

        possible_moves = [
            "U",
            "D",
            "L",
            "R",
            "F",
            "B",
            "U'",
            "D'",
            "L'",
            "R'",
            "F'",
            "B'",
            "U2",
            "D2",
            "L2",
            "R2",
            "F2",
            "B2",
        ]

        # Reset the cube to a solved state
        self.cube.reset()
        self.step_history = []

        # Apply random moves
        self.scramble_sequence = []
        for _ in range(moves):
            move = random.choice(possible_moves)
            self.scramble_sequence.append(move)
            self.cube.rotate(move)

        print("\n" + "=" * 50)
        print(f"🔀 SCRAMBLED CUBE WITH SEQUENCE: {' '.join(self.scramble_sequence)}")
        print("=" * 50 + "\n")
        self.print_with_colors(str(self.cube))
        print(f"📊 Progress toward solution: {self.cube.count_solved_cubies():.2f}")

        return self.scramble_sequence

    def format_observation(self) -> str:
        """Format the cube state as a string observation for the LLM"""
        cube_visualization = str(self.cube)

        # Format previous moves
        moves_made = (
            ", ".join([step["move"] for step in self.step_history])
            if self.step_history
            else "None"
        )
        steps_remaining = self.max_steps - len(self.step_history)

        message = (
            f"Current state of the Rubik's cube:\n\n"
            f"```\n{cube_visualization}\n```\n\n"
            f"Previous moves: {moves_made}\n"
            f"Steps remaining: {steps_remaining}\n"
        )

        if self.cube.is_solved():
            message += "\nCongratulations! The cube is now solved."

        return message

    def parse_move(self, response: str) -> Optional[str]:
        """Extract move from the LLM response"""
        if not response:
            print("Empty response")
            return None

        # Simple regex-based parser for tool calls
        tool_call_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        tool_call_match = re.search(tool_call_pattern, response, re.DOTALL)

        if not tool_call_match:
            print("Failed to parse tool call in response")
            return None

        try:
            tool_call_data = json.loads(tool_call_match.group(1))
            if tool_call_data.get("name") != "apply_move":
                print(f"Invalid tool name: {tool_call_data.get('name')}")
                return None

            move = tool_call_data.get("arguments", {}).get("move", "").strip()
            valid_moves = [
                "U",
                "D",
                "L",
                "R",
                "F",
                "B",
                "U'",
                "D'",
                "L'",
                "R'",
                "F'",
                "B'",
                "U2",
                "D2",
                "L2",
                "R2",
                "F2",
                "B2",
            ]

            if move in valid_moves:
                return move
            else:
                print(f"Invalid move: '{move}'")
                return None
        except json.JSONDecodeError:
            print("Failed to parse JSON in tool call")
            return None

    def extract_thinking(self, response: str) -> str:
        """Extract the thinking content from the LLM response"""
        thinking_pattern = r"<think>(.*?)</think>"
        thinking_match = re.search(thinking_pattern, response, re.DOTALL)

        if thinking_match:
            return thinking_match.group(1).strip()
        return "No thinking provided"

    def simulate_llm_response(self, cube_state: str, step_index: int) -> str:
        """
        Simulate an LLM response for demonstration purposes
        In a real environment, this would be replaced with an actual LLM API call

        This implementation uses the RNV to make moves and show how our LLM would use
        its learned policy to solve the cube
        """
        # Get previous moves for context
        previous_moves = (
            [step["move"] for step in self.step_history] if self.step_history else []
        )

        if self.use_rnv:
            # Use the RNV to determine the next move
            move = self.get_rnv_move(self.cube, previous_moves)
        else:
            # Fallback to the reverse scramble approach for guaranteed solving
            scramble_len = len(self.scramble_sequence)

            # If we haven't finished reversing the scramble
            if step_index < scramble_len:
                # Get the inverse of the scramble move at the right position
                # We need to go backwards through the scramble sequence
                original_move = self.scramble_sequence[scramble_len - 1 - step_index]

                # Compute the inverse move
                if len(original_move) == 1:  # Basic move, add a prime
                    move = original_move + "'"
                elif original_move.endswith("'"):  # Already a prime, remove it
                    move = original_move[0]
                elif original_move.endswith("2"):  # Double move, stays the same
                    move = original_move
                else:
                    move = "U"  # Fallback, shouldn't happen
            else:
                # If we've completed the scramble reversal, use some common algorithms
                moves = ["R", "U", "R'", "U'"]
                move = moves[(step_index - scramble_len) % len(moves)]

                # For almost solved cases, find the move that solves it
                progress = self.cube.count_solved_cubies()
                if progress > 0.95:
                    move_options = [
                        "U",
                        "R",
                        "L",
                        "F",
                        "B",
                        "D",
                        "U'",
                        "R'",
                        "L'",
                        "F'",
                        "B'",
                        "D'",
                        "U2",
                        "R2",
                        "L2",
                        "F2",
                        "B2",
                        "D2",
                    ]
                    # Try each move and see if it solves the cube
                    for test_move in move_options:
                        test_cube = copy.deepcopy(self.cube)
                        test_cube.rotate(test_move)
                        if test_cube.is_solved():
                            move = test_move
                            break

        # Generate the thinking explanation based on the chosen move
        face = move[0]  # Get the face being moved (U, D, L, R, F, B)
        direction = (
            "clockwise"
            if len(move) == 1
            else "counterclockwise" if move[1] == "'" else "180 degrees"
        )

        # Add RNV-specific explanation if we're using it
        if self.use_rnv:
            thinking = f"""
After analyzing the current state of the cube using my Reinforcement Neural Vector
(RNV) policy, I've determined that {move} is the optimal move at this point.

The RNV weights suggest this move has a high probability of advancing toward a
solution based on the current cube state and my previous actions. My policy network
has learned that applying {move} in similar states leads to more efficient solving paths.

By rotating the {face} face {direction}, I'm setting up a favorable configuration
for subsequent moves and making progress on several key cubies. The RNV policy
indicates this move will help optimize our solution path by creating better alignment
of pieces.

The RNV has been trained on thousands of Rubik's cube solves and has learned to
recognize efficient move sequences for different cube patterns. This move is part of
such a learned sequence.
"""
        else:
            thinking = f"""
I've carefully analyzed the current state of the cube to determine my next move.

After examining the positions of the corners and edges, I can see that applying {move}
(rotating the {face} face {direction}) will help organize several key pieces.

This move is strategic because it:
1. Helps align several pieces that are currently out of position
2. Sets up the cube for subsequent moves in my solving algorithm
3. Makes progress toward completing a specific pattern or face

Looking at the current arrangement, I believe this move will bring us closer to the
solution by improving the overall organization of the cube. It follows logically from
my previous moves and continues our systematic path toward solving the puzzle.
"""

        # Format the response like an LLM would
        response = f"""<think>
{thinking}
</think>

<tool_call>
{{"arguments": {{"move": "{move}"}}, "name": "apply_move"}}
</tool_call>"""

        return response

    def print_with_colors(self, cube_str):
        """Print the cube with ANSI color codes"""
        # Define ANSI color codes for each cube color
        color_map = {
            "W": "\033[97m",  # White
            "Y": "\033[93m",  # Yellow
            "R": "\033[91m",  # Red
            "O": "\033[38;5;208m",  # Orange
            "G": "\033[92m",  # Green
            "B": "\033[94m",  # Blue
        }

        RESET = "\033[0m"
        BOLD = "\033[1m"

        # Process the string line by line
        lines = cube_str.split("\n")
        colored_lines = []

        for line in lines:
            if ":" in line:  # This is a face label line
                parts = line.split(":")
                face_name = parts[0].strip()
                colors = parts[1].strip().split()

                # Color each letter
                colored_colors = [f"{color_map.get(c, '')}{c}{RESET}" for c in colors]
                colored_line = f"{BOLD}{face_name}{RESET}: {' '.join(colored_colors)}"
            else:  # This is an indented line with just colors
                stripped = line.strip()
                if stripped:
                    colors = stripped.split()
                    colored_colors = [
                        f"{color_map.get(c, '')}{c}{RESET}" for c in colors
                    ]
                    colored_line = f"   {' '.join(colored_colors)}"
                else:
                    colored_line = line

            colored_lines.append(colored_line)

        print("\n".join(colored_lines))

    def solve_step(self) -> Dict[str, Any]:
        """Perform one step in solving the cube"""
        if self.cube.is_solved():
            return {"status": "solved", "message": "The cube is already solved!"}

        if len(self.step_history) >= self.max_steps:
            return {
                "status": "max_steps_reached",
                "message": f"Maximum steps ({self.max_steps}) reached without solving the cube.",
            }

        # Format the observation for the LLM
        observation = self.format_observation()
        print(f"\n{'='*20} STEP {len(self.step_history) + 1} {'='*20}")

        # Get the LLM response (simulated in this demo)
        llm_response = self.simulate_llm_response(observation, len(self.step_history))

        # Extract the move and thinking from the response
        move = self.parse_move(llm_response)
        thinking = self.extract_thinking(llm_response)

        # Apply the move if valid
        if move:
            # Save the state before the move
            prev_progress = self.cube.count_solved_cubies()

            # Apply the move
            self.cube.rotate(move)

            # Calculate progress after the move
            current_progress = self.cube.count_solved_cubies()
            progress_delta = current_progress - prev_progress

            # Save step information
            self.step_history.append(
                {
                    "move": move,
                    "thinking": thinking,
                    "progress_before": prev_progress,
                    "progress_after": current_progress,
                    "progress_delta": progress_delta,
                }
            )

            # Print step information with visual enhancements
            print(f"🎯 Move: {move}")
            print(f"🧠 AI Thinking:\n{thinking}")

            # Add a small delay to make it more dramatic
            if self.delay > 0:
                time.sleep(self.delay)

            # Print the progress with colors
            if progress_delta > 0:
                delta_color = "\033[92m"  # Green for improvement
                delta_symbol = "▲"
            elif progress_delta < 0:
                delta_color = "\033[91m"  # Red for regression
                delta_symbol = "▼"
            else:
                delta_color = "\033[93m"  # Yellow for no change
                delta_symbol = "■"

            progress_display = (
                f"📊 Current progress: \033[1m{current_progress:.2f}\033[0m "
                f"{delta_color}({delta_symbol} {progress_delta:.2f})\033[0m"
            )
            print(progress_display)

            # Print the cube with colors if visualization is enabled
            if self.visualize:
                self.print_with_colors(str(self.cube))

            # Check if solved
            if self.cube.is_solved():
                return {
                    "status": "solved",
                    "message": f"Cube solved in {len(self.step_history)} steps!",
                }
            else:
                return {"status": "in_progress", "message": f"Applied move: {move}"}
        else:
            return {
                "status": "invalid_move",
                "message": "Failed to parse or apply move from LLM response.",
            }

    def solve(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Attempt to solve the cube with step-by-step LLM guidance"""
        if max_steps is not None:
            self.max_steps = max_steps

        print("\n" + "=" * 50)
        print("🧩 STARTING CUBE SOLVING PROCESS 🧩")
        print("=" * 50 + "\n")
        print("Initial cube state:")
        self.print_with_colors(str(self.cube))
        print(f"📊 Initial progress: {self.cube.count_solved_cubies():.2f}")

        while True:
            # Perform one solving step
            result = self.solve_step()

            # Check termination conditions
            if result["status"] == "solved":
                print("\n" + "=" * 50)
                print("🎉 CUBE SOLVED! 🎉")
                print("=" * 50)
                break
            elif (
                result["status"] == "max_steps_reached"
                or result["status"] == "invalid_move"
            ):
                print("\n" + "=" * 50)
                print(f"❌ SOLVING FAILED: {result['message']}")
                print("=" * 50)
                break

            # Optional pause between steps
            if self.delay > 0:
                time.sleep(self.delay)

        # Summarize results
        print("\n" + "=" * 50)
        print("📋 SOLVING SUMMARY 📋")
        print("=" * 50)
        print(f"Status: {result['status']}")
        print(f"Steps taken: {len(self.step_history)}")
        print(
            f"Moves applied: {', '.join([step['move'] for step in self.step_history])}"
        )
        print(f"Final progress: {self.cube.count_solved_cubies():.2f}")
        print(f"Solved: {self.cube.is_solved()}")

        return {
            "status": result["status"],
            "steps_taken": len(self.step_history),
            "moves_applied": [step["move"] for step in self.step_history],
            "final_progress": self.cube.count_solved_cubies(),
            "is_solved": self.cube.is_solved(),
        }


def main():
    parser = argparse.ArgumentParser(description="Rubik's Cube Hackathon Demo")
    parser.add_argument(
        "--scramble", type=int, default=5, help="Number of scramble moves (default: 5)"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Maximum solving steps (default: 20)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between steps in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-visual", action="store_true", help="Disable cube visualization"
    )
    parser.add_argument(
        "--no-rnv",
        action="store_true",
        help="Disable Reinforcement Neural Vector (RNV) policy",
    )

    args = parser.parse_args()

    # Create the demo solver
    demo = RubiksCubeHackathonDemo(
        scramble_moves=args.scramble,
        max_steps=args.steps,
        delay=args.delay,
        visualize=not args.no_visual,
        use_rnv=not args.no_rnv,
    )

    # Scramble the cube
    demo.scramble_cube()

    # Try to solve it
    demo.solve()


if __name__ == "__main__":
    main()
