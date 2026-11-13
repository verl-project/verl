#!/usr/bin/env python3
"""
RubiksCubeStrategies: Library of solving strategies for the Rubik's cube environment

This module provides a collection of solving strategies for Rubik's cube, along with
explanations and examples for each. These strategies can be used to guide the LLM's
solving approach and provide structured learning.
"""

from typing import Dict, List, Optional


class SolvingStrategy:
    """Base class for Rubik's cube solving strategies"""

    def __init__(
        self,
        name: str,
        description: str,
        difficulty: int,
        steps: List[str],
        example_algorithms: List[Dict[str, str]] = None,
        tips: List[str] = None,
    ):
        """
        Initialize a solving strategy

        Args:
            name: Strategy name
            description: Detailed description of the strategy
            difficulty: Difficulty level (1-5)
            steps: Ordered list of steps to follow
            example_algorithms: Common algorithms used in this strategy
            tips: Tips for using this strategy effectively
        """
        self.name = name
        self.description = description
        self.difficulty = difficulty
        self.steps = steps
        self.example_algorithms = example_algorithms or []
        self.tips = tips or []

    def get_prompt_section(self) -> str:
        """Get formatted prompt section for this strategy"""
        prompt = f"""
STRATEGY: {self.name} (Difficulty: {self.difficulty}/5)

DESCRIPTION:
{self.description}

STEPS:
"""
        for i, step in enumerate(self.steps, 1):
            prompt += f"{i}. {step}\n"

        if self.example_algorithms:
            prompt += "\nCOMMON ALGORITHMS:\n"
            for algo in self.example_algorithms:
                prompt += f"- {algo['name']}: {algo['moves']} - {algo['purpose']}\n"

        if self.tips:
            prompt += "\nTIPS:\n"
            for tip in self.tips:
                prompt += f"- {tip}\n"

        return prompt

    def __str__(self) -> str:
        return f"{self.name} (Difficulty: {self.difficulty}/5)"


# Define common strategies
LAYER_BY_LAYER = SolvingStrategy(
    name="Layer-by-Layer Method",
    description=(
        "The beginner-friendly approach that solves the cube one layer at a time. "
        "It's intuitive and requires memorizing only a few algorithms."
    ),
    difficulty=1,
    steps=[
        "Solve the white cross on the top face",
        "Place the white corner pieces to complete the first layer",
        "Solve the middle layer edges",
        "Create a yellow cross on the bottom face",
        "Position the yellow edges correctly",
        "Position the yellow corners correctly",
        "Orient the yellow corners correctly",
    ],
    example_algorithms=[
        {
            "name": "Sexy Move",
            "moves": "R U R' U'",
            "purpose": "Used for placing corners in the first layer",
        },
        {
            "name": "Middle Layer Edge - Left",
            "moves": "U' L' U L U F U' F'",
            "purpose": "Insert edge piece into the middle layer from the left",
        },
        {
            "name": "Middle Layer Edge - Right",
            "moves": "U R U' R' U' F' U F",
            "purpose": "Insert edge piece into the middle layer from the right",
        },
        {
            "name": "Orient Yellow Edges",
            "moves": "F R U R' U' F'",
            "purpose": "Create a yellow cross on the last layer",
        },
    ],
    tips=[
        "Always keep the white face on top when solving the first layer",
        "Look ahead to plan edge placement before executing moves",
        "Pay attention to where pieces need to go before applying algorithms",
        "Break down the solution into manageable steps",
    ],
)

CFOP_METHOD = SolvingStrategy(
    name="CFOP Method (Fridrich Method)",
    description=(
        "An advanced method used by speedcubers. CFOP stands for Cross, F2L (First Two Layers), "
        "OLL (Orient Last Layer), and PLL (Permute Last Layer). It's efficient but requires "
        "memorizing many algorithms."
    ),
    difficulty=4,
    steps=[
        "Solve the cross on the bottom face (usually white)",
        "Solve the First Two Layers (F2L) by pairing corners with edges and inserting them",
        "Orient the Last Layer (OLL) to make the top face all one color",
        "Permute the Last Layer (PLL) to arrange all pieces correctly",
    ],
    example_algorithms=[
        {
            "name": "F2L Case 1",
            "moves": "R U R'",
            "purpose": "Basic F2L insertion when corner and edge are paired",
        },
        {
            "name": "F2L Case 2",
            "moves": "y' U' L' U L",
            "purpose": "Basic F2L insertion (mirror of case 1)",
        },
        {
            "name": "Sune",
            "moves": "R U R' U R U2 R'",
            "purpose": "Common OLL algorithm used to orient corners",
        },
        {
            "name": "T Permutation",
            "moves": "R U R' U' R' F R2 U' R' U' R U R' F'",
            "purpose": "PLL algorithm that swaps two corners and two edges",
        },
    ],
    tips=[
        "Practice F2L intuitively before learning algorithms",
        "Solve the cross on the bottom to see the F2L pairs more easily",
        "Learn to recognize F2L cases from different angles",
        "Group PLL algorithms by similar patterns to make memorization easier",
    ],
)

ROUX_METHOD = SolvingStrategy(
    name="Roux Method",
    description=(
        "A method focused on building blocks and using M-slice moves. It's very efficient "
        "and requires fewer algorithm memorizations than CFOP but demands good spatial intuition."
    ),
    difficulty=3,
    steps=[
        "Build a 1x2x3 block on the left side (First Block)",
        "Build a 1x2x3 block on the right side (Second Block)",
        "Orient the corners of the top layer and permute the corners of the top layer (CMLL)",
        "Orient the edges of the last layer and permute the M-slice (L6E)",
    ],
    example_algorithms=[
        {
            "name": "CMLL - O Case",
            "moves": "R U R' F' R U R' U' R' F R2 U' R'",
            "purpose": "Orient and permute corners when all corners are oriented incorrectly",
        },
        {
            "name": "EO - Arrow",
            "moves": "M U M'",
            "purpose": "Edge orientation during L6E phase",
        },
        {
            "name": "UL/UR Edge Swap",
            "moves": "M' U2 M U2",
            "purpose": "Swap the UL and UR edges during L6E phase",
        },
    ],
    tips=[
        "Focus on block-building efficiency for the first two blocks",
        "Use inspection time to plan the first block completely",
        "Practice M-slice moves to develop speed and accuracy",
        "Learn to recognize CMLL cases quickly to reduce pauses",
    ],
)

ZZ_METHOD = SolvingStrategy(
    name="ZZ Method",
    description=(
        "A method that focuses on solving edges early to enable rotationless solving. "
        "It orients all edges first, then solves the cube without F or B moves."
    ),
    difficulty=3,
    steps=[
        "Orient all edges (EOLine) while placing DF and DB edges",
        "Build the F2L on the left and right sides (ZZF2L)",
        "Orient the corners of the last layer (OCLL)",
        "Permute the last layer (PLL)",
    ],
    example_algorithms=[
        {
            "name": "EOLine Example",
            "moves": "F L' U B' D'",
            "purpose": "Orient all edges and place DF and DB edges",
        },
        {
            "name": "ZZF2L Pair",
            "moves": "U L U' L'",
            "purpose": "Insert corner-edge pair during F2L",
        },
        {
            "name": "OCLL - Sune",
            "moves": "R U R' U R U2 R'",
            "purpose": "Orient three corners in the last layer",
        },
    ],
    tips=[
        "Practice EOLine recognition to improve planning during inspection",
        "Take advantage of the rotationless solving after EOLine",
        "Use block-building techniques similar to Petrus for F2L",
        "Learn to recognize edge orientation quickly",
    ],
)

BEGINNER_METHOD = SolvingStrategy(
    name="Beginner Method",
    description=(
        "The simplest approach for complete beginners. Uses very intuitive steps and minimal algorithm "
        "memorization, focusing on understanding the cube's mechanics rather than speed."
    ),
    difficulty=1,
    steps=[
        "Solve the white cross",
        "Solve the white corners one by one",
        "Solve the middle layer edges one by one",
        "Make a yellow cross on the top",
        "Solve the yellow edges around the top",
        "Position the yellow corners",
        "Orient the yellow corners",
    ],
    example_algorithms=[
        {
            "name": "White Corner Insertion",
            "moves": "R U R' U'",
            "purpose": "Move a white corner piece into position",
        },
        {
            "name": "Edge Insertion",
            "moves": "U R U' R' U' F' U F",
            "purpose": "Insert a middle layer edge piece",
        },
        {
            "name": "Yellow Cross",
            "moves": "F R U R' U' F'",
            "purpose": "Form a yellow cross on the top face",
        },
    ],
    tips=[
        "Focus on understanding what each move does rather than memorizing algorithms",
        "Take your time and think about where pieces need to go",
        "Keep track of important pieces while executing algorithms",
        "Practice the fundamentals until they become natural",
    ],
)

CORNERS_FIRST = SolvingStrategy(
    name="Corners-First Method",
    description=(
        "Solve all corner pieces first, then solve the edges. This approach is less common "
        "but offers a different perspective on solving the cube."
    ),
    difficulty=2,
    steps=[
        "Orient the corners to get white and yellow on top and bottom",
        "Permute the corners to their correct positions",
        "Solve the middle layer edges",
        "Solve the last layer edges",
    ],
    example_algorithms=[
        {
            "name": "Corner Orientation",
            "moves": "R' D' R D",
            "purpose": "Orient a corner in place",
        },
        {
            "name": "Corner 3-Cycle",
            "moves": "R U' R' D2 R U R' D2",
            "purpose": "Cycle three corners",
        },
        {
            "name": "Edge 3-Cycle",
            "moves": "L' R U2 L R' F' L' R U2 L R' F",
            "purpose": "Cycle three edges",
        },
    ],
    tips=[
        "Use commutators for corner manipulation",
        "Pay attention to corner orientation as it affects the later steps",
        "Learn to visualize corner pieces and their correct positions",
        "Practice edge insertion techniques for the final steps",
    ],
)


def get_strategy_by_name(name: str) -> Optional[SolvingStrategy]:
    """Get a strategy by name"""
    all_strategies = [
        LAYER_BY_LAYER,
        CFOP_METHOD,
        ROUX_METHOD,
        ZZ_METHOD,
        BEGINNER_METHOD,
        CORNERS_FIRST,
    ]

    for strategy in all_strategies:
        if strategy.name.lower() == name.lower():
            return strategy

    return None


def get_strategy_by_difficulty(difficulty: int) -> List[SolvingStrategy]:
    """Get all strategies at a specific difficulty level"""
    all_strategies = [
        LAYER_BY_LAYER,
        CFOP_METHOD,
        ROUX_METHOD,
        ZZ_METHOD,
        BEGINNER_METHOD,
        CORNERS_FIRST,
    ]

    return [
        strategy for strategy in all_strategies if strategy.difficulty == difficulty
    ]


def get_all_strategies() -> List[SolvingStrategy]:
    """Get all available strategies"""
    return [
        LAYER_BY_LAYER,
        CFOP_METHOD,
        ROUX_METHOD,
        ZZ_METHOD,
        BEGINNER_METHOD,
        CORNERS_FIRST,
    ]


def get_strategy_prompt_for_level(level: int) -> str:
    """Get a formatted prompt with strategies appropriate for the curriculum level"""
    if level <= 2:
        # Beginner levels - show only simpler strategies
        strategies = [BEGINNER_METHOD, LAYER_BY_LAYER]
    elif level == 3:
        # Intermediate level
        strategies = [LAYER_BY_LAYER, CORNERS_FIRST, ROUX_METHOD]
    else:
        # Advanced levels - show all strategies
        strategies = get_all_strategies()

    prompt = "# RUBIK'S CUBE SOLVING STRATEGIES\n\nBelow are strategies you can use to solve the cube:\n\n"

    for strategy in strategies:
        prompt += strategy.get_prompt_section() + "\n\n"

    prompt += """
When solving the cube, you can use any of these strategies. Make sure to:
1. Choose a strategy that fits your understanding and the current cube state
2. Explain your thought process using the <think> tags
3. Follow the steps of your chosen strategy systematically
4. Apply the appropriate algorithms for your current situation
5. Track your progress toward the solution
"""

    return prompt
