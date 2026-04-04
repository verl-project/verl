import random
from typing import Any, Dict, List, Optional, Tuple

import mathgenerator


class MathCurriculum:
    """
    A curriculum manager for the mathgenerator library.

    This class organizes math problems by difficulty and provides methods
    to generate problems of appropriate difficulty based on the learner's
    performance.
    """

    # Define difficulty levels and map generator IDs to each level
    DIFFICULTY_LEVELS = {
        # Level 1: Basic arithmetic operations
        1: [
            0,  # Addition
            1,  # Subtraction
            2,  # Multiplication
            3,  # Division
            8,  # Square
            31,  # Factorial
            71,  # Absolute difference between two numbers
            80,  # Percentage of a number
            90,  # isprime
        ],
        # Level 2: Basic operations with fractions and pre-algebra
        2: [
            6,  # Square Root
            11,  # Basic Algebra
            13,  # Fraction to Decimal
            16,  # Fraction Division
            28,  # Fraction Multiplication
            44,  # Compare Fractions
            47,  # Cube Root
            53,  # Exponentiation
            97,  # Power of Powers
            118,  # Percentage difference
            119,  # Percentage error
            124,  # Is Composite
        ],
        # Level 3: Basic geometry and more algebra
        3: [
            18,  # Area of Triangle
            19,  # Triangle exists check
            22,  # Third Angle of Triangle
            24,  # Distance between 2 points
            25,  # Pythagorean Theorem
            49,  # Fourth Angle of Quadrilateral
            58,  # Sum of Angles of Polygon
            75,  # Area of a Sector
            96,  # Perimeter of Polygons
            104,  # Circumference
            108,  # Arc length of Angle
            112,  # Area of Circle
            115,  # Area of Circle given center and a point on circle
        ],
        # Level 4: More advanced algebra and basic statistics
        4: [
            9,  # LCM (Least Common Multiple)
            10,  # GCD (Greatest Common Denominator)
            20,  # Midpoint of the two point
            21,  # Factoring Quadratic
            23,  # Solve a System of Equations in R^2
            26,  # Linear Equations
            40,  # Common Factors
            41,  # Intersection of Two Lines
            45,  # Simple Interest
            50,  # Quadratic Equation
            76,  # Mean and Median
            78,  # Compound Interest
            105,  # Combine Like terms
        ],
        # Level 5: Vectors, matrices, and solid geometry
        5: [
            17,  # Integer Multiplication with 2x2 Matrix
            32,  # Surface Area of Cube
            33,  # Surface Area of Cuboid
            34,  # Surface Area of Cylinder
            35,  # Volume of Cube
            36,  # Volume of Cuboid
            37,  # Volume of cylinder
            38,  # Surface Area of cone
            39,  # Volume of cone
            43,  # Cross Product of 2 Vectors
            46,  # Multiplication of two matrices
            60,  # Surface Area of Sphere
            61,  # Volume of Sphere
            70,  # Angle between 2 vectors
            72,  # Dot Product of 2 Vectors
            77,  # Determinant to 2x2 Matrix
            95,  # Curved surface area of a cylinder
            113,  # Volume of frustum
            117,  # Volume of Hemisphere
            122,  # Volume of pyramid
            123,  # Surface area of pyramid
        ],
        # Level 6: Advanced topics (calculus, statistics, computer science)
        6: [
            4,  # Binary Complement 1s
            5,  # Modulo Division
            7,  # Power Rule Differentiation
            12,  # Logarithm
            14,  # Decimal to Binary
            15,  # Binary to Decimal
            27,  # Prime Factorisation
            30,  # Combinations of Objects
            42,  # Permutations
            48,  # Power Rule Integration
            52,  # Probability of a certain sum appearing on faces of dice
            54,  # Confidence interval For sample S
            55,  # Comparing surds
            56,  # Fibonacci Series
            59,  # Mean,Standard Deviation,Variance
            62,  # nth Fibonacci number
            64,  # Binary to Hexidecimal
            73,  # Binary 2's Complement
            79,  # Decimal to Hexadecimal
            84,  # Converts decimal to octal
            88,  # Trigonometric Differentiation
            89,  # Definite Integral of Quadratic Equation
            91,  # Binary Coded Decimal to Integer
            103,  # Decimal to Binary Coded Decimal
            107,  # Conditional Probability
            110,  # Stationary Points
        ],
        # Level 7: Most complex topics
        7: [
            65,  # Multiplication of 2 complex numbers
            66,  # Geometric Progression
            67,  # Geometric Mean of N Numbers
            68,  # Harmonic Mean of N Numbers
            69,  # Euclidian norm or L2 norm of a vector
            74,  # Inverse of a Matrix
            85,  # Converts decimal to Roman Numerals
            92,  # Complex To Polar Form
            93,  # Union,Intersection,Difference of Two Sets
            94,  # Base Conversion
            98,  # Quotient of Powers with Same Base
            99,  # Quotient of Powers with Same Power
            100,  # complex Quadratic Equation
            101,  # Leap Year or Not
            106,  # signum function
            109,  # Binomial distribution
            111,  # Expanding Factored Binomial
            121,  # Product of scientific notations
        ],
    }

    def __init__(
        self,
        starting_level: int = 1,
        progress_threshold: float = 0.8,
        min_evaluations: int = 5,
    ):
        """
        Initialize the curriculum manager.

        Args:
            starting_level: The difficulty level to start with (default: 1)
            progress_threshold: The success rate required to advance to the next level (default: 0.8)
            min_evaluations: Minimum number of evaluations needed before considering level advancement (default: 5)
        """
        self.current_level = starting_level
        self.progress_threshold = progress_threshold
        self.min_evaluations = min_evaluations

        # Performance tracking
        self.performance_history = {
            level: [] for level in self.DIFFICULTY_LEVELS.keys()
        }

        # Ensure starting level is valid
        if starting_level not in self.DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid starting level: {starting_level}. Available levels: {list(self.DIFFICULTY_LEVELS.keys())}"
            )

    def get_problem(self) -> Tuple[str, str, int]:
        """
        Generate a math problem at the current difficulty level.

        Returns:
            Tuple containing (problem_text, solution_text, generator_id)
        """
        # Get the available generator IDs for the current level
        available_generators = self.DIFFICULTY_LEVELS[self.current_level]

        # Try generators until one works
        max_attempts = 5  # Limit the number of attempts to avoid infinite loops
        attempts = 0

        while attempts < max_attempts:
            # Get a random generator ID from the current level
            generator_id = random.choice(available_generators)

            try:
                # Generate the problem
                problem, solution = mathgenerator.genById(generator_id)
                return problem, solution, generator_id
            except Exception as e:
                # Log the error and try another generator
                print(f"Error with generator {generator_id}: {str(e)}")
                attempts += 1

                # Remove the problematic generator from the available list for this session
                if generator_id in available_generators:
                    available_generators.remove(generator_id)

                # If we've exhausted all generators in this level, move to an adjacent level
                if not available_generators:
                    fallback_level = max(
                        1, min(7, self.current_level + random.choice([-1, 1]))
                    )
                    available_generators = self.DIFFICULTY_LEVELS[fallback_level].copy()

        # If all attempts fail, return a simple addition problem as fallback
        return "What is $2 + 2$?", "4", 0

    def record_performance(self, generator_id: int, is_correct: bool) -> None:
        """
        Record the performance on a specific problem.

        Args:
            generator_id: The ID of the generator used
            is_correct: Whether the answer was correct
        """
        # Find which level this generator belongs to
        level = None
        for lvl, generator_ids in self.DIFFICULTY_LEVELS.items():
            if generator_id in generator_ids:
                level = lvl
                break

        if level is not None:
            # Add the result to the performance history
            self.performance_history[level].append(is_correct)

    def get_success_rate(self, level: int) -> Optional[float]:
        """
        Calculate the success rate for a specific level.

        Args:
            level: The difficulty level

        Returns:
            Success rate as a float between 0 and 1, or None if not enough data
        """
        history = self.performance_history[level]

        if len(history) < self.min_evaluations:
            return None

        # Calculate success rate from recent evaluations
        recent_history = history[-self.min_evaluations :]
        return sum(recent_history) / len(recent_history)

    def should_advance(self) -> bool:
        """
        Determine if the learner should advance to the next level.

        Returns:
            Boolean indicating whether to advance
        """
        success_rate = self.get_success_rate(self.current_level)

        # If not enough data or below threshold, don't advance
        if success_rate is None or success_rate < self.progress_threshold:
            return False

        # Check if there's a next level to advance to
        return self.current_level < max(self.DIFFICULTY_LEVELS.keys())

    def advance_difficulty(self) -> bool:
        """
        Advance to the next difficulty level if appropriate.

        Returns:
            Boolean indicating whether advancement occurred
        """
        if self.should_advance():
            self.current_level += 1
            return True
        return False

    def get_current_level(self) -> int:
        """
        Get the current difficulty level.

        Returns:
            Current level as an integer
        """
        return self.current_level

    def get_num_levels(self) -> int:
        """
        Get the total number of difficulty levels.

        Returns:
            Total number of levels
        """
        return len(self.DIFFICULTY_LEVELS)

    def get_level_description(self, level: Optional[int] = None) -> str:
        """
        Get a description of the specified difficulty level.

        Args:
            level: The level to describe (default: current level)

        Returns:
            String description of the level
        """
        if level is None:
            level = self.current_level

        level_descriptions = {
            1: "Basic arithmetic operations (addition, subtraction, multiplication, division)",
            2: "Basic operations with fractions and pre-algebra",
            3: "Basic geometry and more algebra",
            4: "More advanced algebra and basic statistics",
            5: "Vectors, matrices, and solid geometry",
            6: "Advanced topics (calculus, statistics, computer science)",
            7: "Most complex topics (complex numbers, advanced operations)",
        }

        return level_descriptions.get(
            level, f"Custom level with IDs: {self.DIFFICULTY_LEVELS.get(level, [])}"
        )

    def reset(self, level: int = 1) -> None:
        """
        Reset the curriculum to a specific level and clear performance history.

        Args:
            level: The level to reset to (default: 1)
        """
        if level not in self.DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid level: {level}. Available levels: {list(self.DIFFICULTY_LEVELS.keys())}"
            )

        self.current_level = level
        self.performance_history = {lvl: [] for lvl in self.DIFFICULTY_LEVELS.keys()}

    def get_generator_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available generators.

        Returns:
            List of dictionaries containing generator information
        """
        generators = []
        gen_list = mathgenerator.getGenList()

        for gen in gen_list:
            # Find which level this generator belongs to
            level = None
            for lvl, generator_ids in self.DIFFICULTY_LEVELS.items():
                if gen[0] in generator_ids:
                    level = lvl
                    break

            generators.append(
                {
                    "id": gen[0],
                    "name": gen[1],
                    "function": gen[3],
                    "subject": gen[4],
                    "params": gen[5],
                    "difficulty_level": level,
                }
            )

        return generators
