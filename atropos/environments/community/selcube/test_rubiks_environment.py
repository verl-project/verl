#!/usr/bin/env python3
"""
Test script for the Rubik's Cube environment
"""

# Import Cube class directly from rubiks_cube_environment.py
from rubiks_cube_environment import Cube


def test_basic_moves():
    """Test basic moves and their inverses"""
    print("=== TESTING BASIC MOVES ===")

    # Test each basic move and its inverse
    for move, inverse in [
        ("R", "R'"),
        ("L", "L'"),
        ("U", "U'"),
        ("D", "D'"),
        ("F", "F'"),
        ("B", "B'"),
    ]:
        cube = Cube()
        cube.rotate(move)
        cube.rotate(inverse)
        solved = cube.is_solved()

        print(f"Move {move} followed by {inverse}: {'PASS' if solved else 'FAIL'}")

        if not solved:
            print("Final cube state:")
            print(str(cube))


def test_double_moves():
    """Test double (180°) moves"""
    print("\n=== TESTING DOUBLE MOVES ===")

    # Test each double move applied twice
    for move in ["U2", "D2", "L2", "R2", "F2", "B2"]:
        cube = Cube()
        cube.rotate(move)
        cube.rotate(move)
        solved = cube.is_solved()

        print(f"Double move {move} applied twice: {'PASS' if solved else 'FAIL'}")

        if not solved:
            print("Final cube state:")
            print(str(cube))


def test_complex_algorithms():
    """Test more complex algorithms"""
    print("\n=== TESTING COMPLEX ALGORITHMS ===")

    # Test algorithms
    algorithms = [
        {
            "name": "Sexy Move (R U R' U') × 6",
            "moves": ["R", "U", "R'", "U'"] * 6,
            "should_solve": True,
        },
        {
            "name": "Scramble + Inverse",
            "moves": ["R", "U", "F'", "L", "D2"] + ["D2", "L'", "F", "U'", "R'"],
            "should_solve": True,
        },
        {
            "name": "Sune Algorithm (R U R' U R U2 R')",
            "moves": ["R", "U", "R'", "U", "R", "U2", "R'"],
            "should_solve": False,
        },
    ]

    for algo in algorithms:
        cube = Cube()
        print(f"\nTesting: {algo['name']}")

        # Apply moves
        for move in algo["moves"]:
            cube.rotate(move)

        # Check result
        is_solved = cube.is_solved()
        expected = algo["should_solve"]

        if is_solved == expected:
            expected_str = "solved" if expected else "not solved"
            result_str = "solved" if is_solved else "not solved"
            print(f"Result: PASS (Expected {expected_str}, Got {result_str})")
        else:
            expected_str = "solved" if expected else "not solved"
            result_str = "solved" if is_solved else "not solved"
            print(f"Result: FAIL (Expected {expected_str}, Got {result_str})")
            print("Final cube state:")
            print(str(cube))

        # Show progress percentage if not solved
        if not is_solved:
            progress = cube.count_solved_cubies()
            print(f"Progress toward solution: {progress:.2f}")


def test_scramble_and_count():
    """Test scrambling and counting progress"""
    print("\n=== TESTING SCRAMBLING AND PROGRESS TRACKING ===")

    # Create a cube and apply random-like scramble
    cube = Cube()
    print("Solved cube:")
    print(str(cube))
    print(f"Is solved: {cube.is_solved()}")
    print(f"Progress: {cube.count_solved_cubies():.2f}")

    # Apply a sequence of moves to scramble
    scramble = ["R", "U", "F", "D", "L", "B'", "R'", "U2"]

    print(f"\nApplying scramble: {' '.join(scramble)}")
    for move in scramble:
        cube.rotate(move)

    print("Scrambled cube:")
    print(str(cube))
    print(f"Is solved: {cube.is_solved()}")
    print(f"Progress: {cube.count_solved_cubies():.2f}")


if __name__ == "__main__":
    test_basic_moves()
    test_double_moves()
    test_complex_algorithms()
    test_scramble_and_count()
