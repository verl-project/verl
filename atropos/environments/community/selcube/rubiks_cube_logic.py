#!/usr/bin/env python3
"""
Rubik's Cube logic extracted from the environment for independent testing
"""

# Define the face colors for visualization
UP_COLOR = "W"  # White
DOWN_COLOR = "Y"  # Yellow
RIGHT_COLOR = "R"  # Red
LEFT_COLOR = "O"  # Orange
FRONT_COLOR = "G"  # Green
BACK_COLOR = "B"  # Blue


class Cube:
    """
    A Rubik's cube implementation with accurate move handling.
    """

    def __init__(self):
        # Initialize a solved cube
        self.reset()

    def reset(self):
        """Reset the cube to solved state"""
        # Initialize the cube as a 3D array [face][row][col]
        # Faces: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=FRONT, 5=BACK
        self.cube = [
            [[UP_COLOR for _ in range(3)] for _ in range(3)],  # UP
            [[DOWN_COLOR for _ in range(3)] for _ in range(3)],  # DOWN
            [[LEFT_COLOR for _ in range(3)] for _ in range(3)],  # LEFT
            [[RIGHT_COLOR for _ in range(3)] for _ in range(3)],  # RIGHT
            [[FRONT_COLOR for _ in range(3)] for _ in range(3)],  # FRONT
            [[BACK_COLOR for _ in range(3)] for _ in range(3)],  # BACK
        ]

    def is_solved(self) -> bool:
        """Check if the cube is solved"""
        for face in self.cube:
            center_color = face[1][1]  # Center color never changes
            for row in face:
                for color in row:
                    if color != center_color:
                        return False
        return True

    def count_solved_cubies(self) -> float:
        """
        Count the number of stickers in their correct position
        Returns a normalized score between 0 and 1
        """
        # Create a solved reference cube
        reference = Cube()

        # Count matching stickers
        total_stickers = 6 * 9  # 6 faces, 9 stickers per face
        match_count = 0

        for face_idx in range(6):
            for i in range(3):
                for j in range(3):
                    if self.cube[face_idx][i][j] == reference.cube[face_idx][i][j]:
                        match_count += 1

        return match_count / total_stickers

    def rotate(self, move: str):
        """
        Perform a move on the cube using standard notation
        U, D, L, R, F, B are clockwise rotations of respective faces
        U', D', L', R', F', B' are counterclockwise rotations
        U2, D2, L2, R2, F2, B2 are double (180°) rotations
        """
        # Map move notation to face index and rotation count
        face_map = {"U": 0, "D": 1, "L": 2, "R": 3, "F": 4, "B": 5}

        # Parse the move
        if len(move) == 0:
            raise ValueError("Empty move")

        face = move[0]
        if face not in face_map:
            raise ValueError(f"Invalid face: {face}")

        face_idx = face_map[face]

        # Handle rotation direction
        if len(move) == 1:
            # Clockwise rotation
            count = 1
        elif len(move) == 2:
            if move[1] == "'":
                # Counterclockwise rotation
                count = 3
            elif move[1] == "2":
                # Double rotation
                count = 2
            else:
                raise ValueError(f"Invalid move modifier: {move[1]}")
        else:
            raise ValueError(f"Invalid move format: {move}")

        # Apply the rotation 'count' times
        for _ in range(count):
            self._rotate_face_clockwise(face_idx)
            self._rotate_adjacent_faces(face_idx)

    def _rotate_face_clockwise(self, face_idx: int):
        """Rotate a face clockwise"""
        face = self.cube[face_idx]
        new_face = [[None for _ in range(3)] for _ in range(3)]

        # Copy with 90-degree clockwise rotation
        for i in range(3):
            for j in range(3):
                new_face[j][2 - i] = face[i][j]

        self.cube[face_idx] = new_face

    def _rotate_adjacent_faces(self, face_idx: int):
        """Rotate the appropriate edges on adjacent faces"""
        if face_idx == 0:  # UP face
            # Rotate the top edges of FRONT, RIGHT, BACK, LEFT
            temp = self.cube[4][0][:]  # Save FRONT top edge
            self.cube[4][0] = self.cube[2][0][:]  # FRONT <- LEFT
            self.cube[2][0] = self.cube[5][0][:]  # LEFT <- BACK
            self.cube[5][0] = self.cube[3][0][:]  # BACK <- RIGHT
            self.cube[3][0] = temp  # RIGHT <- FRONT

        elif face_idx == 1:  # DOWN face
            # Rotate the bottom edges of FRONT, LEFT, BACK, RIGHT
            temp = self.cube[4][2][:]  # Save FRONT bottom edge
            self.cube[4][2] = self.cube[3][2][:]  # FRONT <- RIGHT
            self.cube[3][2] = self.cube[5][2][:]  # RIGHT <- BACK
            self.cube[5][2] = self.cube[2][2][:]  # BACK <- LEFT
            self.cube[2][2] = temp  # LEFT <- FRONT

        elif face_idx == 2:  # LEFT face
            # Rotate the left edges of UP, FRONT, DOWN, BACK
            # Need to extract and set columns, not rows
            temp = [self.cube[0][i][0] for i in range(3)]  # Save UP left column

            # UP left <- BACK right (reversed)
            for i in range(3):
                self.cube[0][i][0] = self.cube[5][2 - i][2]

            # BACK right <- DOWN left (reversed)
            for i in range(3):
                self.cube[5][i][2] = self.cube[1][2 - i][0]

            # DOWN left <- FRONT left
            for i in range(3):
                self.cube[1][i][0] = self.cube[4][i][0]

            # FRONT left <- UP left
            for i in range(3):
                self.cube[4][i][0] = temp[i]

        elif face_idx == 3:  # RIGHT face
            # Rotate the right edges of UP, BACK, DOWN, FRONT
            temp = [self.cube[0][i][2] for i in range(3)]  # Save UP right column

            # UP right <- FRONT right
            for i in range(3):
                self.cube[0][i][2] = self.cube[4][i][2]

            # FRONT right <- DOWN right
            for i in range(3):
                self.cube[4][i][2] = self.cube[1][i][2]

            # DOWN right <- BACK left (reversed)
            for i in range(3):
                self.cube[1][i][2] = self.cube[5][2 - i][0]

            # BACK left <- UP right (reversed)
            for i in range(3):
                self.cube[5][i][0] = temp[2 - i]

        elif face_idx == 4:  # FRONT face
            # Rotate the edges of UP bottom, RIGHT left, DOWN top, LEFT right
            # UP bottom row
            temp = self.cube[0][2][:]

            # UP bottom <- LEFT right (rotated)
            for i in range(3):
                self.cube[0][2][i] = self.cube[2][2 - i][2]

            # LEFT right <- DOWN top (rotated)
            for i in range(3):
                self.cube[2][i][2] = self.cube[1][0][i]

            # DOWN top <- RIGHT left (rotated)
            for i in range(3):
                self.cube[1][0][i] = self.cube[3][2 - i][0]

            # RIGHT left <- UP bottom (rotated)
            for i in range(3):
                self.cube[3][i][0] = temp[i]

        elif face_idx == 5:  # BACK face
            # Rotate the edges of UP top, LEFT left, DOWN bottom, RIGHT right
            # UP top row
            temp = self.cube[0][0][:]

            # UP top <- RIGHT right (rotated)
            for i in range(3):
                self.cube[0][0][i] = self.cube[3][2 - i][2]

            # RIGHT right <- DOWN bottom (rotated)
            for i in range(3):
                self.cube[3][i][2] = self.cube[1][2][i]

            # DOWN bottom <- LEFT left (rotated)
            for i in range(3):
                self.cube[1][2][i] = self.cube[2][2 - i][0]

            # LEFT left <- UP top (rotated)
            for i in range(3):
                self.cube[2][i][0] = temp[i]

    def __str__(self) -> str:
        """Convert cube to string representation"""
        face_names = ["U", "D", "L", "R", "F", "B"]
        result = []

        for i, face in enumerate(self.cube):
            result.append(f"{face_names[i]}: {' '.join(face[0])}")
            result.append(f"   {' '.join(face[1])}")
            result.append(f"   {' '.join(face[2])}")

        return "\n".join(result)


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
            status = "solved" if expected else "not solved"
            result_status = "solved" if is_solved else "not solved"
            print(f"Result: PASS (Expected {status}, Got {result_status})")
        else:
            status = "solved" if expected else "not solved"
            result_status = "solved" if is_solved else "not solved"
            print(f"Result: FAIL (Expected {status}, Got {result_status})")
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
