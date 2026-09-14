#!/usr/bin/env python3
"""
Enhanced Rubik's Cube Visualizer

This module provides improved visualization tools for the Rubik's cube environment,
including progress tracking, move history visualization, and interactive elements.
"""

import base64
import datetime
import os
import random
import re
from io import BytesIO
from typing import Dict, List

import matplotlib.pyplot as plt


def generate_progress_chart(
    move_history: List[str] = None,
    progress_history: List[float] = None,
    rewards_history: List[float] = None,
    solved_at_move: int = None,
    title: str = "Cube Solving Progress",
) -> str:
    """
    Generate a chart showing progress over solving steps

    Args:
        move_history: List of moves applied
        progress_history: List of progress values (0.0-1.0) after each move
        rewards_history: Optional list of rewards for each move
        solved_at_move: Index of the move that solved the cube (if solved)
        title: Chart title

    Returns:
        Base64-encoded PNG image of the chart
    """
    plt.figure(figsize=(12, 6))

    # Ensure we have data to plot and initialize if None
    if move_history is None:
        move_history = []

    if progress_history is None or not progress_history:
        plt.text(
            0.5,
            0.5,
            "No progress data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )

        # Save the figure to a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(image_png).decode("utf-8")

    # Plot progress
    # Make sure move_indices and progress_history have the same length
    move_indices = list(range(len(progress_history)))
    plt.plot(move_indices, progress_history, "b-", linewidth=2, label="Progress")

    # Add markers for each move
    plt.plot(move_indices, progress_history, "bo", markersize=6)

    # Plot rewards if provided
    if rewards_history:
        # Ensure rewards_history has the same length as progress_history
        if len(rewards_history) != len(progress_history):
            # Truncate or extend rewards_history to match progress_history
            if len(rewards_history) > len(progress_history):
                rewards_history = rewards_history[: len(progress_history)]
            else:
                # Extend with the last value or 0
                last_reward = rewards_history[-1] if rewards_history else 0
                rewards_history.extend(
                    [last_reward] * (len(progress_history) - len(rewards_history))
                )

        # Normalize rewards to 0-1 range for comparison
        if rewards_history:
            min_reward = min(rewards_history)
            max_reward = max(rewards_history)
            reward_range = max_reward - min_reward

            if reward_range > 0:
                normalized_rewards = [
                    (r - min_reward) / reward_range for r in rewards_history
                ]
            else:
                normalized_rewards = [0.5] * len(rewards_history)

            plt.plot(
                move_indices, normalized_rewards, "r--", linewidth=1.5, label="Reward"
            )

    # Highlight the solving move if provided
    if solved_at_move is not None and 0 <= solved_at_move < len(progress_history):
        plt.axvline(
            x=solved_at_move,
            color="g",
            linestyle="--",
            label=f"Solved at move {solved_at_move+1}",
        )
        plt.plot(
            [solved_at_move],
            [progress_history[solved_at_move]],
            "g*",
            markersize=15,
            label="Solution",
        )

    # Add move labels (only if we have moves to label)
    if move_history:
        # Ensure move_history has the same length as progress_history
        if len(move_history) > len(progress_history):
            move_history = move_history[: len(progress_history)]
        elif len(move_history) < len(progress_history):
            # Extend with empty strings
            move_history.extend([""] * (len(progress_history) - len(move_history)))

        for i, move in enumerate(move_history):
            if move:  # Only annotate non-empty moves
                plt.annotate(
                    move,
                    (i, progress_history[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )

    # Add grid and labels
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Move Number")
    plt.ylabel("Progress / Normalized Reward")
    plt.title(title)
    plt.ylim(-0.05, 1.05)  # Ensure there's space for annotations
    plt.legend()

    # Save the figure to a base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    return base64.b64encode(image_png).decode("utf-8")


def parse_cube_state(cube_state: str) -> Dict[str, List[str]]:
    """
    Parse the cube state string into a dictionary of faces.

    Args:
        cube_state: String representation of the cube

    Returns:
        Dictionary with keys 'up', 'down', 'left', 'right', 'front', 'back'
        Each containing a 3x3 grid represented as a flattened list of colors
    """
    # Initialize faces dictionary
    faces = {"up": [], "down": [], "left": [], "right": [], "front": [], "back": []}

    # Extract face information
    current_face = None
    for line in cube_state.strip().split("\n"):
        line = line.strip()
        if line.startswith("U:"):
            current_face = "up"
            faces[current_face] = [c for c in line[2:].split() if c]
        elif line.startswith("D:"):
            current_face = "down"
            faces[current_face] = [c for c in line[2:].split() if c]
        elif line.startswith("L:"):
            current_face = "left"
            faces[current_face] = [c for c in line[2:].split() if c]
        elif line.startswith("R:"):
            current_face = "right"
            faces[current_face] = [c for c in line[2:].split() if c]
        elif line.startswith("F:"):
            current_face = "front"
            faces[current_face] = [c for c in line[2:].split() if c]
        elif line.startswith("B:"):
            current_face = "back"
            faces[current_face] = [c for c in line[2:].split() if c]
        elif current_face and line:
            # Continue accumulating colors for the current face
            colors = [c for c in line.split() if c]
            if colors:
                faces[current_face].extend(colors)

    # Ensure each face has exactly 9 elements (3x3 grid)
    for face in faces:
        # If we have too few colors, pad with a placeholder
        while len(faces[face]) < 9:
            faces[face].append("?")
        # If we have too many, truncate
        faces[face] = faces[face][:9]

    return faces


def generate_enhanced_cube_html(
    cube_state: str,
    move_history: List[str] = None,
    progress_history: List[float] = None,
    rewards_history: List[float] = None,
    thinking_history: List[str] = None,
    scramble_sequence: List[str] = None,
    is_solved: bool = False,
    curriculum_level: int = None,
    curriculum_description: str = None,
) -> str:
    """
    Generate enhanced HTML visualization of a Rubik's cube solve attempt

    Args:
        cube_state: String representation of the cube's current state
        move_history: List of moves applied during solving
        progress_history: List of progress values after each move
        rewards_history: List of rewards for each move
        thinking_history: List of thinking steps from the LLM
        scramble_sequence: List of moves used to scramble the cube
        is_solved: Whether the cube is solved
        curriculum_level: Current curriculum level (if using curriculum)
        curriculum_description: Description of the current curriculum level

    Returns:
        HTML string for visualization
    """
    # Extract the colors from the cube state string
    faces = parse_cube_state(cube_state)

    # Generate progress chart if data is available
    progress_chart_base64 = None
    if progress_history:
        solved_at_move = None
        if is_solved and move_history:
            solved_at_move = len(move_history) - 1

        # Make sure we have move_history even if it's empty
        if move_history is None:
            move_history = []

        progress_chart_base64 = generate_progress_chart(
            move_history=move_history,
            progress_history=progress_history,
            rewards_history=rewards_history,
            solved_at_move=solved_at_move,
            title="Rubik's Cube Solving Progress",
        )

    # Generate the HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced Rubik's Cube Visualizer</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                margin: 0;
            }
            .cube-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .cube-row {
                display: flex;
            }
            .face {
                margin: 5px;
            }
            .face-title {
                text-align: center;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .face-grid {
                display: grid;
                grid-template-columns: repeat(3, 40px);
                grid-template-rows: repeat(3, 40px);
                gap: 2px;
            }
            .cubie {
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                border: 1px solid #333;
                box-shadow: inset 0 0 5px rgba(0,0,0,0.2);
                border-radius: 3px;
            }
            .chart-container {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
            }
            .move-history {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .move-container {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                margin-top: 10px;
            }
            .move {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                background-color: #e0e0e0;
                border-radius: 5px;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s, background-color 0.2s;
            }
            .move:hover {
                background-color: #d0d0d0;
                transform: scale(1.1);
            }
            .move.U { background-color: #f5f5f5; color: #333; }
            .move.D { background-color: #ffeb3b; color: #333; }
            .move.L { background-color: #ff9800; color: white; }
            .move.R { background-color: #f44336; color: white; }
            .move.F { background-color: #4caf50; color: white; }
            .move.B { background-color: #2196f3; color: white; }

            .thinking-container {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-height: 300px;
                overflow-y: auto;
            }
            .thinking-step {
                margin-bottom: 15px;
                padding: 10px;
                border-left: 3px solid #2196f3;
                background-color: #f9f9f9;
            }
            .badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                margin-left: 10px;
            }
            .level-badge {
                background-color: #3498db;
                color: white;
            }
            .status-badge {
                background-color: #2ecc71;
                color: white;
            }
            .status-badge.unsolved {
                background-color: #e74c3c;
            }
            footer {
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                color: #7f8c8d;
                font-size: 0.9em;
            }
            .tabs {
                display: flex;
                border-bottom: 1px solid #ddd;
                margin-bottom: 15px;
            }
            .tab {
                padding: 8px 16px;
                cursor: pointer;
                background-color: #f1f1f1;
                border: 1px solid #ddd;
                border-bottom: none;
                margin-right: 5px;
                border-radius: 5px 5px 0 0;
            }
            .tab.active {
                background-color: white;
                border-bottom: 2px solid white;
                position: relative;
                top: 1px;
                font-weight: bold;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }

            @media (max-width: 768px) {
                .cube-row {
                    flex-direction: column;
                }
                .face-grid {
                    grid-template-columns: repeat(3, 30px);
                    grid-template-rows: repeat(3, 30px);
                }
                .cubie {
                    width: 30px;
                    height: 30px;
                    font-size: 0.8em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Enhanced Rubik's Cube Visualization</h1>
    """

    # Add status badges
    if is_solved:
        html += '<span class="status-badge">SOLVED</span>'
    else:
        html += '<span class="status-badge unsolved">UNSOLVED</span>'

    if curriculum_level is not None:
        html += f'<span class="badge level-badge">LEVEL {curriculum_level}</span>'

    if curriculum_description:
        html += f"<p>{curriculum_description}</p>"

    html += """
            </header>

            <div class="cube-container">
                <h2>Current State</h2>
    """

    # Create a 3D layout of the cube faces
    # Row 1: Empty, Up, Empty, Empty
    html += '<div class="cube-row">'
    html += '<div class="face"></div>'  # Empty space
    html += generate_face_html("Up", faces["up"])
    html += '<div class="face"></div>'  # Empty space
    html += '<div class="face"></div>'  # Empty space
    html += "</div>"

    # Row 2: Left, Front, Right, Back
    html += '<div class="cube-row">'
    html += generate_face_html("Left", faces["left"])
    html += generate_face_html("Front", faces["front"])
    html += generate_face_html("Right", faces["right"])
    html += generate_face_html("Back", faces["back"])
    html += "</div>"

    # Row 3: Empty, Down, Empty, Empty
    html += '<div class="cube-row">'
    html += '<div class="face"></div>'  # Empty space
    html += generate_face_html("Down", faces["down"])
    html += '<div class="face"></div>'  # Empty space
    html += '<div class="face"></div>'  # Empty space
    html += "</div>"

    html += """
            </div>

            <div class="tabs">
                <div class="tab active" onclick="openTab(event, 'tab-progress')">Progress</div>
                <div class="tab" onclick="openTab(event, 'tab-moves')">Move History</div>
                <div class="tab" onclick="openTab(event, 'tab-thinking')">Thinking Steps</div>
            </div>
    """

    # Progress Chart Tab
    html += '<div id="tab-progress" class="tab-content active">'
    html += '<div class="chart-container">'
    html += "<h2>Solving Progress</h2>"

    if progress_chart_base64:
        img_src = f"data:image/png;base64,{progress_chart_base64}"
        html += f'<img src="{img_src}" alt="Solving Progress Chart" style="max-width:100%;">'
    else:
        html += "<p>No progress data available.</p>"

    html += "</div></div>"

    # Moves History Tab
    html += '<div id="tab-moves" class="tab-content">'
    html += '<div class="move-history">'
    html += "<h2>Move History</h2>"

    # Add playback controls if moves exist
    if move_history:
        html += """
        <div class="controls">
            <button id="play-button" class="control-button" onclick="playSolution()">▶ Play Solution</button>
            <button id="pause-button" class="control-button" onclick="pauseSolution()" disabled>⏸ Pause</button>
            <button id="step-button" class="control-button" onclick="stepSolution()">⏭ Step</button>
            <button id="reset-button" class="control-button" onclick="resetSolution()">⟲ Reset</button>
            <span id="move-counter" style="margin-left: 10px; line-height: 32px;">0 / 0</span>
        </div>
        """

    if scramble_sequence:
        html += "<h3>Scramble Sequence</h3>"
        html += '<div class="move-container">'
        for move in scramble_sequence:
            move_class = move[0] if len(move) >= 1 else ""
            html += f'<div class="move {move_class}">{move}</div>'
        html += "</div>"

    if move_history:
        html += "<h3>Solving Moves</h3>"
        html += '<div class="move-container">'
        for i, move in enumerate(move_history):
            move_class = move[0] if len(move) >= 1 else ""
            html += f'<div class="move {move_class}" title="Move #{i+1}">{move}</div>'
        html += "</div>"
    else:
        html += "<p>No moves have been made yet.</p>"

    html += "</div></div>"

    # Thinking Steps Tab
    html += '<div id="tab-thinking" class="tab-content">'
    html += '<div class="thinking-container">'
    html += "<h2>Thinking Process</h2>"

    if thinking_history:
        for i, thinking in enumerate(thinking_history):
            html += f'<div class="thinking-step"><strong>Step {i+1}:</strong> {thinking}</div>'
    else:
        html += "<p>No thinking steps recorded.</p>"

    html += "</div></div>"

    # Add JavaScript for interactivity
    html += """
            <footer>
                Generated with Atropos Enhanced Rubik's Cube Visualizer
            </footer>
        </div>

        <script>
        // Tab functionality
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;

            // Hide all tab content
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].className = tabcontent[i].className.replace(" active", "");
            }

            // Remove active class from all tabs
            tablinks = document.getElementsByClassName("tab");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }

            // Show the current tab and add an active class
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";
        }

        // Interactive cube visualization
        document.addEventListener('DOMContentLoaded', function() {
            // Add hover effects to moves
            const moveElements = document.querySelectorAll('.move');
            moveElements.forEach(function(move) {
                move.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.2)';
                    this.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';

                    // Display move description
                    const moveText = this.textContent;
                    const description = getMoveDescription(moveText);

                    // Create or update tooltip
                    let tooltip = document.getElementById('move-tooltip');
                    if (!tooltip) {
                        tooltip = document.createElement('div');
                        tooltip.id = 'move-tooltip';
                        tooltip.style.position = 'fixed';
                        tooltip.style.backgroundColor = 'rgba(0,0,0,0.8)';
                        tooltip.style.color = 'white';
                        tooltip.style.padding = '5px 10px';
                        tooltip.style.borderRadius = '5px';
                        tooltip.style.zIndex = '1000';
                        tooltip.style.pointerEvents = 'none';
                        document.body.appendChild(tooltip);
                    }

                    tooltip.textContent = description;
                    tooltip.style.display = 'block';

                    // Position tooltip near the move
                    const rect = this.getBoundingClientRect();
                    tooltip.style.left = rect.right + 10 + 'px';
                    tooltip.style.top = rect.top + 'px';
                });

                move.addEventListener('mouseleave', function() {
                    this.style.transform = '';
                    this.style.boxShadow = '';

                    // Hide tooltip
                    const tooltip = document.getElementById('move-tooltip');
                    if (tooltip) {
                        tooltip.style.display = 'none';
                    }
                });

                // Make moves clickable to show animation
                move.addEventListener('click', function() {
                    animateMove(this.textContent);
                });
            });

            // Add animation capability to cube
            function animateMove(moveText) {
                // Flash the move to indicate it's being applied
                const moveElements = document.querySelectorAll('.move');
                moveElements.forEach(function(move) {
                    if (move.textContent === moveText) {
                        // Add animation class
                        move.classList.add('move-animate');

                        // Remove animation class after animation completes
                        setTimeout(function() {
                            move.classList.remove('move-animate');
                        }, 500);
                    }
                });

                // Animate cubies (simplified version - just highlights affected face)
                const faceToAnimate = moveText[0]; // Get the face letter (U, D, L, R, F, B)
                const cubies = document.querySelectorAll('.cubie');

                // Map face letter to face name
                const faceMap = {
                    'U': 'Up',
                    'D': 'Down',
                    'L': 'Left',
                    'R': 'Right',
                    'F': 'Front',
                    'B': 'Back'
                };

                // Find the face container
                const faceTitles = document.querySelectorAll('.face-title');
                let targetFace = null;

                faceTitles.forEach(function(title) {
                    if (title.textContent === faceMap[faceToAnimate]) {
                        targetFace = title.parentElement;
                    }
                });

                if (targetFace) {
                    // Add animation to the face
                    targetFace.classList.add('face-animate');

                    // Remove animation class after it completes
                    setTimeout(function() {
                        targetFace.classList.remove('face-animate');
                    }, 500);
                }
            }

            // Get description for a move
            function getMoveDescription(moveText) {
                const moveDescriptions = {
                    'U': 'Up face clockwise',
                    'D': 'Down face clockwise',
                    'L': 'Left face clockwise',
                    'R': 'Right face clockwise',
                    'F': 'Front face clockwise',
                    'B': 'Back face clockwise',
                    "U'": 'Up face counter-clockwise',
                    "D'": 'Down face counter-clockwise',
                    "L'": 'Left face counter-clockwise',
                    "R'": 'Right face counter-clockwise',
                    "F'": 'Front face counter-clockwise',
                    "B'": 'Back face counter-clockwise',
                    'U2': 'Up face 180 degrees',
                    'D2': 'Down face 180 degrees',
                    'L2': 'Left face 180 degrees',
                    'R2': 'Right face 180 degrees',
                    'F2': 'Front face 180 degrees',
                    'B2': 'Back face 180 degrees'
                };

                return moveDescriptions[moveText] || moveText;
            }

            // Solution playback functionality
            let moveIndex = 0;
            let playInterval = null;
            let solvingMoves = [];

            // Initialize the moves array and counter
            function initSolutionPlayer() {
                // Collect all solving moves
                const moveElements = document.querySelectorAll('.move-container:last-child .move');
                solvingMoves = Array.from(moveElements);

                // Update move counter
                const moveCounter = document.getElementById('move-counter');
                if (moveCounter) {
                    moveCounter.textContent = `0 / ${solvingMoves.length}`;
                }
            }

            // Play the solution automatically
            function playSolution() {
                // Initialize if needed
                if (solvingMoves.length === 0) {
                    initSolutionPlayer();
                }

                if (solvingMoves.length === 0) return;

                // Update button states
                document.getElementById('play-button').disabled = true;
                document.getElementById('pause-button').disabled = false;

                // Start playback interval
                playInterval = setInterval(function() {
                    if (moveIndex < solvingMoves.length) {
                        // Animate the current move
                        const currentMove = solvingMoves[moveIndex];
                        animateMove(currentMove.textContent);

                        // Highlight the current move
                        solvingMoves.forEach(move => move.classList.remove('current-move'));
                        currentMove.classList.add('current-move');

                        // Scroll to the move if needed
                        currentMove.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

                        // Update move counter
                        const moveCounter = document.getElementById('move-counter');
                        if (moveCounter) {
                            moveCounter.textContent = `${moveIndex + 1} / ${solvingMoves.length}`;
                        }

                        moveIndex++;

                        // If we reached the end, stop playback
                        if (moveIndex >= solvingMoves.length) {
                            pauseSolution();
                            document.getElementById('play-button').disabled = true;
                        }
                    } else {
                        pauseSolution();
                    }
                }, 1000); // Play one move per second
            }

            // Pause the solution playback
            function pauseSolution() {
                clearInterval(playInterval);
                playInterval = null;

                // Update button states
                document.getElementById('play-button').disabled = false;
                document.getElementById('pause-button').disabled = true;
            }

            // Step through the solution one move at a time
            function stepSolution() {
                // Initialize if needed
                if (solvingMoves.length === 0) {
                    initSolutionPlayer();
                }

                if (solvingMoves.length === 0 || moveIndex >= solvingMoves.length) return;

                // Pause any ongoing playback
                pauseSolution();

                // Animate the current move
                const currentMove = solvingMoves[moveIndex];
                animateMove(currentMove.textContent);

                // Highlight the current move
                solvingMoves.forEach(move => move.classList.remove('current-move'));
                currentMove.classList.add('current-move');

                // Scroll to the move if needed
                currentMove.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

                // Update move counter
                const moveCounter = document.getElementById('move-counter');
                if (moveCounter) {
                    moveCounter.textContent = `${moveIndex + 1} / ${solvingMoves.length}`;
                }

                moveIndex++;

                // If we reached the end, disable play button
                if (moveIndex >= solvingMoves.length) {
                    document.getElementById('play-button').disabled = true;
                }
            }

            // Reset the solution playback
            function resetSolution() {
                pauseSolution();
                moveIndex = 0;

                // Remove highlighting from all moves
                solvingMoves.forEach(move => move.classList.remove('current-move'));

                // Update move counter
                const moveCounter = document.getElementById('move-counter');
                if (moveCounter) {
                    moveCounter.textContent = `0 / ${solvingMoves.length}`;
                }

                // Enable play button
                document.getElementById('play-button').disabled = false;
            }

            // Initialize the solution player when the page loads
            initSolutionPlayer();
        });
        </script>

        <style>
        /* Animation styles */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.3); background-color: #ffeb3b; }
            100% { transform: scale(1); }
        }

        @keyframes rotateFace {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(90deg); }
        }

        .move-animate {
            animation: pulse 0.5s ease;
        }

        .face-animate {
            animation: pulse 0.5s ease;
        }

        /* Interactive hover styles */
        .cubie:hover {
            transform: scale(1.1);
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        /* Current move highlight */
        .current-move {
            background-color: #3498db !important;
            color: white;
            transform: scale(1.2);
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
            font-weight: bold;
        }

        /* Add a button to play through the solution */
        .controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
        }

        .control-button {
            padding: 8px 16px;
            background-color: #2c3e50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        .control-button:hover {
            background-color: #1a2530;
        }

        .control-button:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        </style>
    </body>
    </html>
    """

    return html


def generate_face_html(face_name: str, face_colors: List[str]) -> str:
    """
    Generate HTML for a single face of the cube

    Args:
        face_name: Name of the face (Up, Down, Left, Right, Front, Back)
        face_colors: List of 9 colors for the face

    Returns:
        HTML string for the face
    """
    # Color mapping
    color_map = {
        "W": "#FFFFFF",  # White
        "Y": "#FFFF00",  # Yellow
        "R": "#FF0000",  # Red
        "O": "#FFA500",  # Orange
        "B": "#0000FF",  # Blue
        "G": "#00FF00",  # Green
        "?": "#CCCCCC",  # Gray (unknown/placeholder)
    }

    html = f'<div class="face"><div class="face-title">{face_name}</div><div class="face-grid">'

    for color in face_colors:
        bg_color = color_map.get(
            color[0], "#CCCCCC"
        )  # Get first char of color, default to gray
        html += (
            f'<div class="cubie" style="background-color: {bg_color};">{color}</div>'
        )

    html += "</div></div>"
    return html


def extract_thinking_from_history(message_history: List[Dict]) -> List[str]:
    """
    Extract thinking steps from LLM message history

    Args:
        message_history: List of message dictionaries with roles and content

    Returns:
        List of extracted thinking content
    """
    thinking_steps = []

    for message in message_history:
        if message.get("role") == "agent" and isinstance(message.get("content"), str):
            content = message["content"]
            thinking_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if thinking_match:
                thinking_text = thinking_match.group(1).strip()
                if thinking_text:
                    thinking_steps.append(thinking_text)

    return thinking_steps


def save_enhanced_visualization(
    cube_state: str,
    move_history: List[str] = None,
    progress_history: List[float] = None,
    rewards_history: List[float] = None,
    thinking_history: List[str] = None,
    message_history: List[Dict] = None,
    scramble_sequence: List[str] = None,
    is_solved: bool = False,
    curriculum_level: int = None,
    curriculum_description: str = None,
    output_path: str = None,
) -> str:
    """
    Generate and save an enhanced HTML visualization of a Rubik's cube solving attempt

    Args:
        cube_state: String representation of the cube's current state
        move_history: List of moves applied during solving
        progress_history: List of progress values after each move
        rewards_history: List of rewards for each move
        thinking_history: List of thinking steps extracted from LLM
        message_history: Optional full message history to extract thinking from
        scramble_sequence: List of moves used to scramble the cube
        is_solved: Whether the cube is solved
        curriculum_level: Current curriculum level (if using curriculum)
        curriculum_description: Description of the current curriculum level
        output_path: Optional file path to save the HTML

    Returns:
        Path to the saved HTML file
    """
    # Extract thinking from message history if provided and not already extracted
    if message_history and not thinking_history:
        thinking_history = extract_thinking_from_history(message_history)

    # Generate the HTML
    html = generate_enhanced_cube_html(
        cube_state=cube_state,
        move_history=move_history,
        progress_history=progress_history,
        rewards_history=rewards_history,
        thinking_history=thinking_history,
        scramble_sequence=scramble_sequence,
        is_solved=is_solved,
        curriculum_level=curriculum_level,
        curriculum_description=curriculum_description,
    )

    # Set default output path if not provided
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        solved_status = "solved" if is_solved else "unsolved"
        output_path = f"rubiks_visualization_{solved_status}_{timestamp}.html"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Save the HTML to a file
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


if __name__ == "__main__":
    # Example usage
    example_cube_state = """
    U: W W W
       W W W
       W W W
    D: Y Y Y
       Y Y Y
       Y Y Y
    L: O O O
       O O O
       O O O
    R: R R R
       R R R
       R R R
    F: G G G
       G G G
       G G G
    B: B B B
       B B B
       B B B
    """

    # Create example data
    example_moves = [
        "R",
        "U",
        "R'",
        "U'",
        "F",
        "R",
        "U",
        "R'",
        "U'",
        "F'",
        "U",
        "R",
        "U2",
        "R'",
    ]

    # Generate sample progress history
    progress_vals = [0.5]
    for i in range(1, len(example_moves)):
        # Random progress that generally increases
        next_progress = min(1.0, progress_vals[-1] + random.uniform(-0.05, 0.15))
        progress_vals.append(next_progress)

    # Generate sample rewards
    rewards = [random.uniform(0.1, 0.9) for _ in range(len(example_moves))]

    # Sample thinking steps
    sample_thinking = [
        "I see the cube has a white cross on top already. I'll focus on solving the first layer corners.",
        "I'll use the sequence R U R' U' to position the corner piece without disrupting the cross.",
        "Now I need to solve the middle layer edges. I'll use the appropriate algorithm based on the edge orientation.",
        "Looking at the last layer, I need to orient the yellow face first, then permute the corners and edges.",
    ]

    # Sample scramble
    sample_scramble = ["F", "R", "U'", "B", "L2", "D"]

    # Generate and save the visualization
    html_path = save_enhanced_visualization(
        cube_state=example_cube_state,
        move_history=example_moves,
        progress_history=progress_vals,
        rewards_history=rewards,
        thinking_history=sample_thinking,
        scramble_sequence=sample_scramble,
        is_solved=True,
        curriculum_level=2,
        curriculum_description="Level 2: Easy - Learn basic patterns and simple sequences",
        output_path="example_enhanced_rubiks.html",
    )

    print(f"Enhanced visualization saved to {html_path}")
