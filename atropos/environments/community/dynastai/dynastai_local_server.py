#!/usr/bin/env python3
"""
DynastAI Local Server - For local development and testing

This script starts both the backend API server and serves the frontend static files.
"""

import argparse
import os
import sys
import time
from threading import Thread

from src.config import get_config

# Ensure src directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Add parent directory to path to allow standalone execution without atroposlib
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DynastAI Local Development Server")
    parser.add_argument(
        "--api-port", type=int, default=9001, help="Port for the API server"
    )
    parser.add_argument(
        "--web-port", type=int, default=3000, help="Port for the web server"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open the web browser automatically",
    )
    return parser.parse_args()


def run_api_server(port):
    """Run the API server"""
    from src.web.server import run_server

    run_server(host="localhost", port=port)


def run_web_server(port, static_dir):
    """Run a simple HTTP server for the frontend"""
    import http.server
    import socket
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=static_dir, **kwargs)

    # Try ports in sequence until one works
    max_port_attempts = 10
    current_port = port

    for attempt in range(max_port_attempts):
        try:
            with socketserver.TCPServer(("", current_port), Handler) as httpd:
                print(f"Web server running at http://localhost:{current_port}")
                httpd.serve_forever()
                break
        except socket.error:
            if attempt < max_port_attempts - 1:
                print(f"Port {current_port} is in use, trying port {current_port + 1}")
                current_port += 1
            else:
                raise Exception(
                    f"Could not find an available port after {max_port_attempts} attempts"
                )


def main():
    """Main entry point"""
    args = parse_args()
    get_config()  # Initialize config

    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "src/data"), exist_ok=True)

    # Initialize the web directory if not already set up
    web_dir = os.path.join(os.path.dirname(__file__), "src/web")
    static_dir = os.path.join(web_dir, "static")

    if not os.path.exists(static_dir):
        os.makedirs(static_dir, exist_ok=True)

        # Create basic HTML, CSS, JS files if they don't exist
        html_file = os.path.join(static_dir, "index.html")
        css_file = os.path.join(static_dir, "styles.css")
        js_file = os.path.join(static_dir, "game.js")

        if not os.path.exists(html_file):
            with open(html_file, "w") as f:
                f.write(
                    """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DynastAI - Medieval Kingdom Management</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <h1>DynastAI</h1>
        <p>Rule wisely or suffer the consequences</p>
    </header>

    <main>
        <div id="metrics-container">
            <div class="metric">
                <h3>Power</h3>
                <div class="meter-container">
                    <div class="meter power" id="power-meter"></div>
                </div>
                <span id="power-value">50</span>
            </div>
            <div class="metric">
                <h3>Stability</h3>
                <div class="meter-container">
                    <div class="meter stability" id="stability-meter"></div>
                </div>
                <span id="stability-value">50</span>
            </div>
            <div class="metric">
                <h3>Piety</h3>
                <div class="meter-container">
                    <div class="meter piety" id="piety-meter"></div>
                </div>
                <span id="piety-value">50</span>
            </div>
            <div class="metric">
                <h3>Wealth</h3>
                <div class="meter-container">
                    <div class="meter wealth" id="wealth-meter"></div>
                </div>
                <span id="wealth-value">50</span>
            </div>
        </div>

        <div id="card-container" class="hidden">
            <div id="card">
                <div id="card-text">Welcome to your kingdom, Your Majesty. Make your choices wisely...</div>
                <div id="card-options">
                    <button id="yes-button">Yes</button>
                    <button id="no-button">No</button>
                </div>
            </div>
        </div>

        <div id="start-screen">
            <h2>Begin Your Reign</h2>
            <button id="start-game">Start New Game</button>
        </div>

        <div id="game-over" class="hidden">
            <h2>Your Reign Has Ended</h2>
            <p id="game-over-reason"></p>
            <p id="reign-summary"></p>
            <p id="detailed-ending" class="detailed-ending"></p>
            <p id="legacy-message" class="legacy-message"></p>
            <div id="reign-options">
                <button id="continue-game">Continue Playing</button>
                <button id="new-game">Start New Reign</button>
            </div>
        </div>
    </main>

    <footer>
        <p>Year <span id="reign-year">1</span> of your reign</p>
    </footer>

    <script src="game.js"></script>
</body>
</html>
                """
                )

        if not os.path.exists(css_file):
            with open(css_file, "w") as f:
                f.write(
                    """
:root {
    --power-color: #e74c3c;
    --stability-color: #2ecc71;
    --piety-color: #f1c40f;
    --wealth-color: #3498db;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Georgia', serif;
    background-color: #2c3e50;
    color: #ecf0f1;
    line-height: 1.6;
    /* Removed stock photo background */
}

header, footer {
    text-align: center;
    padding: 1rem;
    background-color: rgba(0, 0, 0, 0.7);
    color: #fff;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

main {
    max-width: 800px;
    margin: 2rem auto;
    min-height: calc(100vh - 12rem);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

#metrics-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    margin-bottom: 2rem;
    background-color: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.metric {
    flex: 1;
    min-width: 150px;
    margin: 0.5rem;
    text-align: center;
}

.meter-container {
    background-color: #ddd;
    height: 20px;
    border-radius: 10px;
    margin: 0.5rem 0;
    overflow: hidden;
}

.meter {
    height: 100%;
    transition: width 0.5s ease-in-out;
}

.power { background-color: var(--power-color); width: 50%; }
.stability { background-color: var(--stability-color); width: 50%; }
.piety { background-color: var(--piety-color); width: 50%; }
.wealth { background-color: var(--wealth-color); width: 50%; }

#card-container {
    display: flex;
    justify-content: center;
    margin: 2rem 0;
}

#card {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    width: 100%;
    max-width: 600px;
}

#card-text {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    line-height: 1.6;
}

#card-options {
    display: flex;
    justify-content: space-between;
}

button {
    padding: 0.8rem 2rem;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: bold;
    transition: all 0.2s ease;
}

#yes-button {
    background-color: #2ecc71;
    color: white;
}

#no-button {
    background-color: #e74c3c;
    color: white;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

#start-screen, #game-over {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    text-align: center;
    margin: 2rem auto;
    max-width: 500px;
}

#start-game, #new-game {
    background-color: #3498db;
    color: white;
    margin-top: 1rem;
    padding: 1rem 2rem;
}

footer {
    margin-top: auto;
}

.hidden {
    display: none !important;
}

.detailed-ending {
    background-color: rgba(30, 30, 30, 0.7);
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
    line-height: 1.7;
    max-width: 800px;
    text-align: justify;
}

.legacy-message {
    font-style: italic;
    margin-bottom: 20px;
}

#reign-options {
    display: flex;
    gap: 15px;
    justify-content: center;
    margin-top: 20px;
}
                """
                )

        if not os.path.exists(js_file):
            with open(js_file, "w") as f:
                f.write(
                    """
// Constants
const API_URL = 'http://localhost:9001/api';
let sessionId = null;
let currentCard = null;
let gameOver = false;

// DOM Elements
const powerMeter = document.getElementById('power-meter');
const stabilityMeter = document.getElementById('stability-meter');
const pietyMeter = document.getElementById('piety-meter');
const wealthMeter = document.getElementById('wealth-meter');

const powerValue = document.getElementById('power-value');
const stabilityValue = document.getElementById('stability-value');
const pietyValue = document.getElementById('piety-value');
const wealthValue = document.getElementById('wealth-value');

const reignYear = document.getElementById('reign-year');
const cardContainer = document.getElementById('card-container');
const cardText = document.getElementById('card-text');
const yesButton = document.getElementById('yes-button');
const noButton = document.getElementById('no-button');
const startScreen = document.getElementById('start-screen');
const startGameButton = document.getElementById('start-game');
const gameOverScreen = document.getElementById('game-over');
const gameOverReason = document.getElementById('game-over-reason');
const reignSummary = document.getElementById('reign-summary');
const detailedEnding = document.getElementById('detailed-ending');
const legacyMessage = document.getElementById('legacy-message');
const continueGameButton = document.getElementById('continue-game');
const newGameButton = document.getElementById('new-game');

// Game state
let metrics = {
    power: 50,
    stability: 50,
    piety: 50,
    wealth: 50,
    reign_year: 1
};

let trajectory = [];

// Event listeners
startGameButton.addEventListener('click', startGame);
yesButton.addEventListener('click', () => makeChoice('yes'));
noButton.addEventListener('click', () => makeChoice('no'));
continueGameButton.addEventListener('click', continueGame);
newGameButton.addEventListener('click', startGame);

// Helper functions
function updateMeters() {
    powerMeter.style.width = `${metrics.power}%`;
    stabilityMeter.style.width = `${metrics.stability}%`;
    pietyMeter.style.width = `${metrics.piety}%`;
    wealthMeter.style.width = `${metrics.wealth}%`;

    powerValue.textContent = metrics.power;
    stabilityValue.textContent = metrics.stability;
    pietyValue.textContent = metrics.piety;
    wealthValue.textContent = metrics.wealth;

    reignYear.textContent = metrics.reign_year;

    // Change colors when values get dangerous
    if (metrics.power <= 20 || metrics.power >= 80) {
        powerMeter.style.backgroundColor = '#ff5252';
    } else {
        powerMeter.style.backgroundColor = 'var(--power-color)';
    }

    if (metrics.stability <= 20 || metrics.stability >= 80) {
        stabilityMeter.style.backgroundColor = '#ff9800';
    } else {
        stabilityMeter.style.backgroundColor = 'var(--stability-color)';
    }

    if (metrics.piety <= 20 || metrics.piety >= 80) {
        pietyMeter.style.backgroundColor = '#ff9800';
    } else {
        pietyMeter.style.backgroundColor = 'var(--piety-color)';
    }

    if (metrics.wealth <= 20 || metrics.wealth >= 80) {
        wealthMeter.style.backgroundColor = '#ff9800';
    } else {
        wealthMeter.style.backgroundColor = 'var(--wealth-color)';
    }
}

async function startGame() {
    try {
        // Reset game state
        gameOver = false;
        trajectory = [];

        // Create new game session
        const response = await fetch(`${API_URL}/new_game`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();
        sessionId = data.session_id;
        metrics = data.metrics;

        updateMeters();

        // Hide start screen, show game screen
        startScreen.classList.add('hidden');
        gameOverScreen.classList.add('hidden');
        cardContainer.classList.remove('hidden');

        // Generate first card
        await generateCard();

    } catch (error) {
        console.error("Error starting game:", error);
        alert("Failed to start game. Please check your connection to the game server.");
    }
}

async function generateCard() {
    try {
        const response = await fetch(`${API_URL}/generate_card`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        currentCard = await response.json();

        // Update card UI
        cardText.textContent = currentCard.text;
        yesButton.textContent = currentCard.yes_option;
        noButton.textContent = currentCard.no_option;

    } catch (error) {
        console.error("Error generating card:", error);
        cardText.textContent = "Something went wrong. Please try again.";
    }
}

async function makeChoice(choice) {
    try {
        const response = await fetch(`${API_URL}/card_choice`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                choice: choice
            })
        });

        const data = await response.json();

        // Record this move in trajectory with all required fields
        trajectory.push({
            card_id: currentCard.id || "unknown",
            category: currentCard.category || "unknown",
            choice: choice,
            effects: {
                power: currentCard.effects[choice].power || 0,
                stability: currentCard.effects[choice].stability || 0,
                piety: currentCard.effects[choice].piety || 0,
                wealth: currentCard.effects[choice].wealth || 0
            },
            post_metrics: data.metrics
        });

        // Update game state
        metrics = data.metrics;
        updateMeters();

        // Check for game over conditions
        const reignEnded = (
            data.game_over ||
            metrics.power <= 0 || metrics.power >= 100 ||
            metrics.stability <= 0 || metrics.stability >= 100 ||
            metrics.piety <= 0 || metrics.piety >= 100 ||
            metrics.wealth <= 0 || metrics.wealth >= 100
        );

        console.log("Checking game over conditions:", reignEnded, metrics);

        if (reignEnded) {
            await endReign();
            return;
        }

        // Generate next card
        await generateCard();

    } catch (error) {
        console.error("Error processing choice:", error);
        cardText.textContent = "Something went wrong. Please try again.";
    }
}

async function continueGame() {
    try {
        // Reset game state but keep session ID for continuity
        gameOver = false;
        trajectory = [];

        // Create new game session with the same session ID to maintain reign history
        const response = await fetch(`${API_URL}/new_game`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        const data = await response.json();
        metrics = data.metrics;

        updateMeters();

        // Hide game over screen, show game screen
        gameOverScreen.classList.add('hidden');
        cardContainer.classList.remove('hidden');

        // Generate first card
        await generateCard();
    } catch (error) {
        console.error("Error continuing game:", error);
        gameOverReason.textContent = "Something went wrong when starting a new reign.";
    }
}

async function endReign() {
    try {
        // Determine cause of end
        let cause = null;

        // Log current metrics for debugging
        console.log("End reign metrics:", metrics);

        if (metrics.power <= 0) cause = "power_low";
        else if (metrics.power >= 100) cause = "power_high";
        else if (metrics.stability <= 0) cause = "stability_low";
        else if (metrics.stability >= 100) cause = "stability_high";
        else if (metrics.piety <= 0) cause = "piety_low";
        else if (metrics.piety >= 100) cause = "piety_high";
        else if (metrics.wealth <= 0) cause = "wealth_low";
        else if (metrics.wealth >= 100) cause = "wealth_high";
        else cause = "old_age";

        console.log("Determined cause of end:", cause);

        // Send end reign data to server with complete information
        const response = await fetch(`${API_URL}/end_reign`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                trajectory: trajectory,
                final_metrics: {
                    power: metrics.power,
                    stability: metrics.stability,
                    piety: metrics.piety,
                    wealth: metrics.wealth,
                    reign_year: metrics.reign_year
                },
                reign_length: metrics.reign_year,
                cause_of_end: cause
            })
        });

        let data = { reward: 0 };

        if (response.ok) {
            data = await response.json();
        } else {
            console.error("End reign failed with status:", response.status);
        }

        // Show game over screen
        cardContainer.classList.add('hidden');
        gameOverScreen.classList.remove('hidden');

        // Set reason based on metrics and get detailed ending
        let reason = "";
        let detailedText = "";
        let legacyText = "";

        // Set reason based on cause
        if (cause === "power_low") {
            reason = "You lost all authority. The nobles overthrew you!";
            detailedText = "Years of concessions and weak leadership eroded your authority. " +
                "The nobles, seeing your weakness, formed a coalition against you. " +
                "After a brief struggle, you were deposed and exiled, remembered as a ruler " +
                "who couldn't maintain the respect of the nobility.";
            legacyText = determineRulerLegacy("weak");
        } else if (cause === "power_high") {
            reason = "Your absolute power made you a tyrant. You were assassinated!";
            detailedText = "Your iron-fisted rule and consolidation of power bred resentment " +
                "among the nobility. As your authority grew unchecked, many feared for their " +
                "own positions. A conspiracy formed in the shadows, and despite your vigilance, " +
                "an assassin's blade found its mark. You died as you ruled - alone and feared.";
            legacyText = determineRulerLegacy("tyrant");
        } else if (cause === "stability_low") {
            reason = "The people revolted against your rule!";
            detailedText = "The cries of the hungry and oppressed grew too loud to ignore. " +
                "Years of neglect and harsh policies turned the populace against you. " +
                "What began as isolated protests quickly spread across the kingdom. " +
                "The uprising was swift and merciless, with angry mobs storming the palace. " +
                "Your reign ended at the hands of those you failed to serve.";
            legacyText = determineRulerLegacy("hated");
        } else if (cause === "stability_high") {
            reason = "The people loved you so much they established a republic!";
            detailedText = "The common folk adored you for your generosity and fairness. " +
                "However, your popularity threatened the traditional power structure. " +
                "As people began calling for democratic reforms and greater representation, " +
                "the nobles and church became alarmed. They orchestrated your removal, " +
                "claiming the kingdom needed 'proper governance, not popularity.' " +
                "The republic that followed bore your name, though you did not live to see it flourish.";
            legacyText = determineRulerLegacy("beloved");
        } else if (cause === "piety_low") {
            reason = "The church declared you a heretic and had you executed!";
            detailedText = "Your dismissal of religious traditions and constant conflicts with " +
                "church authorities were deemed heretical. The Grand Inquisitor publicly " +
                "denounced you, turning religious sentiment against the crown. " +
                "Priests preached against you from every pulpit until the faithful rose up " +
                "in a holy crusade. Declared a heretic, you faced the ultimate punishment " +
                "for challenging divine authority.";
            legacyText = determineRulerLegacy("heretic");
        } else if (cause === "piety_high") {
            reason = "The church became too powerful and took control of your kingdom!";
            detailedText = "You allowed religious authorities too much influence, and the " +
                "church's power grew unchecked. Gradually, religious law superseded royal edicts, " +
                "and church officials began overruling your decisions. Eventually, the Archbishop " +
                "declared divine right to rule, and with popular support, established a theocracy. " +
                "You were permitted to retain your title in name only - a figurehead in a kingdom " +
                "ruled by the cloth.";
            legacyText = determineRulerLegacy("pious");
        } else if (cause === "wealth_low") {
            reason = "Your kingdom went bankrupt and you were deposed!";
            detailedText = "Years of extravagance and financial mismanagement emptied the royal " +
                "coffers. Unable to pay the army or maintain the kingdom's infrastructure, " +
                "your rule collapsed under mounting debts. Foreign creditors seized royal assets, " +
                "while unpaid servants and soldiers abandoned their posts. With nothing left to rule, " +
                "you were quietly removed from the throne, your name becoming synonymous with " +
                "fiscal irresponsibility.";
            legacyText = determineRulerLegacy("poor");
        } else if (cause === "wealth_high") {
            reason = "Your vast wealth attracted invaders who conquered your kingdom!";
            detailedText = "Your kingdom's legendary wealth attracted unwanted attention. " +
                "Neighboring rulers looked upon your treasuries with envy, and despite your " +
                "diplomatic efforts, greed won out. A coalition of foreign powers, using your " +
                "hoarding of wealth as justification, invaded with overwhelming force. " +
                "Your vast riches funded your enemies' armies, and your kingdom was divided " +
                "among the victors.";
            legacyText = determineRulerLegacy("wealthy");
        } else {
            reason = "You died of natural causes after a long reign.";
            detailedText = `After ${metrics.reign_year} years of rule, age finally caught up with you. ` +
                `Your legacy secured, you passed peacefully in your sleep, surrounded by ` +
                `generations of family. The kingdom mourned for forty days, and your achievements ` +
                `were recorded in detail by royal historians. Few monarchs are fortunate enough ` +
                `to meet such a natural end, a testament to your balanced approach to leadership.`;
            legacyText = determineRulerLegacy("balanced");
        }

        gameOverReason.textContent = reason;
        detailedEnding.textContent = detailedText;
        legacyMessage.textContent = legacyText;
        reignSummary.textContent = `You ruled for ${metrics.reign_year} years. ` +
            `Final reward: ${data.reward.toFixed(2)}`;

    } catch (error) {
        console.error("Error ending reign:", error);
        gameOverReason.textContent = "Something went wrong when calculating your legacy.";
    }
}

function determineRulerLegacy(rulerType) {
    // Generate a legacy message based on reign length and ruler type
    const reignLength = metrics.reign_year;
    let legacy = "";

    if (reignLength < 5) {
        legacy = "Your brief rule will be barely a footnote in the kingdom's history.";
    } else if (reignLength > 30) {
        switch (rulerType) {
            case "balanced":
                legacy = "Your long and balanced reign will be remembered as a golden age " +
                    "of prosperity and peace.";
                break;
            case "tyrant":
                legacy = "Your decades of tyrannical rule have left a permanent scar on the " +
                    "kingdom's history. Your name will be used to frighten children for generations.";
                break;
            case "beloved":
                legacy = "Your generous and fair leadership established a cultural renaissance " +
                    "that will be studied for centuries to come.";
                break;
            default:
                legacy = "Your long reign, despite its end, has made an indelible mark " +
                    "on the kingdom's history.";
        }
    } else {
        switch (rulerType) {
            case "weak":
                legacy = "History will remember you as a monarch who failed to maintain " +
                    "control of their own court.";
                break;
            case "tyrant":
                legacy = "You will be remembered as a harsh and unforgiving ruler who " +
                    "sought power above all else.";
                break;
            case "hated":
                legacy = "Your name will be spoken with contempt by commoners for " +
                    "generations to come.";
                break;
            case "beloved":
                legacy = "The people will sing songs of your kindness and fairness for " +
                    "many years.";
                break;
            case "heretic":
                legacy = "Religious texts will cite you as an example of the dangers of " +
                    "straying from the faith.";
                break;
            case "pious":
                legacy = "You will be remembered as a devout ruler who perhaps trusted " +
                    "the clergy too much.";
                break;
            case "poor":
                legacy = "Future monarchs will study your reign as a cautionary tale of " +
                    "financial mismanagement.";
                break;
            case "wealthy":
                legacy = "Tales of your kingdom's riches will become legendary, though they " +
                    "ultimately led to your downfall.";
                break;
            case "balanced":
                legacy = "Your rule will be remembered as a time of reasonable balance " +
                    "and steady progress.";
                break;
            default:
                legacy = `You ruled for ${reignLength} years, leaving behind a mixed legacy ` +
                    `of successes and failures.`;
        }
    }

    return legacy;
}

// Check if API is available when page loads
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}`);
        await response.json();
        console.log("API connection successful");
    } catch (error) {
        console.error("API connection failed:", error);
        cardText.textContent = "Cannot connect to game server. Please ensure the server is running.";
    }
});
                """
                )

    # Start API server in a separate thread
    print(f"Starting API server on port {args.api_port}")
    api_thread = Thread(target=run_api_server, args=(args.api_port,))
    api_thread.daemon = True
    api_thread.start()

    # Give the API server time to start
    time.sleep(2)

    # Start web server for frontend
    print(f"Starting web server on port {args.web_port}")

    try:
        # Run the web server in the main thread
        run_web_server(args.web_port, static_dir)
    except Exception as e:
        print(f"Error starting web server: {e}")
        print("Please try running the server again with a different port:")
        print("python3 environments/dynastai/dynastai_local_server.py --web-port 8080")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down servers...")
        sys.exit(0)
