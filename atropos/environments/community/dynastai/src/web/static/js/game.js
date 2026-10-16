/**
 * DynastAI - Game Client
 *
 * This JavaScript file handles the client-side logic for the DynastAI game:
 * - Communicating with the API
 * - Updating the UI based on game state
 * - Handling user interactions
 */

// Configuration
const API_URL = 'http://localhost:9001/api/'; // Added trailing slash
let sessionId = null;
let currentCard = null;
let gameOver = false;

// Track game state
let metrics = {
    power: 50,
    stability: 50,
    piety: 50,
    wealth: 50,
    reign_year: 1
};

// Track dynasty timeline
let dynastyYear = 0;
let rulerName = "Anonymous Ruler";
let trajectory = [];

// DOM Elements
const powerMeter = document.getElementById('power-meter');
const stabilityMeter = document.getElementById('stability-meter');
const pietyMeter = document.getElementById('piety-meter');
const wealthMeter = document.getElementById('wealth-meter');

const powerValue = document.getElementById('power-value');
const stabilityValue = document.getElementById('stability-value');
const pietyValue = document.getElementById('piety-value');
const wealthValue = document.getElementById('wealth-value');

const powerEffect = document.getElementById('power-effect');
const stabilityEffect = document.getElementById('stability-effect');
const pietyEffect = document.getElementById('piety-effect');
const wealthEffect = document.getElementById('wealth-effect');

const dynastyYearSpan = document.getElementById('dynasty-year');
const rulerNameInput = document.getElementById('ruler-name');

const effectsDisplay = document.getElementById('effects-display');
const categoryIndicator = document.getElementById('category-indicator');
const reignYear = document.getElementById('reign-year');
const cardContainer = document.getElementById('card-container');
const cardText = document.getElementById('card-text');
const yesButton = document.getElementById('yes-button');
const noButton = document.getElementById('no-button');
const startScreen = document.getElementById('start-screen');
const startGameButton = document.getElementById('start-game');
const gameOverScreen = document.getElementById('game-over');
const gameOverReason = document.getElementById('game-over-reason');
const finalPower = document.getElementById('final-power');
const finalStability = document.getElementById('final-stability');
const finalPiety = document.getElementById('final-piety');
const finalWealth = document.getElementById('final-wealth');
const reignSummary = document.getElementById('reign-summary');
const detailedEnding = document.getElementById('detailed-ending');
const legacyMessage = document.getElementById('legacy-message');
const continueGameButton = document.getElementById('continue-game');
const newGameButton = document.getElementById('new-game');
const apiStatus = document.getElementById('api-status');

// Event listeners
startGameButton.addEventListener('click', startGame);
yesButton.addEventListener('click', () => makeChoice('yes'));
noButton.addEventListener('click', () => makeChoice('no'));
continueGameButton.addEventListener('click', continueGame);
newGameButton.addEventListener('click', startGame);

/**
 * Check API availability and update the status indicator
 */
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_URL}`);
        await response.json();
        apiStatus.textContent = 'Connected';
        apiStatus.classList.add('online');
        return true;
    } catch (error) {
        console.error("API connection failed:", error);
        apiStatus.textContent = 'Disconnected';
        apiStatus.classList.add('offline');
        return false;
    }
}

/**
 * Update UI meters and values based on current metrics
 */
function updateMeters() {
    // Update meter widths
    powerMeter.style.width = `${metrics.power}%`;
    stabilityMeter.style.width = `${metrics.stability}%`;
    pietyMeter.style.width = `${metrics.piety}%`;
    wealthMeter.style.width = `${metrics.wealth}%`;

    // Update displayed values
    powerValue.textContent = metrics.power;
    stabilityValue.textContent = metrics.stability;
    pietyValue.textContent = metrics.piety;
    wealthValue.textContent = metrics.wealth;
    reignYear.textContent = metrics.reign_year;

    // Update dynasty year (base year + reign year)
    const dynastyYearElement = document.getElementById('dynasty-year');
    if (dynastyYearElement) {
        dynastyYearElement.textContent = dynastyYear + metrics.reign_year;
    }

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

/**
 * Display choice effects on the UI
 */
function displayEffects(effects) {
    // Check if the elements exist
    if (!powerEffect || !stabilityEffect || !pietyEffect || !wealthEffect || !effectsDisplay) {
        console.error("Effect display elements not found");
        return;
    }

    // Update effect values
    powerValue.innerHTML = `${metrics.power} <span class="stat-change ${effects.power >= 0 ? 'positive' : 'negative'}">${formatEffect(effects.power)}</span>`;
    stabilityValue.innerHTML = `${metrics.stability} <span class="stat-change ${effects.stability >= 0 ? 'positive' : 'negative'}">${formatEffect(effects.stability)}</span>`;
    pietyValue.innerHTML = `${metrics.piety} <span class="stat-change ${effects.piety >= 0 ? 'positive' : 'negative'}">${formatEffect(effects.piety)}</span>`;
    wealthValue.innerHTML = `${metrics.wealth} <span class="stat-change ${effects.wealth >= 0 ? 'positive' : 'negative'}">${formatEffect(effects.wealth)}</span>`;

    // Hide effects display since we're showing them inline
    effectsDisplay.classList.add('hidden');

    // Hide the inline effects after 3 seconds
    setTimeout(() => {
        powerValue.textContent = metrics.power;
        stabilityValue.textContent = metrics.stability;
        pietyValue.textContent = metrics.piety;
        wealthValue.textContent = metrics.wealth;
    }, 3000);
}

/**
 * Format effect for display (+5, -3, etc)
 */
function formatEffect(value) {
    if (value > 0) return '+' + value;
    return value.toString();
}

/**
 * Decide which choice to make based on balancing metrics
 */
async function decideChoice(card, currentMetrics) {
    console.log("AI analyzing choices for card:", card);

    // Get effects for both choices
    const yesEffects = card.effects.yes;
    const noEffects = card.effects.no;

    // Calculate how "balanced" each choice would make the metrics
    const yesScore = calculateBalanceScore(currentMetrics, yesEffects);
    const noScore = calculateBalanceScore(currentMetrics, noEffects);

    console.log("Balance scores - Yes:", yesScore, "No:", noScore);

    // Choose the option with the better balance score
    const choice = yesScore >= noScore ? 'yes' : 'no';
    console.log("AI chose:", choice);
    return choice;
}

/**
 * Calculate how well-balanced the metrics would be after applying effects
 * Returns a score where higher is better (more balanced)
 */
function calculateBalanceScore(metrics, effects) {
    // Predict new metric values
    const newMetrics = {
        power: metrics.power + (effects.power || 0),
        stability: metrics.stability + (effects.stability || 0),
        piety: metrics.piety + (effects.piety || 0),
        wealth: metrics.wealth + (effects.wealth || 0)
    };

    // Calculate how far each metric is from the ideal range (20-80)
    // Lower penalty score is better
    let penaltyScore = 0;

    Object.values(newMetrics).forEach(value => {
        if (value < 20) {
            penaltyScore += (20 - value) * 2; // Penalize low values more
        } else if (value > 80) {
            penaltyScore += (value - 80) * 2; // Penalize high values more
        }
        // Values between 20-80 add no penalty
    });

    // Return inverted penalty score so higher is better
    return 1000 - penaltyScore;
}

/**
 * Highlight the selected choice button
 */
function highlightChoice(choice) {
    // Remove any existing highlights and indicators
    yesButton.classList.remove('highlight');
    noButton.classList.remove('highlight');

    // Add highlight to the chosen button with pixel art style
    const buttonToHighlight = choice === 'yes' ? yesButton : noButton;
    buttonToHighlight.classList.add('highlight');

    // Remove highlight after 1.5 seconds (before the next card appears)
    setTimeout(() => {
        buttonToHighlight.classList.remove('highlight');
    }, 1500);
}

/**
 * Start a new game session
 */
async function startGame() {
    try {
        // Check API availability
        const apiAvailable = await checkApiStatus();
        if (!apiAvailable) {
            cardText.textContent = "Cannot connect to game server. Please ensure the server is running.";
            return;
        }

        // Reset game state
        gameOver = false;
        trajectory = [];

        // Get ruler name from input
        rulerName = rulerNameInput.value.trim() || "Anonymous Ruler";

        // Get play mode
        const playMode = document.getElementById('play-mode').value;
        console.log("Starting game in mode:", playMode);

        // Reset dynasty year only if this is a new game (button text check)
        if (startGameButton.textContent !== "Continue Dynasty") {
            dynastyYear = 0;
        }

        // Create new game session
        const response = await fetch(`${API_URL}new_game`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId
            })
        });

        const data = await response.json();
        sessionId = data.session_id;
        metrics = data.metrics;

        updateMeters();

        // Hide start screen, show game screen
        startScreen.classList.add('hidden');
        gameOverScreen.classList.add('hidden');
        cardContainer.classList.remove('hidden');

        // Reset the start button text for future new games
        startGameButton.textContent = "Start New Game";

        // Show/hide AI choice button based on play mode
        const aiChoiceButton = document.getElementById('ai-choice-button');
        if (aiChoiceButton) {
            aiChoiceButton.style.display = playMode === 'ai' ? 'none' : 'block';
        }

        // Generate first card
        await generateCard();

        // If AI mode is selected, start making choices automatically after a 2-second delay
        if (playMode === 'ai') {
            console.log("AI mode selected, making first choice...");
            setTimeout(async () => {
                await letAIMakeChoice();
            }, 1400);
        }

    } catch (error) {
        console.error("Error starting game:", error);
        alert("Failed to start game. Please check your connection to the game server.");
    }
}

/**
 * Add this mapping function before generateCard()
 */
function getCharacterImageFilename(characterName) {
    // Map of titles to image filenames (without .png extension)
    const titleMap = {
        // Compound titles (must come before single-word titles)
        'Guild Master': 'guild-merchant',
        'Royal Advisor': 'royal-advisor',
        'Master of Coin': 'master-of-coin',
        'Master of Laws': 'master-of-laws',
        'Master of Arms': 'master-of-arms',
        'Master of Spies': 'master-of-spies',
        'Court Jester': 'court-jester',
        'Master Builder': 'master-builder',
        'Fleet Admiral': 'fleet-admiral',
        'Royal Seneschal': 'royal-seneschal',
        'Grand Inquisitor': 'grand-inquisitor',
        'Court Astronomer': 'court-astronomer',
        'Master Blacksmith': 'master-blacksmith',
        'Court Physician': 'court-physician',
        'Royal Diplomat': 'royal-diplomat',
        'Court Scribe': 'court-scribe',
        'Royal Librarian': 'royal-librarian',
        'Captain of the Guard': 'captain-of-the-guard',
        'Plague Doctor': 'plague-doctor',
        'Foreign Ambassador': 'foreign-ambassador',
        'Peasant': 'peasant',
        'Court Philosopher': 'court-philosopher',

        // Single-word titles
        'Bishop': 'cardinal',
        'General': 'general',
        'Abbot': 'abbot',
        'Sheriff': 'sheriff',
        'Queen': 'queen',
        'Prince': 'prince',
        'Princess': 'princess'
    };

    // Try to match the full title first
    for (const [title, filename] of Object.entries(titleMap)) {
        if (characterName.startsWith(title)) {
            return filename;
        }
    }

    // If no match found, return no-image
    console.warn(`No image mapping found for character: ${characterName}`);
    return 'no-image';
}

/**
 * Generate a new card
 */
async function generateCard() {
    try {
        // Clear any existing highlights first
        yesButton.classList.remove('highlight');
        noButton.classList.remove('highlight');

        const response = await fetch(`${API_URL}generate_card`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        currentCard = await response.json();

        // Update card UI
        let cardContent = currentCard.text;

        // Handle character image
        const characterImage = document.getElementById('character-image');
        if (currentCard.character_name) {
            // Get the correct image filename based on character title
            const imageFilename = getCharacterImageFilename(currentCard.character_name);

            // Set image source and show it
            characterImage.src = `img/${imageFilename}.png`;
            characterImage.style.display = 'block';

            // Add error handler to fallback to no-image if image fails to load
            characterImage.onerror = () => {
                console.warn(`Failed to load image for ${currentCard.character_name}, falling back to no-image`);
                characterImage.src = 'img/no-image.png';
            };

            cardContent = `<span class="character-name">${currentCard.character_name}:</span> ${cardContent}`;
        } else {
            characterImage.style.display = 'none';
        }

        cardText.innerHTML = cardContent;
        yesButton.textContent = currentCard.yes_option;
        noButton.textContent = currentCard.no_option;

        // Update category indicator
        categoryIndicator.className = '';
        categoryIndicator.classList.add(currentCard.category);

        // Attach AI choice button event listener after rendering
        const aiChoiceButton = document.getElementById('ai-choice-button');
        if (aiChoiceButton) {
            aiChoiceButton.onclick = async () => {
                console.log("AI choice button clicked");
                const choice = await decideChoice(currentCard, metrics);
                console.log("AI chose:", choice);
                highlightChoice(choice);
                setTimeout(() => makeChoice(choice), 1000);
            };
        }
    } catch (error) {
        console.error("Error generating card:", error);
        cardText.textContent = "Something went wrong generating the next scenario. Please try again.";
    }
}

/**
 * Function to let AI make a choice
 */
async function letAIMakeChoice() {
    const playMode = document.getElementById('play-mode').value;
    console.log("Play mode selected:", playMode);
    if (playMode === 'ai') {
        console.log("AI is making a choice...");
        const choice = await decideChoice(currentCard, metrics);
        console.log("AI chose:", choice);

        // Wait for 2 seconds to simulate thinking, then show choice and highlight
        setTimeout(() => {
            // Add AI choice text indicator with text shadow for better readability
            const choiceText = choice === 'yes' ? currentCard.yes_option : currentCard.no_option;
            cardText.innerHTML += `<div class="ai-choice-indicator" style="margin-top: 1rem; color: #ffd700; font-family: var(--header-font); font-size: 0.8em; text-shadow: 2px 2px 0 #000, -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000;">AI chooses: ${choiceText}</div>`;

            // Highlight the chosen button
            highlightChoice(choice);

            // Make the choice after a brief pause
            setTimeout(() => makeChoice(choice), 500);
        }, 1500);
    }
}

/**
 * Modify the makeChoice function to call AI if in AI mode
 */
async function makeChoice(choice) {
    console.log("Making choice:", choice);
    try {
        const response = await fetch(`${API_URL}card_choice`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                choice: choice
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        console.log("Choice processed successfully:", data);

        // Display choice effects
        displayEffects(currentCard.effects[choice]);

        // Record this move in trajectory
        trajectory.push({
            card_id: String(currentCard.id || "unknown"),
            category: String(currentCard.category || "unknown"),
            choice: choice,
            effects: {
                power: Number(currentCard.effects[choice].power || 0),
                stability: Number(currentCard.effects[choice].stability || 0),
                piety: Number(currentCard.effects[choice].piety || 0),
                wealth: Number(currentCard.effects[choice].wealth || 0)
            },
            post_metrics: {
                power: Number(data.metrics.power),
                stability: Number(data.metrics.stability),
                piety: Number(data.metrics.piety),
                wealth: Number(data.metrics.wealth),
                reign_year: Number(data.metrics.reign_year)
            }
        });

        // Update game state
        metrics = data.metrics;
        updateMeters();

        // Debug log metrics
        console.log("Current metrics after choice:", metrics);

        // Check for game over conditions
        let reignEnded = false;

        if (data.game_over === true) {
            console.log("Game over signal from server");
            reignEnded = true;
        }

        if (metrics.power <= 0 || metrics.power >= 100 ||
            metrics.stability <= 0 || metrics.stability >= 100 ||
            metrics.piety <= 0 || metrics.piety >= 100 ||
            metrics.wealth <= 0 || metrics.wealth >= 100) {
            console.log("Game over due to metrics limit reached");
            reignEnded = true;
        }

        if (reignEnded) {
            console.log("Ending reign due to game over condition");
            await endReign();
            return;
        }

        // Generate next card
        await generateCard();

        // If in AI mode, let AI make the next choice after a 2-second delay
        if (document.getElementById('play-mode').value === 'ai') {
            setTimeout(async () => {
                await letAIMakeChoice();
            }, 1400);
        }

    } catch (error) {
        console.error("Error processing choice:", error);
        cardText.textContent = "Something went wrong processing your choice. Please try again.";
    }
}

/**
 * End the current reign and calculate final results
 */
async function endReign() {
    try {
        // Determine cause of end
        let cause = "old_age"; // Default cause

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

        console.log("Determined cause of end:", cause);

        // Debug log trajectory data
        console.log("Trajectory data:", JSON.stringify(trajectory));

        // Ensure trajectory data has the correct structure
        const cleanTrajectory = trajectory.map(item => ({
            card_id: String(item.card_id),
            category: String(item.category),
            choice: String(item.choice),
            effects: {
                power: Number(item.effects.power || 0),
                stability: Number(item.effects.stability || 0),
                piety: Number(item.effects.piety || 0),
                wealth: Number(item.effects.wealth || 0)
            },
            post_metrics: {
                power: Number(item.post_metrics.power || 0),
                stability: Number(item.post_metrics.stability || 0),
                piety: Number(item.post_metrics.piety || 0),
                wealth: Number(item.post_metrics.wealth || 0),
                reign_year: Number(item.post_metrics.reign_year || 1)
            }
        }));

        // Send end reign data to server with all required fields
        const response = await fetch(`${API_URL}end_reign`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                trajectory: cleanTrajectory,
                final_metrics: {
                    power: Number(metrics.power),
                    stability: Number(metrics.stability),
                    piety: Number(metrics.piety),
                    wealth: Number(metrics.wealth),
                    reign_year: Number(metrics.reign_year)
                },
                reign_length: Number(metrics.reign_year),
                cause_of_end: cause
            })
        });

        // Debug log for API response
        console.log("End reign response status:", response.status);

        let data = { reward: 0 };

        if (response.ok) {
            data = await response.json();
            console.log("End reign response data:", data);
        } else {
            console.error("End reign failed with status:", response.status);
            // Even if the API call fails, we should still show the game over screen
        }

        // Force game over state
        gameOver = true;

        // Show game over screen
        cardContainer.classList.add('hidden');
        gameOverScreen.classList.remove('hidden');

        // Set the final metric values
        finalPower.textContent = metrics.power;
        finalStability.textContent = metrics.stability;
        finalPiety.textContent = metrics.piety;
        finalWealth.textContent = metrics.wealth;

        // Set reason based on metrics and get detailed ending
        let reason = "";
        let detailedText = "";
        let legacyText = "";

        // Set reason based on cause
        if (cause === "power_low") {
            reason = "You lost all authority. The nobles overthrew you!";
            detailedText = "Your weak leadership led to a noble uprising. They deposed you and sent you into exile.";
            legacyText = determineRulerLegacy("weak");
        } else if (cause === "power_high") {
            reason = "Your absolute power made you a tyrant. You were assassinated!";
            detailedText = "Your iron rule bred fear and resentment. A conspiracy of nobles ended your tyranny with poison.";
            legacyText = determineRulerLegacy("tyrant");
        } else if (cause === "stability_low") {
            reason = "The people revolted against your rule!";
            detailedText = "Neglect and harsh policies sparked a rebellion. The angry mob stormed your palace.";
            legacyText = determineRulerLegacy("hated");
        } else if (cause === "stability_high") {
            reason = "The people loved you so much they established a republic!";
            detailedText = "Your popularity threatened the nobles. They removed you, but the people created a republic in your name.";
            legacyText = determineRulerLegacy("beloved");
        } else if (cause === "piety_low") {
            reason = "The church declared you a heretic and had you executed!";
            detailedText = "Your defiance of church authority led to excommunication. The faithful rose against you.";
            legacyText = determineRulerLegacy("heretic");
        } else if (cause === "piety_high") {
            reason = "The church became too powerful and took control of your kingdom!";
            detailedText = "Religious law replaced royal authority. The Archbishop now rules, with you as a mere figurehead.";
            legacyText = determineRulerLegacy("pious");
        } else if (cause === "wealth_low") {
            reason = "Your kingdom went bankrupt and you were deposed!";
            detailedText = "Empty coffers and mounting debts doomed your reign. Foreign creditors seized the kingdom's assets.";
            legacyText = determineRulerLegacy("poor");
        } else if (cause === "wealth_high") {
            reason = "Your vast wealth attracted invaders who conquered your kingdom!";
            detailedText = "Your riches drew the envy of neighbors. They invaded with overwhelming force and divided your realm.";
            legacyText = determineRulerLegacy("wealthy");
        } else {
            reason = "You died of natural causes after a long reign.";
            detailedText = `After ${metrics.reign_year} years, you passed peacefully. The kingdom mourns a balanced ruler.`;
            legacyText = determineRulerLegacy("balanced");
        }

        // Generate epithet
        const epithet = generateEpithet(cause, metrics);

        // Make sure to display reward information
        gameOverReason.textContent = reason;
        detailedEnding.textContent = detailedText;
        legacyMessage.textContent = legacyText;

        // Format the reward nicely
        const formattedReward = data.reward !== undefined ? data.reward.toFixed(2) : "0.00";
        reignSummary.textContent = `${rulerName} "${epithet}" ruled for ${metrics.reign_year} years. Final reward: ${formattedReward}`;

        // Display the adaptive weights if available
        if (data.new_weights) {
            console.log("New category weights:", data.new_weights);
            // You could display these weights in the UI if desired
        }

        // Clean up for next game
        currentCard = null;

    } catch (error) {
        console.error("Error ending reign:", error);
        gameOverReason.textContent = "Something went wrong when calculating your legacy.";

        // Force display of game over screen even if there was an error
        cardContainer.classList.add('hidden');
        gameOverScreen.classList.remove('hidden');
    }
}

/**
 * Continue the game with a new ruler in the same dynasty
 */
async function continueGame() {
    try {
        // Reset game state but keep session ID for continuity
        gameOver = false;
        trajectory = [];

        // Update dynasty year before starting new reign
        dynastyYear += metrics.reign_year;

        // Clear the ruler name input to allow entering a new name
        rulerNameInput.value = '';

        // Create new game session with the same session ID to maintain reign history
        const response = await fetch(`${API_URL}new_game`, { // Removed the / for consistency
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        const data = await response.json();
        metrics = data.metrics;

        updateMeters();

        // Hide game over screen, show name input and start screen
        gameOverScreen.classList.add('hidden');
        startScreen.classList.remove('hidden');

        // Focus the name input for convenience
        rulerNameInput.focus();

        // Update start screen text for a new ruler
        const startHeader = document.querySelector('#start-screen h2');
        if (startHeader) {
            startHeader.textContent = "Begin Your New Reign";
        }
        startGameButton.textContent = "Continue Dynasty";
    } catch (error) {
        console.error("Error continuing game:", error);
        gameOverReason.textContent = "Something went wrong when starting a new reign.";
    }
}

/**
 * Generate a legacy message based on reign length and ruler type
 */
function determineRulerLegacy(rulerType) {
    const reignLength = metrics.reign_year;
    let legacy = "";

    if (reignLength < 5) {
        legacy = "A brief footnote in history.";
    } else if (reignLength > 30) {
        switch (rulerType) {
            case "balanced":
                legacy = "A golden age of peace and prosperity.";
                break;
            case "tyrant":
                legacy = "A reign of terror that scarred the kingdom.";
                break;
            case "beloved":
                legacy = "A renaissance of culture and progress.";
                break;
            default:
                legacy = "A long reign that changed the kingdom forever.";
        }
    } else {
        switch (rulerType) {
            case "weak":
                legacy = "A ruler who lost control of their court.";
                break;
            case "tyrant":
                legacy = "A harsh ruler who valued power above all.";
                break;
            case "hated":
                legacy = "A name spoken with contempt by the people.";
                break;
            case "beloved":
                legacy = "A ruler cherished in folk songs.";
                break;
            case "heretic":
                legacy = "A cautionary tale of defying the faith.";
                break;
            case "pious":
                legacy = "A ruler who trusted the clergy too much.";
                break;
            case "poor":
                legacy = "A lesson in financial mismanagement.";
                break;
            case "wealthy":
                legacy = "A tale of riches leading to ruin.";
                break;
            case "balanced":
                legacy = "A time of steady progress.";
                break;
            default:
                legacy = `${reignLength} years of mixed successes and failures.`;
        }
    }

    return legacy;
}

/**
 * Generate a fitting epithet for the ruler based on reign outcomes
 */
function generateEpithet(cause, metrics) {
    // Generate epithets based on cause of end and metrics
    if (metrics.reign_year <= 3) {
        return "the Brief";
    }

    if (metrics.reign_year >= 30) {
        return "the Ancient";
    }

    // Causes of death
    switch(cause) {
        case "power_low":
            return "the Weak";
        case "power_high":
            return "the Tyrant";
        case "stability_low":
            return "the Cruel";
        case "stability_high":
            return "the Beloved";
        case "piety_low":
            return "the Heretic";
        case "piety_high":
            return "the Pious";
        case "wealth_low":
            return "the Bankrupt";
        case "wealth_high":
            return "the Opulent";
        case "old_age":
            // For natural death, base epithet on highest stat
            const stats = [
                { name: "the Just", value: metrics.stability },
                { name: "the Mighty", value: metrics.power },
                { name: "the Wise", value: metrics.piety },
                { name: "the Wealthy", value: metrics.wealth }
            ];

            // Sort stats by value (highest first)
            stats.sort((a, b) => b.value - a.value);

            // Return epithet based on highest stat
            return stats[0].name;
    }

    // Default epithet if no specific condition is met
    return "the Monarch";
}

// Check if API is available when page loads
window.addEventListener('load', async () => {
    await checkApiStatus();
});
