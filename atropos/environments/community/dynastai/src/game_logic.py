"""
Game Logic for DynastAI

This module handles the core game mechanics:
- Game state tracking
- Card generation via OpenRouter/Qwen
- Decision effects processing
- Win/loss condition checking
"""

import json
import os
import random
from typing import Dict, Tuple

import requests
from dotenv import load_dotenv

# Import UUID - it's a built-in module in Python
try:
    import uuid
except ImportError:
    # If somehow uuid is not available, create a simple UUID generator
    class FallbackUUID:
        @staticmethod
        def uuid4():
            # Simple fallback for uuid4 (not as good but functional)
            return f"uuid-{random.randint(10000000, 99999999)}"

    uuid = FallbackUUID

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class GameState:
    """
    Represents the state of a single DynastAI game session.
    """

    def __init__(self):
        # Initialize game metrics (0-100 scale)
        self.power = 50  # Royal authority/Power
        self.stability = 50  # Population happiness/Stability
        self.piety = 50  # Religious influence/Piety
        self.wealth = 50  # Kingdom finances/Wealth

        # Game state tracking
        self.reign_year = 1
        self.current_card = None
        self.card_history = []  # List of played cards
        self.choice_history = []  # List of yes/no choices made
        self.previous_reigns = (
            []
        )  # Track previous reigns in this session for continuity

        # Category counts for adaptive reward calculation
        self.category_counts = {"power": 0, "stability": 0, "piety": 0, "wealth": 0}

    def get_metrics(self) -> Dict[str, int]:
        """Return the current game metrics"""
        return {
            "power": self.power,
            "stability": self.stability,
            "piety": self.piety,
            "wealth": self.wealth,
            "reign_year": self.reign_year,
        }

    def get_category_counts(self) -> Dict[str, int]:
        """Return the count of cards played by category"""
        return self.category_counts

    def record_card_play(self, card, choice):
        """Record a card play and choice"""
        if card and "category" in card:
            # Increment the category count
            category = card["category"]
            if category in self.category_counts:
                self.category_counts[category] += 1

        # Store the card and choice in history
        self.card_history.append(card)
        self.choice_history.append(choice)

        # Increment reign year
        self.reign_year += 1


def generate_card(
    metrics: Dict[str, int], category_weights: Dict[str, int], previous_reigns=None
) -> Dict:
    """
    Generate a new card using either the cards.json file or the OpenRouter API (Qwen 1.7B)

    Parameters:
    - metrics: Current game metrics
    - category_weights: Weights for selecting card categories
    - previous_reigns: Optional list of previous reign outcomes to influence card generation

    Returns:
    - card: A card object with text, options and effects
    """
    # Try to load cards from the cards.json file first
    try:
        if previous_reigns and len(previous_reigns) > 0:
            # If there are previous reigns, occasionally generate continuity cards
            if random.random() < 0.2:  # 20% chance of generating a continuity card
                return generate_continuity_card(
                    metrics, category_weights, previous_reigns
                )

        return select_card_from_file(metrics, category_weights)
    except Exception as e:
        print(f"Error selecting card from file: {e}")
        # Fall back to OpenRouter API or mock cards
        return generate_api_card(metrics, category_weights, previous_reigns)


def select_card_from_file(
    metrics: Dict[str, int], category_weights: Dict[str, int]
) -> Dict:
    """
    Select a card from the cards.json file based on category weights
    """
    # Path to cards.json file
    cards_file = os.path.join(os.path.dirname(__file__), "data", "cards.json")

    if not os.path.exists(cards_file):
        raise FileNotFoundError(f"Cards file not found: {cards_file}")

    with open(cards_file, "r") as f:
        cards_data = json.load(f)

    # Check if it's the new format (direct array) or old format (with "cards" key)
    if isinstance(cards_data, list):
        cards = cards_data
    elif "cards" in cards_data and cards_data["cards"]:
        cards = cards_data["cards"]
    else:
        raise ValueError("No valid cards found in cards.json")

    if not cards:
        raise ValueError("Empty card list in cards.json")

    # Map the new format fields to the expected format
    formatted_cards = []
    for card in cards:
        # Check if it's already in the expected format
        if all(key in card for key in ["text", "yes_option", "no_option", "category"]):
            formatted_cards.append(card)
        else:
            # Convert new format to expected format
            formatted_card = {
                "id": card.get("ID", f"card_{str(uuid.uuid4())[:8]}"),
                "text": card.get("Prompt", ""),
                "yes_option": card.get("Left_Choice", "Yes"),
                "no_option": card.get("Right_Choice", "No"),
                "category": determine_category_from_effects(card),
                "effects": {
                    "yes": {
                        "power": int(card.get("Left_Power", 0)),
                        "stability": int(card.get("Left_Stability", 0)),
                        "piety": int(card.get("Left_Piety", 0)),
                        "wealth": int(card.get("Left_Wealth", 0)),
                    },
                    "no": {
                        "power": int(card.get("Right_Power", 0)),
                        "stability": int(card.get("Right_Stability", 0)),
                        "piety": int(card.get("Right_Piety", 0)),
                        "wealth": int(card.get("Right_Wealth", 0)),
                    },
                },
                "character_name": card.get("Character", "Royal Advisor"),
            }
            formatted_cards.append(formatted_card)

    # Select a category based on weights
    categories = list(category_weights.keys())
    weights = [category_weights[cat] for cat in categories]
    total_weight = sum(weights)

    # Normalize weights to avoid issues if weights are too small
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = [1 / len(categories)] * len(categories)

    category = random.choices(categories, weights=normalized_weights, k=1)[0]

    # Filter cards by selected category if using the new format
    category_cards = [
        card
        for card in formatted_cards
        if determine_category_from_card(card) == category
    ]

    # If no cards for this category, use all cards
    if not category_cards:
        category_cards = formatted_cards

    # Select a random card
    selected_card = random.choice(category_cards)

    return selected_card


def determine_category_from_effects(card):
    """Determine the category from a card's effects"""
    metrics = {
        "power": abs(int(card.get("Left_Power", 0)))
        + abs(int(card.get("Right_Power", 0))),
        "stability": abs(int(card.get("Left_Stability", 0)))
        + abs(int(card.get("Right_Stability", 0))),
        "piety": abs(int(card.get("Left_Piety", 0)))
        + abs(int(card.get("Right_Piety", 0))),
        "wealth": abs(int(card.get("Left_Wealth", 0)))
        + abs(int(card.get("Right_Wealth", 0))),
    }

    # Return the metric with the highest absolute effect
    return max(metrics.items(), key=lambda x: x[1])[0]


def determine_category_from_card(card):
    """Determine the category from a card object"""
    if "category" in card:
        return card["category"]

    if "effects" in card:
        effects = card["effects"]
        total_effects = {}

        for choice in ["yes", "no"]:
            if choice in effects:
                for metric, value in effects[choice].items():
                    if metric not in total_effects:
                        total_effects[metric] = 0
                    total_effects[metric] += abs(value)

        if total_effects:
            return max(total_effects.items(), key=lambda x: x[1])[0]

    return "stability"  # Default category


def generate_continuity_card(
    metrics: Dict[str, int], category_weights: Dict[str, int], previous_reigns
) -> Dict:
    """
    Generate a card that references the previous reign's outcome for continuity
    """
    # Get the most recent reign
    last_reign = previous_reigns[-1]
    cause_of_end = last_reign.get("cause_of_end", "unknown")
    reign_length = last_reign.get("reign_length", 0)

    # Generate a unique card ID
    # Generate a unique card ID for continuity cards
    # card_id = f"continuity_card_{str(uuid.uuid4())[:8]}"

    # Create a continuity event based on how the previous reign ended
    if OPENROUTER_API_KEY:
        try:
            # Create a prompt with previous reign details
            prompt_system = (
                "You are generating JSON event cards for a medieval kingdom management game "
                "that reference past gameplay."
            )
            prompt_user = (
                f"Create a card that references the previous ruler's downfall. "
                f"The previous ruler reigned for {reign_length} years and was ended due to '{cause_of_end}'. "
                f"Output ONLY a JSON event card object where the scenario references the previous ruler's fate."
            )
            prompt = f'System: "{prompt_system}"\n\nUser: "{prompt_user}"'
            # Send to OpenRouter API
            response = call_openrouter(prompt)
            try:
                card_data = json.loads(response)
                if validate_card(card_data):
                    return card_data
            except json.JSONDecodeError:
                # Fall back to mock continuity card on error
                pass
        except Exception as e:
            print(f"Error generating continuity card: {e}")

    # If API fails or isn't available, use mock continuity card
    return generate_mock_continuity_card(metrics, cause_of_end, reign_length)


def generate_mock_continuity_card(metrics, cause_of_end, reign_length):
    """Generate a mock continuity card based on previous reign"""
    category = "stability"  # Default category

    # Create text based on previous reign's end cause
    if cause_of_end == "power_low":
        text = (
            f"Advisors remind you that the previous ruler was overthrown by nobles "
            f"after {reign_length} years of weak leadership."
        )
        category = "power"
    elif cause_of_end == "power_high":
        text = (
            f"You visit the tomb of your predecessor, an infamous tyrant who was "
            f"assassinated after {reign_length} years of iron-fisted rule."
        )
        category = "power"
    elif cause_of_end == "stability_low":
        text = (
            f"The kingdom still bears scars from the peasant revolt that deposed "
            f"the previous monarch after {reign_length} years."
        )
        category = "stability"
    elif cause_of_end == "stability_high":
        text = (
            f"Citizens talk fondly of your predecessor who was so loved they "
            f"established a republic after {reign_length} years."
        )
        category = "stability"
    elif cause_of_end == "piety_low":
        text = (
            f"The church reminds you that the previous ruler was declared a heretic "
            f"and executed after {reign_length} years."
        )
        category = "piety"
    elif cause_of_end == "piety_high":
        text = (
            f"The Archbishop speaks of reclaiming authority that the church gained "
            f"under the previous ruler's {reign_length}-year reign."
        )
        category = "piety"
    elif cause_of_end == "wealth_low":
        text = (
            f"The treasury still suffers from the bankruptcy that ended "
            f"the previous {reign_length}-year reign."
        )
        category = "wealth"
    elif cause_of_end == "wealth_high":
        text = (
            f"Neighboring kingdoms remain wary after invading to seize the vast wealth "
            f"accumulated during the previous {reign_length}-year reign."
        )
        category = "wealth"
    else:
        text = (
            f"Your predecessor ruled for {reign_length} years before their demise. "
            f"Their decisions still affect the kingdom."
        )

    # Generate effect values
    effects = {
        "yes": {category: 5, "stability": -2},
        "no": {category: -2, "stability": 2},
    }

    # Character names for continuity cards
    continuity_characters = {
        "power": "Royal Historian",
        "stability": "Elder Villager",
        "piety": "Ancient Priest",
        "wealth": "Treasury Keeper",
    }

    return {
        "id": f"continuity_{str(uuid.uuid4())[:8]}",
        "text": text,
        "yes_option": "Learn from their mistakes",
        "no_option": "Forge your own path",
        "effects": effects,
        "category": category,
        "character_name": continuity_characters.get(category, "Court Advisor"),
    }


def generate_api_card(
    metrics: Dict[str, int], category_weights: Dict[str, int], previous_reigns=None
) -> Dict:
    """
    Generate a new card using the OpenRouter API (Qwen 1.7B)
    """
    # Select a category based on weights
    categories = list(category_weights.keys())
    weights = [category_weights[cat] for cat in categories]
    total_weight = sum(weights)

    # Normalize weights to avoid issues if weights are too small
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = [1 / len(categories)] * len(categories)

    category = random.choices(categories, weights=normalized_weights, k=1)[0]

    # Generate a unique card ID
    card_id = f"card_{str(uuid.uuid4())[:8]}"

    # Add context from previous reigns if available
    previous_reign_context = ""
    if previous_reigns and len(previous_reigns) > 0:
        last_reign = previous_reigns[-1]
        reign_length = last_reign.get("reign_length", 0)
        cause = last_reign.get("cause_of_end", "unknown")
        previous_reign_context = (
            f"Note: The previous ruler reigned for {reign_length} years "
            f"before falling due to {cause}. "
        )

    # Create a card prompt
    prompt_system = (
        "You are generating JSON event cards for a medieval kingdom management game."
    )

    metrics_str = (
        f"Power:{metrics['power']}, Stability:{metrics['stability']}, "
        f"Piety:{metrics['piety']}, Wealth:{metrics['wealth']}"
    )

    prompt_user = (
        f"Create a {category} focused event card for a medieval ruler.\n"
        f"{previous_reign_context}Current metrics: {metrics_str}.\n"
        f"Output ONLY a JSON event card object like this:\n"
        f"{{\n"
        f"  'id': '{card_id}',\n"
        f"  'text': 'Card scenario description...',\n"
        f"  'yes_option': 'First option text...',\n"
        f"  'no_option': 'Second option text...',\n"
        f"  'effects': {{\n"
        f"    'yes': {{'power': int, 'stability': int, 'piety': int, 'wealth': int}},\n"
        f"    'no': {{'power': int, 'stability': int, 'piety': int, 'wealth': int}}\n"
        f"  }},\n"
        f"  'category': '{category}'\n"
        f"}}\n"
        f"Make sure the effects are integers between -20 and +20, with most being -10 to +10."
    )

    prompt = f'System: "{prompt_system}"\n\nUser: "{prompt_user}"'

    # Call the OpenRouter API
    try:
        if not OPENROUTER_API_KEY:
            # If no API key, generate a mock card for testing
            return generate_mock_card(metrics, category)

        response = call_openrouter(prompt)

        # Parse the JSON content from the response
        try:
            card_data = json.loads(response)
            # Validate that the card has all required fields
            if validate_card(card_data):
                return card_data
            else:
                # If validation fails, fall back to a mock card
                return generate_mock_card(metrics, category)
        except json.JSONDecodeError:
            # If JSON parsing fails, fall back to a mock card
            print(f"Error parsing card JSON: {response}")
            return generate_mock_card(metrics, category)

    except Exception as e:
        print(f"Error generating card: {e}")
        # Fall back to a mock card in case of any error
        return generate_mock_card(metrics, category)


def call_openrouter(prompt, model="qwen/Qwen1.5-7B"):
    """
    Send a prompt to the OpenRouter API and return the response
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"].strip()


def validate_card(card: Dict) -> bool:
    """
    Validate that the card has all required fields and proper structure
    """
    required_fields = ["id", "text", "yes_option", "no_option", "effects", "category"]
    if not all(field in card for field in required_fields):
        return False

    if not all(choice in card["effects"] for choice in ["yes", "no"]):
        return False

    metrics = ["power", "stability", "piety", "wealth"]
    for choice in ["yes", "no"]:
        if not all(metric in card["effects"][choice] for metric in metrics):
            return False

    return True


def generate_mock_card(metrics: Dict[str, int], category: str) -> Dict:
    """
    Generate a mock card for testing when other card sources are unavailable
    First tries to select from the built-in templates, then generates a new card
    """
    # Try to use the card file first if it exists
    try:
        # Path to cards.json file
        cards_file = os.path.join(os.path.dirname(__file__), "data", "cards.json")

        if os.path.exists(cards_file):
            with open(cards_file, "r") as f:
                cards_data = json.load(f)

            # Handle both new format (direct array) and old format (with "cards" key)
            cards = []
            if isinstance(cards_data, list):
                cards = cards_data
            elif "cards" in cards_data and cards_data["cards"]:
                cards = cards_data["cards"]

            if cards:
                # Determine category for each card (if needed)
                formatted_cards = []
                for card in cards:
                    # Convert card if needed
                    if all(
                        key in card
                        for key in ["text", "yes_option", "no_option", "category"]
                    ):
                        formatted_cards.append(card)
                    else:
                        # Convert new format to expected format
                        formatted_card = {
                            "id": card.get("ID", f"card_{str(uuid.uuid4())[:8]}"),
                            "text": card.get("Prompt", ""),
                            "yes_option": card.get("Left_Choice", "Yes"),
                            "no_option": card.get("Right_Choice", "No"),
                            "category": determine_category_from_effects(card),
                            "effects": {
                                "yes": {
                                    "power": int(card.get("Left_Power", 0)),
                                    "stability": int(card.get("Left_Stability", 0)),
                                    "piety": int(card.get("Left_Piety", 0)),
                                    "wealth": int(card.get("Left_Wealth", 0)),
                                },
                                "no": {
                                    "power": int(card.get("Right_Power", 0)),
                                    "stability": int(card.get("Right_Stability", 0)),
                                    "piety": int(card.get("Right_Piety", 0)),
                                    "wealth": int(card.get("Right_Wealth", 0)),
                                },
                            },
                            "character_name": card.get("Character", "Royal Advisor"),
                        }
                        formatted_cards.append(formatted_card)

                # Filter cards by selected category
                category_cards = [
                    card
                    for card in formatted_cards
                    if determine_category_from_card(card) == category
                ]

                # If no cards for this category, use all cards
                if not category_cards:
                    category_cards = formatted_cards

                # Select a random card
                if category_cards:
                    return random.choice(category_cards)
    except Exception as e:
        print(f"Error selecting from card file: {e}")

    # If we can't use the card file, generate a random card
    effect_range = (-10, 10)

    # Create effects for yes and no choices
    yes_effects = {
        metric: random.randint(*effect_range)
        for metric in ["power", "stability", "piety", "wealth"]
    }
    no_effects = {
        metric: random.randint(*effect_range)
        for metric in ["power", "stability", "piety", "wealth"]
    }

    # Ensure category effect is positive for 'yes' and negative for 'no'
    yes_effects[category] = random.randint(5, 15)
    no_effects[category] = random.randint(-15, -5)

    # Character names for each category
    character_names = {
        "power": [
            "General Blackstone",
            "Captain of the Guard",
            "Lord Commander",
            "Duke of Westbridge",
        ],
        "stability": ["Village Elder", "Town Crier", "Guild Master", "Court Jester"],
        "piety": [
            "High Priest",
            "Bishop Aurelius",
            "Sister Margaery",
            "Oracle of the Temple",
        ],
        "wealth": [
            "Royal Treasurer",
            "Master of Coin",
            "Merchant Guild Leader",
            "Foreign Diplomat",
        ],
    }

    # Pick a random character name for the category
    character_name = random.choice(character_names.get(category, ["Royal Advisor"]))

    # Create mock scenarios based on category
    scenarios = {
        "power": "The Royal General requests funds to expand the army.",
        "stability": "Peasants from the northern province complain about high taxes.",
        "piety": "The Cardinal proposes building a new cathedral in the capital.",
        "wealth": "The Master of Coin suggests a new trade agreement with a neighboring kingdom.",
    }

    yes_options = {
        "power": "Strengthen our military",
        "stability": "Reduce their tax burden",
        "piety": "Fund the cathedral project",
        "wealth": "Approve the trade agreement",
    }

    no_options = {
        "power": "Maintain current military size",
        "stability": "Keep the tax rates as they are",
        "piety": "Reject the cathedral project",
        "wealth": "Decline the trade agreement",
    }

    return {
        "id": f"mock_card_{str(uuid.uuid4())[:8]}",
        "text": scenarios.get(
            category, f"A {category} related scenario has emerged in your kingdom."
        ),
        "yes_option": yes_options.get(category, "Approve"),
        "no_option": no_options.get(category, "Decline"),
        "effects": {"yes": yes_effects, "no": no_effects},
        "category": category,
        "character_name": character_name,
    }


def apply_choice_effects(
    game_state: GameState, choice: str
) -> Tuple[bool, Dict[str, int], Dict]:
    """
    Apply the effects of a player's choice to the game state

    Parameters:
    - game_state: The current game state
    - choice: "yes" or "no"

    Returns:
    - is_game_over: Whether the game has ended
    - new_metrics: Updated metrics
    - effects: The effects that were applied
    """
    if not game_state.current_card:
        raise ValueError("No current card in game state")

    # Get the effects based on the choice
    if choice not in ["yes", "no"]:
        raise ValueError(f"Invalid choice: {choice}. Must be 'yes' or 'no'")

    effects = game_state.current_card["effects"][choice]

    # Apply effects to game metrics
    game_state.power = max(0, min(100, game_state.power + effects["power"]))
    game_state.stability = max(0, min(100, game_state.stability + effects["stability"]))
    game_state.piety = max(0, min(100, game_state.piety + effects["piety"]))
    game_state.wealth = max(0, min(100, game_state.wealth + effects["wealth"]))

    # Record the card play and update category counts
    game_state.record_card_play(game_state.current_card, choice)

    # Check for game over conditions
    is_game_over = check_game_over(game_state)

    # Return updated metrics
    new_metrics = game_state.get_metrics()

    return is_game_over, new_metrics, effects


def check_game_over(game_state: GameState) -> bool:
    """
    Check if the game is over based on the current metrics

    The game ends if any metric reaches 0 or 100
    """
    metrics = [
        game_state.power,
        game_state.stability,
        game_state.piety,
        game_state.wealth,
    ]

    return any(metric <= 0 or metric >= 100 for metric in metrics)
