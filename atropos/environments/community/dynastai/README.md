# DynastAI

A medieval kingdom management game with an adaptive reinforcement learning environment.

## Overview

DynastAI is an Atropos-compatible Python RL environment integrated with a web frontend. The game challenges players to rule a medieval kingdom by balancing four key metrics:

- **Power** - Royal authority
- **Stability** - Population happiness
- **Piety** - Religious influence
- **Wealth** - Kingdom finances

Each turn, players are presented with scenario cards generated using Qwen 1.7B via OpenRouter. Every decision affects metrics and contributes to an adaptive reward system that evolves gameplay based on previous reigns.

## Key Features

- **Atropos-Compatible Environment**: Implements the BaseEnv interface for training with Atropos
- **FastAPI Backend**: REST endpoints for game state management
- **HTML/CSS/JS Frontend**: Modern, responsive web interface
- **Adaptive Rewards**: Reward calculation that adapts to player choices and outcomes
- **Card Generation**: Uses both pre-defined cards from JSON and dynamic generation via Qwen 1.7B
- **OpenRouter Integration**: Dynamic card generation using Qwen 1.7B language model (optional)

## Project Structure

```
dynastai/
│
├── src/
│   ├── __init__.py
│   ├── dynastai_env.py      # Atropos environment class
│   ├── config.py            # Configuration management
│   ├── game_logic.py        # Core game mechanics
│   ├── util.py              # Utility functions
│   ├── data/                # Game data storage
│   │   ├── game_data.json   # Game configuration data
│   │   └── cards.json       # Card templates
│   └── web/                 # Web interface
│       ├── __init__.py
│       ├── api.py           # FastAPI endpoints
│       ├── server.py        # Server initialization
│       └── static/          # Frontend assets
│           ├── css/
│           ├── js/
│           └── index.html
│
├── dynastai_server.py       # Main server entry point
├── dynastai_local_server.py # Local development server
├── test_dynastai_env.py     # Environment test script
├── test_dynastai_api.py     # API test script
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Adaptive Reward Mechanism

DynastAI implements a novel adaptive reward mechanism that evolves based on gameplay:

```
R = power_final * P + stability_final * S + piety_final * Pi + wealth_final * W
```

Where:
- `power_final`, `stability_final`, `piety_final`, `wealth_final` are the final metric values
- `P`, `S`, `Pi`, `W` are the counts of cards played in each category

This creates a dynamic reward system that adapts to each player's style and decisions.

## Getting Started

### Prerequisites

- Python 3.8+
- OpenRouter API key (set in `.env` file)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dynastai.git
   cd dynastai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. The system includes a pre-configured `cards.json` file with 400+ scenario cards. If you don't set an OpenRouter API key, the game will exclusively use these pre-defined cards.

### Running the Server

To run the full server with API endpoints:

```bash
python dynastai_server.py
```

For local development with both API and web server:

```bash
python dynastai_local_server.py
```

### Testing the Installation

DynastAI includes several test scripts to verify that everything is working correctly:

1. **Installation Verification**: Check if all required packages are installed properly
   ```bash
   python verify_install.py
   ```

2. **Quick Test**: Run the game with the web interface
   ```bash
   python run_dynastai.py
   ```

3. **Card Generation Test**: Test that cards are generated correctly
   ```bash
   python test_card_generation.py
   ```

4. **Environment Test**: Test the Atropos environment integration
   ```bash
   python test_dynastai_env.py
   ```

Then access the web interface at http://localhost:3000

## API Endpoints

The game exposes the following REST API endpoints:

- `GET /api/`: Root endpoint with API status
- `POST /api/new_game`: Create a new game session
- `GET /api/state/{session_id}`: Get the current game state
- `POST /api/generate_card`: Generate a new scenario card
- `POST /api/card_choice`: Submit a player decision
- `POST /api/end_reign`: End a reign and calculate final rewards

## Integration with Atropos

DynastAI fully integrates with the Atropos RL framework. There are two ways to use it:

### Method 1: Using the Environment Entry Point

```bash
# From the atropos root directory
python environments/dynastai_environment.py serve --slurm False
```

### Method 2: Direct Integration in Trainers

```python
from atroposlib.envs.base import BaseEnv, ServerBaseline
from environments.dynastai.src.dynastai_env import DynastAIEnv, DynastAIEnvConfig

# Create and configure environment
config = DynastAIEnvConfig(
    api_host="localhost",
    api_port=9001,
    web_enabled=True,
    web_port=3000
)
server_configs = ServerBaseline()
env = DynastAIEnv(config, server_configs)

# Use with Atropos training
observation = await env.reset()
action = {"session_id": observation["session_id"], "choice": "yes"}
observation, reward, done, info = await env.step(action)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the legacy command-line DynastAI game
- Uses Qwen 1.7B from OpenRouter for card generation
- Built with FastAPI, Uvicorn, and modern web technologies
## Using with Atropos

To use DynastAI with Atropos for training RL models:

```python
from atroposlib.envs.base import BaseEnv
from atroposlib.envs.server_handling.server_baseline import ServerBaseline
from src.dynastai_env import DynastAIEnv, DynastAIEnvConfig

# Create and configure environment
config = DynastAIEnvConfig(
    api_host="localhost",
    api_port=9001,
    web_ui=True,
    web_port=3000,
    openrouter_api_key="your_api_key"
)
server_configs = ServerBaseline()
env = DynastAIEnv(config, server_configs)

# Use with Atropos training
observation = await env.reset()
action = {"session_id": observation["session_id"], "choice": "yes"}
observation, reward, done, info = await env.step(action)
```

## Testing

To run the local development server and test the game:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your OpenRouter API key is set in the `.env` file or environment:
   ```bash
   export OPENROUTER_API_KEY=your_api_key_here
   ```

3. Run the local development server:
   ```bash
   python dynastai_local_server.py
   ```

4. Open your browser and navigate to `http://localhost:3000` to play the game

## Future Enhancements

Potential improvements for future versions:

- Enhanced card generation with more varied scenarios
- Multi-agent gameplay for competitive kingdom management
- Persistent game state and user accounts
- More complex game mechanics (resource management, diplomacy)
- Improved UI with animations and visual history

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
