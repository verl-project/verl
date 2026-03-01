## Pokemon Showdown Environment

This is a Pokemon Showdown environment that allows you to play Pokemon battles with the Pokemon Showdown Battle Simulator.

### Overview Video
Video: TBD

### Quickstart
1. Set up the Pokemon Showdown Battle Simulator environment. This is necessary for the pokemon players to be able to connect to the battle simulator.
```
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
node pokemon-showdown start --no-security
```

2. Run the RL trainer
```
python environments/game_environments/pokemon-showdown/pokemon_environment.py process
```


TLDR: GPT player vs Max Damage Player
- GPT player is a player that uses a GPT model to decide what to do in a battle.
   - GPT player received battle history state at the end (binary win/loss) as feedback to the Atropos RL environment.
- Max Damage Player is a player that always pick the maximum damage move.

### Wandb Run

Wandb Run: https://wandb.ai/ajayuppili/atropos-environments_game_environments_pokemon-showdown?nw=nwuserajayuppili
