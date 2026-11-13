# Cat Behavior Communication Environment

**Author**: [krishpop](https://github.com/krishpop)

## Overview

This environment trains language models to communicate as cats with their caretakers. The model must learn to express cat needs and desires through authentic cat behaviors and sounds, while caretakers attempt to interpret and respond to these communications.

## Environment Structure

### Core Components

- **`cat_server.py`**: Main environment implementation with cat-caretaker interaction logic
- **`catbot_arena.py`**: Alternative arena-style environment (appears to be GSM8k-based placeholder)
- **`cat_behaviors.json`**: Comprehensive database of 35 authentic cat behaviors and their meanings
- **`cat_scenarios.json`**: 61 different scenarios representing cat needs (food, comfort, health, etc.)

### Cat Behaviors Dataset

The environment includes detailed cat behaviors such as:
- **Communication**: Meowing, purring, trilling, yowling, hissing
- **Body Language**: Tail position, ear position, back arching, slow blinking
- **Physical Actions**: Kneading, head butting, rubbing, scratching
- **Behavioral Indicators**: Hiding, following, bringing gifts, litter box changes

### Scenarios

Cats must communicate needs across categories:
- **Nutrition**: Food, water, treats, supplements
- **Health**: Grooming, veterinary care, medication
- **Comfort**: Sleeping areas, temperature, privacy
- **Safety**: Secure environment, escape-proofing
- **Enrichment**: Play, mental stimulation, social interaction

## Training Mechanics

### Communication Rules
- **No English**: Cats cannot speak human language
- **No Emojis**: Must use realistic cat sounds and behaviors
- **Format**: `Sound! (Context)` or `~Silent~ (Context)`
- **Examples**:
  - `Mew! (Looks up at you)`
  - `Hiss! (Stares at the litterbox)`
  - `~Silent~ (Rubs against your legs)`

### Scoring System

The environment uses a unique "purrfect" evaluation:
- **Purr**: Perfect caretaker response (1.0 score) - reserved for exceptional care
- **Meow**: Room for improvement (0.0 score) - indicates unmet needs

The cat evaluates whether the caretaker addressed all needs perfectly with no possible improvements.

## Features

- **Multi-turn Interaction**: 5-turn conversations between cat and caretaker
- **Authentic Behavior Modeling**: Based on real cat behavioral science
- **Nuanced Evaluation**: Cats are trained to be discerning critics
- **Rich Scenario Diversity**: Covers full spectrum of cat care needs

## Usage

```bash
python environments/community/cat_behavior_env/cat_server.py
```

## Requirements

- Standard Atropos dependencies
- JSON file handling
- Multi-turn conversation support

## Status

⚠️ **Development Note**: This environment appears to be in active development. The main server file contains some placeholder code from GSM8k environment that may need refinement for full cat behavior functionality.

## Research Applications

This environment is valuable for:
- **Multi-modal Communication**: Training models to express needs without direct language
- **Behavioral Modeling**: Understanding animal-human interaction patterns
- **Empathy Training**: Teaching AI to recognize and respond to non-verbal communication
- **Creative AI**: Developing models that can roleplay and stay in character
