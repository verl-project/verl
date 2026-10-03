#!/bin/bash

# Activate the Python virtual environment
source .router_env/bin/activate

# Install required dependencies if needed
pip install librosa numpy

# Set the LiveKit server URL and API keys (update these with your actual values)
export LIVEKIT_URL="pass"
export LIVEKIT_API_KEY="pass"
export LIVEKIT_API_SECRET="pass"

# Set OpenAI API key (replace with your actual key)
export OPENAI_API_KEY="pass"

# Set the room name to match the one in token-server.js
export LIVEKIT_ROOM="stone-router-voice-agent"

# Start the agent
cd engine
python agents/stone_agent.py
