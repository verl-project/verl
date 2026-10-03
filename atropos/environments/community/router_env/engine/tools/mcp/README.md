# MCP Servers Directory

This directory contains all Model Context Protocol (MCP) servers used by the Stone AIOS engine.

## Directory Structure

- `perplexity/`: Perplexity API integration for web search
  - `perplexity-ask/`: The MCP server for Perplexity's Ask functionality
- `spotify/`: Spotify API integration for music playback and control
- Additional MCP servers can be added in their own directories

## Important Notes

1. The code in `engine/agents/` is configured to look for MCP servers in this exact location (`engine/tools/mcp/`).

2. The MCP servers are initially defined as git submodules in `stone_aios/tools/mcp/` but are copied here during setup:
   - The `start.sh` script copies the servers from their submodule location to this directory.
   - It then builds the servers in this location to make them available to the engine.

3. When adding new MCP servers:
   - Add them as submodules in `stone_aios/tools/mcp/`
   - Update `start.sh` to copy and build them in `engine/tools/mcp/`
   - Update the agent code to look for them in this location

## Usage

The MCP servers are automatically started when needed by the engine's agent code through the `run_mcp_servers()` context manager in Pydantic-AI.

# Model Context Protocol (MCP) Submodules

This directory contains various Model Context Protocol (MCP) implementations that Stone AIOS uses to interact with different services.

## Submodules

### Perplexity MCP
- Repository: https://github.com/ppl-ai/modelcontextprotocol.git
- Purpose: Provides integration with Perplexity's search functionality

### Spotify MCP
- Repository: https://github.com/varunneal/spotify-mcp.git
- Purpose: Enables interaction with Spotify's music service

### Basic Memory MCP
- Repository: https://github.com/basicmachines-co/basic-memory.git
- Purpose: Provides memory capabilities for agents

### Google Maps MCP
- Repository: (Google Maps implementation)
- Purpose: Enables interaction with Google Maps for location-based services

### Google Calendar MCP
- Repository: https://github.com/nspady/google-calendar-mcp.git
- Purpose: Provides integration with Google Calendar for managing events and schedules

### Calculator MCP Server
- Repository: https://github.com/githejie/mcp-server-calculator.git
- Purpose: Offers calculation capabilities through the MCP protocol

## Usage

These submodules are reference implementations that can be used by Stone AIOS tools. To update all submodules, run:

```bash
git submodule update --init --recursive
```

## Adding New MCP Implementations

To add a new MCP implementation:

1. Add it as a git submodule:
   ```
   git submodule add <repository-url> tools/mcp/<service-name>
   ```

2. Update this README.md file to include information about the new submodule
