# Google Maps Integration for Stone AIOS

This module enables Stone AIOS to provide location information, directions, and other map-related services using Google Maps.

## Features

- **Location Search**: Find detailed information about places
- **Directions**: Get directions between locations with different transport modes
- **Distance Calculation**: Calculate distances and travel times
- **Place Details**: Get information about businesses, landmarks, etc.

## Requirements

- Google Cloud account with Maps API enabled
- Google Maps API key with the following APIs enabled:
  - Maps JavaScript API
  - Places API
  - Directions API
  - Distance Matrix API
  - Geocoding API

## Configuration

1. Set up a Google Cloud project and enable the necessary Google Maps APIs
2. Create an API key and restrict it to the Google Maps APIs
3. Configure your `.env` file with:

```
GOOGLE_MAPS_API_KEY="your_google_maps_api_key"
```

## Integration Details

The Google Maps integration uses an MCP server implemented in JavaScript that runs as a subprocess when needed. This ensures the maps service only consumes resources when actively being used.

### Supported Commands

- "Where is the Eiffel Tower?" - Get location information
- "How do I get from New York to Boston?" - Get directions
- "How far is it from Los Angeles to San Francisco?" - Calculate distances
- "What restaurants are near me?" - Find nearby places (requires user location)

## Implementation Notes

The integration is implemented in `agents/stone_agent.py` within the `delegate_to_go_agent` function, which handles:

1. Verifying the presence of a valid Google Maps API key
2. Starting the Maps MCP server as a subprocess
3. Processing the query through Claude with map tools access
4. Returning structured results with location information

## Testing

Tests for the Google Maps integration are available in `tests/ai/test_maps_integration.py`.
