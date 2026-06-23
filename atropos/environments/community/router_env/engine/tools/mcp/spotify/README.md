<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 10px;">
<img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg" width="30" height="30">
<h1>Spotify MCP Server</h1>
</div>

A lightweight [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that enables AI assistants like Cursor & Claude to control Spotify playback and manage playlists.

<details>
<summary>Contents</summary>

- [Example Interactions](#example-interactions)
- [Tools](#tools)
  - [Read Operations](#read-operations)
  - [Play / Create Operations](#play--create-operations)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Creating a Spotify Developer Application](#creating-a-spotify-developer-application)
  - [Spotify API Configuration](#spotify-api-configuration)
  - [Authentication Process](#authentication-process)
- [Integrating with Claude Desktop, Cursor, and VsCode (Cline)](#integrating-with-claude-desktop-and-cursor)
</details>

## Example Interactions

- _"Play Elvis's first song"_
- _"Create a Taylor Swift / Slipknot fusion playlist"_
- _"Copy all the techno tracks from my workout playlist to my work playlist"_

## Tools

### Read Operations

1. **searchSpotify**

   - **Description**: Search for tracks, albums, artists, or playlists on Spotify
   - **Parameters**:
     - `query` (string): The search term
     - `type` (string): Type of item to search for (track, album, artist, playlist)
     - `limit` (number, optional): Maximum number of results to return (10-50)
   - **Returns**: List of matching items with their IDs, names, and additional details
   - **Example**: `searchSpotify("bohemian rhapsody", "track", 20)`

2. **getNowPlaying**

   - **Description**: Get information about the currently playing track on Spotify
   - **Parameters**: None
   - **Returns**: Object containing track name, artist, album, playback progress, duration, and playback state
   - **Example**: `getNowPlaying()`

3. **getMyPlaylists**

   - **Description**: Get a list of the current user's playlists on Spotify
   - **Parameters**:
     - `limit` (number, optional): Maximum number of playlists to return (default: 20)
     - `offset` (number, optional): Index of the first playlist to return (default: 0)
   - **Returns**: Array of playlists with their IDs, names, track counts, and public status
   - **Example**: `getMyPlaylists(10, 0)`

4. **getPlaylistTracks**

   - **Description**: Get a list of tracks in a specific Spotify playlist
   - **Parameters**:
     - `playlistId` (string): The Spotify ID of the playlist
     - `limit` (number, optional): Maximum number of tracks to return (default: 100)
     - `offset` (number, optional): Index of the first track to return (default: 0)
   - **Returns**: Array of tracks with their IDs, names, artists, album, duration, and added date
   - **Example**: `getPlaylistTracks("37i9dQZEVXcJZyENOWUFo7")`

5. **getRecentlyPlayed**

   - **Description**: Retrieves a list of recently played tracks from Spotify.
   - **Parameters**:
     - `limit` (number, optional): A number specifying the maximum number of tracks to return.
   - **Returns**: If tracks are found it returns a formatted list of recently played tracks else a message stating: "You don't have any recently played tracks on Spotify".
   - **Example**: `getRecentlyPlayed({ limit: 10 })`

6. **getRecentlyPlayed**

   - **Description**: Retrieves a list of recently played tracks from Spotify.
   - **Parameters**:
     - `limit` (number, optional): A number specifying the maximum number of tracks to return.
   - **Returns**: If tracks are found it returns a formatted list of recently played tracks else a message stating: "You don't have any recently played tracks on Spotify".
   - **Example**: `getRecentlyPlayed({ limit: 10 })`

### Play / Create Operations

1. **playMusic**

   - **Description**: Start playing a track, album, artist, or playlist on Spotify
   - **Parameters**:
     - `uri` (string, optional): Spotify URI of the item to play (overrides type and id)
     - `type` (string, optional): Type of item to play (track, album, artist, playlist)
     - `id` (string, optional): Spotify ID of the item to play
     - `deviceId` (string, optional): ID of the device to play on
   - **Returns**: Success status
   - **Example**: `playMusic({ uri: "spotify:track:6rqhFgbbKwnb9MLmUQDhG6" })`
   - **Alternative**: `playMusic({ type: "track", id: "6rqhFgbbKwnb9MLmUQDhG6" })`

2. **pausePlayback**

   - **Description**: Pause the currently playing track on Spotify
   - **Parameters**:
     - `deviceId` (string, optional): ID of the device to pause
   - **Returns**: Success status
   - **Example**: `pausePlayback()`

3. **skipToNext**

   - **Description**: Skip to the next track in the current playback queue
   - **Parameters**:
     - `deviceId` (string, optional): ID of the device
   - **Returns**: Success status
   - **Example**: `skipToNext()`

4. **skipToPrevious**

   - **Description**: Skip to the previous track in the current playback queue
   - **Parameters**:
     - `deviceId` (string, optional): ID of the device
   - **Returns**: Success status
   - **Example**: `skipToPrevious()`

5. **createPlaylist**

   - **Description**: Create a new playlist on Spotify
   - **Parameters**:
     - `name` (string): Name for the new playlist
     - `description` (string, optional): Description for the playlist
     - `public` (boolean, optional): Whether the playlist should be public (default: false)
   - **Returns**: Object with the new playlist's ID and URL
   - **Example**: `createPlaylist({ name: "Workout Mix", description: "Songs to get pumped up", public: false })`

6. **addTracksToPlaylist**

   - **Description**: Add tracks to an existing Spotify playlist
   - **Parameters**:
     - `playlistId` (string): ID of the playlist
     - `trackUris` (array): Array of track URIs or IDs to add
     - `position` (number, optional): Position to insert tracks
   - **Returns**: Success status and snapshot ID
   - **Example**: `addTracksToPlaylist({ playlistId: "3cEYpjA9oz9GiPac4AsH4n", trackUris: ["spotify:track:4iV5W9uYEdYUVa79Axb7Rh"] })`

7. **addToQueue**

   - **Description**: Adds a track, album, artist or playlist to the current playback queue
   - - **Parameters**:
     - `uri` (string, optional): Spotify URI of the item to add to queue (overrides type and id)
     - `type` (string, optional): Type of item to queue (track, album, artist, playlist)
     - `id` (string, optional): Spotify ID of the item to queue
     - `deviceId` (string, optional): ID of the device to queue on
   - **Returns**: Success status
   - **Example**: `addToQueue({ uri: "spotify:track:6rqhFgbbKwnb9MLmUQDhG6" })`
   - **Alternative**: `addToQueue({ type: "track", id: "6rqhFgbbKwnb9MLmUQDhG6" })`

## Setup

### Prerequisites

- Node.js v16+
- A Spotify Premium account
- A registered Spotify Developer application

### Installation

```bash
git clone https://github.com/marcelmarais/spotify-mcp-server.git
cd spotify-mcp-server
npm install
npm run build
```

### Creating a Spotify Developer Application

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Log in with your Spotify account
3. Click the "Create an App" button
4. Fill in the app name and description
5. Accept the Terms of Service and click "Create"
6. In your new app's dashboard, you'll see your **Client ID**
7. Click "Show Client Secret" to reveal your **Client Secret**
8. Click "Edit Settings" and add a Redirect URI (e.g., `http://localhost:8888/callback`)
9. Save your changes

### Spotify API Configuration

Create a `spotify-config.json` file in the project root (you can copy and modify the provided example):

```bash
# Copy the example config file
cp spotify-config.example.json spotify-config.json
```

Then edit the file with your credentials:

```json
{
  "clientId": "your-client-id",
  "clientSecret": "your-client-secret",
  "redirectUri": "http://localhost:8888/callback"
}
```

### Authentication Process

The Spotify API uses OAuth 2.0 for authentication. Follow these steps to authenticate your application:

1. Run the authentication script:

```bash
npm run auth
```

2. The script will generate an authorization URL. Open this URL in your web browser.

3. You'll be prompted to log in to Spotify and authorize your application.

4. After authorization, Spotify will redirect you to your specified redirect URI with a code parameter in the URL.

5. The authentication script will automatically exchange this code for access and refresh tokens.

6. These tokens will be saved to your `spotify-config.json` file, which will now look something like:

```json
{
  "clientId": "your-client-id",
  "clientSecret": "your-client-secret",
  "redirectUri": "http://localhost:8888/callback",
  "accessToken": "BQAi9Pn...kKQ",
  "refreshToken": "AQDQcj...7w",
  "expiresAt": 1677889354671
}
```

7. The server will automatically refresh the access token when needed, using the refresh token.

## Integrating with Claude Desktop, Cursor, and VsCode [Via Cline model extension](https://marketplace.visualstudio.com/items/?itemName=saoudrizwan.claude-dev)

To use your MCP server with Claude Desktop, add it to your Claude configuration:

```json
{
  "mcpServers": {
    "spotify": {
      "command": "node",
      "args": ["spotify-mcp-server/build/index.js"]
    }
  }
}
```

For Cursor, go to the MCP tab in `Cursor Settings` (command + shift + J). Add a server with this command:

```bash
node path/to/spotify-mcp-server/build/index.js
```

To set up your MCP correctly with Cline ensure you have the following file configuration set `cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "spotify": {
      "command": "node",
      "args": ["~/../spotify-mcp-server/build/index.js"],
      "autoApprove": ["getListeningHistory", "getNowPlaying"]
    }
  }
}
```

You can add additional tools to the auto approval array to run the tools without intervention.
