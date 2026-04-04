import { handleSpotifyRequest } from './utils.js';
import { z } from 'zod';
import type { SpotifyHandlerExtra, tool } from './types.js';

const playMusic: tool<{
  uri: z.ZodOptional<z.ZodString>;
  type: z.ZodOptional<z.ZodEnum<['track', 'album', 'artist', 'playlist']>>;
  id: z.ZodOptional<z.ZodString>;
  deviceId: z.ZodOptional<z.ZodString>;
}> = {
  name: 'playMusic',
  description: 'Start playing a Spotify track, album, artist, or playlist',
  schema: {
    uri: z
      .string()
      .optional()
      .describe('The Spotify URI to play (overrides type and id)'),
    type: z
      .enum(['track', 'album', 'artist', 'playlist'])
      .optional()
      .describe('The type of item to play'),
    id: z.string().optional().describe('The Spotify ID of the item to play'),
    deviceId: z
      .string()
      .optional()
      .describe('The Spotify device ID to play on'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { uri, type, id, deviceId } = args;

    if (!uri && (!type || !id)) {
      return {
        content: [
          {
            type: 'text',
            text: 'Error: Must provide either a URI or both a type and ID',
            isError: true,
          },
        ],
      };
    }

    let spotifyUri = uri;
    if (!spotifyUri && type && id) {
      spotifyUri = `spotify:${type}:${id}`;
    }

    await handleSpotifyRequest(async (spotifyApi) => {
      const device = deviceId || '';

      if (!spotifyUri) {
        await spotifyApi.player.startResumePlayback(device);
        return;
      }

      if (type === 'track') {
        await spotifyApi.player.startResumePlayback(device, undefined, [
          spotifyUri,
        ]);
      } else {
        await spotifyApi.player.startResumePlayback(device, spotifyUri);
      }
    });

    return {
      content: [
        {
          type: 'text',
          text: `Started playing ${type || 'music'} ${id ? `(ID: ${id})` : ''}`,
        },
      ],
    };
  },
};

const pausePlayback: tool<{
  deviceId: z.ZodOptional<z.ZodString>;
}> = {
  name: 'pausePlayback',
  description: 'Pause Spotify playback on the active device',
  schema: {
    deviceId: z
      .string()
      .optional()
      .describe('The Spotify device ID to pause playback on'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { deviceId } = args;

    await handleSpotifyRequest(async (spotifyApi) => {
      await spotifyApi.player.pausePlayback(deviceId || '');
    });

    return {
      content: [
        {
          type: 'text',
          text: 'Playback paused',
        },
      ],
    };
  },
};

const skipToNext: tool<{
  deviceId: z.ZodOptional<z.ZodString>;
}> = {
  name: 'skipToNext',
  description: 'Skip to the next track in the current Spotify playback queue',
  schema: {
    deviceId: z
      .string()
      .optional()
      .describe('The Spotify device ID to skip on'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { deviceId } = args;

    await handleSpotifyRequest(async (spotifyApi) => {
      await spotifyApi.player.skipToNext(deviceId || '');
    });

    return {
      content: [
        {
          type: 'text',
          text: 'Skipped to next track',
        },
      ],
    };
  },
};

const skipToPrevious: tool<{
  deviceId: z.ZodOptional<z.ZodString>;
}> = {
  name: 'skipToPrevious',
  description:
    'Skip to the previous track in the current Spotify playback queue',
  schema: {
    deviceId: z
      .string()
      .optional()
      .describe('The Spotify device ID to skip on'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { deviceId } = args;

    await handleSpotifyRequest(async (spotifyApi) => {
      await spotifyApi.player.skipToPrevious(deviceId || '');
    });

    return {
      content: [
        {
          type: 'text',
          text: 'Skipped to previous track',
        },
      ],
    };
  },
};

const createPlaylist: tool<{
  name: z.ZodString;
  description: z.ZodOptional<z.ZodString>;
  public: z.ZodOptional<z.ZodBoolean>;
}> = {
  name: 'createPlaylist',
  description: 'Create a new playlist on Spotify',
  schema: {
    name: z.string().describe('The name of the playlist'),
    description: z
      .string()
      .optional()
      .describe('The description of the playlist'),
    public: z
      .boolean()
      .optional()
      .describe('Whether the playlist should be public'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { name, description, public: isPublic = false } = args;

    const result = await handleSpotifyRequest(async (spotifyApi) => {
      const me = await spotifyApi.currentUser.profile();

      return await spotifyApi.playlists.createPlaylist(me.id, {
        name,
        description,
        public: isPublic,
      });
    });

    return {
      content: [
        {
          type: 'text',
          text: `Successfully created playlist "${name}"\nPlaylist ID: ${result.id}`,
        },
      ],
    };
  },
};

const addTracksToPlaylist: tool<{
  playlistId: z.ZodString;
  trackIds: z.ZodArray<z.ZodString>;
  position: z.ZodOptional<z.ZodNumber>;
}> = {
  name: 'addTracksToPlaylist',
  description: 'Add tracks to a Spotify playlist',
  schema: {
    playlistId: z.string().describe('The Spotify ID of the playlist'),
    trackIds: z.array(z.string()).describe('Array of Spotify track IDs to add'),
    position: z
      .number()
      .nonnegative()
      .optional()
      .describe('Position to insert the tracks (0-based index)'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { playlistId, trackIds, position } = args;

    if (trackIds.length === 0) {
      return {
        content: [
          {
            type: 'text',
            text: 'Error: No track IDs provided',
          },
        ],
      };
    }

    try {
      const trackUris = trackIds.map((id) => `spotify:track:${id}`);

      await handleSpotifyRequest(async (spotifyApi) => {
        await spotifyApi.playlists.addItemsToPlaylist(
          playlistId,
          trackUris,
          position,
        );
      });

      return {
        content: [
          {
            type: 'text',
            text: `Successfully added ${trackIds.length} track${
              trackIds.length === 1 ? '' : 's'
            } to playlist (ID: ${playlistId})`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error adding tracks to playlist: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
      };
    }
  },
};

const resumePlayback: tool<{
  deviceId: z.ZodOptional<z.ZodString>;
}> = {
  name: 'resumePlayback',
  description: 'Resume Spotify playback on the active device',
  schema: {
    deviceId: z
      .string()
      .optional()
      .describe('The Spotify device ID to resume playback on'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { deviceId } = args;

    await handleSpotifyRequest(async (spotifyApi) => {
      await spotifyApi.player.startResumePlayback(deviceId || '');
    });

    return {
      content: [
        {
          type: 'text',
          text: 'Playback resumed',
        },
      ],
    };
  },
};

const addToQueue: tool<{
  uri: z.ZodOptional<z.ZodString>;
  type: z.ZodOptional<z.ZodEnum<['track', 'album', 'artist', 'playlist']>>;
  id: z.ZodOptional<z.ZodString>;
  deviceId: z.ZodOptional<z.ZodString>;
}> = {
  name: 'addToQueue',
  description: 'Adds a track, album, artist or playlist to the playback queue',
  schema: {
    uri: z
      .string()
      .optional()
      .describe('The Spotify URI to play (overrides type and id)'),
    type: z
      .enum(['track', 'album', 'artist', 'playlist'])
      .optional()
      .describe('The type of item to play'),
    id: z.string().optional().describe('The Spotify ID of the item to play'),
    deviceId: z
      .string()
      .optional()
      .describe('The Spotify device ID to add the track to'),
  },
  handler: async (args) => {
    const { uri, type, id, deviceId } = args;

    let spotifyUri = uri;
    if (!spotifyUri && type && id) {
      spotifyUri = `spotify:${type}:${id}`;
    }

    if (!spotifyUri) {
      return {
        content: [
          {
            type: 'text',
            text: 'Error: Must provide either a URI or both a type and ID',
            isError: true,
          },
        ],
      };
    }

    await handleSpotifyRequest(async (spotifyApi) => {
      await spotifyApi.player.addItemToPlaybackQueue(
        spotifyUri,
        deviceId || '',
      );
    });

    return {
      content: [
        {
          type: 'text',
          text: `Added item ${spotifyUri} to queue`,
        },
      ],
    };
  },
};

export const playTools = [
  playMusic,
  pausePlayback,
  skipToNext,
  skipToPrevious,
  createPlaylist,
  addTracksToPlaylist,
  resumePlayback,
  addToQueue,
];
