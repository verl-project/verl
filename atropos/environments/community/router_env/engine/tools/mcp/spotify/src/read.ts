import type { RequestHandlerExtra } from '@modelcontextprotocol/sdk/shared/protocol.js';
import type { MaxInt } from '@spotify/web-api-ts-sdk';
import { z } from 'zod';
import type { SpotifyHandlerExtra, SpotifyTrack, tool } from './types.js';
import { formatDuration, handleSpotifyRequest } from './utils.js';

function isTrack(item: any): item is SpotifyTrack {
  return (
    item &&
    item.type === 'track' &&
    Array.isArray(item.artists) &&
    item.album &&
    typeof item.album.name === 'string'
  );
}

const searchSpotify: tool<{
  query: z.ZodString;
  type: z.ZodEnum<['track', 'album', 'artist', 'playlist']>;
  limit: z.ZodOptional<z.ZodNumber>;
}> = {
  name: 'searchSpotify',
  description: 'Search for tracks, albums, artists, or playlists on Spotify',
  schema: {
    query: z.string().describe('The search query'),
    type: z
      .enum(['track', 'album', 'artist', 'playlist'])
      .describe(
        'The type of item to search for either track, album, artist, or playlist',
      ),
    limit: z
      .number()
      .min(1)
      .max(50)
      .optional()
      .describe('Maximum number of results to return (10-50)'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { query, type, limit } = args;
    const limitValue = limit ?? 10;

    try {
      const results = await handleSpotifyRequest(async (spotifyApi) => {
        return await spotifyApi.search(
          query,
          [type],
          undefined,
          limitValue as MaxInt<50>,
        );
      });

      let formattedResults = '';

      if (type === 'track' && results.tracks) {
        formattedResults = results.tracks.items
          .map((track, i) => {
            const artists = track.artists.map((a) => a.name).join(', ');
            const duration = formatDuration(track.duration_ms);
            return `${i + 1}. "${
              track.name
            }" by ${artists} (${duration}) - ID: ${track.id}`;
          })
          .join('\n');
      } else if (type === 'album' && results.albums) {
        formattedResults = results.albums.items
          .map((album, i) => {
            const artists = album.artists.map((a) => a.name).join(', ');
            return `${i + 1}. "${album.name}" by ${artists} - ID: ${album.id}`;
          })
          .join('\n');
      } else if (type === 'artist' && results.artists) {
        formattedResults = results.artists.items
          .map((artist, i) => {
            return `${i + 1}. ${artist.name} - ID: ${artist.id}`;
          })
          .join('\n');
      } else if (type === 'playlist' && results.playlists) {
        formattedResults = results.playlists.items
          .map((playlist, i) => {
            return `${i + 1}. "${playlist?.name ?? 'Unknown Playlist'} (${
              playlist?.description ?? 'No description'
            } tracks)" by ${playlist?.owner?.display_name} - ID: ${
              playlist?.id
            }`;
          })
          .join('\n');
      }

      return {
        content: [
          {
            type: 'text',
            text:
              formattedResults.length > 0
                ? `# Search results for "${query}" (type: ${type})\n\n${formattedResults}`
                : `No ${type} results found for "${query}"`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error searching for ${type}s: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
      };
    }
  },
};

const getNowPlaying: tool<Record<string, never>> = {
  name: 'getNowPlaying',
  description: 'Get information about the currently playing track on Spotify',
  schema: {},
  handler: async (args, extra: SpotifyHandlerExtra) => {
    try {
      const currentTrack = await handleSpotifyRequest(async (spotifyApi) => {
        return await spotifyApi.player.getCurrentlyPlayingTrack();
      });

      if (!currentTrack || !currentTrack.item) {
        return {
          content: [
            {
              type: 'text',
              text: 'Nothing is currently playing on Spotify',
            },
          ],
        };
      }

      const item = currentTrack.item;

      if (!isTrack(item)) {
        return {
          content: [
            {
              type: 'text',
              text: 'Currently playing item is not a track (might be a podcast episode)',
            },
          ],
        };
      }

      const artists = item.artists.map((a) => a.name).join(', ');
      const album = item.album.name;
      const duration = formatDuration(item.duration_ms);
      const progress = formatDuration(currentTrack.progress_ms || 0);
      const isPlaying = currentTrack.is_playing;

      return {
        content: [
          {
            type: 'text',
            text:
              `# Currently ${isPlaying ? 'Playing' : 'Paused'}\n\n` +
              `**Track**: "${item.name}"\n` +
              `**Artist**: ${artists}\n` +
              `**Album**: ${album}\n` +
              `**Progress**: ${progress} / ${duration}\n` +
              `**ID**: ${item.id}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error getting current track: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
      };
    }
  },
};

const getMyPlaylists: tool<{
  limit: z.ZodOptional<z.ZodNumber>;
}> = {
  name: 'getMyPlaylists',
  description: "Get a list of the current user's playlists on Spotify",
  schema: {
    limit: z
      .number()
      .min(1)
      .max(50)
      .optional()
      .describe('Maximum number of playlists to return (1-50)'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { limit = 50 } = args;

    const playlists = await handleSpotifyRequest(async (spotifyApi) => {
      return await spotifyApi.currentUser.playlists.playlists(
        limit as MaxInt<50>,
      );
    });

    if (playlists.items.length === 0) {
      return {
        content: [
          {
            type: 'text',
            text: "You don't have any playlists on Spotify",
          },
        ],
      };
    }

    const formattedPlaylists = playlists.items
      .map((playlist, i) => {
        const tracksTotal = playlist.tracks?.total ? playlist.tracks.total : 0;
        return `${i + 1}. "${playlist.name}" (${tracksTotal} tracks) - ID: ${
          playlist.id
        }`;
      })
      .join('\n');

    return {
      content: [
        {
          type: 'text',
          text: `# Your Spotify Playlists\n\n${formattedPlaylists}`,
        },
      ],
    };
  },
};

const getPlaylistTracks: tool<{
  playlistId: z.ZodString;
  limit: z.ZodOptional<z.ZodNumber>;
}> = {
  name: 'getPlaylistTracks',
  description: 'Get a list of tracks in a Spotify playlist',
  schema: {
    playlistId: z.string().describe('The Spotify ID of the playlist'),
    limit: z
      .number()
      .min(1)
      .max(50)
      .optional()
      .describe('Maximum number of tracks to return (1-50)'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { playlistId, limit = 50 } = args;

    const playlistTracks = await handleSpotifyRequest(async (spotifyApi) => {
      return await spotifyApi.playlists.getPlaylistItems(
        playlistId,
        undefined,
        undefined,
        limit as MaxInt<50>,
      );
    });

    if ((playlistTracks.items?.length ?? 0) === 0) {
      return {
        content: [
          {
            type: 'text',
            text: "This playlist doesn't have any tracks",
          },
        ],
      };
    }

    const formattedTracks = playlistTracks.items
      .map((item, i) => {
        const { track } = item;
        if (!track) return `${i + 1}. [Removed track]`;

        if (isTrack(track)) {
          const artists = track.artists.map((a) => a.name).join(', ');
          const duration = formatDuration(track.duration_ms);
          return `${i + 1}. "${track.name}" by ${artists} (${duration}) - ID: ${track.id}`;
        }

        return `${i + 1}. Unknown item`;
      })
      .join('\n');

    return {
      content: [
        {
          type: 'text',
          text: `# Tracks in Playlist\n\n${formattedTracks}`,
        },
      ],
    };
  },
};

const getRecentlyPlayed: tool<{
  limit: z.ZodOptional<z.ZodNumber>;
}> = {
  name: 'getRecentlyPlayed',
  description: 'Get a list of recently played tracks on Spotify',
  schema: {
    limit: z
      .number()
      .min(1)
      .max(50)
      .optional()
      .describe('Maximum number of tracks to return (1-50)'),
  },
  handler: async (args, extra: SpotifyHandlerExtra) => {
    const { limit = 50 } = args;

    const history = await handleSpotifyRequest(async (spotifyApi) => {
      return await spotifyApi.player.getRecentlyPlayedTracks(
        limit as MaxInt<50>,
      );
    });

    if (history.items.length === 0) {
      return {
        content: [
          {
            type: 'text',
            text: "You don't have any recently played tracks on Spotify",
          },
        ],
      };
    }

    const formattedHistory = history.items
      .map((item, i) => {
        const track = item.track;
        if (!track) return `${i + 1}. [Removed track]`;

        if (isTrack(track)) {
          const artists = track.artists.map((a) => a.name).join(', ');
          const duration = formatDuration(track.duration_ms);
          return `${i + 1}. "${track.name}" by ${artists} (${duration}) - ID: ${track.id}`;
        }

        return `${i + 1}. Unknown item`;
      })
      .join('\n');

    return {
      content: [
        {
          type: 'text',
          text: `# Recently Played Tracks\n\n${formattedHistory}`,
        },
      ],
    };
  }
}

export const readTools = [
  searchSpotify,
  getNowPlaying,
  getMyPlaylists,
  getPlaylistTracks,
  getRecentlyPlayed,
];
