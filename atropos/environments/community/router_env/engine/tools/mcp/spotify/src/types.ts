import type { RequestHandlerExtra } from '@modelcontextprotocol/sdk/shared/protocol.js';
import { ServerNotification, ServerRequest } from '@modelcontextprotocol/sdk/types.js';
import type { z } from 'zod';

export type SpotifyHandlerExtra = RequestHandlerExtra<ServerRequest, ServerNotification>;

export type tool<Args extends z.ZodRawShape> = {
  name: string;
  description: string;
  schema: Args;
  handler: (
    args: z.infer<z.ZodObject<Args>>,
    extra: SpotifyHandlerExtra,
  ) =>
    | Promise<{
        content: Array<{
          type: 'text';
          text: string;
        }>;
      }>
    | {
        content: Array<{
          type: 'text';
          text: string;
        }>;
      };
};

export interface SpotifyArtist {
  id: string;
  name: string;
}

export interface SpotifyAlbum {
  id: string;
  name: string;
  artists: SpotifyArtist[];
}

export interface SpotifyTrack {
  id: string;
  name: string;
  type: string;
  duration_ms: number;
  artists: SpotifyArtist[];
  album: SpotifyAlbum;
}
