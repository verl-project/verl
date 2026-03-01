import { SpotifyApi } from '@spotify/web-api-ts-sdk';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath, URL } from 'node:url';
import open from 'open';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_FILE = path.join(__dirname, '../spotify-config.json');

export interface SpotifyConfig {
  clientId: string;
  clientSecret: string;
  redirectUri: string;
  accessToken?: string;
  refreshToken?: string;
}

export function loadSpotifyConfig(): SpotifyConfig {
  if (!fs.existsSync(CONFIG_FILE)) {
    throw new Error(
      `Spotify configuration file not found at ${CONFIG_FILE}. Please create one with clientId, clientSecret, and redirectUri.`,
    );
  }

  try {
    const config = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
    if (!config.clientId || !config.clientSecret || !config.redirectUri) {
      throw new Error(
        'Spotify configuration must include clientId, clientSecret, and redirectUri.',
      );
    }
    return config;
  } catch (error) {
    throw new Error(
      `Failed to parse Spotify configuration: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

export function saveSpotifyConfig(config: SpotifyConfig): void {
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2), 'utf8');
}

let cachedSpotifyApi: SpotifyApi | null = null;

export function createSpotifyApi(): SpotifyApi {
  if (cachedSpotifyApi) {
    return cachedSpotifyApi;
  }

  const config = loadSpotifyConfig();

  if (config.accessToken && config.refreshToken) {
    const accessToken = {
      access_token: config.accessToken,
      token_type: 'Bearer',
      expires_in: 3600 * 24 * 30, // Default to 1 month
      refresh_token: config.refreshToken,
    };

    cachedSpotifyApi = SpotifyApi.withAccessToken(config.clientId, accessToken);
    return cachedSpotifyApi;
  }

  cachedSpotifyApi = SpotifyApi.withClientCredentials(
    config.clientId,
    config.clientSecret,
  );

  return cachedSpotifyApi;
}

function generateRandomString(length: number): string {
  const array = new Uint8Array(length);
  crypto.getRandomValues(array);
  return Array.from(array)
    .map((b) =>
      'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'.charAt(
        b % 62,
      ),
    )
    .join('');
}

function base64Encode(str: string): string {
  return Buffer.from(str).toString('base64');
}

async function exchangeCodeForToken(
  code: string,
  config: SpotifyConfig,
): Promise<{ access_token: string; refresh_token: string }> {
  const tokenUrl = 'https://accounts.spotify.com/api/token';
  const authHeader = `Basic ${base64Encode(`${config.clientId}:${config.clientSecret}`)}`;

  const params = new URLSearchParams();
  params.append('grant_type', 'authorization_code');
  params.append('code', code);
  params.append('redirect_uri', config.redirectUri);

  const response = await fetch(tokenUrl, {
    method: 'POST',
    headers: {
      Authorization: authHeader,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: params,
  });

  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(`Failed to exchange code for token: ${errorData}`);
  }

  const data = await response.json();
  return {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
  };
}

export async function authorizeSpotify(): Promise<void> {
  const config = loadSpotifyConfig();

  const redirectUri = new URL(config.redirectUri);
  if (
    redirectUri.hostname !== 'localhost' &&
    redirectUri.hostname !== '127.0.0.1'
  ) {
    console.error(
      'Error: Redirect URI must use localhost for automatic token exchange',
    );
    console.error(
      'Please update your spotify-config.json with a localhost redirect URI',
    );
    console.error('Example: http://127.0.0.1:8888/callback');
    process.exit(1);
  }

  const port = redirectUri.port || '80';
  const callbackPath = redirectUri.pathname || '/callback';

  const state = generateRandomString(16);

  const scopes = [
    'user-read-private',
    'user-read-email',
    'user-read-playback-state',
    'user-modify-playback-state',
    'user-read-currently-playing',
    'playlist-read-private',
    'playlist-modify-private',
    'playlist-modify-public',
    'user-library-read',
    'user-library-modify',
    'user-read-recently-played',
    'user-modify-playback-state',
    'user-read-playback-state',
    'user-read-currently-playing'
  ];

  const authParams = new URLSearchParams({
    client_id: config.clientId,
    response_type: 'code',
    redirect_uri: config.redirectUri,
    scope: scopes.join(' '),
    state: state,
    show_dialog: 'true',
  });

  const authorizationUrl = `https://accounts.spotify.com/authorize?${authParams.toString()}`;

  const authPromise = new Promise<void>((resolve, reject) => {
    // Create HTTP server to handle the callback
    const server = http.createServer(async (req, res) => {
      if (!req.url) {
        return res.end('No URL provided');
      }

      const reqUrl = new URL(req.url, `http://localhost:${port}`);

      if (reqUrl.pathname === callbackPath) {
        const code = reqUrl.searchParams.get('code');
        const returnedState = reqUrl.searchParams.get('state');
        const error = reqUrl.searchParams.get('error');

        res.writeHead(200, { 'Content-Type': 'text/html' });

        if (error) {
          console.error(`Authorization error: ${error}`);
          res.end(
            '<html><body><h1>Authentication Failed</h1><p>Please close this window and try again.</p></body></html>',
          );
          server.close();
          reject(new Error(`Authorization failed: ${error}`));
          return;
        }

        if (returnedState !== state) {
          console.error('State mismatch error');
          res.end(
            '<html><body><h1>Authentication Failed</h1><p>State verification failed. Please close this window and try again.</p></body></html>',
          );
          server.close();
          reject(new Error('State mismatch'));
          return;
        }

        if (!code) {
          console.error('No authorization code received');
          res.end(
            '<html><body><h1>Authentication Failed</h1><p>No authorization code received. Please close this window and try again.</p></body></html>',
          );
          server.close();
          reject(new Error('No authorization code received'));
          return;
        }

        try {
          const tokens = await exchangeCodeForToken(code, config);

          config.accessToken = tokens.access_token;
          config.refreshToken = tokens.refresh_token;
          saveSpotifyConfig(config);

          res.end(
            '<html><body><h1>Authentication Successful!</h1><p>You can now close this window and return to the application.</p></body></html>',
          );
          console.log(
            'Authentication successful! Access token has been saved.',
          );

          server.close();
          resolve();
        } catch (error) {
          console.error('Token exchange error:', error);
          res.end(
            '<html><body><h1>Authentication Failed</h1><p>Failed to exchange authorization code for tokens. Please close this window and try again.</p></body></html>',
          );
          server.close();
          reject(error);
        }
      } else {
        res.writeHead(404);
        res.end();
      }
    });

    server.listen(Number.parseInt(port), '127.0.0.1', () => {
      console.log(
        `Listening for Spotify authentication callback on port ${port}`,
      );
      console.log('Opening browser for authorization...');

      open(authorizationUrl).catch((error: Error) => {
        console.log(
          'Failed to open browser automatically. Please visit this URL to authorize:',
        );
        console.log(authorizationUrl);
      });
    });

    server.on('error', (error) => {
      console.error(`Server error: ${error.message}`);
      reject(error);
    });
  });

  await authPromise;
}

export function formatDuration(ms: number): string {
  const minutes = Math.floor(ms / 60000);
  const seconds = ((ms % 60000) / 1000).toFixed(0);
  return `${minutes}:${seconds.padStart(2, '0')}`;
}

export async function handleSpotifyRequest<T>(
  action: (spotifyApi: SpotifyApi) => Promise<T>,
): Promise<T> {
  try {
    const spotifyApi = createSpotifyApi();
    return await action(spotifyApi);
  } catch (error) {
    // Skip JSON parsing errors as these are actually successful operations
    const errorMessage = error instanceof Error ? error.message : String(error);
    if (
      errorMessage.includes('Unexpected token') ||
      errorMessage.includes('Unexpected non-whitespace character') ||
      errorMessage.includes('Exponent part is missing a number in JSON')
    ) {
      return undefined as T;
    }
    // Rethrow other errors
    throw error;
  }
}
