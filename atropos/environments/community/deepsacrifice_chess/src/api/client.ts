// API client for DeepSacrifice frontend

export async function fetchLatestGames() {
  // TODO: Replace with real fetch
  return [{ fen: "startpos", moves: [] }];
}

export async function fetchAgentStatus() {
  // TODO: Replace with real fetch
  return { gamesPlayed: 0, avgReward: 0 };
}

export async function startTraining() {
  // TODO: Replace with real POST
  return { started: true };
}
