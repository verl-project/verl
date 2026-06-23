import { Chess } from "chess.js";
import { AttackerAgent } from "../agents/attacker_agent";
import { ChessEnv } from "../env/chess_env";

// Simulate user move by picking a random legal move
function getRandomUserMove(fen: string): { from: string; to: string } | null {
  const chess = new Chess(fen);
  const moves = chess.moves({ verbose: true });
  if (moves.length === 0) return null;
  const move = moves[Math.floor(Math.random() * moves.length)];
  return { from: move.from, to: move.to };
}

export async function runUserVsAgentGame() {
  const env = new ChessEnv();
  const agent = new AttackerAgent();
  let done = false;
  let moveCount = 0;
  let fen = env.reset();
  let userTurn = true;
  const gameData: any[] = [];

  while (!done && moveCount < 100) {
    let move;
    let player;
    if (userTurn) {
      move = getRandomUserMove(fen); // Placeholder for real user input
      player = "user";
      console.log(`User move:`, move);
    } else {
      move = agent.getMove(fen);
      player = "agent";
      console.log(`Agent move:`, move);
    }
    if (!move) break;
    // Placeholder for LLM feedback (to be replaced with real call)
    const llmFeedback = { score: 7, justification: "Placeholder feedback" };
    const { fen: newFen, reward, done: isDone } = env.step(move);
    moveCount++;
    gameData.push({
      moveNumber: moveCount,
      player,
      move,
      fen: newFen,
      reward,
      llmFeedback,
    });
    console.log("FEN:", newFen);
    console.log("Reward:", reward);
    fen = newFen;
    done = isDone;
    userTurn = !userTurn;
  }
  console.log(`User-vs-Agent game finished after ${moveCount} moves.`);
  console.log("Game data:", gameData);
}

// If run directly, play a game
(async () => {
  await runUserVsAgentGame();
})();
