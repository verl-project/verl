import { Chess } from "chess.js";

export function getAggressiveMove(
  fen: string,
): { from: string; to: string } | null {
  const chess = new Chess(fen);
  const moves = chess.moves({ verbose: true });
  if (moves.length === 0) return null;

  // Prefer captures
  const captures = moves.filter(
    (m) => m.flags.includes("c") || m.flags.includes("e"),
  );
  if (captures.length > 0) {
    const move = captures[Math.floor(Math.random() * captures.length)];
    return { from: move.from, to: move.to };
  }

  // Prefer checks
  const checks = moves.filter((m) => {
    chess.move({ from: m.from, to: m.to });
    const isCheck = chess.inCheck();
    chess.undo();
    return isCheck;
  });
  if (checks.length > 0) {
    const move = checks[Math.floor(Math.random() * checks.length)];
    return { from: move.from, to: move.to };
  }

  // Otherwise, pick random
  const move = moves[Math.floor(Math.random() * moves.length)];
  return { from: move.from, to: move.to };
}

// Minimal AttackerAgent class with placeholder learning
export class AttackerAgent {
  getMove(fen: string): { from: string; to: string } | null {
    return getAggressiveMove(fen);
  }

  learnFromGame(gameData: any, llmFeedback: any): void {
    // TODO: Implement incremental learning from LLM feedback
    // For now, this is a placeholder
  }
}
