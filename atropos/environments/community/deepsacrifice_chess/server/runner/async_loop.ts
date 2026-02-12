import { getAggressiveMove } from "../agents/attacker_agent";
import { ChessEnv } from "../env/chess_env";

export async function runTrainingLoop(episodes = 1) {
  const env = new ChessEnv();
  for (let ep = 0; ep < episodes; ep++) {
    env.reset();
    let done = false;
    while (!done) {
      const move = getAggressiveMove(env.getFEN());
      if (!move) break;
      const { done: isDone } = env.step(move);
      done = isDone;
    }
    // Log or store game
    console.log(`Episode ${ep + 1} finished. FEN: ${env.getFEN()}`);
  }
}
