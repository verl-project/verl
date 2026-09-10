import { Elysia } from "elysia";
import { getAggressiveMove } from "../agents/attacker_agent";
import { ChessEnv } from "../env/chess_env";
import { scoreAndJustifyGame } from "../llm/llm_feedback";
import { computeReward } from "../reward/reward_fn";

const env = new ChessEnv();
const games: any[] = [];
let currentGame: any[] = [];

const app = new Elysia()
  .get("/ping", () => "pong")
  .get("/api/games/latest", () => games.slice(-5))
  .post("/api/move", async ({ body }) => {
    const { from, to, san, color } = body as {
      from: string;
      to: string;
      san?: string;
      color?: string;
    };
    const currentFen = env.getFEN();
    const currentColor = currentFen.split(" ")[1] === "b" ? "black" : "white";
    if (color && color !== currentColor) {
      return { error: `It's not ${color}'s turn.` };
    }
    // User move
    const { fen: userFen, done: userDone } = env.step({ from, to });
    const userMoveData = {
      fen: userFen,
      move: { from, to, san },
      reward: null, // Placeholder, to be filled after scoring
      llmFeedback: { score: null, justification: null },
    };
    currentGame.push(userMoveData);
    // If game is over after user move, return
    if (userDone) {
      games.push([...currentGame]);
      const moves = [...currentGame];
      currentGame = [];
      return {
        moves,
        done: true,
      };
    }
    // Agent move (as black)
    const agentMove = getAggressiveMove(userFen);
    let agentMoveData = null;
    let agentDone = false;
    if (agentMove) {
      const agentPrevFen = env.getFEN();
      const { fen: agentFen, done: agentIsDone } = env.step(agentMove);
      agentMoveData = {
        fen: agentFen,
        move: agentMove,
        reward: null, // Placeholder
        llmFeedback: { score: null, justification: null },
      };
      currentGame.push(agentMoveData);
      agentDone = agentIsDone;
      if (agentIsDone) {
        games.push([...currentGame]);
        const moves = [...currentGame];
        currentGame = [];
        return {
          moves,
          done: true,
        };
      }
    }
    return {
      moves: [userMoveData, agentMoveData].filter(Boolean),
      done: agentDone,
    };
  })
  // .get("/evaluate", async () => {
  //   const fen = env.getFEN();
  //   const evalResult = await evaluatePosition(fen);
  //   return evalResult;
  // })
  .post("/api/train/start", () => ({ started: true }))
  .get("/api/agent/status", () => ({ gamesPlayed: games.length, avgReward: 0 }))
  .post("/api/reset", () => {
    env.reset();
    currentGame.length = 0;
    return { fen: env.getFEN() };
  })
  .post("/api/game/llm_feedback", async ({ body }) => {
    // Expects: { moves: [{ fen, move: { from, to, san } }] }
    const { moves } = body as {
      moves: {
        fen: string;
        move: { from: string; to: string; san?: string };
      }[];
    };
    if (!Array.isArray(moves)) {
      return { error: "Missing or invalid moves array" };
    }
    const fenHistory = moves.map((m) => m.fen);
    const moveSANs = moves.map(
      (m) => m.move.san || `${m.move.from}-${m.move.to}`,
    );
    // Only score agent moves (even indices)
    const agentMoveIndices = moves
      .map((_, idx) => idx)
      .filter((idx) => idx % 2 === 1);
    const agentFens = agentMoveIndices.map((idx) => fenHistory[idx]);
    const agentSANs = agentMoveIndices.map((idx) => moveSANs[idx]);
    let feedbackArr = [];
    try {
      feedbackArr = await scoreAndJustifyGame(agentFens, agentSANs);
    } catch (e) {
      return { error: "LLM feedback failed", details: String(e) };
    }
    const scoredMoves = await Promise.all(
      moves.map(async (moveData, idx) => {
        if (idx % 2 === 1) {
          // Agent move: fill in feedback
          const { fen, move } = moveData;
          const moveSAN = String(move.san ?? `${move.from}-${move.to}`);
          const feedback = feedbackArr.shift() || {
            score: null,
            justification: null,
          };
          const reward = await computeReward(
            fen,
            moveSAN,
            feedback.score ?? "",
            feedback.justification ?? "",
          );
          return {
            ...moveData,
            reward,
            llmFeedback: feedback,
          };
        } else {
          // User move: leave feedback/reward as null
          return {
            ...moveData,
            reward: null,
            llmFeedback: { score: null, justification: null },
          };
        }
      }),
    );
    return { moves: scoredMoves };
  })
  .listen(3001);

console.log("API running on http://localhost:3001");
