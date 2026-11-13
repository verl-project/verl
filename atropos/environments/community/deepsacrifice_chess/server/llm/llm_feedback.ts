/**
 * Placeholder for LLM-based move aggression/brilliance scoring.
 * Will use OpenAI GPT-4o-mini via callOpenAI in the future.
 */
import { callOpenAI } from "./openai_client";

export async function scoreMoveAggression(
  fenHistory: string[],
  moveIdx: number,
  moveSAN: string,
): Promise<string> {
  const prompt = `Given the following chess game FEN history (one FEN per move):\n${fenHistory.map((f, i) => `${i + 1}: ${f}`).join("\n")}\nEvaluate the aggression or brilliance of move #${moveIdx + 1} (${moveSAN}). Respond ONLY with a single digit from 1 (not aggressive) to 10 (extremely aggressive/brilliant). Be fast and concise.`;
  const response = await callOpenAI(
    [
      {
        role: "system",
        content:
          "You are a chess grandmaster evaluating move aggression and brilliance. Respond as quickly and concisely as possible.",
      },
      { role: "user", content: prompt },
    ],
    16,
  );
  return response;
}

/**
 * Placeholder for LLM-based sacrifice justification.
 * Will use OpenAI GPT-4o-mini via callOpenAI in the future.
 */
export async function justifySacrifice(
  fenHistory: string[],
  moveIdx: number,
  moveSAN: string,
): Promise<string> {
  const prompt = `Given the following chess game FEN history (one FEN per move):\n${fenHistory.map((f, i) => `${i + 1}: ${f}`).join("\n")}\nWas the sacrifice in move #${moveIdx + 1} (${moveSAN}) justified? Reply in 1 short sentence: is the sacrifice justified or not, and why. Be fast and concise.`;
  const response = await callOpenAI(
    [
      {
        role: "system",
        content:
          "You are a chess grandmaster evaluating sacrifices. Respond as quickly and concisely as possible.",
      },
      { role: "user", content: prompt },
    ],
    16,
  );
  return response;
}

export async function scoreAndJustifyGame(
  fenHistory: string[],
  moveSANs: string[],
): Promise<{ score: string; justification: string }[]> {
  const prompt = `Given the following chess game FEN history (one FEN per move) and the corresponding SAN moves, evaluate each move for aggression/brilliance and sacrifice justification.\n\nFEN history (one per move):\n${fenHistory.map((f, i) => `${i + 1}: ${f}`).join("\n")}\n\nSAN moves (one per move):\n${moveSANs.map((san, i) => `${i + 1}: ${san}`).join("\n")}\n\nFor each move, respond with a JSON array of objects, each with:\n- score: a single digit from 1 (not aggressive) to 10 (extremely aggressive/brilliant)\n- justification: 1 short sentence on whether the move is a justified sacrifice or not, and why.\n\nExample:\n[{"score": "7", "justification": "The sacrifice is risky but justified."}, ...]\n\nRespond ONLY with the JSON array, nothing else.`;
  const response = await callOpenAI(
    [
      {
        role: "system",
        content:
          "You are a chess grandmaster evaluating a full game for aggression and sacrifice justification. Respond as quickly and concisely as possible.",
      },
      { role: "user", content: prompt },
    ],
    512,
  );
  try {
    const parsed = JSON.parse(response);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    throw new Error("Response is not an array");
  } catch (e) {
    throw new Error("Failed to parse LLM response as JSON: " + response);
  }
}
