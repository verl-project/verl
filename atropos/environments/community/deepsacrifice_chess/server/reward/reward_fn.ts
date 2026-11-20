/**
 * Placeholder reward function using LLM feedback.
 * Will use OpenAI GPT-4o-mini feedback in the future.
 */
export async function computeReward(
  fen: string,
  moveSAN: string,
  llmScore?: string,
  llmJustification?: string,
): Promise<number> {
  // TODO: Use real LLM feedback to compute reward
  // For now, return a dummy reward
  return 0.5;
}
