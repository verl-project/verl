/**
 * OpenAI API client for GPT-4o-mini feedback.
 * Expects to run in Bun/Node.js where process.env is available.
 * If you see a type error for 'process', install @types/node as a dev dependency.
 */

import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const MODEL = "gpt-4o-mini"; // Use gpt-4o-mini if available, else gpt-4o

if (!OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is not set");
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

export async function callOpenAI(
  messages: ChatCompletionMessageParam[],
  max_tokens = 64,
): Promise<string> {
  const response = await openai.chat.completions.create({
    model: MODEL,
    messages,
    max_tokens,
  });
  const content = response.choices[0]?.message?.content;
  if (!content) throw new Error("No content returned from OpenAI");
  return content.trim();
}
