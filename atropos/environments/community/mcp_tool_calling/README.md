# Readme

## 1-Minute Demo Video
Watch the demo on YouTube
https://www.loom.com/share/44c793c47e7d45eaaf02bac7c168a10d?sid=4ff3d95f-701f-4d11-be3f-aa89f8fa2f0d

## Environment Design & Motivation
NousWhiteHouse is a reinforcement learning (RL) project focused on improving agent tool calls using the Model Context Protocol (MCP). The goal is to enable agents to dynamically discover and invoke tools more effectively, leveraging MCP for context-aware decision-making.

After replicating RESTGPT, we noticed that LLMs struggled to find the right tools to call, such as finding Gims songs on Spotify. Instead of manually matching multiple APIs, the recent advent of MCP inspires us to double down on tool-calling efforts.

Our Dataset uses a format like-
{
  "user_prompt_text": "What is the current stock price of AAPL?",
  "expected_mcp_call": {
    "tool_name": "getStockPrice",
    "arguments": {
      "tickerSymbol": "AAPL"
    }
  }
}

the return promts are compared with the expected_mcp_call

Our main task or challenge that our environment presented-
Help LLMs use MCPs

Why is this environment interesting or useful for RL research-
this environment will result in super fast tool calling with more accurate results and allow for more seamless integrations of tools with LLMs

Framework-
we used the Single Tool Environment as a framework for the MCP env

Challenge-
Finding existing large datasets with MCP calls was extermely difficult.

## Estimate
### 🧪 Zero-Training Test Results
Results of running the example trainer on the gsm8k server via Lambda:

W&B Link: https://api.wandb.ai/links/l-a-t-hacken-tu-eindhoven/nqjy1v4b
