# Wikipedia Article Creator: OpenAI Integration and Environment Setup Plan

## 1. Overview

This document outlines the plan for integrating OpenAI models and proper environment configuration with the existing WikipediaArticleCreatorEnv implementation. The goal is to create a robust setup that can leverage OpenAI's models (like GPT-4, GPT-3.5-turbo) while maintaining a secure configuration for API credentials through environment variables.

## 2. Key Components and Changes

### 2.1 Environment Variables (.env) Setup

We need to properly load and manage API credentials from a .env file:

```python
# At the top of wikipedia_article_creator.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Later, access environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")
```

### 2.2 OpenAI API Integration

Update the `config_init` method in the `WikipediaArticleCreatorEnv` class to support OpenAI models:

```python
@classmethod
def config_init(cls) -> Tuple[WikipediaArticleCreatorConfig, List[APIServerConfig]]:
    env_config = WikipediaArticleCreatorConfig(
        tokenizer_name="gpt-4-turbo",  # Use any OpenAI model name or a different tokenizer for local processing
        group_size=4,
        use_wandb=True,
        rollout_server_url="http://localhost:8000",
        total_steps=1000,
        batch_size=128,
        steps_per_eval=20,
        max_token_length=1024 * 16,
        inference_weight=1.0,
        wandb_name="wikipedia_article_creator",
        eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
        eval_limit_ratio=0.1,
        max_steps=10,
        temperature=0.7,
        thinking_active=True,
        eval_topics=5,
        tool_timeout=15.0,
        tavily_api_key=None,  # Falls back to environment variable
        min_article_sections=3,
        max_article_tokens=2048,
        topics_file="topics.json",
        logging_active=True,
    )

    # Configure OpenAI server
    server_configs = [
        APIServerConfig(
            model_name="gpt-4o",  # or any other OpenAI model
            base_url=None,  # Use default OpenAI base URL
            api_key=os.environ.get("OPENAI_API_KEY"),
            num_max_requests_at_once=4,
            num_requests_for_eval=16,
        ),
    ]

    return env_config, server_configs
```

### 2.3 OpenAI Token Length Handling

Since OpenAI models handle token length limits differently, we need to make some adjustments to ensure proper tokenization and context management:

```python
async def _get_model_response(self, messages: List[Dict]) -> str:
    """Get a response from the model for the current conversation state"""
    try:
        # For OpenAI models, we pass the messages directly
        if self.server.config.base_url is None or 'openai' in (self.server.config.base_url or ''):
            completion = await self.server.chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=min(4096, self.config.max_token_length),  # Ensure within OpenAI limits
            )
            # Extract the text from response
            return completion.choices[0].message.content
        else:
            # For non-OpenAI models (e.g., local models via vLLM)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            completion = await self.server.completion(
                prompt=prompt,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=self.config.temperature,
            )
            return completion.choices[0].text
    except Exception as e:
        logger.error(f"Error getting model response: {e}")
        return ""
```

### 2.4 Testing Runner Script

Create a simple runner script to test the environment with OpenAI models:

```python
#!/usr/bin/env python3
"""
Test runner for Wikipedia Article Creator with OpenAI models
"""
import asyncio
import logging
import os
from dotenv import load_dotenv

from environments.hack0.wikipedia.wikipedia_article_creator import WikipediaArticleCreatorEnv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def main():
    """Run the environment with a specified model and topic"""
    # Create and initialize the environment
    env_config, server_configs = WikipediaArticleCreatorEnv.config_init()

    # Override with command-line arguments if needed
    # env_config.max_steps = 5
    # server_configs[0].model_name = "gpt-3.5-turbo"

    # Initialize the environment
    env = WikipediaArticleCreatorEnv(env_config, server_configs, slurm=False, testing=True)
    await env.setup()

    # Run a single episode with a specific topic
    topic = "Climate change in Antarctica"
    episode_id = 1

    logger.info(f"Starting research on topic: {topic}")
    episode = env._get_or_create_episode(episode_id, topic)

    # Run until terminal state
    while not episode.is_terminal:
        is_terminal, step_data = await env._next_step(episode)
        response = step_data.get("response", "")
        tool_calls = step_data.get("tool_calls", [])

        logger.info(f"Step {episode.steps_taken} completed")
        logger.info(f"Tool calls: {len(tool_calls)}")

        if is_terminal and episode.final_article:
            logger.info("Article generated successfully")
            logger.info(f"Article length: {len(episode.final_article)} characters")

            # Save the article to a file
            with open(f"article_{topic.replace(' ', '_')}.md", "w") as f:
                f.write(episode.final_article)

            # Evaluate article quality
            quality_metrics = env._assess_article_quality(
                episode.final_article, episode.research_facts
            )
            logger.info(f"Article quality metrics: {quality_metrics}")
        elif is_terminal:
            logger.info("Episode terminated without article")

if __name__ == "__main__":
    asyncio.run(main())
```

## 3. Required Environment Setup

### 3.1 .env File Structure

Create a `.env` file in the project root with the necessary API keys:

```
# OpenAI API Key for model access
OPENAI_API_KEY=sk-...your-openai-key...

# Tavily API Key for web search and content extraction
TAVILY_API_KEY=tvly-...your-tavily-key...

# Optional WandB configuration
WANDB_API_KEY=...your-wandb-key...
WANDB_PROJECT=wikipedia_article_creator
```

### 3.2 Dependencies

Ensure the following packages are installed:

```
python-dotenv>=1.0.0
openai>=1.10.0
tavily-python>=0.2.2
wandb>=0.16.0
```

## 4. Implementation Plan

### 4.1 Phase 1: Environment Setup and .env Integration

1. Add the python-dotenv package to project dependencies
2. Create a `.env.template` file with placeholder values
3. Update README to explain the required API keys and environment setup
4. Modify `wikipedia_article_creator.py` to load environment variables

### 4.2 Phase 2: OpenAI Integration

1. Update the `config_init` method to use OpenAI models
2. Modify the `_get_model_response` method to handle OpenAI models properly
3. Adjust tokenization and context length handling for OpenAI models
4. Test with smaller models (GPT-3.5-turbo) before moving to GPT-4

### 4.3 Phase 3: Testing and Validation

1. Create a test runner script to validate the environment with OpenAI models
2. Test with a variety of topics to ensure the environment works correctly
3. Monitor API usage and optimize requests to minimize token usage
4. Validate article quality with different models and parameters

### 4.4 Phase 4: Performance Optimization

1. Implement caching for API responses to reduce duplicate requests
2. Optimize API call batching for evaluation runs
3. Add error handling and retry logic for API rate limits
4. Document best practices for working with OpenAI models in this environment

## 5. Potential Challenges and Solutions

### 5.1 API Rate Limits

**Challenge:** OpenAI API has rate limits that could affect research speed.
**Solution:** Implement exponential backoff retry logic and request batching.

### 5.2 Token Context Length

**Challenge:** Different OpenAI models have different context length limits.
**Solution:** Implement model-specific context management to stay within limits.

### 5.3 Cost Management

**Challenge:** API usage costs can accumulate quickly with multiple research steps.
**Solution:** Implement cost tracking, efficient caching, and configurable limits.

### 5.4 Model Differences

**Challenge:** Different models may have different capabilities and response formats.
**Solution:** Add model-specific parsing and prompt adjustments.

## 6. Testing Metrics

1. **Research Efficiency:**
   - Average tool calls per completed article
   - Percentage of relevant vs. irrelevant searches
   - Time to completion

2. **Article Quality:**
   - Structure compliance (section count, references)
   - Content relevance to topic
   - Factual accuracy
   - Information density

3. **API Usage:**
   - Total tokens per article
   - Cost per article
   - Rate limit errors encountered

## 7. Next Steps

After successful implementation of OpenAI integration and environment setup:

1. Optimize prompts specifically for OpenAI models
2. Implement comparative evaluation across different models
3. Add support for multi-model fallback (e.g., start with GPT-3.5, escalate to GPT-4 for complex topics)
4. Create a comprehensive benchmark suite for Wikipedia article creation
