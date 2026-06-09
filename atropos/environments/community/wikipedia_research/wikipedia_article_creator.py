#!/usr/bin/env python3
"""
WikipediaArticleCreatorEnv: Environment for training an LLM to research and create Wikipedia-style articles

This environment uses web search and content extraction tools to enable multi-step research
and article generation on arbitrary topics.
"""

import json
import logging
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning(
        "python-dotenv not installed, environment variables must be set manually"
    )

import wandb

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call
from environments.community.wikipedia_research.tools.tavily_tools import (
    TavilyExtractTool,
    TavilySearchTool,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress logging from tool_call_parser module
logging.getLogger("atroposlib.utils.tool_call_parser").setLevel(logging.ERROR)

# System prompt for the Wikipedia article creation task
SYSTEM_PROMPT = """
You are a skilled researcher and writer who creates accurate, neutral, and comprehensive
Wikipedia-style articles.

Your task is to research the given topic using web search and content extraction tools, and then write a
well-structured Wikipedia article based on your findings.

Follow these guidelines when creating your article:
1. Research thoroughly using the tools provided
2. Maintain a Neutral Point of View (NPOV) - present all significant viewpoints fairly
3. Structure your article with a clear introduction, organized sections, and a conclusion if appropriate
4. Cite reliable sources for factual claims
5. Use formal, encyclopedic language
6. Format your article in Markdown
7. IMPORTANT: Do not try to visit Wikipedia pages directly - they are blocked. Instead, search for
   information from other reputable sources

During your work, you may:
1. Think through your research strategy and article planning
2. Search for information using web_search
3. Extract content from specific webpages using visit_page
4. Organize and synthesize information from multiple sources
5. Create a final Wikipedia-style article when you have sufficient information

You should enclose your thoughts and internal monologue inside <think> </think> tags, and then use tools or
provide your final output.

IMPORTANT: When you have completed your research and are ready to provide the final article, format it
as follows:
Final Step: ```markdown
[Your complete article in markdown format]
```

For tool calls, you MUST use <tool_call> </tool_call> tags with valid JSON inside. Always format exactly
as shown:

For web search:
<tool_call>
{"name": "web_search", "arguments": {"query": "example search query", "num_results": 5}}
</tool_call>

For webpage visits:
<tool_call>
{"name": "visit_page", "arguments": {"url": "https://example.com/page"}}
</tool_call>

The JSON structure is critical - it must be valid JSON with double quotes around all keys and string values.
Always enclose your tool calls between <tool_call> and </tool_call> tags, and make sure the JSON is
correctly formatted.
"""


class WikipediaArticleCreatorConfig(BaseEnvConfig):
    """Configuration for the WikipediaArticleCreator environment"""

    max_steps: int = 10  # Maximum research steps per article
    temperature: float = 0.7  # Sampling temperature
    thinking_active: bool = True  # Enable thinking tags
    eval_topics: int = 30  # Number of topics for evaluation
    tool_timeout: float = 15.0  # Timeout for tool execution (seconds)
    tavily_api_key: Optional[str] = None  # API key for Tavily (falls back to env var)
    min_article_sections: int = 3  # Minimum number of sections in final article
    max_article_tokens: int = 2048  # Maximum tokens in final article
    topics_file: str = "topics.json"  # File containing research topics
    logging_active: bool = True  # Enable detailed logging


class EpisodeState:
    """
    Maintains state for a single episode (article creation task)
    """

    def __init__(self, episode_id: int, topic: str):
        self.episode_id = episode_id
        self.topic = topic  # The research topic for this episode
        self.message_history: List[Dict] = []  # Stores all interactions
        self.tool_calls: List[Dict] = []  # Records tool calls made
        self.tool_results: List[Dict] = []  # Records tool results returned
        self.steps_taken: int = 0  # Number of steps in this episode
        self.is_terminal: bool = False  # Whether episode has terminated
        self.final_article: Optional[str] = None  # Final Wikipedia article in markdown
        self.research_facts: List[str] = (
            []
        )  # Important facts discovered during research
        self.score: float = 0.0  # Score for this episode


class WikipediaArticleCreatorEnv(BaseEnv):
    """
    Environment for training an LLM to research and create Wikipedia-style articles

    This environment:
    - Presents the model with a topic to research
    - Allows multi-step interactions using web_search and visit_page tools
    - Tracks research process and article quality
    - Rewards comprehensive, well-structured, and accurate articles
    """

    def __init__(
        self,
        config: WikipediaArticleCreatorConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        # Initialize environment
        self.config = config
        self.episodes: Dict[int, EpisodeState] = {}
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb = []

        # Set up tools
        tavily_key = config.tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not tavily_key:
            logger.warning(
                "No Tavily API key provided - tools will not function properly"
            )

        self.search_tool = TavilySearchTool(api_key=tavily_key)
        self.extract_tool = TavilyExtractTool(api_key=tavily_key)

        # Tool definitions for the LLM
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information on a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to perform.",
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5, max: 10)",
                                "default": 5,
                            },
                            "filter_year": {
                                "type": "string",
                                "description": "Filter results to a specific year",
                                "nullable": True,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "visit_page",
                    "description": "Extract content from a specific webpage",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the webpage to visit",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

        # Load topics if file exists
        self.topics = self._load_topics()
        self.iter = 0

        self.article_quality_metrics: List[Dict[str, float]] = []

    def _load_topics(self) -> List[str]:
        """Load research topics from wikipedia_articles.json or use defaults if file doesn't exist"""
        try:
            articles_path = os.path.join(
                os.path.dirname(__file__), "wikipedia_articles.json"
            )
            if os.path.exists(articles_path):
                # The file is large, so we'll read it in chunks and extract just the titles
                topics = []
                with open(articles_path, "r") as f:
                    # Read opening bracket
                    char = f.read(1)
                    if char != "[":
                        raise ValueError("Expected JSON array to start with '['")

                    # Process articles one by one
                    count = 0
                    max_topics = 100  # Limit to 100 topics

                    while count < max_topics:
                        article_json = ""
                        brace_count = 0
                        in_article = False

                        # Read until we find a complete article JSON object
                        while True:
                            char = f.read(1)
                            if not char:  # End of file
                                break

                            if char == "{" and not in_article:
                                in_article = True
                                brace_count = 1
                                article_json = "{"
                            elif in_article:
                                article_json += char
                                if char == "{":
                                    brace_count += 1
                                elif char == "}":
                                    brace_count -= 1
                                    if brace_count == 0:
                                        # Found complete article
                                        break

                        if not article_json:
                            break

                        try:
                            article = json.loads(article_json)
                            title = article.get("title", "")
                            if title and len(title) < 100:  # Skip very long titles
                                topics.append(title)
                                count += 1
                        except json.JSONDecodeError:
                            continue

                if topics:
                    logger.info(
                        f"Loaded {len(topics)} topics from wikipedia_articles.json"
                    )
                    return topics

        except Exception as e:
            logger.warning(f"Failed to load topics from wikipedia_articles.json: {e}")

        # Default topics if file doesn't exist or loading fails
        default_topics = [
            "History of artificial intelligence",
            "Climate change in the Arctic",
            "The Great Barrier Reef ecosystem",
            "Quantum computing principles",
            "Anti-black racism in the Arab World",
            "History of cryptography",
            "Renewable energy in developing countries",
            "Space exploration in the 21st century",
            "Traditional medicine systems around the world",
            "The evolution of human language",
        ]
        logger.info(f"Using {len(default_topics)} default topics")
        return default_topics

    @classmethod
    def config_init(cls) -> Tuple[WikipediaArticleCreatorConfig, List[APIServerConfig]]:
        """Initialize default configuration"""
        # Read environment variables (with defaults if not present)
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")
        max_steps = int(os.environ.get("MAX_STEPS", "10"))
        temperature = float(os.environ.get("TEMPERATURE", "0.7"))

        # Determine if we're using an OpenAI model or a local model
        is_openai_model = model_name.startswith(("gpt-", "text-"))

        # Always use a standard HuggingFace tokenizer that's available
        # gpt2 is a good option for estimating OpenAI tokens
        tokenizer_name = "gpt2"

        env_config = WikipediaArticleCreatorConfig(
            tokenizer_name=tokenizer_name,
            group_size=1,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=512,
            steps_per_eval=20,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="wikipedia_article_creator",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            max_steps=max_steps,
            temperature=temperature,
            thinking_active=True,
            eval_topics=5,
            tool_timeout=15.0,
            tavily_api_key=os.environ.get("TAVILY_API_KEY"),  # Load from environment
            min_article_sections=3,
            max_article_tokens=2048,
            topics_file="topics.json",
            logging_active=True,  # Enable message history in the output for wandb logging
            include_messages=True,  # Enable message history in the output for wandb logging
            num_rollouts_to_keep=32,  # Keep enough conversations for good logging samples
        )

        # Configure servers based on model type
        if is_openai_model:
            # OpenAI API configuration
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables.")

            server_configs = [
                APIServerConfig(
                    model_name=model_name,
                    base_url=None,  # Use default OpenAI base URL
                    api_key=openai_api_key,
                    num_max_requests_at_once=4,
                    num_requests_for_eval=16,
                ),
            ]
        else:
            # Local model configuration
            server_configs = [
                APIServerConfig(
                    model_name=model_name,
                    base_url="http://localhost:9004/v1",
                    api_key="x",
                    num_max_requests_at_once=8,
                    num_requests_for_eval=64,
                ),
            ]

        return env_config, server_configs

    def _get_or_create_episode(
        self, episode_id: int, topic: Optional[str] = None
    ) -> EpisodeState:
        """Get an existing episode or create a new one"""
        if episode_id not in self.episodes:
            if topic is None:
                topic = random.choice(self.topics)

            ep = EpisodeState(episode_id, topic)

            # Initialize with system prompt
            ep.message_history = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Add initial user prompt with the topic
            ep.message_history.append(
                {
                    "role": "user",
                    "content": f'Research and write a comprehensive Wikipedia-style article about: "{topic}"',
                }
            )

            self.episodes[episode_id] = ep

        return self.episodes[episode_id]

    def _parse_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from model response"""
        tool_calls = []

        logger.info("\n==== PARSING TOOL CALLS ====")

        # Try to find tool calls using regex first
        tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        raw_tool_calls = re.findall(tool_call_pattern, response, re.DOTALL)

        logger.info(f"Found {len(raw_tool_calls)} tool call tags in response")

        if raw_tool_calls:
            for i, raw_call in enumerate(raw_tool_calls):
                # Print with line numbers to see where newlines and other issues might be
                lines = raw_call.split("\n")
                logger.info(f"RAW TOOL CALL #{i+1} (multiline format):")
                for line_num, line in enumerate(lines):
                    logger.info(f"    Line {line_num+1}: {repr(line)}")

                # Also print the raw string representation
                logger.info(f"RAW TOOL CALL #{i+1} (repr): {repr(raw_call)}")
                try:
                    # Clean up the raw call string - fix known issues from GPT-4 responses
                    # 1. Remove extra closing braces that sometimes appear
                    clean_call = re.sub(r"\}\s*\}", "}", raw_call)

                    # 2. Try to extract just the valid JSON using regex if there are still issues
                    json_pattern = r'(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]+\}\s*\})'
                    json_match = re.search(json_pattern, clean_call)
                    if json_match:
                        clean_call = json_match.group(1)
                        logger.info(f"Extracted cleaner JSON: {clean_call}")

                    # Try to parse the cleaned JSON
                    call_data = json.loads(clean_call)
                    name = call_data.get("name")
                    args = call_data.get("arguments", {})

                    # Validate that the tool exists
                    if any(tool["function"]["name"] == name for tool in self.tools):
                        logger.info(f"Parsed tool call: {name}, {args}")
                        tool_calls.append({"name": name, "arguments": args})
                    else:
                        logger.warning(f"Unknown tool name: {name}")
                except json.JSONDecodeError as e:
                    # Only log this at INFO level to reduce verbosity in normal output
                    if (
                        self.config.logging_active
                        and hasattr(self, "process_mode")
                        and not self.process_mode
                    ):
                        logger.warning(f"Failed to parse tool call JSON: {e}")
                    else:
                        logger.debug(f"Failed to parse tool call JSON: {e}")

        # Fallback to the library parser if no tool calls were found
        if not tool_calls:
            logger.info(
                "No tool calls found with regex, falling back to library parser"
            )
            for tool in self.tools:
                name = tool["function"]["name"]
                parsed_name, parsed_args, is_error = parse_tool_call(
                    response, [tool], ["tool_call"]
                )

                if not is_error and parsed_name == name:
                    # Only log detailed parsing in non-process mode to reduce verbosity
                    if not hasattr(self, "process_mode") or not self.process_mode:
                        logger.debug(
                            f"Parsed tool call with library: {name}, {parsed_args}"
                        )
                    tool_calls.append({"name": name, "arguments": parsed_args})
                elif parsed_name and parsed_name != "-ERROR-":
                    # Only log parsing failures for non-obvious errors in non-process mode
                    if not hasattr(self, "process_mode") or not self.process_mode:
                        logger.debug(
                            f"Failed tool call parse: {parsed_name}, error: {is_error}"
                        )

        logger.info(f"Final parsed tool calls: {len(tool_calls)}")
        return tool_calls

    def _extract_final_article(self, response: str) -> Optional[str]:
        """Extract final Wikipedia article markdown if present"""
        # Regular expression to match content between Final Step: ```markdown and ``` tags
        pattern = r"Final Step:\s*```markdown\s*(.*?)```"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()
        return None

    def _format_tool_results(self, tool_results: List[Dict]) -> str:
        """Format tool results as a user message"""
        if not tool_results:
            return "No results found."

        formatted_results = ["==== TOOL RESULTS ===="]

        for result in tool_results:
            tool_name = result.get("name", "unknown_tool")
            args = result.get("arguments", {})
            data = result.get("data", [])

            if tool_name == "web_search":
                query = args.get("query", "")
                num_results = args.get("num_results", 5)
                formatted_results.append(
                    f'[WEB SEARCH] query="{query}", num_results={num_results}\n'
                )

                if isinstance(data, list):
                    formatted_results.append(json.dumps(data, indent=2))
                else:
                    formatted_results.append("No results found.")

            elif tool_name == "visit_page":
                url = args.get("url", "")
                formatted_results.append(f'[PAGE EXTRACT] url="{url}"\n')

                if isinstance(data, dict):
                    content = data.get("content", "")
                    title = data.get("title", "")
                    success = data.get("success", False)

                    if success:
                        formatted_results.append(f"Title: {title}")
                        formatted_results.append(f"Content:\n{content[:2000]}...")
                        if len(content) > 2000:
                            formatted_results.append(
                                "\n[Content truncated due to length]"
                            )
                    else:
                        error = data.get("error", "Unknown error")
                        formatted_results.append(f"Error: {error}")
                else:
                    formatted_results.append("Failed to retrieve page content.")

        formatted_results.append("==== END TOOL RESULTS ====")
        return "\n\n".join(formatted_results)

    def _extract_research_facts(self, tool_results: List[Dict], facts: List[str]):
        """Extract important facts from tool results for later evaluation"""
        for result in tool_results:
            tool_name = result.get("name", "")
            data = result.get("data", None)

            if tool_name == "web_search" and isinstance(data, list):
                for item in data:
                    content = item.get("content", "")
                    if content:
                        # Simple sentence extraction - could be enhanced with NLP
                        sentences = re.split(r"(?<=[.!?])\s+", content)
                        for sentence in sentences:
                            if len(sentence) > 30 and sentence not in facts:
                                facts.append(sentence)

            elif tool_name == "visit_page" and isinstance(data, dict):
                content = data.get("content", "")
                if content:
                    paragraphs = content.split("\n\n")
                    for paragraph in paragraphs:
                        if len(paragraph) > 50 and paragraph not in facts:
                            facts.append(paragraph)

    async def _execute_tool_call(self, tool_call: Dict) -> Dict:
        """Execute a tool call and return the result"""
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        result = {"name": tool_name, "arguments": arguments, "data": None}

        try:
            if tool_name == "web_search":
                query = arguments.get("query", "")
                num_results = min(
                    arguments.get("num_results", 5), 10
                )  # Limit to 10 max
                filter_year = arguments.get("filter_year", None)

                # If query is about Wikipedia, provide a helpful message
                if "wikipedia" in query.lower():
                    logger.info(
                        "Query contains 'wikipedia' - providing guidance message"
                    )
                    wikipedia_message = [
                        {
                            "title": "Wikipedia Research Notice",
                            "url": "https://example.com/wikipedia-notice",
                            "content": (
                                "Instead of searching for Wikipedia articles directly, try searching for the "
                                "actual topic or subject. The goal is to create a Wikipedia-style article "
                                "using information from various reliable sources."
                            ),
                            "snippet": (
                                "Instead of searching for Wikipedia articles directly, try searching for "
                                "the actual topic or subject."
                            ),
                            "date": None,
                        }
                    ]
                    result["data"] = wikipedia_message
                else:
                    search_results = self.search_tool.forward(
                        query=query, num_results=num_results, filter_year=filter_year
                    )

                    # Filter out Wikipedia URLs from search results
                    filtered_results = []
                    for item in search_results:
                        url = item.get("url", "").lower()
                        if "wikipedia.org" not in url:
                            filtered_results.append(item)
                        else:
                            logger.info(
                                f"Filtered out Wikipedia URL from search results: {url}"
                            )

                    # Add a notice if results were filtered
                    if len(filtered_results) < len(search_results):
                        logger.info(
                            f"Filtered out {len(search_results) - len(filtered_results)} Wikipedia results"
                        )
                        # Add a notice as the last result if we filtered anything
                        if filtered_results:
                            filtered_results.append(
                                {
                                    "title": "Search Results Notice",
                                    "url": "https://example.com/search-notice",
                                    "content": (
                                        "Some Wikipedia results were automatically filtered out. Please focus "
                                        "on using other reliable sources for your research."
                                    ),
                                    "snippet": (
                                        "Wikipedia results were filtered. Use other reliable sources for "
                                        "your research."
                                    ),
                                    "date": None,
                                }
                            )

                    result["data"] = filtered_results

            elif tool_name == "visit_page":
                url = arguments.get("url", "")

                # Check if the URL is from Wikipedia and block it
                if "wikipedia.org" in url.lower():
                    logger.info(f"Blocking Wikipedia URL: {url}")
                    result["data"] = {
                        "url": url,
                        "title": "Page Not Found",
                        "content": (
                            "Wikipedia pages are not available in this environment. Please search for "
                            "information from other sources."
                        ),
                        "success": False,
                        "error": "Wikipedia pages are blocked in this environment.",
                    }
                else:
                    try:
                        logger.info(f"Attempting to extract content from URL: {url}")
                        page_data = self.extract_tool.forward(url=url)

                        # Log success or partial success
                        if page_data.get("success", False):
                            content_length = len(page_data.get("content", ""))
                            logger.info(
                                f"Successfully extracted {content_length} characters from {url}"
                            )
                        else:
                            error_msg = page_data.get("error", "Unknown error")
                            logger.error(
                                f"Extraction reported failure: {error_msg} for URL: {url}"
                            )

                        result["data"] = page_data
                    except Exception as e:
                        logger.error(
                            f"Exception during content extraction from {url}: {str(e)}"
                        )
                        import traceback

                        logger.error(
                            f"Extraction error traceback: {traceback.format_exc()}"
                        )
                        result["data"] = {
                            "url": url,
                            "title": "Page Extraction Failed",
                            "content": (
                                f"Failed to extract content from the page due to an error: {str(e)}"
                            ),
                            "success": False,
                            "error": f"Exception during extraction: {str(e)}",
                        }

            else:
                logger.warning(f"Unknown tool: {tool_name}")
                result["data"] = {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            result["data"] = {"error": f"Tool execution failed: {str(e)}"}

        return result

    async def _get_model_response(self, messages: List[Dict]) -> str:
        """Get a response from the model for the current conversation state"""
        try:
            # Try to use chat_completion first (which works with OpenAI models)
            try:
                logger.info("Attempting to use chat_completion API")
                completion = await self.server.chat_completion(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=min(
                        4096, self.config.max_token_length
                    ),  # Ensure within OpenAI limits
                )
                return completion.choices[0].message.content
            except (AttributeError, TypeError) as e:
                # If chat_completion fails, fall back to standard completion
                logger.info(
                    f"Chat completion failed: {e}, falling back to standard completion"
                )

                # For non-OpenAI models (local), use the standard completion API with tokenized prompt
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
            import traceback

            logger.error(traceback.format_exc())
            return ""

    async def _next_step(self, episode: EpisodeState) -> Tuple[bool, Dict]:
        """
        Process one step of article research interaction
        Returns (is_terminal, step_data)
        """
        # Get current conversation history
        messages = episode.message_history.copy()

        logger.info("\n==== REQUESTING MODEL RESPONSE ====")
        # Generate model response
        response = await self._get_model_response(messages)

        if not response:
            episode.is_terminal = True
            logger.info("No response received from model")
            return True, {"response": "", "tool_calls": [], "tool_results": []}

        logger.info("\n==== MODEL RESPONSE ====")
        # Print the raw response with repr to see exactly what's in it, including newlines and special chars
        print("\n\n==== RAW MODEL RESPONSE (repr) ====")
        print(repr(response))
        print("==== END RAW MODEL RESPONSE ====\n\n")

        # Also log it normally
        logger.info(response)
        logger.info("==== END MODEL RESPONSE ====")

        # Check for final article
        final_article = self._extract_final_article(response)
        if final_article:
            logger.info("\n==== FINAL ARTICLE DETECTED ====")
            episode.is_terminal = True
            episode.final_article = final_article
            # Add response to history
            episode.message_history.append({"role": "assistant", "content": response})
            return True, {"response": response, "tool_calls": [], "tool_results": []}

        # Extract tool calls for research
        tool_calls = self._parse_tool_calls(response)

        # Hide detailed tool call logging in process mode
        if tool_calls:
            if hasattr(self, "process_mode") and self.process_mode:
                # In process mode, just show a summary
                logger.info(f"Found {len(tool_calls)} tool calls")
            else:
                # In normal mode, show more detailed logging
                logger.info(f"\n==== EXECUTING {len(tool_calls)} TOOL CALLS ====")

        # Execute research tool calls
        tool_results = []
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("name", "unknown")
            logger.info(f"Executing tool call {i+1}: {tool_name}")
            result = await self._execute_tool_call(tool_call)
            tool_results.append(result)

        # Add response and tool results to history
        episode.message_history.append({"role": "assistant", "content": response})

        # Format tool results as a user message
        tool_results_message = self._format_tool_results(tool_results)
        episode.message_history.append(
            {"role": "user", "content": tool_results_message}
        )

        # Update episode state
        episode.steps_taken += 1
        episode.tool_calls.extend(tool_calls)
        episode.tool_results.extend(tool_results)

        # Extract and store research facts for later evaluation
        self._extract_research_facts(tool_results, episode.research_facts)

        # Check if max steps reached
        if episode.steps_taken >= self.config.max_steps:
            logger.info(f"\n==== MAX STEPS REACHED ({self.config.max_steps}) ====")
            episode.is_terminal = True

        return episode.is_terminal, {
            "response": response,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    def _assess_article_quality(
        self, final_article: str, research_facts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate the quality of the final article
        Returns a dictionary of quality metrics
        """
        metrics = {
            "structure_score": 0.0,
            "comprehensiveness_score": 0.0,
            "fact_usage_score": 0.0,
            "overall_quality": 0.0,
        }

        # Basic structure analysis
        if not final_article:
            return metrics

        # Check for section headers
        sections = re.findall(r"^##?\s+.+$", final_article, re.MULTILINE)
        num_sections = len(sections)

        # Check for references
        references = re.findall(
            r"^##?\s*References", final_article, re.MULTILINE | re.IGNORECASE
        )
        has_references = len(references) > 0

        # Calculate structure score
        structure_score = 0.0
        if num_sections >= self.config.min_article_sections:
            structure_score += 0.7
        else:
            structure_score += 0.7 * (num_sections / self.config.min_article_sections)

        if has_references:
            structure_score += 0.3

        metrics["structure_score"] = structure_score

        # Calculate comprehensiveness score based on length and section count
        article_length = len(final_article)
        comp_score = min(1.0, article_length / 3000) * 0.7
        comp_score += min(1.0, num_sections / 5) * 0.3
        metrics["comprehensiveness_score"] = comp_score

        # Calculate fact usage score
        # This is a simplistic approach - could be enhanced with NLP/semantic matching
        fact_usage = 0.0
        if research_facts:
            facts_found = 0
            for fact in research_facts:
                # Check if key phrases from the fact appear in the article
                key_phrases = [p for p in fact.split() if len(p) > 5]
                if key_phrases:
                    for phrase in key_phrases[:5]:  # Use up to 5 phrases per fact
                        if phrase.lower() in final_article.lower():
                            facts_found += 1
                            break

            fact_usage = min(1.0, facts_found / len(research_facts))
        metrics["fact_usage_score"] = fact_usage

        # Calculate overall quality
        overall = structure_score * 0.3 + comp_score * 0.4 + fact_usage * 0.3
        metrics["overall_quality"] = overall

        return metrics

    async def collect_trajectories(
        self, item: Tuple[int, str]
    ) -> Tuple[List[ScoredDataGroup], List]:
        """
        Manage full research trajectory collection

        Args:
            item: Tuple containing (episode_id, topic)

        Returns:
            Tuple of:
            - List of ScoredDataGroup objects: Scored data for training
            - List: Empty list (no backlog items)
        """
        episode_id, topic = item

        # Get or create episode state
        episode = self._get_or_create_episode(episode_id, topic)

        # Detect if we're in process mode
        is_process_mode = hasattr(self, "process_mode") and self.process_mode

        trajectory_data: List[ScoredDataGroup] = []

        # Run episode until terminal state
        while not episode.is_terminal:
            is_terminal, step_data = await self._next_step(episode)

            # Skip steps with no response or no tools used (unless terminal)
            response = step_data.get("response", "")
            if not response:
                continue

            # Create scored data for this step
            step_score = ScoredDataGroup()

            # Tokenize conversation up to this point
            tokenized = tokenize_for_trainer(self.tokenizer, episode.message_history)

            # Score based on tool usage (basic heuristic, improve in future)
            tool_calls = step_data.get("tool_calls", [])
            tool_results = step_data.get("tool_results", [])

            if is_terminal and episode.final_article:
                # Terminal step with article - score based on article quality
                quality_metrics = self._assess_article_quality(
                    episode.final_article, episode.research_facts
                )
                step_score["tokens"] = [tokenized["tokens"]]
                step_score["masks"] = [tokenized["masks"]]
                step_score["scores"] = [
                    quality_metrics["overall_quality"] * 2 - 1
                ]  # Scale to [-1, 1]

                # Record metrics for logging
                quality_metrics["topic"] = episode.topic
                quality_metrics["steps_taken"] = episode.steps_taken
                self.article_quality_metrics.append(quality_metrics)

                # If we're in process mode, perform factual accuracy evaluation
                if is_process_mode:
                    try:
                        # Import here to avoid circular imports
                        import json

                        from environments.hack0.wikipedia.article_evaluator import (
                            ArticleEvaluator,
                        )

                        # Check if OpenAI API key is available
                        openai_api_key = os.environ.get("OPENAI_API_KEY")
                        if openai_api_key:
                            # Initialize article evaluator
                            evaluator = ArticleEvaluator(openai_api_key)

                            # Load reference articles
                            articles_path = os.path.join(
                                os.path.dirname(__file__), "wikipedia_articles.json"
                            )
                            if os.path.exists(articles_path):
                                with open(articles_path, "r") as f:
                                    articles_data = json.load(f)

                                topic = episode.topic
                                generated_article = episode.final_article

                                # Retrieve reference article content
                                reference_content = evaluator.get_reference_article(
                                    articles_data, topic
                                )

                                if reference_content:
                                    # Evaluate article factual accuracy - changed from async to sync
                                    evaluation_results = (
                                        evaluator.evaluate_article_accuracy(
                                            reference_content=reference_content,
                                            generated_article=generated_article,
                                        )
                                    )

                                    # Calculate accuracy score
                                    accuracy_score = evaluator.calculate_accuracy_score(
                                        evaluation_results
                                    )

                                    # Print statistics for this evaluation
                                    if (
                                        evaluation_results
                                        and "statistics" in evaluation_results
                                    ):
                                        stats = evaluation_results["statistics"]
                                        print("\n" + "=" * 80)
                                        print(
                                            f"FACTUAL ACCURACY EVALUATION FOR: {topic}"
                                        )
                                        print("=" * 80)
                                        print(
                                            f"CORRECT:   {stats.get('correct_count', 0)} statements "
                                            f"({stats.get('pct_correct', 0):.1f}%)"
                                        )
                                        print(
                                            f"INCORRECT: {stats.get('incorrect_count', 0)} statements "
                                            f"({stats.get('pct_incorrect', 0):.1f}%)"
                                        )
                                        print(
                                            f"UNKNOWN:   {stats.get('unknown_count', 0)} statements "
                                            f"({stats.get('pct_unknown', 0):.1f}%)"
                                        )
                                        print(
                                            f"TOTAL:     {stats.get('total_count', 0)} statements evaluated"
                                        )
                                        print("-" * 80)
                                        # Remove duplicate raw scores since we're keeping everything in [-1, 1] now

                                        # Keep original scores in their native ranges
                                        # Original quality_score is in [0,1] range
                                        # Convert to [-1,1] range for consistency with accuracy score
                                        quality_score_scaled = (
                                            quality_metrics["overall_quality"] * 2 - 1
                                        )

                                        # Accuracy score is already in [-1,1] range

                                        # Calculate combined score (simple average of the two scores)
                                        combined_score = (
                                            quality_score_scaled + accuracy_score
                                        ) / 2

                                        # This is already in [-1,1] range for ScoredDataGroup
                                        scaled_score = combined_score

                                        print(
                                            f"Original Quality Score: {quality_score_scaled:.4f} (range [-1, 1])"
                                        )
                                        print(
                                            f"Factual Accuracy Score: {accuracy_score:.4f} (range [-1, 1])"
                                        )
                                        print(
                                            f"Combined Final Score:   {combined_score:.4f} (range [-1, 1])"
                                        )
                                        print("=" * 80 + "\n")

                                        # Update the score in step_score
                                        step_score["scores"] = [scaled_score]

                                    # Add accuracy metrics to article_quality_metrics for wandb logging
                                    if (
                                        evaluation_results
                                        and "statistics" in evaluation_results
                                    ):
                                        stats = evaluation_results["statistics"]
                                        accuracy_metrics = {
                                            "pct_correct": stats.get("pct_correct", 0),
                                            "pct_incorrect": stats.get(
                                                "pct_incorrect", 0
                                            ),
                                            "pct_unknown": stats.get("pct_unknown", 0),
                                            "accuracy_score": accuracy_score,
                                        }
                                        # Update the last added quality metrics entry
                                        self.article_quality_metrics[-1].update(
                                            accuracy_metrics
                                        )
                    except Exception as e:
                        print(f"Error evaluating article factual accuracy: {e}")
                        import traceback

                        print(traceback.format_exc())

            elif tool_calls:
                # Non-terminal step with tool usage - score based on usefulness
                step_score["tokens"] = [tokenized["tokens"]]
                step_score["masks"] = [tokenized["masks"]]

                # Simple usefulness heuristic:
                # - Higher score for visiting pages than generic searches
                # - Higher score if results were found than if errors
                usefulness = 0.0
                for result in tool_results:
                    name = result.get("name", "")
                    data = result.get("data", None)

                    if name == "web_search" and isinstance(data, list) and data:
                        usefulness = max(usefulness, 0.6)
                    elif (
                        name == "visit_page"
                        and isinstance(data, dict)
                        and data.get("success", False)
                    ):
                        usefulness = max(usefulness, 0.8)

                step_score["scores"] = [usefulness * 2 - 1]  # Scale to [-1, 1]

            else:
                # Step with no tool usage - low score
                step_score["tokens"] = [tokenized["tokens"]]
                step_score["masks"] = [tokenized["masks"]]
                step_score["scores"] = [-0.5]  # Slight negative score

            # Add messages to the step_score to make them available for wandb logging and HTML rendering
            # We do this through the messages key which is supported in ScoredDataGroup and HTML rendering
            if self.config.include_messages:
                # For HTML rendering, we need to combine all messages into a single markdown string
                # This ensures the entire conversation appears as a single content item

                # First, create the complete conversation as one big markdown document
                # This is what will be shown in the HTML output
                complete_conversation = []

                # Add the topic
                complete_conversation.append(f"# Wikipedia Article: {episode.topic}\n")

                # Include tool calls and research steps
                for i, msg in enumerate(episode.message_history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    # Skip system messages for cleaner output
                    if role == "system":
                        continue

                    # Handle normal messages
                    complete_conversation.append(f"## {role.upper()}")

                    # Special handling for final article
                    if role == "assistant" and "Final Step: ```markdown" in content:
                        article_content = self._extract_final_article(content)
                        if article_content:
                            complete_conversation.append(
                                content.split("Final Step: ```markdown")[0]
                            )  # Add thinking/research
                            complete_conversation.append("### FINAL ARTICLE")
                            complete_conversation.append(
                                f"```markdown\n{article_content}\n```"
                            )
                        else:
                            complete_conversation.append(content)
                    else:
                        complete_conversation.append(content)

                    # If this is an assistant message that triggered tools, add the tool calls
                    if role == "assistant" and i < len(episode.message_history) - 1:
                        next_msg = episode.message_history[i + 1]
                        if next_msg.get(
                            "role"
                        ) == "user" and "==== TOOL RESULTS ====" in next_msg.get(
                            "content", ""
                        ):
                            # Extract tool name from the message
                            tool_content = next_msg.get("content", "")
                            if "[WEB SEARCH]" in tool_content:
                                complete_conversation.append("### 🔍 SEARCH RESULTS")
                            elif "[PAGE EXTRACT]" in tool_content:
                                complete_conversation.append("### 📄 PAGE EXTRACT")
                            complete_conversation.append(
                                "```\n" + tool_content + "\n```"
                            )

                # Join everything into a single string with double newlines between sections
                full_conversation_markdown = "\n\n".join(complete_conversation)

                # Store the full conversation as a single message (for HTML rendering)
                step_score["messages"] = [full_conversation_markdown]

            # For process mode, we only want to keep the final state
            # This ensures we get a single group in the HTML output
            if is_process_mode:
                if is_terminal:
                    # For terminal steps, keep only this step which has the full conversation
                    trajectory_data = [step_score]
                else:
                    # For intermediate steps in process mode, don't add to trajectory_data
                    pass
            else:
                # Normal training mode - add all steps
                trajectory_data.append(step_score)

        # Don't delete the episode yet - we need it for wandb logging
        # Instead, mark it for deletion after wandb logging is complete
        # We'll actually clean it up after handle_send_to_api in handle_env

        return trajectory_data, []

    async def score(
        self, rollout_group_data: List[ScoredDataGroup]
    ) -> List[ScoredDataGroup]:
        """
        Enhanced scoring function that incorporates factual accuracy evaluation.

        Uses OpenAI models to evaluate the factual accuracy of the generated articles
        against reference articles from Wikipedia.
        """
        try:
            # Import here to avoid circular imports
            import json

            from environments.hack0.wikipedia.article_evaluator import ArticleEvaluator

            # Check if OpenAI API key is available
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning(
                    "OPENAI_API_KEY not found. Skipping factual accuracy evaluation."
                )
                return rollout_group_data

            # Initialize article evaluator
            evaluator = ArticleEvaluator(openai_api_key)

            # Load reference articles
            articles_path = os.path.join(
                os.path.dirname(__file__), "wikipedia_articles.json"
            )
            if not os.path.exists(articles_path):
                logger.warning(
                    f"Wikipedia articles file not found at {articles_path}. Skipping factual accuracy evaluation."
                )
                return rollout_group_data

            with open(articles_path, "r") as f:
                articles_data = json.load(f)

            # Process each ScoredDataGroup
            for group in rollout_group_data:
                for i in range(len(group["tokens"])):
                    # Check if this is a terminal step with a final article
                    episode_id = i  # Use index as a proxy for episode_id
                    episode = self.episodes.get(episode_id)

                    if episode and episode.is_terminal and episode.final_article:
                        topic = episode.topic
                        generated_article = episode.final_article

                        # Retrieve reference article content
                        reference_content = evaluator.get_reference_article(
                            articles_data, topic
                        )

                        if reference_content:
                            # Evaluate article factual accuracy
                            evaluation_results = (
                                await evaluator.evaluate_article_accuracy(
                                    reference_content=reference_content,
                                    generated_article=generated_article,
                                )
                            )

                            # Calculate accuracy score
                            accuracy_score = evaluator.calculate_accuracy_score(
                                evaluation_results
                            )

                            # Combine with existing quality metrics
                            quality_metrics = self._assess_article_quality(
                                final_article=generated_article,
                                research_facts=episode.research_facts,
                            )

                            # Adjust the overall quality to include factual accuracy
                            # Original score is in [0,1], we'll combine it with accuracy_score [-1,1]
                            combined_score = (
                                quality_metrics["overall_quality"]
                                + (accuracy_score + 1) / 2
                            ) / 2

                            # Scale to [-1, 1] for compatibility with existing scoring
                            scaled_score = combined_score * 2 - 1

                            # Update the score in the ScoredDataGroup
                            group["scores"][i] = scaled_score

                            # Print statistics for this evaluation
                            if (
                                evaluation_results
                                and "statistics" in evaluation_results
                            ):
                                stats = evaluation_results["statistics"]
                                print("\n" + "=" * 80)
                                print(f"FACTUAL ACCURACY EVALUATION FOR: {topic}")
                                print("=" * 80)
                                print(
                                    f"CORRECT:   {stats.get('correct_count', 0)} statements "
                                    f"({stats.get('pct_correct', 0):.1f}%)"
                                )
                                print(
                                    f"INCORRECT: {stats.get('incorrect_count', 0)} statements "
                                    f"({stats.get('pct_incorrect', 0):.1f}%)"
                                )
                                print(
                                    f"UNKNOWN:   {stats.get('unknown_count', 0)} statements "
                                    f"({stats.get('pct_unknown', 0):.1f}%)"
                                )
                                print(
                                    f"TOTAL:     {stats.get('total_count', 0)} statements evaluated"
                                )
                                print("-" * 80)
                                print(
                                    f"Factual Accuracy Score: {accuracy_score:.4f} (range [-1, 1])"
                                )
                                print(
                                    f"Original Quality Score: {quality_metrics['overall_quality']:.4f} (range [0, 1])"
                                )
                                print(
                                    f"Combined Final Score:   {scaled_score:.4f} (range [-1, 1])"
                                )
                                print("=" * 80 + "\n")

                            # Add accuracy metrics to article_quality_metrics for wandb logging
                            if (
                                evaluation_results
                                and "statistics" in evaluation_results
                            ):
                                stats = evaluation_results["statistics"]

                                accuracy_metrics = {
                                    "pct_correct": stats.get("pct_correct", 0),
                                    "pct_incorrect": stats.get("pct_incorrect", 0),
                                    "pct_unknown": stats.get("pct_unknown", 0),
                                    "accuracy_score": accuracy_score,
                                }

                                # Find the corresponding metrics entry and update it
                                for metrics in self.article_quality_metrics:
                                    if metrics.get("topic") == topic:
                                        metrics.update(accuracy_metrics)
                                        break
                        else:
                            logger.warning(
                                f"No reference article found for topic: {topic}"
                            )

        except Exception as e:
            logger.error(f"Error during factual accuracy evaluation: {e}")
            import traceback

            logger.error(traceback.format_exc())

        return rollout_group_data

    async def setup(self):
        """Set up the environment - load topics, etc."""
        pass

    async def get_next_item(self) -> Tuple[int, str]:
        """Get next episode ID and topic"""
        # Select a random topic
        topic = random.choice(self.topics)
        episode_id = self.iter
        self.iter += 1

        return (episode_id, topic)

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on test set of topics"""
        if not self.config.use_wandb:
            logger.info("Skipping evaluation as wandb is not enabled")
            return

        num_eval = min(self.config.eval_topics, len(self.topics))
        eval_topics = random.sample(self.topics, num_eval)

        logger.info(f"Starting evaluation on {num_eval} topics")

        eval_metrics = {
            "avg_steps": 0.0,
            "avg_quality": 0.0,
            "avg_structure": 0.0,
            "avg_comprehensiveness": 0.0,
            "avg_fact_usage": 0.0,
            "completion_rate": 0.0,
        }

        completed_count = 0
        total_steps = 0
        quality_scores = {
            "overall": [],
            "structure": [],
            "comprehensiveness": [],
            "fact_usage": [],
        }

        # Run evaluation episodes
        for eval_idx, topic in enumerate(eval_topics):
            episode_id = 1000000 + eval_idx  # High range for eval episodes
            episode = self._get_or_create_episode(episode_id, topic)

            # Run episode until terminal
            while not episode.is_terminal:
                is_terminal, _ = await self._next_step(episode)
                if is_terminal:
                    break

            # Record metrics
            total_steps += episode.steps_taken

            if episode.final_article:
                completed_count += 1
                quality_metrics = self._assess_article_quality(
                    episode.final_article, episode.research_facts
                )

                quality_scores["overall"].append(quality_metrics["overall_quality"])
                quality_scores["structure"].append(quality_metrics["structure_score"])
                quality_scores["comprehensiveness"].append(
                    quality_metrics["comprehensiveness_score"]
                )
                quality_scores["fact_usage"].append(quality_metrics["fact_usage_score"])

            # Clean up episode
            if episode_id in self.episodes:
                del self.episodes[episode_id]

        # Calculate averages
        if num_eval > 0:
            eval_metrics["avg_steps"] = total_steps / num_eval
            eval_metrics["completion_rate"] = completed_count / num_eval

            if completed_count > 0:
                eval_metrics["avg_quality"] = (
                    sum(quality_scores["overall"]) / completed_count
                )
                eval_metrics["avg_structure"] = (
                    sum(quality_scores["structure"]) / completed_count
                )
                eval_metrics["avg_comprehensiveness"] = (
                    sum(quality_scores["comprehensiveness"]) / completed_count
                )
                eval_metrics["avg_fact_usage"] = (
                    sum(quality_scores["fact_usage"]) / completed_count
                )

        # Store metrics for wandb logging
        self.eval_metrics = [
            ("eval/avg_steps", eval_metrics["avg_steps"]),
            ("eval/completion_rate", eval_metrics["completion_rate"]),
            ("eval/avg_quality", eval_metrics["avg_quality"]),
            ("eval/avg_structure", eval_metrics["avg_structure"]),
            ("eval/avg_comprehensiveness", eval_metrics["avg_comprehensiveness"]),
            ("eval/avg_fact_usage", eval_metrics["avg_fact_usage"]),
        ]

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        """
        Save complete conversation histories to wandb

        This captures the full research and article creation process,
        including all tool calls and intermediate steps
        """
        # Use the base implementation first for basic text and scores
        await super().add_rollouts_for_wandb(scored_data, item)

        # Now also save the complete conversation history if we have it
        if item is not None and isinstance(item, tuple) and len(item) > 0:
            episode_id = item[0]
            episode = self.episodes.get(episode_id)

            if episode and hasattr(episode, "message_history"):
                # Format the conversation with relevant metadata
                num_keep = self.config.num_rollouts_per_group_for_logging
                if num_keep == -1:
                    num_keep = self.config.group_size

                # Add detailed conversation data to rollouts
                # We'll extract this in create_rollout_table to create a more detailed table
                for i in range(min(num_keep, len(scored_data["tokens"]))):
                    # Add chat history to the most recent entry in rollouts_for_wandb
                    if len(self.rollouts_for_wandb) > 0 and i < len(
                        self.rollouts_for_wandb[-1]
                    ):
                        entry = list(self.rollouts_for_wandb[-1][i])
                        # Append the message history to the existing tuple
                        entry.append(
                            {
                                "topic": episode.topic,
                                "steps_taken": episode.steps_taken,
                                "is_terminal": episode.is_terminal,
                                "message_history": episode.message_history,
                                "tool_calls": episode.tool_calls,
                                "tool_results": episode.tool_results,
                            }
                        )
                        # Replace the tuple with our updated entry
                        self.rollouts_for_wandb[-1][i] = tuple(entry)

    async def create_rollout_table(self, wandb_metrics):
        """
        Create a detailed wandb table with complete conversation histories

        This expands on the base implementation to include full chat histories
        and research steps in a structured format
        """
        if len(self.rollouts_for_wandb) > 0:
            # First create the basic table with text and scores
            basic_table = wandb.Table(columns=["text", "score"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    # Check if this is a basic entry (just text and score) or has chat history
                    if len(item) == 2:
                        basic_table.add_data(item[0], item[1])
                    else:
                        basic_table.add_data(item[0], item[1])

            wandb_metrics["train/rollouts"] = basic_table

            # Create a detailed table with conversation histories
            # This will only include entries that have chat history
            detailed_table = wandb.Table(
                columns=[
                    "topic",
                    "steps_taken",
                    "score",
                    "full_conversation",
                    "tool_calls_count",
                    "has_final_article",
                ]
            )

            for group in self.rollouts_for_wandb:
                for item in group:
                    # Check if this entry has chat history
                    if len(item) > 2:
                        conversation_data = item[2]

                        # Extract conversation metadata
                        topic = conversation_data.get("topic", "Unknown")
                        steps_taken = conversation_data.get("steps_taken", 0)
                        tool_calls = conversation_data.get("tool_calls", [])
                        message_history = conversation_data.get("message_history", [])

                        # Format full conversation as a string
                        conversation_text = "\n\n".join(
                            [
                                f"[{msg.get('role', 'unknown')}]\n{msg.get('content', '')}"
                                for msg in message_history
                            ]
                        )

                        # Check if there's a final article
                        has_final_article = (
                            "Final Step: ```markdown" in conversation_text
                        )

                        detailed_table.add_data(
                            topic,
                            steps_taken,
                            item[1],  # Score
                            conversation_text,
                            len(tool_calls),
                            has_final_article,
                        )

            wandb_metrics["train/detailed_conversations"] = detailed_table

        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb"""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add eval metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value

        # Clear metrics for next round
        self.eval_metrics = []

        # Add article quality metrics if available
        if self.article_quality_metrics:
            # Calculate averages
            avg_quality = sum(
                m["overall_quality"] for m in self.article_quality_metrics
            ) / len(self.article_quality_metrics)
            avg_steps = sum(
                m["steps_taken"] for m in self.article_quality_metrics
            ) / len(self.article_quality_metrics)

            wandb_metrics["train/avg_article_quality"] = avg_quality
            wandb_metrics["train/avg_steps_per_article"] = avg_steps

            # Convert to [-1, 1] range for consistency with other metrics
            wandb_metrics["train/article_quality_score"] = avg_quality * 2 - 1

            # Add factual accuracy metrics if available
            if any("accuracy_score" in m for m in self.article_quality_metrics):
                # Calculate average accuracy metrics
                accuracy_metrics = [
                    m for m in self.article_quality_metrics if "accuracy_score" in m
                ]
                if accuracy_metrics:
                    # Calculate raw accuracy statistics
                    avg_accuracy_score = sum(
                        m["accuracy_score"] for m in accuracy_metrics
                    ) / len(accuracy_metrics)
                    avg_pct_correct = sum(
                        m.get("pct_correct", 0) for m in accuracy_metrics
                    ) / len(accuracy_metrics)
                    avg_pct_incorrect = sum(
                        m.get("pct_incorrect", 0) for m in accuracy_metrics
                    ) / len(accuracy_metrics)
                    avg_pct_unknown = sum(
                        m.get("pct_unknown", 0) for m in accuracy_metrics
                    ) / len(accuracy_metrics)

                    # Log raw factual accuracy metrics
                    wandb_metrics["train/avg_factual_accuracy"] = avg_accuracy_score
                    wandb_metrics["train/avg_pct_correct"] = avg_pct_correct
                    wandb_metrics["train/avg_pct_incorrect"] = avg_pct_incorrect
                    wandb_metrics["train/avg_pct_unknown"] = avg_pct_unknown

                    # Calculate combined scores
                    combined_scores = []
                    for m in accuracy_metrics:
                        # Convert quality score from [0,1] to [-1,1]
                        quality_score_scaled = m["overall_quality"] * 2 - 1
                        # Take average of quality and accuracy scores
                        combined_score = (
                            quality_score_scaled + m["accuracy_score"]
                        ) / 2
                        combined_scores.append(combined_score)

                    # Calculate average combined score
                    if combined_scores:
                        avg_combined_score = sum(combined_scores) / len(combined_scores)
                        wandb_metrics["train/avg_combined_score"] = avg_combined_score

                        # Add a summary metric that includes both article quality and factual accuracy
                        # This provides a comprehensive score for overall article quality including factual accuracy
                        wandb_metrics["train/overall_article_score"] = (
                            avg_combined_score
                        )

            # Create a table of article metrics
            if wandb.run is not None:
                # Add factual accuracy columns if available
                columns = [
                    "topic",
                    "steps",
                    "overall_quality",
                    "structure",
                    "comprehensiveness",
                    "fact_usage",
                ]

                # Check if we have factual accuracy metrics
                if any("accuracy_score" in m for m in self.article_quality_metrics):
                    columns.extend(
                        [
                            "factual_accuracy",
                            "pct_correct",
                            "pct_incorrect",
                            "pct_unknown",
                            "combined_score",
                        ]
                    )

                table = wandb.Table(columns=columns)

                for metric in self.article_quality_metrics:
                    row_data = [
                        metric["topic"],
                        metric["steps_taken"],
                        metric["overall_quality"],
                        metric["structure_score"],
                        metric["comprehensiveness_score"],
                        metric["fact_usage_score"],
                    ]

                    # Add factual accuracy metrics if available
                    if "accuracy_score" in metric:
                        # Calculate combined score
                        quality_score_scaled = metric["overall_quality"] * 2 - 1
                        combined_score = (
                            quality_score_scaled + metric["accuracy_score"]
                        ) / 2

                        row_data.extend(
                            [
                                metric.get("accuracy_score", 0),
                                metric.get("pct_correct", 0),
                                metric.get("pct_incorrect", 0),
                                metric.get("pct_unknown", 0),
                                combined_score,
                            ]
                        )

                    table.add_data(*row_data)

                wandb_metrics["train/article_quality"] = table

            # Clear for next round
            self.article_quality_metrics = []

        await super().wandb_log(wandb_metrics)

    @classmethod
    def cli(cls):
        """Command-line interface entry point"""
        super().cli()


if __name__ == "__main__":
    WikipediaArticleCreatorEnv.cli()
