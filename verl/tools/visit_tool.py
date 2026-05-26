# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Website Visit Tool for extracting webpage content.

This module provides the VisitTool for building agents that can visit webpages
and extract their content, optionally converting to markdown format using services
like Jina Reader.
"""

import time
import asyncio
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, List
from uuid import uuid4
import tempfile

from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from markdownify import markdownify as md
import tiktoken
import pymupdf4llm  # isort: skip
import json
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options

import aiohttp
import ray

from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")
PAGE_SIZE = 95000

SUMMARY_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
4. **If goal is empty, just extract the most relevant information from the content, and summarize it in a concise paragraph(s).

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisitExecutionWorker:
    """Worker for executing website visit operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="visit-rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing visit: {e}")
                    raise
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visit_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visit execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisitExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class VisitTool(BaseTool):
    """Website Visit tool for extracting content from web pages.

    This tool provides functionality to visit URLs and extract their content,
    optionally converting to markdown format. It supports integration with
    services like Jina Reader or custom browser wrappers.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the visit tool to fetch webpage content
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize VisitTool with configuration and schema.

        Args:
            config: Configuration dictionary containing:
                - visit_service_url: URL of the website visit service (e.g., Jina Reader)
                - num_workers: Number of concurrent workers (default: 10)
                - rate_limit: Rate limit for requests (default: 10)
                - timeout: Request timeout in seconds (default: 30)
                - enable_global_rate_limit: Enable rate limiting (default: True)
                - use_jina: Whether to use Jina Reader API (default: True)
                - jina_api_key: API key for Jina (optional, can use env JINA_API_KEY)
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "visit_webpage",
                    "description": "Visits a webpage and returns its content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to visit"
                            },
                            "goal": {
                                "type": "string",
                                "description": "The goal of the visit for webpage(s)."
                            }
                        },
                        "required": ["url", "goal"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.use_summary = config.get("use_summary", False)

        self.execution_pool = init_visit_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # Visit service configuration
        self.visit_service_url = config.get("visit_service_url", "")
        self.use_jina = config.get("use_jina", True)
        self.jina_api_key = config.get("jina_api_key") or os.getenv("JINA_API_KEY")

        # if not self.visit_service_url and not self.use_jina:
        #     raise ValueError("Either visit_service_url or use_jina must be configured")
        
        self.openai_api_key = config.get("summary_api_key") or os.environ.get("OPENAI_API_KEY")
        self.summary_base_url = config.get("summary_base_url") or os.environ.get("OPENAI_BASE_URL")
        self.summary_model_name = config.get("summary_model_name", "gpt-4o-mini")
        
        logger.info(f"Initialized VisitTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_visit(self, instance_id: str, url: str, goal: str, timeout: int) -> tuple[str, dict]:
        """Execute website visit operation.

        Args:
            instance_id: Tool instance ID
            url: URL to visit
            goal: The goal or purpose of the visit
            timeout: Request timeout in seconds

        Returns:
            Tuple of (content_text, metadata)
        """
        import asyncio
        url = url.replace("https://r.jina.ai/", "")
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Allow disabling Crawl4AI via env var (recommended for training due to
            # Playwright fork-safety issues with Ray workers).
            use_crawl4ai = os.environ.get("VISIT_USE_CRAWL4AI", "0") == "1"

            if self.use_jina:
                content, metadata = loop.run_until_complete(self._fetch_webpage_content_with_jina(url, timeout))
                # If Jina fails, fallback chain: [Crawl4AI?] -> trafilatura -> custom service
                if metadata.get("status") != "success":
                    if use_crawl4ai:
                        logger.info(f"[visit] Jina failed for {url}, trying Crawl4AI fallback")
                        content, metadata = self._fetch_webpage_with_crawl4ai(url, timeout)
                    if metadata.get("status") != "success":
                        logger.info(f"[visit] Falling back to trafilatura for {url}")
                        content, metadata = self._fetch_webpage_with_trafilatura(url, timeout)
                        if metadata.get("status") != "success":
                            content, metadata = loop.run_until_complete(
                                self._fetch_webpage_content_with_custom_service(url, timeout=timeout)
                            )
            else:
                # No Jina: prefer trafilatura (lightweight, fork-safe)
                content, metadata = self._fetch_webpage_with_trafilatura(url, timeout)
                if metadata.get("status") != "success":
                    if use_crawl4ai:
                        content, metadata = self._fetch_webpage_with_crawl4ai(url, timeout)
                    if metadata.get("status") != "success":
                        content, metadata = loop.run_until_complete(
                            self._fetch_webpage_content_with_custom_service(url, timeout=timeout)
                        )
            
            if self.use_summary and metadata.get("status") == "success":
                result, metadata = self.process_webpage_content(content, url, goal, metadata, timeout)
            else:
                result, metadata = self.process_webpage_content(content, url, goal, metadata, timeout)
            return result, metadata
        finally:
            loop.close()

    def _fetch_webpage_with_crawl4ai(self, url: str, timeout: int = 30) -> tuple[str, dict]:
        """Fetch webpage using Crawl4AI (open-source, no API key needed).
        Crawl4AI provides intelligent web crawling with markdown conversion.
        Falls back gracefully if crawl4ai is not installed.
        """
        metadata = {"url": url, "status": "unknown", "error": None}
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        except ImportError:
            logger.warning("[visit] crawl4ai not installed, skipping")
            metadata["status"] = "error"
            metadata["error"] = "crawl4ai not installed"
            return "[visit] Failed to read page.", metadata

        async def _crawl():
            browser_config = BrowserConfig(headless=True)
            run_config = CrawlerRunConfig(
                word_count_threshold=10,
                page_timeout=timeout * 1000,  # ms
            )
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)
                if result.success and result.markdown and len(result.markdown.strip()) > 50:
                    return result.markdown[:PAGE_SIZE], {
                        **metadata,
                        "status": "success",
                        "content_length": len(result.markdown),
                    }
                else:
                    error = result.error_message if hasattr(result, 'error_message') else "empty content"
                    return "[visit] Failed to read page.", {
                        **metadata,
                        "status": "error",
                        "error": error,
                    }

        try:
            # Run in a new event loop since we may be in a sync context
            loop = asyncio.new_event_loop()
            try:
                content, meta = loop.run_until_complete(_crawl())
                return content, meta
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"[visit] Crawl4AI error for {url}: {e}")
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return "[visit] Failed to read page.", metadata

    def _fetch_webpage_with_trafilatura(self, url: str, timeout: int = 30) -> tuple[str, dict]:
        """Fetch webpage using trafilatura (open-source, no API key needed).
        Falls back to BeautifulSoup if trafilatura extraction fails.
        """
        metadata = {"url": url, "status": "unknown", "error": None}
        try:
            import trafilatura
        except ImportError:
            logger.warning("[visit] trafilatura not installed, skipping")
            metadata["status"] = "error"
            metadata["error"] = "trafilatura not installed"
            return "[visit] Failed to read page.", metadata

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                metadata["status"] = "error"
                metadata["error"] = f"HTTP {response.status_code}"
                return f"[visit] Failed to fetch page (HTTP {response.status_code}).", metadata

            html = response.text
            content = trafilatura.extract(
                html,
                include_links=True,
                include_tables=True,
                favor_recall=True,
                output_format="txt",
            )
            if content and len(content.strip()) > 50:
                metadata["status"] = "success"
                metadata["content_length"] = len(content)
                logger.debug(f"[visit] trafilatura extraction OK for {url}")
                return content[:PAGE_SIZE], metadata

            # trafilatura returned nothing useful, try BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if text and len(text.strip()) > 50:
                metadata["status"] = "success"
                metadata["content_length"] = len(text)
                logger.debug(f"[visit] BeautifulSoup extraction OK for {url}")
                return text[:PAGE_SIZE], metadata

            metadata["status"] = "error"
            metadata["error"] = "empty content"
            return "[visit] Empty content.", metadata
        except Exception as e:
            logger.warning(f"[visit] trafilatura error for {url}: {e}")
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return "[visit] Failed to read page.", metadata
    
    def truncate_to_tokens(self, text: str, max_tokens: int = 95000) -> str:
        encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    
    def pdf_to_markdown(self, pdf_content: bytes) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_content)
                tmp_path = tmp_file.name

            try:
                markdown_content = pymupdf4llm.to_markdown(tmp_path)
                return markdown_content
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except requests.exceptions.RequestException as e:
            return f"Error downloading PDF: {str(e)}"
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
         
    def url_to_markdown(self, url: str) -> str | List[Dict[str, Any]]:
    # for arxiv pdf, use html link instead of simpler scraping
        if url.startswith("https://arxiv.org/pdf/"):
            url = url.replace("arxiv.org/pdf/", "arxiv.org/html/").removesuffix(".pdf")

        try:
            # Fetch the web page
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type")
            if "pdf" in content_type:
                return self.pdf_to_markdown(response.content)

            # For images, directly return the image URL as message content
            if content_type.startswith("image/"):
                return f"Its an image_url, we can't process image inputs at the moment. Here is the image url: {url}"

            # Parse with Beautiful Soup
            soup = BeautifulSoup(response.content, "html.parser")

            # Optional: Remove unwanted elements (scripts, styles, etc.)
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Convert to Markdown
            markdown_content = MarkdownConverter().convert_soup(soup)

            return markdown_content
       
        except Exception as e:
            logging.error(f"[visit] Failed to read page via webpage scrapper service: {e}")
            return ""
        
    # def url_to_markdown_js(sel, url, wait_time=3):
    #     """
    #     Convert URL content to Markdown (with JavaScript rendering using Selenium)

    #     Args:
    #         url: The URL to convert
    #         wait_time: Time to wait for JavaScript to render (seconds)
    #     """
    #     chrome_options = Options()
    #     chrome_options.add_argument("--headless")
    #     chrome_options.add_argument("--no-sandbox")
    #     chrome_options.add_argument("--disable-dev-shm-usage")
    #     chrome_options.add_argument("--disable-gpu")
    #     chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    #     driver = None
    #     try:
    #         driver = webdriver.Chrome(options=chrome_options)
    #         driver.get(url)

    #         time.sleep(wait_time)

    #         # Get page source after JavaScript rendering
    #         html_content = driver.page_source

    #         # Convert to Markdown
    #         markdown_content = md(html_content, heading_style="ATX")

    #         return markdown_content
    #     except Exception as e:
    #         return f"Error: {str(e)}"
    #     finally:
    #         if driver:
    #             driver.quit()
                
    def call_summary_service(self, msgs, max_retries=2):
        client = OpenAI(api_key=self.openai_api_key or "EMPTY", base_url=self.summary_base_url or None)
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=self.summary_model_name,
                    messages=msgs,
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except Exception as e:
                logging.debug(f"[visit] Summary attempt {attempt+1} failed: {e}")
                if attempt == (max_retries - 1):
                    return ""
                continue
                
    def process_webpage_content(self, content: str, url: str, goal: str, metadata: dict, timeout: int=10) -> tuple[str, dict]:
        """Process a webpage content and return the summary."""
        
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 3))
        if metadata.get("status") == "success":
            content = self.truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role":"user","content": SUMMARY_PROMPT.format(webpage_content=content, goal=goal)}]
            parse_retry_times = 0
            raw = self.call_summary_service(messages, max_retries=max_retries)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[visit] Summary url[{url}] " 
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                logging.info(status_msg)
                content = content[:truncate_length]
                extraction_prompt = SUMMARY_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = self.call_summary_service(messages, max_retries=max_retries)
                summary_retries -= 1

            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = self.call_summary_service(messages, max_retries=max_retries)
                    parse_retry_times += 1
            
            if parse_retry_times >= 3 or summary_retries < 0:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                metadata["status"] = "failed"
                logger.debug("[visit] Could not generate valid summary after maximum retries but webpage was accessed")
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"
                metadata["status"] = "success"
                logger.debug(f"Successfully fetched content from {url} via visit_tool service")

            metadata["content_length"] = len(useful_information)
            return useful_information, metadata

        # If no valid content was obtained after all retries
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            metadata["status"] = "failed"
            metadata["content_length"] = len(useful_information)
            logger.debug(f"Unsuccessfully fetched content from {url} via visit_tool service")
            return useful_information, metadata
            
    async def _fetch_webpage_content_with_custom_service(self, url: str, timeout: int=10) -> tuple[str, dict]:
        """Read a webpage and return the content."""
        metadata = {"url": url, "status": "unknown", "error": None}
        try:
            page = await asyncio.to_thread(self.url_to_markdown, url)
            # page = self.url_to_markdown_js(url=url)
           
            if page == "":
                metadata["status"] = "failed"
                logger.debug(f"Unsuccessfully fetched content from {url} via custom webpage scrapper service")
                return "[visit] Failed to read page.", metadata
            elif page.startswith("image_url:"):
                metadata["status"] = "success"
                metadata["content_length"] = len(page)
                logger.debug(f"Successfully fetched image URL from {url} via custom webpage scrapper service")
                return page, metadata
            else:
                metadata["status"] = "success"
                metadata["content_length"] = len(page)
                logger.debug(f"Successfully fetched content from {url} via custom webpage scrapper service")
            content = page[:PAGE_SIZE]
            return content, metadata
            
        except Exception as e:
            logging.error("[visit] failed with exception: " + str(e))
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return "[visit] Failed to read page.", metadata
        
        

    async def _fetch_webpage_content_with_jina(self, url: str, timeout: int) -> tuple[str, dict]:
        """Fetch webpage content using configured service.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Tuple of (content, metadata)
        """
        metadata = {"url": url, "status": "unknown", "error": None}

        try:
            # Use Jina Reader API
            jina_url = f"https://r.jina.ai/{url}"
            headers = {}
            if self.jina_api_key:
                headers["Authorization"] = f"Bearer {self.jina_api_key}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    jina_url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        metadata["status"] = "success"
                        metadata["content_length"] = len(content)
                        logger.debug(f"Successfully fetched content from {url} via Jina Reader API")
                        return content, metadata
                    else:
                        error_msg = f"HTTP {response.status}: {await response.text()}"
                        metadata["status"] = "error"
                        metadata["error"] = error_msg
                        logger.error(f"Unsuccessfully fetched content from {url} via Jina Reader API - {error_msg}")
                        return f"[visit] Failed to read page.", metadata
        
        except Exception as e:
            error_msg = str(e)
            metadata["status"] = "error"
            metadata["error"] = error_msg
            logger.error(f"Exception fetching {url} via Jina Reader API -  {error_msg}")
            return f"[visit] Failed to read page.", metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the visit tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing url (string or list of strings), goal, and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response containing webpage content
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        raw_url = parameters.get("url")
        goal = parameters.get("goal", "")
        timeout = parameters.get("timeout", self.timeout)

        if isinstance(raw_url, str):
            urls: list[str] = [raw_url]
        elif isinstance(raw_url, list):
            urls = raw_url
        else:
            error_msg = "Error: 'url' is missing or has invalid type; expected string or list of strings."
            logger.error(f"[VisitTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), 0.0, {"error": "invalid_url"}

        if not urls:
            error_msg = "Error: 'url' list is empty."
            logger.error(f"[VisitTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), 0.0, {"error": "invalid_url"}

        if not all(isinstance(u, str) for u in urls):
            error_msg = "Error: All entries in 'url' list must be strings."
            logger.error(f"[VisitTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), 0.0, {"error": "invalid_url"}

        invalid_formats = [u for u in urls if not u.startswith(("http://", "https://"))]
        if invalid_formats:
            first_invalid = invalid_formats[0]
            error_msg = f"Error: Invalid URL format: {first_invalid}"
            logger.error(f"[VisitTool] {error_msg}")
            return ToolResponse(text=error_msg), 0.0, {"error": "invalid_url_format"}

        # Execute visit using Ray execution pool
        try:
            tasks = [
                self.execution_pool.execute.remote(self.execute_visit, instance_id, url, goal, timeout)
                for url in urls
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            combined_contents: list[str] = []
            metrics_list: list[dict[str, Any]] = []

            for url, result in zip(urls, results):

                if isinstance(result, Exception):
                    error_msg = f"Visit execution failed for {url}: {result}"
                    logger.error(f"[VisitTool] {error_msg}")
                    combined_contents.append(error_msg)
                    metrics_list.append({
                        "url": url,
                        "status": "error",
                        "content_length": 0,
                        "error": str(result),
                    })
                    continue

                content, metadata = result
                combined_contents.append(content)
                metrics_list.append({
                    "url": metadata.get("url", url),
                    "status": metadata.get("status", "unknown"),
                    "content_length": metadata.get("content_length", 0),
                    "error": metadata.get("error"),
                })
                

            combined_response = "\n=======\n".join(combined_contents)
            self._instance_dict[instance_id]["reward"].append(combined_response)
            metrics: dict[str, Any]
            metrics = metrics_list[0] if len(metrics_list) == 1 else {"results": metrics_list}

            return ToolResponse(text=combined_response), 0.0, metrics

        except Exception as e:
            error_msg = f"Visit execution failed: {e}"
            logger.error(f"[VisitTool] {error_msg}")
            return ToolResponse(text=error_msg), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        """Calculate reward for the tool instance."""
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


def _build_cli_visit_tool() -> VisitTool:
    """Create a VisitTool instance for manual URL inspection."""

    schema = OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="visit_webpage",
            description="Fetch a URL and convert it to markdown for inspection.",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "url": OpenAIFunctionPropertySchema(
                        type="string",
                        description="URL to fetch for markdown preview.",
                    )
                },
                required=["url"],
            ),
        ),
    )

    # Disable Jina usage so we exercise the local markdown conversion path.
    return VisitTool(config={"use_jina": True, "use_summary": True, "summary_model_name": "gpt-5-mini"}, tool_schema=schema)


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Inspect url_to_markdown output for a URL.")
    # parser.add_argument("url", help="URL to fetch and convert to markdown")
    # args = parser.parse_args()
    url = "https://en.wikipedia.org/wiki/Whitemantle_Range"
    tool = _build_cli_visit_tool()
    content, metadata = tool.execute_visit(instance_id="cli-test", url=url, goal="", timeout=30)
    print(content)
    print(metadata)