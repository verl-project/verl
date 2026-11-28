"""
Tavily integration tools for SmolAgents.
These tools replace the SerpAPI and SimpleTextBrowser based tools with Tavily's content extraction service.
"""

import os
from typing import Any, Dict, List, Optional

from smolagents import Tool
from tavily import TavilyClient


class TavilyExtractTool(Tool):
    name = "visit_page"
    description = """Visit a webpage at a given URL and extract its content.

    Returns an object containing:
    - url: The URL that was visited
    - title: The title of the webpage
    - content: The full text content of the webpage
    - success: Boolean indicating if the extraction was successful
    - error: Error message if extraction failed (null if successful)
    """
    inputs = {"url": {"type": "string", "description": "The URL to visit"}}
    output_type = "object"

    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key)

    def forward(self, url: str) -> Dict[str, Any]:
        """
        Visit a webpage and extract its content.

        Args:
            url: The URL to visit

        Returns:
            A dictionary containing:
            - url: The URL that was visited
            - title: The title of the webpage
            - content: The text content of the webpage
            - success: Boolean indicating if the extraction was successful
            - error: Error message if extraction failed

        Note: This function returns the extracted content without printing it.
        """
        try:
            print(f"\n🌐 VISIT PAGE: {url}")
            response = self.client.extract(
                urls=url, include_images=False, extract_depth="basic"
            )

            if not response or not response.get("results"):
                error_msg = f"No results in Tavily extraction response for {url}"
                print(f"Error: {error_msg}")
                print(f"Full response: {response}")

                # Provide user-friendly content with the error
                return {
                    "url": url,
                    "title": "Content Extraction Failed",
                    "content": (
                        "The content extraction service couldn't retrieve information from this page. "
                        "This can happen with certain websites that have access restrictions or complex "
                        "layouts. Try using a different source for this information."
                    ),
                    "success": False,
                    "error": error_msg,
                }

            # Extract content, handle potential missing fields
            try:
                content = response["results"][0]["raw_content"]
                content_length = len(content)
                print(f"Successfully extracted {content_length} characters")
            except (KeyError, IndexError) as ke:
                error_msg = f"Missing data in Tavily response for {url}: {str(ke)}"
                print(f"Error: {error_msg}")
                print(f"Response structure: {response.keys()}")
                if "results" in response and response["results"]:
                    print(f"Result keys: {response['results'][0].keys()}")

                # Return partial data if available
                content = response.get("results", [{}])[0].get("raw_content", "")
                if not content:
                    content = (
                        "The extraction service was able to access the page but returned "
                        "incomplete content."
                    )

            title = response.get("results", [{}])[0].get("title", "Unknown title")

            # Format the content as a structured object
            return {
                "url": url,
                "title": title,
                "content": content,
                "success": bool(content),  # Only mark as successful if we have content
                "error": None if content else "Extracted content was empty",
            }
        except Exception as e:
            import traceback

            error_msg = f"Error extracting content from {url}: {str(e)}"
            trace = traceback.format_exc()
            print(f"Error: {error_msg}")
            print(f"Traceback: {trace}")

            # Provide a user-friendly error message
            return {
                "url": url,
                "title": "Page Access Error",
                "content": (
                    "There was a technical problem accessing this page. This might be due to the "
                    "website blocking automated access or requiring authentication. Try using a "
                    "different source for this information."
                ),
                "success": False,
                "error": error_msg,
            }


class TavilySearchTool(Tool):
    name = "web_search"
    description = """Perform a web search query and return the search results.

    Returns an array of search result objects, each containing:
    - title: The title of the search result
    - url: The URL of the search result
    - content: The full text content from the search result
    - snippet: A text snippet from the content (same as content field)
    - date: The publication date if available (may be null)
    """
    inputs = {
        "query": {"type": "string", "description": "The web search query to perform."},
        "num_results": {
            "type": "integer",
            "description": "Number of results to return (default: 10)",
            "nullable": True,
        },
        "filter_year": {
            "type": "string",
            "description": "Filter results to a specific year",
            "nullable": True,
        },
    }
    output_type = "array"

    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key)

    def forward(
        self,
        query: str,
        num_results: Optional[int] = 10,
        filter_year: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search.

        Args:
            query: The search query
            num_results: Number of results to return (default: 10)
            filter_year: Filter results to a specific year (optional)

        Returns:
            A list of search result objects, each containing:
            - title: The title of the search result
            - url: The URL of the search result
            - content: The content snippet from the search result
            - date: The date of the content if available
        """
        try:
            # Add search emoji
            print(f"\n🔍 WEB SEARCH: {query} (max results: {num_results})")
            search_params = {
                "query": query,
                "search_depth": "advanced",
                "max_results": num_results,  # Default is already handled in the function signature
            }

            # Add year filter if provided
            if filter_year:
                search_params["query"] += f" {filter_year}"

            # Use Tavily's search API
            response = self.client.search(**search_params)

            if not response.get("results"):
                return []

            # Convert Tavily results to the expected format for the agent
            formatted_results = []
            for result in response["results"]:
                formatted_results.append(
                    {
                        "title": result.get("title", "No title"),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "snippet": result.get(
                            "content", ""
                        ),  # For compatibility with expected format
                        "date": result.get("published_date", None),
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")
            return []
