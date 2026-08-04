import logging
import os

from mcp.server.fastmcp import FastMCP

# Setup logging to a file
# Adjust the log file path if necessary, perhaps to be relative to this script's location
# or a dedicated logs directory.
log_file_path = os.path.join(os.path.dirname(__file__), "math_server_official.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="w"),  # 'w' to overwrite each run
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

mcp = FastMCP("Official Math Server 🚀")


@mcp.tool()
def add(a: int, b: int) -> int:  # Changed return type hint to int
    """Add two numbers and return the result"""
    logger.info(f"Executing add tool with a={a}, b={b}")
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:  # Changed return type hint to int
    """Multiply two numbers and return the result"""
    logger.info(f"Executing multiply tool with a={a}, b={b}")
    return a * b


if __name__ == "__main__":
    logger.info(
        f"Starting Official MCP math_server.py with STDIO transport... Log file: {log_file_path}"
    )
    mcp.run(transport="stdio")  # Ensure stdio transport is used as in server_stdio.py
