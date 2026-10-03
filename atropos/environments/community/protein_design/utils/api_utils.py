import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def load_api_key() -> Optional[str]:
    """
    Load the NVIDIA NIM API key from environment variables.

    Returns:
        The API key from environment variables, or None if not found
    """
    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        logger.error(
            "NVIDIA_NIM_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
        return None

    return api_key
