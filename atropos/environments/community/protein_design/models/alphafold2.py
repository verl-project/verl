import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://health.api.nvidia.com/v1/biology/deepmind/alphafold2"
DEFAULT_STATUS_URL = "https://health.api.nvidia.com/v1/status"


async def call_alphafold2(
    sequence: str,
    api_key: str,
    algorithm: str = "mmseqs2",
    e_value: float = 0.0001,
    iterations: int = 1,
    databases: List[str] = ["small_bfd"],
    relax_prediction: bool = False,
    skip_template_search: bool = True,
    url: str = DEFAULT_URL,
    status_url: str = DEFAULT_STATUS_URL,
    polling_interval: int = 10,
    timeout: int = 600,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Call the NVIDIA NIM AlphaFold2 API.

    Args:
        sequence: Protein sequence in one-letter code
        api_key: NVIDIA NIM API key
        algorithm: MSA search algorithm, "mmseqs2" or "jackhmmer"
        e_value: E-value threshold for template search
        iterations: Number of iterations for template search
        databases: List of databases to search
        relax_prediction: Whether to relax the prediction
        skip_template_search: Whether to skip template search
        url: API endpoint URL
        status_url: Status URL for checking job completion
        polling_interval: Seconds between status checks
        timeout: Request timeout in seconds

    Returns:
        API response or None on failure
    """
    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "NVCF-POLL-SECONDS": "300",
    }

    data = {
        "sequence": sequence,
        "algorithm": algorithm,
        "e_value": e_value,
        "iterations": iterations,
        "databases": databases,
        "relax_prediction": relax_prediction,
        "skip_template_search": skip_template_search,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=data, headers=headers, timeout=timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 202:
                    req_id = response.headers.get("nvcf-reqid")
                    if req_id:
                        logger.info(f"AlphaFold2 job submitted, request ID: {req_id}")
                        return await _poll_job_status(
                            req_id=req_id,
                            headers=headers,
                            status_url=status_url,
                            polling_interval=polling_interval,
                            timeout=timeout,
                        )
                    else:
                        logger.error("No request ID in response headers")
                        return None
                else:
                    logger.error(f"Error calling AlphaFold2 API: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return None
    except Exception as e:
        import traceback

        logger.error(f"Error calling AlphaFold2 API: {e}")
        logger.error(traceback.format_exc())
        return None


async def _poll_job_status(
    req_id: str,
    headers: Dict[str, str],
    status_url: str,
    polling_interval: int = 10,
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    Poll the status endpoint until the job completes.

    Args:
        req_id: The request ID to check
        headers: Request headers
        status_url: Status URL for checking job completion
        polling_interval: Seconds between status checks
        timeout: Request timeout in seconds

    Returns:
        The final response or None on failure
    """
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{status_url}/{req_id}", headers=headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        logger.info(f"AlphaFold2 job {req_id} completed")
                        return await response.json()
                    elif response.status == 202:
                        logger.debug(
                            f"AlphaFold2 job {req_id} still running, polling..."
                        )
                        await asyncio.sleep(polling_interval)
                    else:
                        logger.error(
                            f"Error checking AlphaFold2 job status: {response.status}"
                        )
                        text = await response.text()
                        logger.error(f"Response: {text}")
                        return None
        except Exception as e:
            logger.error(f"Error polling AlphaFold2 job status: {e}")
            return None
