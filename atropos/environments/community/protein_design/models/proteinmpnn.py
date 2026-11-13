import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict"
DEFAULT_STATUS_URL = "https://health.api.nvidia.com/v1/status"


async def call_proteinmpnn(
    input_pdb: str,
    api_key: str,
    ca_only: bool = False,
    use_soluble_model: bool = False,
    sampling_temp: List[float] = [0.1],
    url: str = DEFAULT_URL,
    status_url: str = DEFAULT_STATUS_URL,
    polling_interval: int = 10,
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    Call the NVIDIA NIM ProteinMPNN API.

    Args:
        input_pdb: PDB structure as a string
        api_key: NVIDIA NIM API key
        ca_only: Whether to use only Cα atoms
        use_soluble_model: Whether to use the soluble model
        sampling_temp: List of sampling temperatures
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
        "input_pdb": input_pdb,
        "ca_only": ca_only,
        "use_soluble_model": use_soluble_model,
        "sampling_temp": sampling_temp,
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
                        logger.info(f"ProteinMPNN job submitted, request ID: {req_id}")
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
                    logger.error(f"Error calling ProteinMPNN API: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return None
    except Exception as e:
        logger.error(f"Error calling ProteinMPNN API: {e}")
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
                        logger.info(f"ProteinMPNN job {req_id} completed")
                        return await response.json()
                    elif response.status == 202:
                        logger.debug(
                            f"ProteinMPNN job {req_id} still running, polling..."
                        )
                        await asyncio.sleep(polling_interval)
                    else:
                        logger.error(
                            f"Error checking ProteinMPNN job status: {response.status}"
                        )
                        text = await response.text()
                        logger.error(f"Response: {text}")
                        return None
        except Exception as e:
            logger.error(f"Error polling ProteinMPNN job status: {e}")
            return None
