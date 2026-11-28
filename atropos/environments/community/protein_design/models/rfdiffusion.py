import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://health.api.nvidia.com/v1/biology/ipd/rfdiffusion/generate"
DEFAULT_STATUS_URL = "https://health.api.nvidia.com/v1/status"


async def call_rfdiffusion(
    input_pdb: str,
    api_key: str,
    contigs: str = None,
    hotspot_res: List[str] = None,
    diffusion_steps: int = 15,
    url: str = DEFAULT_URL,
    status_url: str = DEFAULT_STATUS_URL,
    polling_interval: int = 10,
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    Call the NVIDIA NIM RFDiffusion API.

    Args:
        input_pdb: PDB structure as a string
        api_key: NVIDIA NIM API key
        contigs: Contig string (e.g. "A20-60/0 50-100")
        hotspot_res: List of hotspot residues (e.g. ["A50","A51"])
        diffusion_steps: Number of diffusion steps
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

    data = {"input_pdb": input_pdb, "diffusion_steps": diffusion_steps}

    if contigs:
        data["contigs"] = contigs
    if hotspot_res:
        data["hotspot_res"] = hotspot_res

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
                        logger.info(f"RFDiffusion job submitted, request ID: {req_id}")
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
                    logger.error(f"Error calling RFDiffusion API: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return None
    except Exception as e:
        logger.error(f"Error calling RFDiffusion API: {e}")
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
                        logger.info(f"RFDiffusion job {req_id} completed")
                        return await response.json()
                    elif response.status == 202:
                        logger.debug(
                            f"RFDiffusion job {req_id} still running, polling..."
                        )
                        await asyncio.sleep(polling_interval)
                    else:
                        logger.error(
                            f"Error checking RFDiffusion job status: {response.status}"
                        )
                        text = await response.text()
                        logger.error(f"Response: {text}")
                        return None
        except Exception as e:
            logger.error(f"Error polling RFDiffusion job status: {e}")
            return None
