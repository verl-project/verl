# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import multiprocessing
import os
import time

import ray
import requests
from sglang_router.launch_server import RouterArgs, launch_router

from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def launch_router_process(
    worker_urls: list[str],
    request_timeout: int = 180,
    max_wait_time: int = 300,
    timeout: int = 30,
) -> str:
    router_ip = ray.util.get_node_ip_address().strip("[]")
    router_port, _ = get_free_port(router_ip)
    router_address = (
        f"[{router_ip}]:{router_port}" if is_valid_ipv6_address(router_ip) else f"{router_ip}:{router_port}"
    )
    router_args = RouterArgs(
        host=router_ip,
        port=router_port,
        worker_urls=worker_urls,
        balance_abs_threshold=0,
        log_level="warn",
        request_timeout_secs=request_timeout,
    )
    router_process = multiprocessing.Process(target=launch_router, args=(router_args,))
    router_process.daemon = True
    router_process.start()
    time.sleep(3)
    assert router_process.is_alive()

    # health check
    start_time = time.time()
    url = f"http://{router_address}/health"
    with requests.Session() as session:
        while time.time() - start_time < max_wait_time:
            try:
                response = session.get(url, timeout=timeout)
                if response.status_code == 200:
                    break
            except requests.RequestException as e:
                logger.debug(f"Health check failed: {e}")

            time.sleep(2)
        else:
            router_process.terminate()
            raise RuntimeError(f"Router health check failed after {max_wait_time} seconds.")

    logger.info(f"Router is running on {router_address}")
    return router_address, router_process
