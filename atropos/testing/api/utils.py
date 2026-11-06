import multiprocessing
import time

import requests

from atroposlib.cli.run_api import main as run_api_main


def check_api_running() -> bool:
    try:
        data = requests.get("http://localhost:8000/info")
        return data.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def launch_api_for_testing(max_wait_for_api: int = 10) -> multiprocessing.Process:
    api_proc = multiprocessing.Process(target=run_api_main)
    api_proc.start()
    counter = 0
    while not check_api_running():
        time.sleep(1)
        counter += 1
        if counter > max_wait_for_api:
            raise TimeoutError("API server did not start in time.")
    print("API server started for testing.")
    return api_proc
