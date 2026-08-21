import argparse
import time

import requests
import wandb


def update_wandb(health_statuses):
    wandb.log(health_statuses)


def run(api_addr, tp, node_num):
    print(f"Starting up with {api_addr}, {tp}, {node_num}", flush=True)
    while True:
        try:
            data = requests.get(f"{api_addr}/wandb_info").json()
            wandb_group = data["group"]
            wandb_project = data["project"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            wandb_project = None
            wandb_group = None
            print("Waiting for init...")

        if wandb_project is None:
            time.sleep(1)
        else:
            wandb.init(
                project=wandb_project, group=wandb_group, name=f"inf_node_{node_num}"
            )
            break
    curr_step = 0
    health_statuses = {
        f"server/server_heath_{node_num}_{i}": 0.0 for i in range(8 // tp)
    }
    while True:
        data = requests.get(f"{api_addr}/status").json()
        step = data["current_step"]
        if step > curr_step:
            wandb.log(health_statuses, step=step)
            curr_step = step
        time.sleep(60)
        # Check on each server
        for i in range(8 // tp):
            try:
                health_status = requests.get(
                    f"http://localhost:{9000 + i}/health_generate"
                ).status_code
                health_statuses[f"server/server_heath_{node_num}_{i}"] = (
                    1 if health_status == 200 else 0
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                health_statuses[f"server/server_heath_{node_num}_{i}"] = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_addr", type=str, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--node_num", type=int, required=True)
    args = parser.parse_args()
    run(args.api_addr, args.tp, args.node_num)


if __name__ == "__main__":
    main()
