import os
import fire
import json
import wandb


def main(path, project):
    name = path.split("/")[-3]
    # name = os.path.basename(path).split(".")[0]
    wandb.init(project=project, name=name)
    previous_step = 0
    log_cache = {}
    for line in open(path):
        log = json.loads(line)
        parsed_log = {}
        for key, value in log.items():
            if key != "rejected_score_mean":
                key = key.replace("rejected_score", "rejected_score/")
            if "/" in key:
                split_key = key.split("/")
                new_key = "_".join(split_key[1:]) + "/" + split_key[0]
                parsed_log[new_key] = value
            else:
                parsed_log[key] = value
        print(parsed_log)
        step = parsed_log.pop("step")
        if step != previous_step:
            wandb.log(log_cache, commit=True, step=previous_step)
            log_cache = {}
            previous_step = step
        log_cache.update(parsed_log)
    if log_cache:
        wandb.log(log_cache, commit=True, step=previous_step)
        

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
