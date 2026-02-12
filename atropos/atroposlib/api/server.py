import time
import uuid
from typing import Any, List, Optional

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from atroposlib.api.utils import grab_exact_from_heterogeneous_queue

app = FastAPI(title="AtroposLib API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "AtroposLib API"}


class Registration(BaseModel):
    wandb_group: str
    wandb_project: str
    batch_size: int
    max_token_len: int
    checkpoint_dir: str
    save_checkpoint_interval: int
    starting_step: int
    num_steps: int


class RegisterEnv(BaseModel):
    max_token_length: int
    desired_name: str
    weight: float


class EnvIdentifier(BaseModel):
    env_id: int


class ScoredData(BaseModel):
    tokens: List[List[int]]
    masks: List[List[int]]
    scores: List[float]
    ref_logprobs: Optional[List[List[float]]] = None
    overrides: Optional[List[dict]] = None
    group_overrides: Optional[dict] = None
    images: Optional[Any] = None


class Status(BaseModel):
    """
    basemodel for status information of the current server
    """

    current_step: int
    queue_size: int


class Info(BaseModel):
    """
    basemodel for useful information
    """

    batch_size: int = -1


@app.post("/register")
async def register(registration: Registration):
    try:
        isinstance(app.state.queue, list)
    except AttributeError:
        app.state.queue = []
        app.state.group = registration.wandb_group
        app.state.project = registration.wandb_project
        app.state.batchsize = int(registration.batch_size)
        app.state.max_token_len = int(registration.max_token_len)
        app.state.status_dict = {"step": registration.starting_step}
        app.state.checkpoint_dir = registration.checkpoint_dir
        app.state.save_checkpoint_interval = registration.save_checkpoint_interval
        app.state.num_steps = registration.num_steps
        app.state.curr_batch = []
        app.state.started = False
        app.state.envs = []
    try:
        app.state.requesters.append(uuid.uuid4().int)
    except AttributeError:
        # If requesters doesn't exist, create it
        app.state.requesters = [uuid.uuid4().int]
    return {"uuid": app.state.requesters[-1]}


@app.post("/register-env")
async def register_env_url(register_env: RegisterEnv):
    try:
        if not app.state.started:
            return {
                "status": "wait for trainer to start",
            }
    except AttributeError:
        return {
            "status": "wait for trainer to start",
        }
    try:
        isinstance(app.state.envs, list)
    except AttributeError:
        app.state.envs = []
    checkpoint_dir = ""
    try:
        checkpoint_dir = app.state.checkpoint_dir
    except AttributeError:
        pass
    real_name = (
        f"{register_env.desired_name}_"
        f"{len([x for x in app.state.envs if x['desired_name'] == register_env.desired_name])}"
    )
    registered_id = len(app.state.envs)
    app.state.envs.append(
        {
            "max_context_len": register_env.max_token_length,
            "weight": register_env.weight if register_env.weight is not None else 1.0,
            "desired_name": register_env.desired_name,
            "real_name": real_name,
            "registered_id": registered_id,
            "last_update": time.time(),
            "connected": True,
        }
    )
    return {
        "status": "success",
        "env_id": registered_id,
        "wandb_name": real_name,
        "checkpoint_dir": checkpoint_dir,
        "starting_step": app.state.status_dict["step"],
        "checkpoint_interval": app.state.save_checkpoint_interval,
        "num_steps": app.state.num_steps,
    }


@app.post("/disconnect-env")
async def disconnect_env(disconnect_env: EnvIdentifier):
    try:
        app.state.envs[disconnect_env.env_id]["connected"] = False
        return {"status": "success"}
    except (AttributeError, IndexError) as e:
        return {"status": "failure", "error": str(e)}


@app.get("/wandb_info")
async def wandb_info():
    try:
        return {"group": app.state.group, "project": app.state.project}
    except AttributeError:
        return {"group": None, "project": None}


@app.get("/info")
async def info():
    try:
        return {
            "batch_size": app.state.batchsize,
            "max_token_len": app.state.max_token_len,
        }
    except AttributeError:
        return {"batch_size": -1, "max_token_len": -1}


@app.get("/batch")
async def get_batch():
    if not app.state.started:
        app.state.started = True

    if len(app.state.curr_batch) > 0:
        return {"batch": app.state.curr_batch.pop()}
    else:
        new_batches = []
        batch, app.state.queue = grab_exact_from_heterogeneous_queue(
            app.state.queue, app.state.batchsize
        )
        while batch is not None:
            new_batches.append(batch)
            batch, app.state.queue = grab_exact_from_heterogeneous_queue(
                app.state.queue, app.state.batchsize
            )
        steps_to_take = len(new_batches)
        if steps_to_take == 0:
            return {"batch": None}
        app.state.status_dict["step"] += steps_to_take
        # chunk it
        for batch in new_batches:
            app.state.curr_batch.append(batch)
        curr_batch = app.state.curr_batch.pop()
        # check length before sending
        print(f"Sending batch of length {sum(len(x['tokens']) for x in curr_batch)}")
        return {"batch": curr_batch}


@app.get("/latest_example")
async def get_latest_example():
    try:
        return app.state.latest
    except AttributeError:
        return {
            "tokens": [],
            "masks": [],
            "scores": [],
            "ref_logprobs": [],
            "images": [],
        }


@app.post("/scored_data")
async def scored_data(scored_data: ScoredData):
    app.state.queue.append(
        {
            "tokens": scored_data.tokens,
            "masks": scored_data.masks,
            "scores": scored_data.scores,
            "ref_logprobs": scored_data.ref_logprobs,
            "overrides": scored_data.overrides,
            "group_overrides": scored_data.group_overrides,
            "images": scored_data.images,
        }
    )
    app.state.latest = app.state.queue[-1]
    return {"status": "received"}


@app.post("/scored_data_list")
async def scored_data_list(scored_data_list: List[ScoredData]):
    """Handle a list of ScoredData objects for step-based learning"""

    for idx, scored_data in enumerate(scored_data_list):

        app.state.queue.append(
            {
                "tokens": scored_data.tokens,
                "masks": scored_data.masks,
                "scores": scored_data.scores,
                "ref_logprobs": scored_data.ref_logprobs,
                "images": scored_data.images,
                "overrides": scored_data.overrides,
                "group_overrides": scored_data.group_overrides,
            }
        )

    if scored_data_list:
        app.state.latest = app.state.queue[-1]

    return {"status": "received", "groups_processed": len(scored_data_list)}


@app.get("/status")
async def get_status():
    try:
        return {
            "current_step": app.state.status_dict["step"],
            "queue_size": len(app.state.queue),
        }
    except AttributeError:
        return {"current_step": 0, "queue_size": 0}


@app.get("/status-env")
async def get_status_env(env: EnvIdentifier):
    total = sum(
        [
            x["max_context_len"] * max(0.0, x["weight"])
            for x in app.state.envs
            if x["connected"]
        ]
    )
    env_weight = (
        app.state.envs[env.env_id]["max_context_len"]
        * app.state.envs[env.env_id]["weight"]
        / total
    )
    env_weight = max(
        0.01, env_weight
    )  # Minimum weight of 0.01 :) TODO: try to figure out a better way to do this

    try:
        ret_dict = {
            "current_step": app.state.status_dict["step"],
            "queue_size": len(app.state.queue),
        }
    except AttributeError:
        ret_dict = {"current_step": 0, "queue_size": 0}
    ret_dict["env_weight"] = env_weight
    return ret_dict


@app.get("/reset_data")
async def reset_data():
    try:
        del app.state.queue
        app.state.group = None
        app.state.project = None
        app.state.batchsize = -1
        app.state.num_steps = -1
        app.state.status_dict = {"step": 0}
        app.state.curr_batch = []
        app.state.started = False
        app.state.requesters = []
        app.state.envs = []
    except KeyError:
        pass
    return PlainTextResponse("Reset successful", status_code=status.HTTP_200_OK)
