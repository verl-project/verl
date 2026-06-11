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
import time

import ray
from ray.util import list_named_actors

from verl.plugin.platform import get_platform


def _get_collective_module():
    mod = get_platform().get_collective_module()
    if mod is None:
        raise RuntimeError(
            f"Platform '{get_platform().device_name}' does not provide a collective communication module. "
            "Override get_collective_module() in your platform implementation."
        )
    return mod


@ray.remote
class NCCLIDStore:
    """Ray actor that holds and serves an NCCL unique ID for rendezvous."""

    def __init__(self, nccl_id):
        self._nccl_id = nccl_id

    def get(self):
        """Return the stored NCCL unique identifier."""
        return self._nccl_id


def get_nccl_id_store_by_name(name):
    """Look up a named NCCLIDStore Ray actor by its registered name.

    Args:
        name: The registered actor name to search for.

    Returns:
        The actor handle if exactly one match is found, otherwise None.

    """
    all_actors = list_named_actors(all_namespaces=True)
    matched_actors = [actor for actor in all_actors if actor.get("name", None) == name]
    if len(matched_actors) == 1:
        actor = matched_actors[0]
        return ray.get_actor(**actor)
    elif len(matched_actors) > 1:
        logging.warning("multiple actors with same name found: %s", matched_actors)
    elif len(matched_actors) == 0:
        logging.info("failed to get any actor named %s", name)
    return None


def create_nccl_communicator_in_ray(
    rank: int, world_size: int, group_name: str, max_retries: int = 100, interval_s: int = 5
):
    """Create an NCCL communicator across Ray workers using a shared ID store.

    Args:
        rank: The rank of the current process in the communicator.
        world_size: Total number of processes in the communicator.
        group_name: Name used to register the NCCL ID store actor.
        max_retries: Maximum number of retries for non-rank-0 processes.
        interval_s: Sleep interval in seconds between retries.

    Returns:
        An initialized NcclCommunicator instance.
    """
    collective = _get_collective_module()
    NcclCommunicator = collective.NcclCommunicator
    get_unique_id = collective.get_unique_id

    if rank == 0:
        nccl_id = get_unique_id()
        nccl_id_store = NCCLIDStore.options(name=group_name).remote(nccl_id)

        assert ray.get(nccl_id_store.get.remote()) == nccl_id
        communicator = NcclCommunicator(
            ndev=world_size,
            commId=nccl_id,
            rank=0,
        )
        return communicator
    else:
        for i in range(max_retries):
            nccl_id_store = get_nccl_id_store_by_name(group_name)
            if nccl_id_store is not None:
                logging.info("nccl_id_store %s got", group_name)
                nccl_id = ray.get(nccl_id_store.get.remote())
                logging.info("nccl id for %s got: %s", group_name, nccl_id)
                communicator = NcclCommunicator(
                    ndev=world_size,
                    commId=nccl_id,
                    rank=rank,
                )
                return communicator
            logging.info("failed to get nccl_id for %d time, sleep for %d seconds", i + 1, interval_s)
            time.sleep(interval_s)
