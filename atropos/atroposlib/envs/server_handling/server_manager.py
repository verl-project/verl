import asyncio
import inspect
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Union

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from pydantic import BaseModel, Field

from atroposlib.envs.server_handling.openai_server import OpenAIServer
from atroposlib.envs.server_handling.server_baseline import (
    APIServer,
    APIServerConfig,
    ServerBaseline,
)
from atroposlib.envs.server_handling.server_harness import ServerHarness
from atroposlib.envs.server_handling.trl_vllm_server import TrlVllmServer


class ServerManagerConfig(BaseModel):
    slurm: bool = Field(
        default=False, description="Whether environment is running on slurm or not."
    )
    testing: bool = Field(
        default=False, description="If set to True, environment uses mock OpenAI data."
    )


class ServerManager:
    def __init__(
        self,
        configs: Union[ServerBaseline, List[APIServerConfig]],
        server_class: APIServer = APIServer,
        slurm=False,
        testing=False,
    ):
        # First we check to see if it's the base server class, and if so, we need to select the appropriate server class
        # You can't use type() to check if it's the base server class, because it's an abstract class, it'll appear as
        # an ABCMeta, not what you're expecting.
        if inspect.isabstract(server_class):
            if not isinstance(configs, list):
                if configs.server_type == "openai":
                    server_class = OpenAIServer
                elif configs.server_type == "trl":
                    server_class = TrlVllmServer
                else:
                    raise ValueError(f"Invalid server type: {configs.server_type}")
            else:
                if configs[0].server_type == "openai":
                    server_class = OpenAIServer
                elif configs[0].server_type == "trl":
                    server_class = TrlVllmServer
                else:
                    raise ValueError(f"Invalid server type: {configs[0].server_type}")
        if testing:
            # testing :)
            self.servers = [ServerHarness()]
            return
        if not isinstance(configs, list):
            urls = []
            if os.environ.get("SLURM_JOB_NODELIST", None) is not None:
                nodelist = (
                    os.popen(
                        f'scontrol show hostnames {os.environ["SLURM_JOB_NODELIST"]}'
                    )
                    .read()
                    .split("\n")
                )
                nodelist = [node for node in nodelist if node != ""]
                if len(nodelist) < 2:
                    # localhost!
                    for i in range(4):
                        urls.append(f"http://localhost:{9000 + i + 4}/v1")
                else:
                    num_training_nodes = int(os.environ.get("NUM_TRAINING_NODES"))
                    for node in nodelist[num_training_nodes:]:
                        for i in range(8 // os.environ.get("INFER_TP", 1)):
                            urls.append(f"http://{node}:{9000 + i}/v1")
                openai_configs = []
            else:
                # localhost!
                for i in range(4):
                    urls.append(f"http://localhost:{9000 + i + 4}/v1")
                openai_configs = []
            for url in urls:
                openai_configs.append(
                    APIServerConfig(
                        base_url=url,
                        timeout=configs.timeout,
                        num_max_requests_at_once=configs.num_max_requests_at_once,
                        num_requests_for_eval=configs.num_requests_for_eval,
                        model_name=configs.model_name,
                        rolling_buffer_length=configs.rolling_buffer_length,
                        api_key="x",
                    )
                )
            self.servers = [server_class(config) for config in openai_configs]
        elif not slurm:
            self.servers = [server_class(config) for config in configs]
        else:
            nodelist = (
                os.popen(f'scontrol show hostnames {os.environ["SLURM_JOB_NODELIST"]}')
                .read()
                .split("\n")
            )
            nodelist = [node for node in nodelist if node != ""]
            if len(nodelist) < 2:
                print(
                    "Not enough nodes to distribute to, assuming single node"
                    " and you've setup your sglang appropriately."
                )
                self.servers = [server_class(config) for config in configs]
                return
            urls = []
            num_training_nodes = int(os.environ.get("NUM_TRAINING_NODES"))
            for node in nodelist[num_training_nodes:]:
                if node == "":
                    continue
                for i in range(8 // os.environ.get("INFER_TP", 1)):
                    urls.append(f"http://{node}:{9000 + i}/v1")
            # assume at least one good config is passed in
            new_configs = []
            for i in range(len(urls)):
                new_conf = configs[0].model_copy(deep=True)
                new_conf.base_url = urls[i]
                new_configs.append(new_conf)
            self.servers = [server_class(config) for config in new_configs]

    async def update_weight(self, weight: float):
        for server in self.servers:
            await server.update_weight(weight)

    async def wait_for_sem(self, is_training):
        """
        Wait for a server to be available. This is used to prevent the client from
        overwhelming the server with requests.
        """
        if is_training:
            eval_vals = [
                (
                    max(0, server.eval_sem._value - server.eval_sem.min_val())
                    if server.eval_sem._value != server.eval_sem.max_val
                    else 0
                )
                for server in self.servers
            ]
            sem_vals = [
                max(0, (server.sem._value - server.sem.min_val()) - eval_val)
                for server, eval_val in zip(self.servers, eval_vals)
            ]
        else:
            sem_vals = [
                max(0, server.eval_sem._value - server.eval_sem.min_val())
                for server in self.servers
            ]
        while all([sem_val <= 0 for sem_val in sem_vals]):
            # None available... wait
            await asyncio.sleep(1)

    async def chat_completion(self, **kwargs) -> ChatCompletion:
        is_train = kwargs.get("split", "train") == "train"
        most_available_server = 0
        most_available_server_num_slots = -1
        await self.wait_for_sem(is_train)
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if (
                server.sem._value if is_train else server.eval_sem._value
            ) > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = (
                    server.sem._value if is_train else server.eval_sem._value
                )
        return await self.servers[most_available_server].chat_completion(**kwargs)

    async def completion(self, **kwargs) -> Completion:
        is_train = kwargs.get("split", "train") == "train"
        most_available_server = 0
        most_available_server_num_slots = -1
        await self.wait_for_sem(is_train)
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if (
                server.sem._value if is_train else server.eval_sem._value
            ) > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = (
                    server.sem._value if is_train else server.eval_sem._value
                )
        return await self.servers[most_available_server].completion(**kwargs)

    @asynccontextmanager
    async def dedicated_server(self) -> AsyncGenerator[OpenAIServer, None]:
        most_available_server = 0
        most_available_server_num_slots = -1
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if server.sem._value > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = server.sem._value
        async with self.servers[most_available_server].sem:
            try:
                yield self.servers[most_available_server]
            finally:
                pass
