# Copyright 2026 Nvidia Corporation
# Licensed under the Apache License, Version 2.0.

"""Ray-managed metadata and node-local FlexKV services."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ray
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _wait_tcp(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {host}:{port}")


def _wait_socket(path: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if Path(path).is_socket():
            return
        rc = process.poll()
        if rc is not None:
            raise RuntimeError(f"FlexKV node server exited during startup (rc={rc})")
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


class FlexKVMetadataService:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.processes: list[subprocess.Popen] = []
        self.log_files = []

    def start(self) -> dict[str, Any]:
        host = ray.util.get_node_ip_address().strip("[]")
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        for name, port in (
            ("flexkv", int(self.config["flexkv_redis_port"])),
            ("mooncake", int(self.config["mooncake_redis_port"])),
        ):
            log = open(log_dir / f"redis-{name}.log", "a", encoding="utf-8")
            cmd = [
                self.config["redis_server_path"],
                "--bind",
                host,
                "--protected-mode",
                "no",
                "--port",
                str(port),
                "--save",
                "",
                "--appendonly",
                "no",
            ]
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            self.processes.append(process)
            self.log_files.append(log)
            _wait_tcp(host, port, float(self.config["startup_timeout_s"]))
            if process.poll() is not None:
                raise RuntimeError(f"redis-server for {name} exited during startup; see {log.name}")
        print(
            f"VERL_FLEXKV_SERVICE phase=METADATA_READY node_id={self.config['node_id']} host={host} "
            f"flexkv_redis={self.config['flexkv_redis_port']} "
            f"mooncake_redis={self.config['mooncake_redis_port']}",
            flush=True,
        )
        return {
            "host": host,
            "flexkv_redis_port": int(self.config["flexkv_redis_port"]),
            "mooncake_redis_port": int(self.config["mooncake_redis_port"]),
        }

    def stop(self) -> None:
        for process in reversed(self.processes):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for log in self.log_files:
            log.close()


class FlexKVNodeService:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def start(self) -> dict[str, str]:
        node_ip = ray.util.get_node_ip_address().strip("[]")
        cfg = self.config
        run_id = cfg["run_id"]
        node_index = int(cfg["node_index"])
        server_path = f"/tmp/verl_fk_{run_id}"
        gpu_register_path = f"{server_path}_gpu_register"
        config_path = f"/tmp/verl_fk_{run_id}.yaml"
        mooncake_path = f"/tmp/verl_fk_{run_id}_mte.json"
        rpc_port = int(cfg["rpc_port_base"]) + node_index
        for path in (
            server_path,
            gpu_register_path,
            f"{gpu_register_path}_control",
        ):
            Path(path).unlink(missing_ok=True)

        Path(config_path).write_text(
            "enable_p2p_cpu: true\n"
            "enable_p2p_ssd: false\n"
            f"cpu_cache_gb: {int(cfg['cpu_cache_gb'])}\n"
            "ssd_cache_gb: 0\n"
            f"local_zmq_ip: \"{node_ip}\"\n"
            f"local_zmq_port: {int(cfg['local_zmq_port_base']) + node_index}\n"
            f"redis_host: \"{cfg['metadata_host']}\"\n"
            f"redis_port: {int(cfg['flexkv_redis_port'])}\n"
            f"local_ip: \"{node_ip}\"\n"
            "redis_password: null\n"
            f"node_ttl_seconds: {int(cfg['node_ttl_seconds'])}\n",
            encoding="utf-8",
        )
        Path(mooncake_path).write_text(
            json.dumps(
                {
                    "engine_ip": node_ip,
                    "engine_port": int(cfg["transfer_engine_port_base"]) + node_index,
                    "metadata_backend": "redis",
                    "metadata_server": f"redis://{cfg['metadata_host']}:{cfg['mooncake_redis_port']}",
                    "metadata_server_auth": "",
                    "protocol": cfg["protocol"],
                    "device_name": cfg["device_name"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        log_dir = Path(cfg["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_dir / f"flexkv-node-{run_id}-{node_index}.log", "a", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(cfg["gpu_ids"]),
                "FLEXKV_CONFIG_PATH": config_path,
                "MOONCAKE_CONFIG_PATH": mooncake_path,
                "FLEXKV_ENABLE_P2P": "1",
                "FLEXKV_SERVER_CLIENT_MODE": "1",
                "FLEXKV_SERVER_RECV_PORT": f"ipc://{server_path}",
                "VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE": "1",
                "FLEXKV_ENABLE_MPS": "0",
                "MC_LEGACY_RPC_PORT_BINDING": str(rpc_port),
            }
        )
        packed_kv = bool(cfg.get("packed_kv", True))
        cmd = [
            sys.executable,
            "-m",
            "verl.workers.rollout.flexkv_service",
            "serve",
            "--server-recv-port",
            f"ipc://{server_path}",
            "--gpu-register-port",
            f"ipc://{gpu_register_path}",
            "--expected-gpus",
            str(cfg["expected_gpus"]),
            "--instance-num",
            str(cfg["instance_num"]),
            "--tp-size",
            str(cfg["tp_size"]),
            "--model-path",
            cfg["model_path"],
            "--tokens-per-block",
            str(cfg["tokens_per_block"]),
        ]
        if packed_kv:
            cmd.append("--packed-kv")
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_socket(server_path, self.process, float(cfg["startup_timeout_s"]))
        print(
            f"VERL_FLEXKV_SERVICE phase=NODE_READY node_id={cfg['node_id']} "
            f"node_ip={node_ip} gpus={cfg['gpu_ids']} socket={server_path}",
            flush=True,
        )
        return {
            "FLEXKV_ENABLE_P2P": "1",
            "FLEXKV_SERVER_CLIENT_MODE": "1",
            "FLEXKV_SERVER_LAUNCH_MODE": "external",
            "FLEXKV_SERVER_RECV_PORT": f"ipc://{server_path}",
            "FLEXKV_INSTANCE_NUM": str(cfg["instance_num"]),
            "FLEXKV_CPU_CACHE_GB": str(cfg["cpu_cache_gb"]),
            "FLEXKV_CONFIG_PATH": config_path,
            "MOONCAKE_CONFIG_PATH": mooncake_path,
            "VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE": "1",
            "FLEXKV_ENABLE_MPS": "0",
            "MC_LEGACY_RPC_PORT_BINDING": str(rpc_port),
        }

    def stop(self) -> None:
        if self.process is not None:
            import psutil

            try:
                root = psutil.Process(self.process.pid)
                processes = root.children(recursive=True) + [root]
            except psutil.NoSuchProcess:
                processes = []
            for process in reversed(processes):
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(processes, timeout=5)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=5)
        if self.log_file is not None:
            self.log_file.close()


class FlexKVServiceManager:
    """Driver-side coordinator shared by vLLM or SGLang rollout replicas."""

    def __init__(self, rollout_config, model_config):
        self.config = OmegaConf.to_container(rollout_config.flexkv_service, resolve=True)
        self.model_path = getattr(model_config, "path")
        self.rollout_name = str(rollout_config.name)
        self.tp_size = int(rollout_config.tensor_model_parallel_size)
        self.expected_gpus = int(self.config["expected_gpus_per_node"] or rollout_config.n_gpus_per_node)
        if self.expected_gpus % self.tp_size:
            raise ValueError("expected_gpus_per_node must be divisible by rollout TP")
        self.metadata_node_id = str(ray.get_runtime_context().get_node_id())
        self.run_id = str(ray.get_runtime_context().get_job_id()).replace(":", "")[:8]
        self.lock = asyncio.Lock()
        self.nodes: dict[str, dict[str, Any]] = {}
        self.metadata_actor = None
        self.metadata = None

    async def _ensure_metadata(self) -> None:
        if self.metadata is not None:
            return
        actor_cls = ray.remote(num_cpus=0.1)(FlexKVMetadataService)
        metadata_config = dict(self.config)
        metadata_config["node_id"] = self.metadata_node_id
        self.metadata_actor = actor_cls.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=self.metadata_node_id, soft=False)
        ).remote(metadata_config)
        self.metadata = await self.metadata_actor.start.remote()

    async def register_node(self, node_id: str, gpu_ids: list[str]) -> dict[str, str]:
        async with self.lock:
            await self._ensure_metadata()
            state = self.nodes.setdefault(
                node_id,
                {
                    "gpus": set(),
                    "event": asyncio.Event(),
                    "env": None,
                    "actor": None,
                    "error": None,
                    "index": len(self.nodes),
                },
            )
            state["gpus"].update(gpu_ids)
            if len(state["gpus"]) > self.expected_gpus:
                raise RuntimeError(f"node {node_id} reported too many GPUs: {state['gpus']}")
            if len(state["gpus"]) == self.expected_gpus and state["actor"] is None:
                node_cfg = dict(self.config)
                node_cfg.update(
                    {
                        "run_id": self.run_id,
                        "node_id": node_id,
                        "node_index": state["index"],
                        "gpu_ids": sorted(state["gpus"], key=int),
                        "expected_gpus": self.expected_gpus,
                        "instance_num": self.expected_gpus // self.tp_size,
                        "tp_size": self.tp_size,
                        "model_path": self.model_path,
                        "packed_kv": self.rollout_name == "vllm",
                        "metadata_host": self.metadata["host"],
                    }
                )
                actor_cls = ray.remote(num_cpus=0.1)(FlexKVNodeService)
                state["actor"] = actor_cls.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                ).remote(node_cfg)
                try:
                    state["env"] = await state["actor"].start.remote()
                except Exception as e:
                    state["error"] = e
                finally:
                    state["event"].set()
            event = state["event"]
        try:
            await asyncio.wait_for(event.wait(), timeout=float(self.config["startup_timeout_s"]))
        except asyncio.TimeoutError as e:
            observed = sorted(self.nodes[node_id]["gpus"])
            raise TimeoutError(
                f"timed out waiting for {self.expected_gpus} FlexKV GPUs on node {node_id}; observed {observed}"
            ) from e
        state = self.nodes[node_id]
        if state["error"] is not None:
            raise RuntimeError(f"failed to start FlexKV service on node {node_id}") from state["error"]
        return dict(state["env"])

    async def shutdown(self) -> None:
        actors = [state["actor"] for state in self.nodes.values() if state["actor"] is not None]
        if actors:
            await asyncio.gather(*(actor.stop.remote() for actor in actors), return_exceptions=True)
        if self.metadata_actor is not None:
            await self.metadata_actor.stop.remote()


def _run_node_server() -> None:
    import argparse

    import torch
    from transformers import AutoConfig
    from mooncake.engine import TransferEngine

    from flexkv.common.config import (
        CacheConfig,
        ModelConfig,
        RankInfo,
        load_user_config_from_file,
        update_default_config_from_user_config,
    )
    from flexkv.server.server import KVServer

    parser = argparse.ArgumentParser()
    parser.add_argument("serve")
    parser.add_argument("--server-recv-port", required=True)
    parser.add_argument("--gpu-register-port", required=True)
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--instance-num", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokens-per-block", type=int, default=16)
    parser.add_argument("--packed-kv", action="store_true")
    args = parser.parse_args()
    assert TransferEngine is not None

    hf = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    text = getattr(hf, "text_config", hf)
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    if args.packed_kv:
        head_dim *= 2
    model_config = ModelConfig(
        num_layers=text.num_hidden_layers,
        num_kv_heads=text.num_key_value_heads,
        head_size=head_dim,
        packed_kv=args.packed_kv,
        dtype=torch.bfloat16,
        tp_size=args.tp_size,
        pp_size=1,
        dp_size=1,
        nnodes=1,
        instance_num=args.instance_num,
    )
    model_config.freeze()
    cache_config = CacheConfig(tokens_per_block=args.tokens_per_block)
    user_config = load_user_config_from_file(os.environ["FLEXKV_CONFIG_PATH"])
    update_default_config_from_user_config(RankInfo(model_config=model_config), cache_config, user_config)
    server = KVServer(
        model_config=model_config,
        cache_config=cache_config,
        gpu_register_port=args.gpu_register_port,
        server_recv_port=args.server_recv_port,
    )
    server.run()


if __name__ == "__main__":
    _run_node_server()
