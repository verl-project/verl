# Copyright (c) InternLM. All rights reserved.
import atexit
import functools
import os
import queue
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue, connection
from multiprocessing.synchronize import Event as EventClass
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    cast,
)
from uuid import uuid4

import loguru
from typing_extensions import NotRequired

from .base_judger import (
    JudgeStatus,
    MessageItem,
    MetaData,
    Reward,
    registered_judgers,
)


class InputData(TypedDict):
    prompt_messages: List[MessageItem]
    completion_messages: List[MessageItem]
    metadata: NotRequired[MetaData]


T = TypeVar("T")


@dataclass
class GenericTask(Generic[T]):
    token: str
    index: int
    judger: str
    content: T


@dataclass
class SubprocessConfig:
    loguru_handlers: Optional[List[dict]] = None
    worker_init_func: Optional[Callable] = None


class ParallelRouter:
    def __init__(
        self,
        judgers_config: Dict[str, dict],
        data_judger_mapping: Dict[str, Optional[List[str]]],
        logger: Optional["loguru.Logger"] = None,
        subprocess_config: Optional[SubprocessConfig] = None,
    ):
        if logger is not None:
            self.logger = logger
        else:
            import mock

            self.logger = mock.Mock()

        if subprocess_config is not None:
            self.subprocess_config = subprocess_config
        else:
            self.subprocess_config = SubprocessConfig()

        if not (
            isinstance(judgers_config, dict)
            and all(
                isinstance(k, str) and isinstance(v, dict)
                for k, v in judgers_config.items()
            )
        ):
            raise TypeError(
                f"Illegal judgers_config: {judgers_config}\n"
                "Should be Dict[str, dict]"
            )
        if "RM" in judgers_config.keys():
            raise KeyError(
                f"'RM' is a reserved judger keywork for {self.__class__.__name__}, "
                f"please remove it from judgers_config: {judgers_config}"
            )
        self.judgers_config = judgers_config

        data_judger_mapping: Dict[str, List[str]] = {
            k: v or [] for k, v in data_judger_mapping.items()
        }  # change None to empty list []
        if not (
            isinstance(data_judger_mapping, dict)
            and all(
                isinstance(k, str)
                and isinstance(v, (list, tuple, set))
                and all(isinstance(vv, str) for vv in v)
                for k, v in data_judger_mapping.items()
            )
        ):
            raise TypeError(
                f"Illegal data_judger_mapping: {data_judger_mapping}\n"
                "Should be Dict[str, List[str]]"
            )
        self.data_judger_mapping = data_judger_mapping

        avail_judgers = set(self.judgers_config.keys()) | {"RM"}
        _used_judgers: List[str] = []
        for v in data_judger_mapping.values():
            _used_judgers.extend(v)
        used_judgers: set = set(_used_judgers)
        if unused := avail_judgers - used_judgers:
            self.logger.warning(
                "Following judgers are available but not "
                f"used in data mapping: {unused}\n"
                "Please make sure this is intended"
            )
            # remove unused configs
            for judger_name in unused:
                self.judgers_config.pop(judger_name, None)
        if missing := used_judgers - avail_judgers:
            self.logger.warning(
                "Following judgers are configured to be used "
                f"but not built in data mapping: {missing}\n"
                "Please make sure this is intended"
            )
            # remove missing judgers from mapping, to prevent potential errors
            for source in list(self.data_judger_mapping.keys()):
                before = set(self.data_judger_mapping[source])
                self.data_judger_mapping[source] = list(before - missing)
            # then filter out data_mapping without available judgers
            self.data_judger_mapping = {
                source: judgers
                for source, judgers in self.data_judger_mapping.items()
                if len(judgers) > 0
            }

        # Try build judgers in __init__ so that raise Exceptions earlly
        for judger_name, judger_conf in self.judgers_config.items():
            _ = self._build_judger(judger_name, judger_conf)

        self._processes: List[Process] = []
        self._stop_event = Event()
        atexit.register(self.shutdown)

        self._input_queues: Dict[str, Queue[GenericTask[InputData]]] = {
            judger_name: Queue() for judger_name in self.judgers_config.keys()
        }
        self._output_queue: Queue[GenericTask[Reward]] = Queue()
        self._exc_queue: Queue[Tuple[str, Exception]] = Queue()
        self._num_tasks: Dict[str, int] = {}  # for each token
        self._num_indexes: Dict[str, int] = {}  # for each token
        self._results_buffer: Dict[str, List[GenericTask[Reward]]] = defaultdict(
            list
        )  # results buffer grouped by the key "token"

    def submit(self, data_batch: List[InputData]):
        indexes_for_ext: List[int] = []
        indexes_for_local: List[int] = []
        tasks_input: List[GenericTask[InputData]] = []
        token = str(uuid4())
        for index, data_item in enumerate(data_batch):
            if (
                not isinstance(data_item, dict)
                or "metadata" not in data_item
                or "prompt_messages" not in data_item
                or "completion_messages" not in data_item
            ):
                indexes_for_local.append(index)
                continue
            source = data_item["metadata"].get("data_source", None)
            if source is None or source not in self.data_judger_mapping:
                indexes_for_local.append(index)
                continue
            indexes_for_ext.append(index)
            for judger in self.data_judger_mapping[source]:
                if judger == "RM":
                    indexes_for_local.append(index)
                else:
                    tasks_input.append(
                        GenericTask(
                            token=token,
                            index=index,
                            judger=judger,
                            content=data_item,
                        )
                    )

        self._num_tasks[token] = len(tasks_input)
        self._num_indexes[token] = len(data_batch)
        for task in tasks_input:
            self._input_queues[task.judger].put(task, block=True, timeout=1)

        if not self._processes:
            self.logger.debug("Starting processes...")
            for judger_name, judger_conf in self.judgers_config.items():
                num_proc = judger_conf.pop("num_processes", 1)
                self._processes.extend(
                    [
                        Process(
                            target=ParallelRouter._safe_process_worker,
                            kwargs={
                                "stop_event": self._stop_event,
                                "judger_name": judger_name,
                                "judger_conf": judger_conf,
                                "input_queue": self._input_queues[judger_name],
                                "output_queue": self._output_queue,
                                "exc_queue": self._exc_queue,
                                "config": self.subprocess_config,
                            },
                            daemon=True,
                        )
                        for _ in range(num_proc)
                    ]
                )
            for p in self._processes:
                p.start()
            self.logger.debug(f"Start processes done, total {len(self._processes)}")

        return token, indexes_for_local

    def query(
        self, token: str, timeout: float = 0
    ) -> Optional[List[Optional[Dict[str, Reward]]]]:
        start = time.time()
        while True:
            self._try_catch_subprocess_exceptions()
            try:
                result = self._output_queue.get(timeout=0.1)
                self._results_buffer[result.token].append(result)
            except queue.Empty:
                pass
            if len(self._results_buffer[token]) == self._num_tasks[token]:
                results = self._results_buffer.pop(token)
                num_tasks = self._num_tasks.pop(token)
                num_indexes = self._num_indexes.pop(token)
                rewards: List[Dict[str, Reward]] = [{} for _ in range(num_indexes)]
                for result in results:
                    reward = result.content
                    if result.judger in rewards[result.index]:
                        self.logger.warning(
                            f"{result.judger} already exists: {rewards[result.index]}, "
                            f"will replace --> {reward}"
                        )
                    rewards[result.index][result.judger] = reward
                # convert empty dicts to None
                return [r or None for r in rewards]
            if timeout > 0 and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Timeout after {timeout} seconds, got {len(self._results_buffer[token])} results, expected {self._num_tasks[token]}"
                )

    @staticmethod
    def _safe_process_worker(
        stop_event: EventClass,
        judger_name: str,
        judger_conf: dict,
        input_queue: "Queue[GenericTask[InputData]]",
        output_queue: "Queue[GenericTask[Reward]]",
        exc_queue: "Queue[Tuple[str, Exception]]",
        config: SubprocessConfig,
    ):
        try:
            ParallelRouter._process_worker(
                stop_event=stop_event,
                judger_name=judger_name,
                judger_conf=judger_conf,
                input_queue=input_queue,
                output_queue=output_queue,
                exc_queue=exc_queue,
                config=config,
            )
        except Exception as e:
            exc_queue.put((judger_name, e), timeout=1)

    @staticmethod
    def _process_worker(
        stop_event: EventClass,
        judger_name: str,
        judger_conf: dict,
        input_queue: "Queue[GenericTask[InputData]]",
        output_queue: "Queue[GenericTask[Reward]]",
        exc_queue: "Queue[Tuple[str, Exception]]",
        config: SubprocessConfig,
    ):
        from xtuner._lite import get_logger

        logger = get_logger()
        if config.loguru_handlers is not None:
            for handler in config.loguru_handlers:
                handler["enqueue"] = True
                logger.add(*handler)
        if config.worker_init_func is not None:
            config.worker_init_func()

        # Infer num threads for each stage according to configs
        _num_threads = judger_conf.pop("concurrency_per_proc", (1, 1))
        if isinstance(_num_threads, (tuple, list)) and len(_num_threads) == 2:
            num_threads_s1, num_threads_s2 = _num_threads
        elif isinstance(_num_threads, int):
            num_threads_s1 = max(1, _num_threads // 2)
            num_threads_s2 = max(1, _num_threads - num_threads_s1)
        else:
            raise TypeError(
                "`concurrency_per_proc` in judger_conf should be int or "
                f"Tuple[int, int], got {type(_num_threads)}: {_num_threads}"
            )

        # Lazy build judgers in subprocesses to avoid serialization errors
        judger = ParallelRouter._build_judger(judger_name, judger_conf)
        # input_queue = self._input_queues[judger_name]
        # output_queue = self._output_queue
        handle_queue: queue.Queue[GenericTask[JudgeStatus]] = queue.Queue()
        log_prefix = f"[pid={os.getpid()},{judger_name}]"

        def report_exc_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    logger.error(
                        f"{log_prefix} "
                        f"Thread worker of {judger_name} raised "
                        f"{type(e).__name__}: {e}",
                        f"Stack trace: {stack_trace}",
                    )
                    exc_queue.put((judger_name, e), timeout=1)

            return wrapper

        # Stage 1: input_queue -> judger.on_data_received -> handle_queue
        @report_exc_wrapper
        def thread_worker_s1():
            while not stop_event.is_set():
                try:
                    task = input_queue.get(timeout=0.1)
                    logger.debug(f"{log_prefix} dequeue input: {task}")
                except queue.Empty:
                    logger.debug(f"{log_prefix} input queue empty")
                    time.sleep(0.1)
                    continue
                data = task.content
                if "metadata" not in data:
                    raise RuntimeError(
                        f"'metadata' not in data.keys(): {list(data.keys())}"
                    )
                logger.debug(f"{log_prefix} on_data_received")
                handle = judger.on_data_received(
                    data["prompt_messages"],
                    data["completion_messages"],
                    cast(dict, data["metadata"]),
                )
                logger.debug(f"{log_prefix} got handle")
                new_task = GenericTask(
                    token=task.token,
                    index=task.index,
                    judger=task.judger,
                    content=handle,
                )
                while True:
                    try:
                        handle_queue.put(
                            new_task,
                            timeout=0.1,
                        )
                        logger.debug(f"{log_prefix} enqueue handle: {new_task}")
                        break
                    except queue.Full:
                        time.sleep(0.1)

        # Stage 2: handle_queue -> judger.on_reward_required -> output_queue
        @report_exc_wrapper
        def thread_worker_s2():
            while not stop_event.is_set():
                try:
                    task = handle_queue.get(timeout=0.1)
                    logger.debug(f"{log_prefix} dequeue handle: {task}")
                except queue.Empty:
                    logger.debug(f"{log_prefix} handle queue empty")
                    time.sleep(0.1)
                    continue
                logger.debug(f"{log_prefix} on_reward_required")
                reward = judger.on_reward_required(task.content)
                logger.info(f"{log_prefix} got result")
                new_task = GenericTask(
                    token=task.token,
                    index=task.index,
                    judger=task.judger,
                    content=reward,
                )
                while True:
                    try:
                        output_queue.put(
                            new_task,
                            timeout=0.1,
                        )
                        logger.debug(f"{log_prefix} enqueue output: {new_task}")
                        break
                    except queue.Full:
                        time.sleep(0.1)

        from threading import Thread

        threads: List[Thread] = []
        for _ in range(num_threads_s1):
            threads.append(Thread(target=thread_worker_s1, daemon=True))
        for _ in range(num_threads_s2):
            threads.append(Thread(target=thread_worker_s2, daemon=True))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    @staticmethod
    def _build_judger(judger_name: str, judger_conf: dict):
        judger_conf = deepcopy(judger_conf)
        judger_conf.pop("num_processes", None)
        judger_conf.pop("concurrency_per_proc", None)
        _type = judger_conf.pop("type", None)
        if _type is None:
            _type = judger_name
        if _type not in registered_judgers:
            raise KeyError(
                f"{judger_name} use unregistered judger type: {_type}. "
                f"Available judgers are: {list(registered_judgers.keys())}"
            )
        cls = registered_judgers[_type]
        return cls(**judger_conf)

    def _try_catch_subprocess_exceptions(self):
        exc_handles: List[Tuple[str, Exception]] = []
        while True:
            try:
                exc_handle = self._exc_queue.get(timeout=0.001)
                exc_handles.append(exc_handle)
            except queue.Empty:
                break
        if exc_handles:
            error_message = "\n".join(
                [
                    f"- [{judger_name}] {type(exc).__name__}: {exc}"
                    for judger_name, exc in exc_handles
                ]
            )
            raise RuntimeError(
                "Following threads/processes raise exceptions unexpectedly:\n"
                f"{error_message}\n"
                "Program terminated"
            )

    def shutdown(self, timeout: float = 2.0):
        if not hasattr(self, "_processes") or not self._processes:
            return
        if not self._stop_event.is_set():
            self._stop_event.set()
        connection.wait([p.sentinel for p in self._processes], timeout=timeout)
        for p in self._processes:
            if p.is_alive():
                p.kill()
                p.join()
        self._processes = []
