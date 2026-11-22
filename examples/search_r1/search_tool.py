"""BaseTool client for the existing sd_search retrieval HTTP service."""

import asyncio
import time
from collections import deque
from functools import partial
from uuid import uuid4

import aiohttp

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class SearchTool(BaseTool):
    def __init__(self, config, tool_schema=None):
        schema = tool_schema or OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search Wikipedia for evidence relevant to a query.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        )
        super().__init__(config, schema)
        self.url = config["retrieval_service_url"]
        self.topk = int(config.get("topk", 3))
        self.timeout = float(config.get("timeout", 120))
        self.max_query_chars = int(config.get("max_query_chars", 4096))
        self.max_concurrency = int(config.get("max_concurrency", 4))
        self.batch_size = int(config.get("batch_size", 16))
        self.batch_wait_s = float(config.get("batch_wait_ms", 2)) / 1000
        if not 1 <= self.topk <= 100 or min(self.timeout, self.max_query_chars, self.max_concurrency) <= 0:
            raise ValueError("Invalid search tool limits")
        if not 1 <= self.batch_size <= 64 or self.batch_wait_s < 0:
            raise ValueError("Invalid retrieval batch limits")
        self._states = {}
        self._session = None
        self._semaphore = None
        self._pending = deque()
        self._dispatcher = None
        self._batch_tasks = set()
        self._closed = False

    async def create(self, instance_id=None, **kwargs):
        # Ground truth and full prompts deliberately are not stored here.
        instance_id = instance_id or uuid4().hex
        if instance_id in self._states:
            raise ValueError("Search tool instance already exists")
        self._states[instance_id] = {"calls": 0}
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        if instance_id not in self._states:
            raise ValueError("Unknown search tool instance")
        query = parameters.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > self.max_query_chars:
            raise ValueError("Search query must be a nonempty string within max_query_chars")
        if self._closed:
            raise RuntimeError("Search tool is closed")
        if self._session is None or self._session.closed:
            # Internal loopback requests must not inherit the development proxy.
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                trust_env=False,
                connector=aiohttp.TCPConnector(limit=self.max_concurrency),
            )
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._states[instance_id]["calls"] += 1
        started = time.perf_counter()
        future = asyncio.get_running_loop().create_future()
        self._pending.append((query.strip(), future))
        if self._dispatcher is None:
            self._dispatcher = asyncio.create_task(self._dispatch())
        hits, profile, batch_count, http_elapsed = await future
        passages = []
        for rank, hit in enumerate(hits[: self.topk], start=1):
            document = hit.get("document", {})
            contents = document.get("contents")
            if not isinstance(contents, str):
                raise ValueError("Retriever document is missing contents")
            title, _, text = contents.partition("\n")
            passages.append(f"Doc {rank}(Title: {title}) {text}")
        metrics = {"retrieval_http_s": time.perf_counter() - started, "retrieved_documents": len(passages)}
        metrics.update(retrieval_batch_size=batch_count, retrieval_batch_http_s=http_elapsed,
                       retrieval_profile_scope="http_batch")
        if isinstance(profile, dict):
            metrics["retrieval_profile"] = profile
        return ToolResponse(text="\n".join(passages).strip() or "No passages found."), 0.0, metrics

    async def _dispatch(self):
        try:
            while self._pending:
                # Reserve sender capacity before taking queries out of the queue.
                # This is the sole dispatcher: at most max_concurrency batches
                # can be in flight, while later queries remain mergeable.
                while len(self._batch_tasks) >= self.max_concurrency:
                    await asyncio.wait(list(self._batch_tasks), return_when=asyncio.FIRST_COMPLETED)
                if len(self._pending) < self.batch_size and self.batch_wait_s:
                    await asyncio.sleep(self.batch_wait_s)
                batch = []
                while self._pending and len(batch) < self.batch_size:
                    item = self._pending.popleft()
                    if not item[1].done():
                        batch.append(item)
                if batch:
                    task = asyncio.create_task(self._post_batch(batch))
                    self._batch_tasks.add(task)
                    task.add_done_callback(partial(self._batch_done, batch))
        finally:
            self._dispatcher = None

    def _batch_done(self, batch, task):
        self._batch_tasks.discard(task)
        # Cancellation before the coroutine starts cannot reach _post_batch's
        # exception handler. Its callers still need their futures cancelled.
        if task.cancelled():
            for _, future in batch:
                if not future.done():
                    future.cancel()

    async def _post_batch(self, batch):
        try:
            payload = {"queries": [query for query, _ in batch], "topk": self.topk, "return_scores": True}
            async with self._semaphore:
                started = time.perf_counter()
                async with self._session.post(self.url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                elapsed = time.perf_counter() - started
            rows = result.get("result")
            if not isinstance(rows, list) or len(rows) != len(batch) or not all(isinstance(row, list) for row in rows):
                raise ValueError("Retriever returned an invalid result batch")
            for (_, future), hits in zip(batch, rows, strict=True):
                if not future.done():
                    future.set_result((hits, result.get("profile", {}), len(batch), elapsed))
        except asyncio.CancelledError:
            for _, future in batch:
                if not future.done():
                    future.cancel()
            raise
        except Exception as error:
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)

    async def release(self, instance_id, **kwargs):
        self._states.pop(instance_id, None)

    async def close(self):
        """Explicit cleanup for tests and any worker shutdown hook."""
        self._closed = True
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
        while self._pending:
            _, future = self._pending.popleft()
            if not future.done():
                future.cancel()
        tasks = list(self._batch_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._session is not None:
            await self._session.close()
