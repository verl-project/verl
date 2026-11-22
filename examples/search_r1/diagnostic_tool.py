"""Opt-in tracing of the unchanged SearchTool batching policy."""

import atexit
import hashlib
import json
import os
from pathlib import Path
import time

from .search_tool import SearchTool


def contents_digest(rows):
    payload = [[hit['document']['contents'] for hit in hits] for hits in rows]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class ProfiledSearchTool(SearchTool):
    def __init__(self, config, tool_schema=None):
        super().__init__(config, tool_schema)
        directory = Path(config.get('diagnostic_trace_dir') or os.environ['SEARCH_R1_RETRIEVAL_TRACE_DIR'])
        directory.mkdir(parents=True, exist_ok=True)
        self._trace = (directory/f'client-{os.getpid()}.jsonl').open('a', buffering=1)
        atexit.register(self._close_trace)

    def _close_trace(self):
        if not self._trace.closed:
            self._trace.close()

    def _record(self, row):
        self._trace.write(json.dumps({'pid': os.getpid(), **row}, ensure_ascii=False)+'\n')

    async def execute(self, instance_id, parameters, **kwargs):
        started_ns, started = time.time_ns(), time.perf_counter()
        result = await super().execute(instance_id, parameters, **kwargs)
        elapsed = time.perf_counter()-started
        info = result[2]
        self._record({'kind': 'query', 'start_ns': started_ns, 'end_ns': time.time_ns(),
                      'query': parameters['query'].strip(), 'wall_s': elapsed,
                      'http_batch_s': info['retrieval_batch_http_s'],
                      'client_non_http_s': max(0, elapsed-info['retrieval_batch_http_s']),
                      'server_request_id': info.get('retrieval_profile', {}).get('request_id')})
        return result

    async def _post_batch(self, batch):
        started_ns, started = time.time_ns(), time.perf_counter()
        await super()._post_batch(batch)
        elapsed = time.perf_counter()-started
        completed = [(q, future.result()) for q, future in batch
                     if future.done() and not future.cancelled() and future.exception() is None]
        if not completed:
            self._record({'kind': 'batch_error', 'start_ns': started_ns, 'end_ns': time.time_ns()})
            return
        profile, http_s = completed[0][1][1], completed[0][1][3]
        self._record({'kind': 'batch', 'start_ns': started_ns, 'end_ns': time.time_ns(),
                      'queries': [q for q, _ in completed], 'query_count': len(batch),
                      'complete_results': len(completed) == len(batch),
                      'post_task_s': elapsed, 'http_batch_s': http_s,
                      'post_task_non_http_s': max(0, elapsed-http_s),
                      'profile': profile,
                      'contents_sha256': contents_digest([row[0] for _, row in completed])})

    async def close(self):
        await super().close()
        self._close_trace()
