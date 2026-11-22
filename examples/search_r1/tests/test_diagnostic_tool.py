import asyncio
import json

import pytest
from aiohttp import web

from examples.search_r1.diagnostic_tool import ProfiledSearchTool, contents_digest
from examples.search_r1.tests.test_retrieval_batching import start_server


@pytest.mark.parametrize('fail', [False, True])
def test_tracing_preserves_results_and_links_requests(tmp_path, fail):
    async def check():
        received = []

        async def handler(request):
            body = await request.json()
            received.append(body)
            if fail:
                return web.Response(status=503)
            return web.json_response({
                'result': [[{'document': {'contents': q+'\nEvidence'}}] for q in body['queries']],
                'profile': {'request_id': f'batch-{len(received)}'},
            })

        runner, url = await start_server(handler)
        tool = ProfiledSearchTool({'retrieval_service_url': url, 'batch_size': 4,
                                   'max_concurrency': 1, 'diagnostic_trace_dir': str(tmp_path)})
        instance, _ = await tool.create()
        try:
            results = await asyncio.gather(*[
                tool.execute(instance, {'query': f' q{i} '}) for i in range(9)
            ], return_exceptions=True)
        finally:
            await tool.close()
            await runner.cleanup()
        traces = [json.loads(line) for path in tmp_path.glob('*.jsonl') for line in path.open()]
        assert [len(x['queries']) for x in received] == [4, 4, 1]
        if fail:
            assert all(isinstance(r, Exception) for r in results)
            assert [r['kind'] for r in traces] == ['batch_error']*3
            return
        assert [r[0].text for r in results] == [f'Doc 1(Title: q{i}) Evidence' for i in range(9)]
        batches = {r['profile']['request_id']: r for r in traces if r['kind'] == 'batch'}
        queries = [r for r in traces if r['kind'] == 'query']
        assert len(batches) == 3 and len(queries) == 9
        for batch in batches.values():
            assert batch['contents_sha256'] == contents_digest([
                [{'document': {'contents': q+'\nEvidence'}}] for q in batch['queries']])
            assert batch['complete_results']
        for query in queries:
            assert query['query'] in batches[query['server_request_id']]['queries']
            assert query['wall_s'] >= query['http_batch_s'] >= 0
    asyncio.run(check())
