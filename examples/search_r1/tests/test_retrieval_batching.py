import asyncio

import pytest
from aiohttp import ClientResponseError, web

from examples.search_r1.search_tool import SearchTool


async def start_server(handler):
    app = web.Application()
    app.router.add_post("/retrieve", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/retrieve"


@pytest.mark.parametrize("mode", ["success", "http_error", "bad_payload"])
def test_batches_preserve_mapping_and_fan_out_errors(mode):
    async def check():
        received, active, peak = [], 0, 0

        async def handler(request):
            nonlocal active, peak
            body = await request.json()
            received.append(body)
            active += 1
            peak = max(peak, active)
            try:
                # Force the first batch to finish after subsequent requests.
                await asyncio.sleep(.02 if "q0" in body["queries"] else .001)
                if mode == "http_error":
                    return web.Response(status=503)
                result = [[{"document": {"contents": f"Title {q}\nEvidence {q}"}, "score": .9}]
                          for q in body["queries"]]
                return web.json_response({"result": result if mode == "success" else [], "profile": {"total_s": .01}})
            finally:
                active -= 1

        runner, url = await start_server(handler)
        tool = SearchTool({"retrieval_service_url": url, "batch_size": 8, "max_concurrency": 2})
        instances = [(await tool.create())[0] for _ in range(33)]
        try:
            results = await asyncio.wait_for(asyncio.gather(*[
                tool.execute(instance, {"query": f" q{i} "}) for i, instance in enumerate(instances)
            ], return_exceptions=True), timeout=3)
            assert len(received) == 5 and max(len(r["queries"]) for r in received) == 8
            assert sorted(q for r in received for q in r["queries"]) == sorted(f"q{i}" for i in range(33))
            assert all(r["topk"] == 3 and r["return_scores"] is True for r in received)
            assert peak <= 2
            for i, result in enumerate(results):
                if mode != "success":
                    assert isinstance(result, ClientResponseError if mode == "http_error" else ValueError)
                else:
                    response, reward, metrics = result
                    assert response.text == f"Doc 1(Title: Title q{i}) Evidence q{i}"
                    assert reward == 0 and metrics["retrieval_profile_scope"] == "http_batch"
                    assert 1 <= metrics["retrieval_batch_size"] <= 8
        finally:
            for instance in instances:
                await tool.release(instance)
            await tool.close()
            await runner.cleanup()
        assert not tool._states and not tool._pending and not tool._batch_tasks

    asyncio.run(check())


def test_cancelled_query_does_not_shift_sibling_results():
    async def check():
        started, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            body = await request.json()
            started.set()
            await release.wait()
            return web.json_response({"result": [[{"document": {"contents": q}}] for q in body["queries"]]})

        runner, url = await start_server(handler)
        tool = SearchTool({"retrieval_service_url": url, "batch_size": 4})
        instance, _ = await tool.create()
        try:
            tasks = [asyncio.create_task(tool.execute(instance, {"query": f"q{i}"})) for i in range(4)]
            await asyncio.wait_for(started.wait(), 2)
            tasks[1].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[1]
            release.set()
            results = await asyncio.wait_for(asyncio.gather(tasks[0], tasks[2], tasks[3]), 2)
            assert [r[0].text for r in results] == [f"Doc 1(Title: q{i})" for i in [0, 2, 3]]
        finally:
            release.set()
            await tool.release(instance)
            await tool.close()
            await runner.cleanup()

    asyncio.run(check())


def test_close_cancels_unflushed_queries():
    async def check():
        tool = SearchTool({"retrieval_service_url": "http://127.0.0.1:1/retrieve", "batch_wait_ms": 1000})
        instance, _ = await tool.create()
        task = asyncio.create_task(tool.execute(instance, {"query": "q"}))
        await asyncio.sleep(.005)
        await tool.close()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        await tool.release(instance)
        assert not tool._pending and not tool._batch_tasks

    asyncio.run(check())


def test_close_cancels_batch_before_http_coroutine_starts():
    async def check():
        tool = SearchTool({"retrieval_service_url": "http://127.0.0.1:1/retrieve", "batch_wait_ms": 0})
        instance, _ = await tool.create()
        task = asyncio.create_task(tool.execute(instance, {"query": "q"}))
        # execute schedules the dispatcher, which then schedules the HTTP task.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(tool._batch_tasks) == 1 and not tool._pending
        await tool.close()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        await tool.release(instance)
        assert not tool._batch_tasks

    asyncio.run(check())


@pytest.mark.parametrize("mode", ["success", "http_error", "bad_payload", "timeout", "close"])
def test_saturated_sender_keeps_queries_available_for_batching(mode):
    async def check():
        received = []
        started, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            body = await request.json()
            received.append(body["queries"])
            if len(received) == 2:
                started.set()
            await release.wait()
            if mode == "http_error":
                return web.Response(status=503)
            rows = [[{"document": {"contents": q}}] for q in body["queries"]]
            return web.json_response({"result": [] if mode == "bad_payload" else rows})

        runner, url = await start_server(handler)
        tool = SearchTool({"retrieval_service_url": url, "batch_size": 4, "max_concurrency": 2,
                           "batch_wait_ms": 0, "timeout": .2 if mode == "timeout" else 2})
        instance, _ = await tool.create()
        tasks = []
        try:
            # Occupy both sending slots before offering more queries.
            tasks.append(asyncio.create_task(tool.execute(instance, {"query": "first"})))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            tasks.append(asyncio.create_task(tool.execute(instance, {"query": "second"})))
            await asyncio.wait_for(started.wait(), 1)
            for i in range(9):
                tasks.append(asyncio.create_task(tool.execute(instance, {"query": f"q{i}"})))
                # Each query arrives in a separate event-loop turn, like model replies.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            assert len(tool._batch_tasks) == 2
            assert len(tool._pending) == 9
            assert received == [["first"], ["second"]]
            # A cancelled, unsent query must not be sent or consume a result slot.
            tasks[5].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[5]
            if mode == "close":
                await tool.close()
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
                assert all(isinstance(r, asyncio.CancelledError) for r in results)
                assert len(received) == 2
            else:
                if mode == "timeout":
                    # Both initial HTTP calls expire; their capacity must become reusable.
                    initial = await asyncio.wait_for(asyncio.gather(*tasks[:2], return_exceptions=True), 1)
                    assert all(isinstance(r, asyncio.TimeoutError) for r in initial)
                release.set()
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
                assert [len(batch) for batch in received] == [1, 1, 4, 4]
                assert [q for batch in received[2:] for q in batch] == [f"q{i}" for i in range(9) if i != 3]
                for index, result in enumerate(results):
                    if index == 5:
                        assert isinstance(result, asyncio.CancelledError)
                    elif mode in ["http_error", "bad_payload"]:
                        assert isinstance(result, ClientResponseError if mode == "http_error" else ValueError)
                    elif mode == "timeout" and index < 2:
                        assert isinstance(result, asyncio.TimeoutError)
                    else:
                        query = ["first", "second"][index] if index < 2 else f"q{index-2}"
                        assert result[0].text == f"Doc 1(Title: {query})"
                # Re-enter after the dispatcher drained its queue.
                if mode in ["success", "timeout"]:
                    result = await asyncio.wait_for(tool.execute(instance, {"query": "next-wave"}), 1)
                    assert result[0].text == "Doc 1(Title: next-wave)"
        finally:
            release.set()
            # Let already-created HTTP tasks enter their cancellation handlers.
            await asyncio.sleep(0)
            await tool.close()
            await asyncio.gather(*tasks, return_exceptions=True)
            await tool.release(instance)
            await runner.cleanup()
        assert not tool._pending and not tool._batch_tasks and tool._dispatcher is None

    asyncio.run(check())
