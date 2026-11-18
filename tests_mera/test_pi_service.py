from __future__ import annotations

import asyncio
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mera_pi_service_test", ROOT / "submission_stage2/inference/pi_service.py")
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


WORKER = r'''
import json,sys
def emit(value):
    print(json.dumps(value),flush=True)
config=json.loads(sys.stdin.readline())
generation=0
replies=0
def operation():
    emit({'type':'operation','id':'op'+str(generation),'operation':'read_file','path':'/testbed/source.py'})
operation()
for line in sys.stdin:
    value=json.loads(line)
    if value.get('type')=='operation_result':
        replies+=1
        emit({'type':'complete','replies':replies,'received_type':value['type'],'result':value.get('output')})
    elif value.get('type')=='prompt':
        generation+=1
        operation()
'''


class PiServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.worker = Path(self.temporary.name) / "worker.py"
        self.worker.write_text(WORKER)
        self.app = service.create_app(worker_path=self.worker, node_binary=sys.executable,
                                      backend_url="http://vllm.invalid:8001", model_alias="adapted")
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False),
                                       base_url="http://bridge.invalid")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)
        self.temporary.cleanup()

    async def start(self):
        response = await self.client.post("/mera/pi/sessions", json={"instruction": "Fix the supplied repository."})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        data["headers"] = {"X-Mera-Session-Token": data["token"]}
        data["path"] = "/mera/pi/sessions/" + data["session_id"]
        return data

    async def events(self, session):
        response = await self.client.get(session["path"] + "/events", headers=session["headers"])
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["events"]

    async def test_twenty_concurrent_sessions_relay_only_their_own_results(self):
        sessions = await asyncio.gather(*(self.start() for _ in range(20)))
        self.assertEqual(len({s["token"] for s in sessions}), 20)
        initial = await asyncio.gather(*(self.events(s) for s in sessions))
        self.assertTrue(all(events[0]["id"] == "op0" for events in initial))
        replies = await asyncio.gather(*(self.client.post(s["path"] + "/result", headers=s["headers"],
                          json={"id": "op0", "output": "task-" + str(i)}) for i, s in enumerate(sessions)))
        self.assertTrue(all(r.status_code == 200 for r in replies))
        results = await asyncio.gather(*(self.events(s) for s in sessions))
        self.assertEqual([events[0]["result"] for events in results], ["task-" + str(i) for i in range(20)])
        for session in sessions:
            response = await self.client.delete(session["path"], headers=session["headers"])
            self.assertEqual(response.status_code, 204)
        self.assertEqual(len(self.app.state.sessions), 0)

    async def test_missing_or_cross_session_tokens_do_not_consume_events(self):
        left, right = await asyncio.gather(self.start(), self.start())
        self.assertEqual((await self.client.get(left["path"] + "/events")).status_code, 404)
        self.assertEqual((await self.client.get(left["path"] + "/events", headers=right["headers"])).status_code, 404)
        self.assertEqual((await self.events(left))[0]["type"], "operation")

    async def test_explicit_allowlisted_base_model_is_preserved_for_honest_comparison(self):
        app = service.create_app(worker_path=self.worker, node_binary=sys.executable,
                                 model_alias="adapted", allowed_model_aliases=["adapted", "base"])
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bridge.invalid") as client:
                health = (await client.get("/mera/pi/health")).json()
                self.assertEqual(health["allowed_models"], ["adapted", "base"])
                response = await client.post("/mera/pi/sessions", json={"instruction": "Fix repository.", "model": "base"})
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(data["model"], "base")
                self.assertEqual(app.state.sessions[data["session_id"]].config["model"], "base")
                unknown = await client.post("/mera/pi/sessions", json={"instruction": "Fix repository.", "model": "another-model"})
                self.assertEqual(unknown.status_code, 400)
                self.assertEqual(len(app.state.sessions), 1)

    async def test_lost_event_response_replays_operation_and_completion(self):
        session = await self.start()
        first = await self.events(session)
        self.assertEqual(await self.events(session), first)
        response = await self.client.post(session["path"] + "/result", headers=session["headers"],
                                          json={"id": "op0", "output": "checked"})
        self.assertEqual(response.status_code, 200)
        complete = await self.events(session)
        self.assertEqual(complete[0]["type"], "complete")
        self.assertEqual(await self.events(session), complete)

    async def test_operation_id_validation_and_concurrent_duplicate_results(self):
        session = await self.start()
        await self.events(session)
        for value in ({}, {"id": []}, {"id": ""}, {"id": "x" * 257}):
            response = await self.client.post(session["path"] + "/result", headers=session["headers"], json=value)
            self.assertEqual(response.status_code, 400, response.text)
        server = self.app.state.sessions[session["session_id"]]
        send = server.send

        async def slow_send(value):
            await asyncio.sleep(0.025)
            await send(value)

        server.send = slow_send
        replies = await asyncio.gather(*(self.client.post(session["path"] + "/result", headers=session["headers"],
                            json={"id": "op0", "output": "once", "type": "prompt"}) for _ in range(2)))
        self.assertEqual([r.status_code for r in replies], [200, 200])
        self.assertEqual(sum(bool(r.json().get("duplicate")) for r in replies), 1)
        complete = (await self.events(session))[0]
        self.assertEqual(complete["replies"], 1)
        self.assertEqual(complete["received_type"], "operation_result")
        response = await self.client.post(session["path"] + "/continue", headers=session["headers"], json={"message": "Review once more."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await self.events(session))[0]["id"], "op1")
        await self.client.post(session["path"] + "/result", headers=session["headers"], json={"id": "op1", "output": "twice"})
        await self.events(session)
        old_retry = await self.client.post(session["path"] + "/result", headers=session["headers"], json={"id": "op0", "output": "once"})
        self.assertEqual(old_retry.status_code, 200)
        self.assertTrue(old_retry.json()["duplicate"])

    async def test_expiry_with_full_event_queue_closes_process_and_releases_capacity(self):
        session = await self.start()
        await self.events(session)
        server = self.app.state.sessions[session["session_id"]]
        while not server.events.full():
            server.events.put_nowait({"type": "progress"})
        server.config["time_budget_sec"] = -30
        await asyncio.wait_for(server.expire(), 2)
        self.assertTrue(server.closed)
        self.assertIsNotNone(server.process.returncode)
        self.assertNotIn(session["session_id"], self.app.state.sessions)

    async def test_cancelled_start_after_spawn_cleans_process_and_session(self):
        started = asyncio.Event()
        original_send = service.Session.send
        processes = []

        async def blocked_send(server, value):
            if value.get("type") == "start":
                processes.append(server.process)
                started.set()
                await asyncio.Event().wait()
            await original_send(server, value)

        with patch.object(service.Session, "send", blocked_send):
            task = asyncio.create_task(self.client.post("/mera/pi/sessions", json={"instruction": "Fix repository."}))
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(self.app.state.sessions), 0)
        self.assertIsNotNone(processes[0].returncode)

    async def test_capacity_counts_inflight_process_spawns(self):
        entered = 0
        all_started = asyncio.Event()
        release = asyncio.Event()
        original = asyncio.create_subprocess_exec

        async def delayed_spawn(*args, **kwargs):
            nonlocal entered
            entered += 1
            if entered == 32:
                all_started.set()
            await release.wait()
            return await original(*args, **kwargs)

        with patch.object(service.asyncio, "create_subprocess_exec", delayed_spawn):
            starts = [asyncio.create_task(self.client.post("/mera/pi/sessions", json={"instruction": "Fix repository."})) for _ in range(33)]
            await asyncio.wait_for(all_started.wait(), 3)
            await asyncio.sleep(0.01)
            release.set()
            responses = await asyncio.gather(*starts)
        self.assertEqual(sum(r.status_code == 200 for r in responses), 32)
        self.assertEqual(sum(r.status_code == 429 for r in responses), 1)

    async def test_proxy_streams_openai_sse_and_preserves_encoded_responses(self):
        requests = []
        streams = []

        class Stream(httpx.AsyncByteStream):
            def __init__(self, blocks):
                self.blocks, self.closed = blocks, False

            async def __aiter__(self):
                for block in self.blocks:
                    await asyncio.sleep(0)
                    yield block

            async def aclose(self):
                self.closed = True

        async def backend(request):
            requests.append(request)
            if request.url.path == "/v1/chat/completions":
                blocks = [b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', b'data: [DONE]\n\n']
                stream = Stream(blocks)
                streams.append(stream)
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
            payload = gzip.compress(b'{"data":[{"id":"adapted"}]}')
            stream = Stream([payload[:10], payload[10:]])
            streams.append(stream)
            return httpx.Response(200, headers={"content-type": "application/json", "content-encoding": "gzip"}, stream=stream)

        previous = self.app.state.client
        self.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        try:
            response = await self.client.post("/v1/chat/completions", headers={"Authorization": "Bearer dummy", "X-Session-ID": "test"},
                                              json={"model": "adapted", "stream": True, "messages": []})
            self.assertEqual(response.status_code, 200)
            self.assertIn('data: [DONE]', response.text)
            response = await self.client.get("/v1/models")
            self.assertEqual(response.json()["data"][0]["id"], "adapted")
            self.assertTrue(all(stream.closed for stream in streams))
            self.assertEqual(requests[0].headers["x-session-id"], "test")
            self.assertEqual(requests[0].url.host, "vllm.invalid")
        finally:
            await self.app.state.client.aclose()
            self.app.state.client = previous

    async def test_cancelled_stream_closes_backend_inference_request(self):
        entered = asyncio.Event()
        closed = asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[]}\n\n'
                entered.set()
                await asyncio.Event().wait()

            async def aclose(self):
                closed.set()

        async def backend(request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

        previous = self.app.state.client
        self.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        try:
            task = asyncio.create_task(self.client.post("/v1/chat/completions", json={"stream": True}))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closed.wait(), 2)
        finally:
            await self.app.state.client.aclose()
            self.app.state.client = previous


if __name__ == "__main__":
    unittest.main()
