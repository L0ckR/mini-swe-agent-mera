"""Pi runs beside vLLM; task operations are relayed to the Harbor worker.

The model sees Pi's native tool protocol. No task source is opened or executed
in this process: the authenticated session client executes each operation in
its own Harbor task environment and sends its result back.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import secrets
import signal
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


class Session:
    def __init__(self, process, config, *, on_close=None):
        self.process, self.config = process, config
        self.token = secrets.token_urlsafe(32)
        self.events = asyncio.Queue(maxsize=512)
        self.pending = set()
        self.pending_events = {}
        self.last_reply = None
        self.completed = OrderedDict()
        self.ack_lock = asyncio.Lock()
        self.finished = False
        self.last_complete_event = None
        self.closed = False
        self.created = time.monotonic()
        self.tasks = []
        self.write_lock = asyncio.Lock()
        self.on_close = on_close
        self._close_task = None

    def terminal_event(self, event):
        """A disconnected consumer must not prevent expiry or cleanup."""
        if event.get("type") in {"complete", "error"}:
            self.finished = True
            self.last_complete_event = event
        if self.events.full():
            self.events.get_nowait()
        self.events.put_nowait(event)

    async def send(self, value):
        async with self.write_lock:
            if self.process.returncode is not None:
                raise RuntimeError("Pi worker has exited")
            self.process.stdin.write((json.dumps(value, ensure_ascii=True) + "\n").encode())
            await self.process.stdin.drain()

    async def read(self):
        try:
            while line := await self.process.stdout.readline():
                event = json.loads(line)
                kind = event.get("type")
                if kind == "operation":
                    identifier = event.get("id")
                    if (not isinstance(identifier, str) or not identifier or len(identifier) > 256
                            or identifier in self.pending or identifier in self.completed):
                        raise ValueError("Pi worker emitted an invalid operation id")
                    self.pending.add(identifier)
                    self.pending_events[identifier] = event
                if kind in {"complete", "error"}:
                    self.finished = True
                    self.last_complete_event = event
                await self.events.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.terminal_event({"type": "error", "error": f"Pi protocol failed: {type(exc).__name__}: {exc}"})
        finally:
            if not self.closed:
                if not self.finished:
                    self.terminal_event({"type": "error", "error": "Pi worker exited without a completion event"})
                self.finished = True
                await self.close()

    async def drain_stderr(self):
        # Keep a bounded diagnostic tail; stderr must not backpressure the worker.
        self.stderr = ""
        while data := await self.process.stderr.read(4096):
            self.stderr = (self.stderr + data.decode(errors="replace"))[-8000:]

    async def expire(self):
        await asyncio.sleep(self.config["time_budget_sec"] + 30)
        self.terminal_event({"type": "error", "error": "Pi session deadline reached"})
        await self.close()

    async def close(self):
        # Cleanup survives cancellation of an HTTP request or the reader task.
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self):
        self.closed = True
        self.finished = True
        for task in self.tasks:
            if task is not asyncio.current_task():
                task.cancel()
        if self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            except ProcessLookupError:
                pass
        await asyncio.gather(*(task for task in self.tasks if task is not asyncio.current_task()), return_exceptions=True)
        if self.on_close is not None:
            self.on_close()


def create_app(*, backend_url=None, worker_path=None, node_binary=None, model_alias=None, allowed_model_aliases=None):
    backend = (backend_url or os.environ.get("MERA_VLLM_BASE_URL", "http://127.0.0.1:8001")).rstrip("/")
    worker = str(worker_path or os.environ.get("MERA_PI_WORKER", "/opt/mera-pi/worker.mjs"))
    node = str(node_binary or os.environ.get("MERA_NODE_BIN", "/usr/local/bin/node"))
    alias = model_alias or os.environ.get("MERA_PI_MODEL", "qwen38-27b-pi-mera")
    if allowed_model_aliases is None:
        allowed_model_aliases = os.environ.get("MERA_PI_ALLOWED_MODELS", "").split(",")
    if isinstance(allowed_model_aliases, str):
        allowed_model_aliases = allowed_model_aliases.split(",")
    allowed_models = {value.strip() for value in allowed_model_aliases if isinstance(value, str) and value.strip()}
    allowed_models.add(alias)
    sessions = {}
    capacity_lock = asyncio.Lock()
    starting = 0

    @asynccontextmanager
    async def lifespan(app):
        app.state.client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(300, connect=10))
        try:
            yield
        finally:
            await asyncio.gather(*(session.close() for session in list(sessions.values())), return_exceptions=True)
            await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.sessions = sessions

    def session_for(identifier, request):
        session = sessions.get(identifier)
        token = request.headers.get("X-Mera-Session-Token", "")
        if session is None or not secrets.compare_digest(token, session.token):
            raise HTTPException(404, "Unknown Pi session")
        return session

    async def body(request, maximum=8 * 1024 * 1024):
        data = await request.body()
        if len(data) > maximum:
            raise HTTPException(413, "Request too large")
        try:
            value = json.loads(data)
        except ValueError:
            raise HTTPException(400, "Invalid JSON") from None
        if not isinstance(value, dict):
            raise HTTPException(400, "JSON object required")
        return value

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def inference_proxy(path: str, request: Request):
        data = await request.body()
        if len(data) > 16 * 1024 * 1024:
            raise HTTPException(413, "Inference request too large")
        headers = {key: value for key, value in request.headers.items()
                   if key.lower() in {"authorization", "content-type", "x-session-id"}}
        try:
            outgoing = app.state.client.build_request(request.method, backend + "/v1/" + path,
                                                      content=data, headers=headers, params=request.query_params)
            response = await app.state.client.send(outgoing, stream=True)
        except httpx.HTTPError:
            return JSONResponse({"error": "Inference backend unavailable"}, status_code=503)
        headers = {key: value for key, value in response.headers.items()
                   if key.lower() in {"content-type", "content-encoding", "cache-control", "x-request-id"}}

        async def stream():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await asyncio.shield(response.aclose())

        return StreamingResponse(stream(), status_code=response.status_code, headers=headers,
                                 background=BackgroundTask(response.aclose))

    @app.get("/mera/pi/health")
    async def health():
        return {"service": "native-pi-harbor-bridge", "worker_present": Path(worker).is_file(),
                "node_present": Path(node).is_file(), "model": alias,
                "allowed_models": sorted(allowed_models),
                "session_count": len(sessions)}

    @app.post("/mera/pi/sessions")
    async def start(request: Request):
        nonlocal starting
        config = await body(request, 256 * 1024)
        instruction = config.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 200000:
            raise HTTPException(400, "Nonempty bounded instruction required")
        if config.get("workdir", "/testbed") != "/testbed":
            raise HTTPException(400, "Pi task workdir must be /testbed")
        requested_model = config.get("model", alias)
        if not isinstance(requested_model, str) or requested_model not in allowed_models:
            raise HTTPException(400, "Model is not in the Pi session allow-list")
        for name, low, high, default in (("max_turns", 1, 200, 100), ("time_budget_sec", 10, 1740, 1680),
                                         ("max_output_tokens", 256, 32768, 8192), ("context_window", 4096, 131072, 65536)):
            value = config.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise HTTPException(400, f"Invalid {name}")
            config[name] = value
        # The caller cannot redirect credentials/model traffic to another URL.
        config.update(base_url=backend + "/v1", api_key="dummy", model=requested_model, workdir="/testbed")
        environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH") if key in os.environ}
        environment.update(HOME="/opt/mera-pi/empty-home", PI_SKIP_VERSION_CHECK="1", NO_COLOR="1")
        async with capacity_lock:
            if len(sessions) + starting >= 32:
                raise HTTPException(429, "Pi session capacity reached")
            starting += 1
        reserved = True
        session = None
        try:
            process = await asyncio.create_subprocess_exec(node, worker, stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                        env=environment, start_new_session=True, limit=8 * 1024 * 1024)
            identifier = secrets.token_hex(16)
            session = Session(process, config, on_close=lambda: sessions.pop(identifier, None))
            async with capacity_lock:
                sessions[identifier] = session
                starting -= 1
                reserved = False
            await session.send({**config, "type": "start"})
            session.tasks = [asyncio.create_task(session.read()), asyncio.create_task(session.drain_stderr()),
                             asyncio.create_task(session.expire())]
            return {"session_id": identifier, "token": session.token, "model": requested_model}
        except BaseException:
            if session is not None:
                await asyncio.shield(session.close())
            raise
        finally:
            if reserved:
                async with capacity_lock:
                    starting -= 1

    @app.get("/mera/pi/sessions/{identifier}/events")
    async def events(identifier: str, request: Request):
        session = session_for(identifier, request)
        result = []
        if session.events.empty():
            # GET can have succeeded server-side while its HTTP response was
            # lost. Replay an unacknowledged operation with the SAME id; the
            # Harbor client caches completed tool outputs by id.
            if session.pending_events:
                return {"events": list(session.pending_events.values())[:32]}
            if session.finished and session.last_complete_event is not None:
                return {"events": [session.last_complete_event]}
        try:
            result.append(await asyncio.wait_for(session.events.get(), timeout=10))
        except TimeoutError:
            return {"events": []}
        while not session.events.empty() and len(result) < 32:
            result.append(session.events.get_nowait())
        return {"events": result}

    @app.post("/mera/pi/sessions/{identifier}/result")
    async def operation_result(identifier: str, request: Request):
        session = session_for(identifier, request)
        value = await body(request)
        operation_id = value.get("id")
        if not isinstance(operation_id, str) or not operation_id or len(operation_id) > 256:
            raise HTTPException(400, "Bounded operation id required")
        # Serialize the membership check and dispatch so concurrent retries
        # cannot send a result twice. Remember earlier replies, not only the last.
        async with session.ack_lock:
            if operation_id in session.completed:
                return {"accepted": True, "duplicate": True}
            if operation_id not in session.pending:
                raise HTTPException(409, "Unknown or completed Pi operation")
            await session.send({**value, "type": "operation_result"})
            session.pending.remove(operation_id)
            session.pending_events.pop(operation_id, None)
            session.completed[operation_id] = None
            if len(session.completed) > 256:
                session.completed.popitem(last=False)
            session.last_reply = operation_id
        return {"accepted": True}

    @app.post("/mera/pi/sessions/{identifier}/continue")
    async def continuation(identifier: str, request: Request):
        session = session_for(identifier, request)
        value = await body(request, 128 * 1024)
        message = value.get("message")
        if not isinstance(message, str) or not message.strip() or len(message) > 100000:
            raise HTTPException(400, "Bounded continuation message required")
        async with session.ack_lock:
            if not session.finished or session.pending:
                raise HTTPException(409, "Pi is still running")
            previous_complete = session.last_complete_event
            session.last_complete_event = None
            session.finished = False
            try:
                await session.send({"type": "prompt", "message": message})
            except BaseException:
                session.finished = True
                session.last_complete_event = previous_complete
                raise
        return {"accepted": True}

    @app.delete("/mera/pi/sessions/{identifier}")
    async def stop(identifier: str, request: Request):
        session = session_for(identifier, request)
        sessions.pop(identifier, None)
        await session.close()
        return Response(status_code=204)

    return app


app = create_app()
