#!/usr/bin/env python3
"""Local-only diagnostic bridge from our vLLM wire protocol to Ollama native chat.

This bridge is not part of the submitted inference image. Ollama 0.17.6 lacks
vLLM's thinking token budget, preserved-thinking template controls and named
Qwen reasoning levels; the local diagnostic records those differences.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from typing import Any
import uuid

import httpx

UNSUPPORTED_FIELDS = {
    "thinking_token_budget": "unsupported; no per-thinking-token limit is enforced",
    "reasoning_effort": "replaced by the bridge's explicit boolean thinking setting",
    "chat_template_kwargs": "vLLM template controls are not available through native Ollama chat",
    "tool_choice": "Ollama 0.17.6 selects tools automatically",
}


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 18000
    upstream: str = "http://127.0.0.1:11435"
    model: str = "qwen3.5:9b-q4_K_M"
    alias: str = "qwen35-local"
    context: int = 16384
    max_tokens: int = 2048
    thinking: bool = False
    timeout: float = 300.0


def _json_object(value: Any, name: str) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _number(value: Any, name: str, *, integer: bool = False, minimum: float = 0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (integer and not float(value).is_integer()):
        raise ValueError(f"{name} is outside the permitted range")
    return int(value) if integer else value


def build_native_request(payload: dict, settings: Settings) -> tuple[dict, list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    if payload.get("model") != settings.alias:
        raise ValueError(f"model must be the configured local alias {settings.alias!r}")
    if payload.get("stream", False) is not False:
        raise ValueError("this local diagnostic bridge supports only non-streaming requests")
    source_messages = payload.get("messages")
    if not isinstance(source_messages, list) or not source_messages:
        raise ValueError("messages must be a nonempty array")
    messages = []
    tool_names: dict[str, str] = {}
    for source in source_messages:
        if not isinstance(source, dict) or source.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("each message requires a supported role")
        content = source.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("local SWE diagnostics support text message content only")
        message = {"role": source["role"], "content": content or ""}
        reasoning = source.get("reasoning_content") or source.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise ValueError("assistant reasoning must be text")
            message["thinking"] = reasoning
        if "tool_calls" in source:
            if source["role"] != "assistant" or not isinstance(source["tool_calls"], list):
                raise ValueError("tool_calls must be an assistant array")
            calls = []
            for call in source["tool_calls"]:
                if not isinstance(call, dict) or call.get("type", "function") != "function":
                    raise ValueError("only function tool calls are supported")
                function = _json_object(call.get("function"), "tool function")
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("tool function name must be nonempty")
                arguments = _json_object(function.get("arguments", {}), "tool arguments")
                native_call = {"function": {"name": name, "arguments": arguments}}
                if isinstance(call.get("id"), str) and call["id"]:
                    native_call["id"] = call["id"]
                    tool_names[call["id"]] = name
                calls.append(native_call)
            message["tool_calls"] = calls
        if source["role"] == "tool":
            identifier = source.get("tool_call_id")
            if isinstance(identifier, str) and identifier:
                message["tool_call_id"] = identifier
            name = source.get("name") or tool_names.get(identifier)
            if name:
                message["tool_name"] = name
        messages.append(message)

    requested_max = _number(payload.get("max_tokens", settings.max_tokens), "max_tokens", integer=True, minimum=1)
    options = {"num_ctx": settings.context, "num_predict": min(requested_max, settings.max_tokens)}
    defaults = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    for field, default in defaults.items():
        options[field] = _number(payload.get(field, default), field, integer=field == "top_k")
    if options["temperature"] > 2 or options["top_p"] > 1 or options["min_p"] > 1:
        raise ValueError("sampling option is outside its supported range")
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("only function tool definitions are supported")
    native = {"model": settings.model, "messages": messages, "tools": deepcopy(tools),
              "stream": False, "think": settings.thinking, "options": options,
              "keep_alive": "10m"}
    ignored = sorted(key for key in UNSUPPORTED_FIELDS if key in payload)
    return native, ignored


def build_openai_response(data: dict, alias: str) -> dict:
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError("upstream did not return a successful native chat object")
    raw = data.get("message")
    if not isinstance(raw, dict):
        raise ValueError("upstream chat response has no message")
    message = {"role": "assistant", "content": raw.get("content") or None}
    thinking = raw.get("thinking")
    if thinking:
        message.update(reasoning_content=thinking, reasoning=thinking)
    calls, seen = [], set()
    for call in raw.get("tool_calls") or []:
        function = _json_object(call.get("function"), "upstream tool function")
        if not isinstance(function.get("name"), str) or not function["name"]:
            raise ValueError("upstream tool call has no name")
        arguments = _json_object(function.get("arguments", {}), "upstream tool arguments")
        identifier = call.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            identifier = "call_" + uuid.uuid4().hex
        seen.add(identifier)
        calls.append({"id": identifier, "type": "function", "function": {
            "name": function["name"], "arguments": json.dumps(arguments, ensure_ascii=False),
        }})
    if calls:
        message["tool_calls"] = calls
    prompt = int(data.get("prompt_eval_count") or 0)
    completion = int(data.get("eval_count") or 0)
    reason = "tool_calls" if calls else "length" if data.get("done_reason") == "length" else "stop"
    return {"id": "chatcmpl-local-" + uuid.uuid4().hex, "object": "chat.completion",
            "created": int(time.time()), "model": alias,
            "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion, "prompt_tokens_details": {"cached_tokens": 0}}}


class BridgeState:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(timeout=settings.timeout, trust_env=False)
        self.owns_client = client is None
        self.lock = threading.Lock()
        self.chat_requests = 0
        self.ignored_fields: set[str] = set()

    def info(self) -> dict:
        try:
            version = self.client.get(self.settings.upstream.rstrip("/") + "/api/version")
            version.raise_for_status()
            engine_version = version.json().get("version")
        except (httpx.HTTPError, ValueError):
            engine_version = None
        with self.lock:
            return {"diagnostic_only": True, "engine": "ollama-native", "engine_version": engine_version,
                    "model": self.settings.model, "served_alias": self.settings.alias,
                    "context_tokens": self.settings.context, "max_output_tokens": self.settings.max_tokens,
                    "thinking": self.settings.thinking, "thinking_token_budget_enforced": False,
                    "production_equivalent": False, "requests": self.chat_requests,
                    "ignored_fields_seen": sorted(self.ignored_fields),
                    "unsupported_controls": UNSUPPORTED_FIELDS}


def create_server(settings: Settings, upstream_client: httpx.Client | None = None) -> ThreadingHTTPServer:
    state = BridgeState(settings, upstream_client)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # No request bodies, credentials or model-generated content in stderr.
            return

        def send_json(self, status: int, body: dict):
            data = json.dumps(body, ensure_ascii=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def error(self, status, message):
            self.send_json(status, {"error": {"message": message, "type": "local_bridge_error"}})

        def do_GET(self):
            if self.path == "/local-info":
                self.send_json(200, state.info())
            elif self.path.rstrip("/") == "/v1/models":
                try:
                    response = state.client.get(settings.upstream.rstrip("/") + "/api/tags")
                    response.raise_for_status()
                    names = {item.get("name") for item in response.json().get("models", [])}
                    available = settings.model in names or (":" not in settings.model and settings.model + ":latest" in names)
                    models = [{"id": settings.alias, "object": "model", "owned_by": "local-ollama"}] if available else []
                    self.send_json(200, {"object": "list", "data": models})
                except (httpx.HTTPError, ValueError):
                    self.error(502, "Cannot discover local Ollama models")
            else:
                self.error(404, "Unknown local bridge endpoint")

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.error(404, "Unknown local bridge endpoint")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("body length must be between 1 byte and 8 MiB")
                payload = json.loads(self.rfile.read(length))
                native, ignored = build_native_request(payload, settings)
            except (ValueError, TypeError, KeyError) as exc:
                self.error(400, str(exc))
                return
            with state.lock:
                state.chat_requests += 1
                state.ignored_fields.update(ignored)
            try:
                response = state.client.post(settings.upstream.rstrip("/") + "/api/chat", json=native)
                if response.status_code >= 400:
                    self.error(502, f"Ollama returned HTTP {response.status_code}; inspect the local Ollama log")
                    return
                result = build_openai_response(response.json(), settings.alias)
                self.send_json(200, result)
            except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
                self.error(502, "Local Ollama request failed or returned an invalid native chat response")

    server = ThreadingHTTPServer((settings.host, settings.port), Handler)
    server.daemon_threads = True
    server.bridge_state = state
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--upstream", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--alias", default="qwen35-local")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking", choices=["off", "on"], default="off")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.context < 1 or args.max_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("context, max tokens and timeout must be positive")
    settings = Settings(args.host, args.port, args.upstream, args.model, args.alias,
                        args.context, args.max_tokens, args.thinking == "on", args.timeout)
    server = create_server(settings)
    print(json.dumps({"bridge": f"http://{args.host}:{args.port}/v1", "model": args.model,
                      "alias": args.alias, "thinking": settings.thinking, "diagnostic_only": True}), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if server.bridge_state.owns_client:
            server.bridge_state.client.close()


if __name__ == "__main__":
    main()
