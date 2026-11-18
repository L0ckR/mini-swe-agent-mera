from __future__ import annotations

from dataclasses import replace
import json
import threading
import unittest

import httpx

from scripts.ollama_bridge import Settings, buffered_sse, build_native_request, build_openai_response, create_server


TOOL = {"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"],
}}}


class ConversionTests(unittest.TestCase):
    def test_native_conversion_preserves_tools_and_whitespace_and_records_unsupported_controls(self):
        command = "python3 - <<'PY'\n    print('literal $HOME')\nPY\n"
        payload = {
            "model": "qwen35-local", "messages": [
                {"role": "system", "content": "Solve issue", "extra": {"internal": True}},
                {"role": "assistant", "content": None, "reasoning_content": "Check file.", "tool_calls": [
                    {"id": "call1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}},
                ]},
                {"role": "tool", "tool_call_id": "call1", "content": "exit_code: 0"},
            ], "tools": [TOOL], "max_tokens": 8192, "temperature": 0.7, "top_p": 0.8,
            "top_k": 20, "min_p": 0.0, "thinking_token_budget": 3072,
            "reasoning_effort": "medium", "chat_template_kwargs": {"enable_thinking": True},
            "tool_choice": "auto",
        }
        native, ignored = build_native_request(payload, Settings())
        self.assertEqual(native["model"], "qwen3.5:9b-q4_K_M")
        self.assertFalse(native["think"])
        self.assertFalse(native["stream"])
        self.assertEqual(native["options"], {"num_ctx": 16384, "num_predict": 2048, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0})
        self.assertEqual(native["messages"][1]["thinking"], "Check file.")
        self.assertEqual(native["messages"][1]["tool_calls"][0]["function"]["arguments"]["command"], command)
        self.assertEqual(native["messages"][2]["tool_name"], "bash")
        self.assertNotIn("extra", native["messages"][0])
        self.assertEqual(ignored, ["chat_template_kwargs", "reasoning_effort", "thinking_token_budget", "tool_choice"])
        self.assertNotIn("reasoning_effort", native)

    def test_response_conversion_generates_unique_ids_and_maps_reasoning_and_usage(self):
        data = {"message": {"role": "assistant", "thinking": "Inspect result", "content": "",
                            "tool_calls": [
                                {"id": "same", "function": {"name": "bash", "arguments": {"command": "pwd"}}},
                                {"id": "same", "function": {"name": "bash", "arguments": {"command": "ls"}}},
                            ]}, "done_reason": "stop", "prompt_eval_count": 123, "eval_count": 42}
        reply = build_openai_response(data, "qwen35-local")
        self.assertEqual(reply["object"], "chat.completion")
        self.assertEqual(reply["model"], "qwen35-local")
        choice = reply["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["reasoning_content"], "Inspect result")
        self.assertEqual(len({call["id"] for call in choice["message"]["tool_calls"]}), 2)
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"command": "pwd"})
        self.assertEqual(reply["usage"]["total_tokens"], 165)

    def test_invalid_alias_stream_or_tool_arguments_are_rejected(self):
        base = {"model": "qwen35-local", "messages": [{"role": "user", "content": "hi"}]}
        for update in ({"model": "wrong"}, {"stream": "yes"}, {"stream": 1}, {"stream_options": []},
                       {"stream_options": {"include_usage": 1}}, {"max_tokens": True}, {"messages": []}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                build_native_request(base | update, Settings())
        with self.assertRaises(ValueError):
            build_native_request(base | {"messages": [{"role": "assistant", "tool_calls": [
                {"function": {"name": "bash", "arguments": "not json"}},
            ]}]}, Settings())

    def test_buffered_sse_preserves_indexed_calls_reasoning_and_final_usage(self):
        native, _ = build_native_request({"model": "qwen35-local", "messages": [{"role": "user", "content": "hi"}],
                                         "stream": True, "stream_options": {"include_usage": True}}, Settings())
        self.assertFalse(native["stream"])
        response = build_openai_response({"message": {"content": "Обнаружены файлы", "thinking": "Read source.", "tool_calls": [
            {"id": "call-a", "function": {"name": "read", "arguments": {"path": "one.py"}}},
            {"id": "call-b", "function": {"name": "bash", "arguments": {"command": "printf 'hello\\n'"}}},
        ]}, "prompt_eval_count": 50, "eval_count": 12}, "local")
        lines = buffered_sse(response).decode().splitlines()
        self.assertEqual(lines[-2], "data: [DONE]")
        chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
        self.assertTrue(all(chunk["object"] == "chat.completion.chunk" for chunk in chunks))
        self.assertEqual({chunk["id"] for chunk in chunks}, {response["id"]})
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
        self.assertEqual("".join(delta.get("reasoning_content", "") for delta in deltas), "Read source.")
        self.assertEqual("".join(delta.get("content", "") for delta in deltas), "Обнаружены файлы")
        calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
        self.assertEqual([call["index"] for call in calls], [0, 1])
        self.assertEqual([call["id"] for call in calls], ["call-a", "call-b"])
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"command": "printf 'hello\\n'"})
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"]["total_tokens"], 62)
        without_usage = buffered_sse(response, include_usage=False).decode()
        self.assertNotIn('"usage"', without_usage)

    def test_thinking_setting_and_length_response_are_explicit(self):
        native, _ = build_native_request({"model": "qwen35-local", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 256}, replace(Settings(), thinking=True))
        self.assertTrue(native["think"])
        self.assertEqual(native["options"]["num_predict"], 256)
        self.assertEqual(build_openai_response({"message": {"content": "partial"}, "done_reason": "length"}, "local")["choices"][0]["finish_reason"], "length")


class HTTPProtocolTests(unittest.TestCase):
    def setUp(self):
        self.requests = []

        def upstream(request):
            self.requests.append(request)
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b-q4_K_M"}]})
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.17.6"})
            if request.url.path == "/api/chat":
                return httpx.Response(200, json={"message": {"tool_calls": [{"function": {
                    "name": "bash", "arguments": {"command": "pwd"},
                }}]}, "prompt_eval_count": 50, "eval_count": 12, "done_reason": "stop"})
            return httpx.Response(404)

        self.upstream = httpx.Client(transport=httpx.MockTransport(upstream))
        self.server = create_server(replace(Settings(), port=0), self.upstream)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_address[1])
        self.client = httpx.Client(base_url=self.base, timeout=3, trust_env=False)

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.upstream.close()

    def test_http_round_trip_and_limitations_metadata(self):
        models = self.client.get("/v1/models").json()
        self.assertEqual(models["data"][0]["id"], "qwen35-local")
        response = self.client.post("/v1/chat/completions", headers={"Authorization": "Bearer dummy"}, json={
            "model": "qwen35-local", "messages": [{"role": "user", "content": "Call bash pwd"}],
            "tools": [TOOL], "thinking_token_budget": 3072, "reasoning_effort": "medium",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "tool_calls")
        native = json.loads(next(r.content for r in self.requests if r.url.path == "/api/chat"))
        self.assertFalse(native["think"])
        self.assertNotIn("reasoning_effort", native)
        info = self.client.get("/local-info").json()
        self.assertEqual(info["engine_version"], "0.17.6")
        self.assertFalse(info["production_equivalent"])
        self.assertFalse(info["thinking_token_budget_enforced"])
        self.assertIn("thinking_token_budget", info["ignored_fields_seen"])
        self.assertEqual(info["requests"], 1)

    def test_http_invalid_payload_and_upstream_error_are_clear(self):
        self.assertEqual(self.client.post("/v1/chat/completions", content="{broken").status_code, 400)
        self.assertEqual(self.client.get("/missing").status_code, 404)
        self.upstream.close()
        self.server.bridge_state.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500, text="internal")))
        try:
            response = self.client.post("/v1/chat/completions", json={"model": "qwen35-local", "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(response.status_code, 502)
            self.assertIn("Ollama returned HTTP 500", response.json()["error"]["message"])
        finally:
            self.server.bridge_state.client.close()

    def test_http_sse_stream_parses_through_httpx_and_keeps_native_upstream_buffered(self):
        with self.client.stream("POST", "/v1/chat/completions", json={
            "model": "qwen35-local", "messages": [{"role": "user", "content": "Call bash pwd"}],
            "tools": [TOOL], "stream": True, "stream_options": {"include_usage": True},
        }) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            events = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(value) for value in events[:-1]]
        call = next(chunk["choices"][0]["delta"]["tool_calls"][0] for chunk in chunks
                    if chunk["choices"] and chunk["choices"][0]["delta"].get("tool_calls"))
        self.assertEqual(call["index"], 0)
        self.assertTrue(call["id"])
        self.assertEqual(call["type"], "function")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "pwd"})
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 12)
        upstream = json.loads(next(request.content for request in self.requests if request.url.path == "/api/chat"))
        self.assertFalse(upstream["stream"])
        info = self.client.get("/local-info").json()
        self.assertEqual(info["stream_mode"], "buffered_sse")
        self.assertFalse(info["upstream_streaming"])
        self.assertEqual(info["streaming_requests"], 1)
