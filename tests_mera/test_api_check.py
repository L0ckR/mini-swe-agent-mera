from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from scripts.check_inference import check, native_tools


class InferenceAPICheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_contract_replay_all_native_tools_and_concurrent_non_thinking_requests(self):
        requests = []

        async def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "mera-model"}]})
            if request.url.path.endswith("/mera/pi/health"):
                return httpx.Response(200, json={"service": "native-pi-harbor-bridge", "model": "mera-model",
                                                "worker_present": True, "node_present": True})
            payload = json.loads(request.content)
            requests.append(payload)
            prompt = payload["messages"][-1]["content"]
            name, arguments = "bash", {"command": "pwd"}
            if "with these exact JSON arguments" in prompt:
                name = prompt.split()[1]
                arguments, _ = json.JSONDecoder().raw_decode(prompt.split("arguments: ", 1)[1])
            elif "The tool operation is complete" in prompt:
                return httpx.Response(200, json={"object": "chat.completion", "model": "mera-model",
                    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "protocol checked"}}]})
            return httpx.Response(200, json={
                "object": "chat.completion", "model": "mera-model",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_probe", "type": "function", "function": {
                        "name": name, "arguments": json.dumps(arguments),
                    }}],
                }}], "usage": {"completion_tokens": 10},
            })

        factory = httpx.AsyncClient
        with patch("scripts.check_inference.httpx.AsyncClient", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)):
            result = await check("http://inference.invalid/v1", "mera-model", "dummy", 4)
        self.assertEqual(result["custom_tool_check"], "passed")
        self.assertEqual(result["multi_turn_check"], "passed")
        self.assertEqual(result["completion_tokens"], 40)
        self.assertEqual(len(requests), 11)
        self.assertEqual(requests[1]["messages"][2]["role"], "tool")
        self.assertEqual(requests[2]["tools"], native_tools())
        self.assertEqual({entry["function"]["name"] for entry in requests[2]["tools"]}, {"bash", "read", "edit", "write"})
        native_bash = next(entry["function"]["parameters"] for entry in requests[2]["tools"] if entry["function"]["name"] == "bash")
        self.assertIn("timeout", native_bash["properties"])
        self.assertNotIn("timeout_sec", native_bash["properties"])
        self.assertEqual(result["pi_service_check"], "passed")
        self.assertEqual(result["completion_check"], "passed")
        self.assertIs(result["thinking"], False)
        self.assertTrue(all(payload["chat_template_kwargs"] == {"enable_thinking": False} for payload in requests))
        self.assertTrue(all("thinking_token_budget" not in payload and "reasoning_effort" not in payload for payload in requests))
        self.assertTrue(all(payload["temperature"] == 0.8 and payload["max_tokens"] == 8192 for payload in requests[-4:]))

    async def test_missing_model_alias_fails_before_completions(self):
        async def handler(request):
            self.assertTrue(request.url.path.endswith("/models"))
            return httpx.Response(200, json={"data": [{"id": "wrong-alias"}]})

        factory = httpx.AsyncClient
        with patch("scripts.check_inference.httpx.AsyncClient", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)):
            with self.assertRaisesRegex(ValueError, "model alias missing"):
                await check("http://inference.invalid/v1", "mera-model", "dummy", 4)

    async def test_missing_pi_runtime_fails_before_any_completion(self):
        async def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "mera-model"}]})
            self.assertTrue(request.url.path.endswith("/mera/pi/health"))
            return httpx.Response(200, json={"service": "native-pi-harbor-bridge", "model": "mera-model",
                                            "worker_present": True, "node_present": False})

        factory = httpx.AsyncClient
        with patch("scripts.check_inference.httpx.AsyncClient", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)):
            with self.assertRaisesRegex(ValueError, "Native Pi service"):
                await check("http://inference.invalid/v1", "mera-model", "dummy", 4)

    async def test_invalid_concurrency_fails_without_network(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "concurrency"):
                await check("http://inference.invalid/v1", "mera-model", "dummy", value)
