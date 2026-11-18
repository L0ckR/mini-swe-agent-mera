from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from scripts.check_inference import check


class InferenceAPICheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_contract_replay_custom_tools_and_concurrent_thinking_requests(self):
        requests = []

        async def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "mera-model"}]})
            payload = json.loads(request.content)
            requests.append(payload)
            prompt = payload["messages"][-1]["content"]
            name, arguments = "bash", {"command": "pwd"}
            if "printf hello" in prompt:
                arguments = {"command": "printf hello", "timeout_sec": 30}
            elif "protocol checked" in prompt:
                name, arguments = "finish", {"summary": "protocol checked"}
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
        self.assertEqual(len(requests), 8)
        self.assertEqual(requests[1]["messages"][2]["role"], "tool")
        self.assertEqual({entry["function"]["name"] for entry in requests[2]["tools"]}, {"bash", "read_file", "edit_file", "create_file", "finish"})
        self.assertTrue(all(payload["thinking_token_budget"] == 3072 for payload in requests[-4:]))

    async def test_missing_model_alias_fails_before_completions(self):
        async def handler(request):
            self.assertTrue(request.url.path.endswith("/models"))
            return httpx.Response(200, json={"data": [{"id": "wrong-alias"}]})

        factory = httpx.AsyncClient
        with patch("scripts.check_inference.httpx.AsyncClient", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)):
            with self.assertRaisesRegex(ValueError, "model alias missing"):
                await check("http://inference.invalid/v1", "mera-model", "dummy", 4)
