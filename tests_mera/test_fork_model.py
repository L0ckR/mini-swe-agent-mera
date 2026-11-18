from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_stage2"))

from custom_agent._vendor.mini_swe_agent.exceptions import FormatError
from custom_agent._vendor.mini_swe_agent.agents.default import DefaultAgent
from custom_agent.model_client import ForkModel, MEMORY_PREFIX
from custom_agent.policy import RunBudget


def completion(*calls, content=None, reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    if calls:
        message["tool_calls"] = [{
            "id": f"call_{index}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        } for index, (name, arguments) in enumerate(calls)]
    return {"choices": [{"finish_reason": "tool_calls" if calls else "stop", "message": message}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 10,
                      "prompt_tokens_details": {"cached_tokens": 7}}}


def history(turns=0, output_size=30):
    messages = [{"role": "system", "content": "Work on the repository"},
                {"role": "user", "content": "Fix encoding handling"}]
    for index in range(turns):
        identifier = f"old_{index}"
        messages.extend([{
            "role": "assistant", "content": None,
            "tool_calls": [{"id": identifier, "type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"command": f"cat file_{index}.py"}),
            }}], "extra": {"private_adapter_state": True},
        }, {"role": "tool", "tool_call_id": identifier, "name": "bash", "content": "x" * output_size}])
    return messages


def assert_protocol(test, messages):
    pending = set()
    for message in messages:
        if message["role"] == "assistant":
            test.assertFalse(pending, "assistant followed incomplete tool results")
            pending.update(call["id"] for call in message.get("tool_calls", []))
        elif message["role"] == "tool":
            test.assertIn(message["tool_call_id"], pending)
            pending.remove(message["tool_call_id"])
        else:
            test.assertFalse(pending, "user/system message orphaned tool results")
    test.assertFalse(pending)


class ForkModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.logs = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_model(self, handler, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        return ForkModel("http://inference.invalid/v1", "dummy", "openai/mera-model", self.logs,
                         kwargs.pop("budget", RunBudget()), client=client, **kwargs)

    def audit(self):
        return json.loads((self.logs / "model-audit.json").read_text())

    async def test_query_normalizes_actions_reasoning_and_usage_without_internal_fields(self):
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=completion(("bash", {"command": "pytest -q"}), reasoning="Inspect the error."))

        model = self.make_model(handler)
        messages = history(1)
        async with model:
            result = await model.query(messages)
        self.assertEqual(result["extra"]["actions"][0]["name"], "bash")
        self.assertTrue(result["extra"]["actions"][0]["only_call"])
        self.assertEqual(result["reasoning_content"], result["reasoning"])
        self.assertEqual(model.usage, {"prompt_tokens": 30, "completion_tokens": 10, "cached_tokens": 7})
        self.assertEqual(requests[0]["model"], "mera-model")
        self.assertEqual(requests[0]["thinking_token_budget"], 3072)
        self.assertTrue(all("extra" not in message for message in requests[0]["messages"]))
        self.assertEqual(self.audit()["requests"][0]["status"], "success")

    async def test_multiple_calls_normalize_duplicate_ids_and_finish_needs_sole_call(self):
        payload = completion(("finish", {"summary": "done"}), ("bash", {"command": "pwd"}))
        payload["choices"][0]["message"]["tool_calls"][1]["id"] = "call_0"

        async def handler(request):
            return httpx.Response(200, json=payload)

        model = self.make_model(handler)
        async with model:
            message = await model.query(history())
        actions = message["extra"]["actions"]
        self.assertEqual(len({action["tool_call_id"] for action in actions}), 2)
        self.assertTrue(all(not action["only_call"] for action in actions))
        observations = model.format_observation_messages(message, [
            {"output": "mixed finish rejected", "submitted": True, "summary": "done"}, {"output": "/testbed"},
        ])
        self.assertEqual([observation["role"] for observation in observations], ["tool", "tool"])

    async def test_finish_observation_closes_protocol_before_native_exit(self):
        async def handler(request):
            return httpx.Response(200, json=completion(("finish", {"summary": "codec tests pass"})))

        model = self.make_model(handler)
        async with model:
            message = await model.query(history())
        observations = model.format_observation_messages(message, [{
            "output": "bounded review", "raw_output": "large raw log", "submitted": True, "summary": "codec tests pass",
        }])
        self.assertEqual([observation["role"] for observation in observations], ["tool", "exit"])
        self.assertEqual(observations[-1]["extra"]["exit_status"], "Submitted")
        self.assertEqual(observations[0]["content"], "bounded review")
        self.assertNotIn("large raw log", json.dumps(observations))

    async def test_missing_tool_call_becomes_recoverable_upstream_format_error(self):
        async def handler(request):
            return httpx.Response(200, json=completion(content="I will inspect the file."))

        model = self.make_model(handler)
        async with model:
            with self.assertRaises(FormatError) as error:
                await model.query(history())
        self.assertEqual([message["role"] for message in error.exception.messages], ["assistant", "user"])
        self.assertIn("tool", error.exception.messages[-1]["content"].lower())
        self.assertEqual(model.usage["completion_tokens"], 10)

    async def test_401_fails_once_and_credentials_are_not_logged(self):
        count = 0

        async def handler(request):
            nonlocal count
            count += 1
            return httpx.Response(401, json={"error": "unauthorized"})

        model = self.make_model(handler)
        model.api_key = "private-key-not-for-logs"
        async with model:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                await model.query(history())
        self.assertEqual(count, 1)
        self.assertNotIn("private-key-not-for-logs", (self.logs / "model-audit.json").read_text())
        self.assertNotIn("api_key", json.dumps(model.serialize()))

    async def test_transient_failure_retries_the_request_and_records_raw_response(self):
        count = 0

        async def handler(request):
            nonlocal count
            count += 1
            if count == 1:
                return httpx.Response(503, json={"error": "busy"})
            return httpx.Response(200, json=completion(("bash", {"command": "pwd"})))

        model = self.make_model(handler)
        with patch("custom_agent.model_client.asyncio.sleep", return_value=None):
            async with model:
                result = await model.query(history())
        self.assertEqual(count, 2)
        self.assertEqual(len(result["extra"]["actions"]), 1)
        self.assertEqual(self.audit()["requests"][0]["response"], {"error": "busy"})

    async def test_retry_budget_is_shared_instead_of_four_full_timeouts(self):
        count = 0

        async def handler(request):
            nonlocal count
            count += 1
            await asyncio.sleep(1)
            return httpx.Response(503)

        model = self.make_model(handler, request_timeout_sec=0.03)
        start = time.monotonic()
        async with model:
            with self.assertRaises(FormatError) as error:
                await model.query(history())
            self.assertIn("shared", error.exception.messages[0]["content"])
        self.assertEqual(count, 1)
        self.assertLess(time.monotonic() - start, 0.25)

    async def test_native_core_recovers_timeout_with_cheaper_request_and_no_duplicate_tools(self):
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                await asyncio.sleep(1)
            return httpx.Response(200, json=completion(
                ("bash", {"command": "pwd"}) if len(requests) == 2 else ("finish", {"summary": "checked"})
            ))

        class Environment:
            def __init__(self):
                self.actions = []
            async def execute(self, action):
                self.actions.append(action)
                return {"output": "/testbed"} if action["name"] == "bash" else {"output": "checked", "submitted": True, "summary": "checked"}
            def get_template_vars(self):
                return {}
            def serialize(self):
                return {}

        model = self.make_model(handler, request_timeout_sec=0.03)
        environment = Environment()
        core = DefaultAgent(model, environment, system_template="Work", instance_template="{{task}}", cost_limit=0)
        async with model:
            result = await core.run("Fix the issue")
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual([action["name"] for action in environment.actions], ["bash", "finish"])
        self.assertEqual(requests[1]["thinking_token_budget"], 1024)
        self.assertEqual(requests[1]["max_tokens"], 4096)
        self.assertEqual(requests[1]["reasoning_effort"], "low")
        self.assertIn("No tool actions", requests[1]["messages"][-1]["content"])
        self.assertEqual(len(self.audit()["recoveries"]), 1)

    async def test_solver_cutoff_becomes_format_feedback_and_review_can_continue(self):
        class BoundaryBudget:
            def __init__(self):
                self.solve_deadline = time.monotonic() + 0.025
            def remaining(self):
                return 100.0
            @property
            def phase(self):
                return "solve" if time.monotonic() < self.solve_deadline else "review"
            def timeout(self, cap):
                return min(cap, self.solve_deadline - time.monotonic()) if self.phase == "solve" else cap

        count = 0

        async def handler(request):
            nonlocal count
            count += 1
            if count == 1:
                await asyncio.sleep(1)
            return httpx.Response(200, json=completion(("finish", {"summary": "review complete"})))

        model = self.make_model(handler, budget=BoundaryBudget())
        async with model:
            with self.assertRaises(FormatError) as error:
                await model.query(history())
            self.assertIn("final-review", error.exception.messages[0]["content"].lower())
            result = await model.query(history())
        self.assertEqual(result["extra"]["actions"][0]["name"], "finish")
        self.assertEqual(count, 2)

    async def test_long_history_checkpoint_preserves_full_native_history_and_protocol(self):
        requests = []

        async def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if "tools" not in payload:
                return httpx.Response(200, json=completion(content="Cause: codec ignores encoding; inspect codec.py next. No edits yet."))
            assert_protocol(self, payload["messages"])
            return httpx.Response(200, json=completion(("bash", {"command": "cat codec.py"})))

        model = self.make_model(handler, context_budget_tokens=2000)
        messages = history(15, output_size=1000)
        original = deepcopy(messages)
        async with model:
            await model.query(messages)
        self.assertEqual(messages, original)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]["messages"][:2], original[:2])
        self.assertTrue(requests[1]["messages"][2]["content"].startswith(MEMORY_PREFIX))
        self.assertEqual(len(self.audit()["checkpoints"]), 1)

    async def test_context_400_emergency_compaction_keeps_complete_tool_turns(self):
        count = 0

        async def handler(request):
            nonlocal count
            count += 1
            payload = json.loads(request.content)
            if count == 1:
                return httpx.Response(400, json={"error": "maximum context length exceeded"})
            if "tools" not in payload:
                return httpx.Response(200, json=completion(content="Earlier observations: inspect file_0.py and file_1.py; no edits confirmed."))
            assert_protocol(self, payload["messages"])
            return httpx.Response(200, json=completion(("bash", {"command": "pwd"})))

        model = self.make_model(handler)
        async with model:
            result = await model.query(history(4))
        self.assertEqual(count, 3)
        self.assertEqual(result["extra"]["actions"][0]["name"], "bash")

    async def test_large_first_output_recovers_without_changing_command_arguments(self):
        count = 0
        command = "printf '" + "code" * 3000 + "'"
        messages = history(1, output_size=50000)
        messages[2]["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": command})

        async def handler(request):
            nonlocal count
            count += 1
            payload = json.loads(request.content)
            if count == 1:
                return httpx.Response(400, json={"error": "maximum context length"})
            assert_protocol(self, payload["messages"])
            self.assertLess(len(payload["messages"][-1]["content"]), 2000)
            arguments = json.loads(payload["messages"][2]["tool_calls"][0]["function"]["arguments"])
            self.assertEqual(arguments["command"], command)
            return httpx.Response(200, json=completion(("bash", {"command": "pwd"})))

        model = self.make_model(handler)
        async with model:
            await model.query(messages)
        self.assertEqual(count, 2)
        self.assertEqual(len(messages[-1]["content"]), 50000)

    async def test_cancellation_preserves_successful_usage_and_pending_request(self):
        count = 0

        async def handler(request):
            nonlocal count
            count += 1
            if count == 1:
                return httpx.Response(200, json=completion(("bash", {"command": "pwd"})))
            raise asyncio.CancelledError()

        model = self.make_model(handler)
        async with model:
            await model.query(history())
            with self.assertRaises(asyncio.CancelledError):
                await model.query(history())
        self.assertEqual(self.audit()["usage"]["prompt_tokens"], 30)
        self.assertEqual(self.audit()["requests"][-1]["status"], "cancelled")

