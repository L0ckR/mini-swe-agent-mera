from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_stage2"))
from custom_agent.policy import BudgetExpired, RunBudget
from custom_agent.mini_agent import ReactBashAgent, SNAPSHOT_COMMAND
from harbor.models.agent.context import AgentContext


def response(name, arguments):
    return {
        "object": "chat.completion", "model": "test-model",
        "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_content": "Use the observed repository state.",
            "tool_calls": [{"id": "call_probe", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}],
        }}], "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "prompt_tokens_details": {"cached_tokens": 3}},
    }


class LocalEnvironment:
    def __init__(self):
        self.commands = []
        self.results = []

    async def exec(self, *, command, cwd, timeout_sec):
        self.commands.append(command)
        process = await asyncio.create_subprocess_shell(
            command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        result = SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())
        self.results.append(result)
        return result


class BudgetTests(unittest.TestCase):
    def test_solve_timeouts_cannot_consume_review_reserve(self):
        budget = RunBudget(1680, 90)
        budget.deadline = time.monotonic() + 110
        self.assertEqual(budget.phase, "solve")
        self.assertLessEqual(budget.timeout(300), 20)

    def test_review_caps_requests_and_keeps_final_snapshot_margin(self):
        budget = RunBudget(1680, 90)
        budget.deadline = time.monotonic() + 80
        self.assertEqual(budget.phase, "review")
        self.assertLessEqual(budget.timeout(300), 45)
        budget.deadline = time.monotonic() + 5
        with self.assertRaises(BudgetExpired):
            budget.timeout(300)
        self.assertGreater(budget.remaining(), 0)


class ForkHarborIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.logs = self.root / "logs"
        self.context = AgentContext()

    def tearDown(self):
        self.temporary.cleanup()

    def agent(self, **options):
        options.setdefault("semantic_review", False)
        return ReactBashAgent(
            logs_dir=self.logs, model_name="openai/test-model", workdir=str(self.repo),
            extra_env={"REACT_AGENT_BASE_URL": "http://inference.invalid/v1", "REACT_AGENT_API_KEY": "dummy"},
            **options,
        )

    async def run_responses(self, agent, env, replies):
        replies = list(replies)
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            self.assertTrue(replies, "Unexpected extra LLM call")
            return httpx.Response(200, json=replies.pop(0))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("custom_agent.model_client.httpx.AsyncClient", return_value=client):
            await agent.run("Fix add(a, b) so it sums both arguments.", env, self.context)
        self.assertFalse(replies)
        return requests

    async def test_actual_fork_loop_repairs_real_repository_and_saves_native_trajectory(self):
        (self.repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (self.repo / "test_calc.py").write_text("import unittest\nfrom calc import add\nclass TestAdd(unittest.TestCase):\n    def test_sum(self):\n        self.assertEqual(add(2, 3), 5)\n")
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "calc.py", "test_calc.py"], cwd=self.repo, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"], cwd=self.repo, check=True)
        env = LocalEnvironment()
        requests = await self.run_responses(self.agent(), env, [
            response("read_file", {"path": "calc.py"}),
            response("bash", {"command": "python3 -B -m unittest -q test_calc"}),
            response("edit_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}),
            response("bash", {"command": "python3 -B -m unittest -q test_calc"}),
            response("finish", {"summary": "Fixed add; test failed before fix and passed afterward."}),
            response("finish", {"summary": "Reviewed the actual patch; focused tests passed."}),
        ])
        self.assertIn("return a + b", (self.repo / "calc.py").read_text())
        test_results = [item for item in env.results if "Ran 1 test" in item.stderr]
        self.assertEqual([item.return_code for item in test_results], [1, 0])
        self.assertEqual(self.context.metadata["stop_reason"], "Submitted")
        self.assertEqual(self.context.n_input_tokens, 60)
        self.assertEqual(self.context.n_output_tokens, 30)
        native = json.loads((self.logs / "mini-swe-agent.json").read_text())
        self.assertEqual(native["trajectory_format"], "mini-swe-agent-1.1")
        self.assertEqual(native["info"]["mini_version"], "2.4.6")
        self.assertEqual(native["info"]["model_stats"]["api_calls"], 6)
        self.assertFalse(any("extra" in msg for payload in requests for msg in payload["messages"]))
        self.assertIn(SNAPSHOT_COMMAND, env.commands)

    async def test_cancellation_during_tool_keeps_usage_and_both_logs(self):
        started = asyncio.Event()

        class BlockingEnvironment:
            async def exec(self, **kwargs):
                if kwargs["command"].startswith("# MERA_REPOSITORY_BOOTSTRAP"):
                    return SimpleNamespace(return_code=0, stdout='{"existing_entries":["calc.py"]}', stderr="")
                started.set()
                await asyncio.Event().wait()

        async def handler(request):
            return httpx.Response(200, json=response("bash", {"command": "sleep 300"}))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("custom_agent.model_client.httpx.AsyncClient", return_value=client):
            task = asyncio.create_task(self.agent().run("Fix issue", BlockingEnvironment(), self.context))
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.context.metadata["stop_reason"], "cancelled")
        self.assertEqual(self.context.n_input_tokens, 10)
        self.assertTrue((self.logs / "mini-swe-agent.json").is_file())
        audit = json.loads((self.logs / "tool-audit.json").read_text())
        self.assertTrue(audit["actions"][-1]["cancelled"])

    async def test_cancellation_during_finally_snapshot_still_persists_usage(self):
        snapshot_started = asyncio.Event()

        class SnapshotBlockingEnvironment:
            async def exec(self, *, command, **kwargs):
                if command == SNAPSHOT_COMMAND:
                    snapshot_started.set()
                    await asyncio.Event().wait()
                return SimpleNamespace(return_code=0, stdout="ok", stderr="")

        async def handler(request):
            return httpx.Response(200, json=response("bash", {"command": "pwd"}))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("custom_agent.model_client.httpx.AsyncClient", return_value=client):
            task = asyncio.create_task(self.agent(max_steps=1).run("Fix issue", SnapshotBlockingEnvironment(), self.context))
            await asyncio.wait_for(snapshot_started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.context.metadata["stop_reason"], "cancelled")
        self.assertEqual(self.context.n_input_tokens, 10)

    def test_bad_integer_config_does_not_disable_upstream_limits(self):
        for value in (True, 0.5, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.agent(max_steps=value)
