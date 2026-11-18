from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_stage2"))
from custom_agent.pi_agent import PiAgent
from harbor.models.agent.context import AgentContext


class LocalEnvironment:
    def __init__(self):
        self.commands = []

    async def exec(self, *, command, cwd, timeout_sec):
        self.commands.append(command)
        process = await asyncio.create_subprocess_shell(command, cwd=cwd,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())


def operation(identifier, name, args):
    return {"type": "operation", "id": identifier, "operation": name, "args": args}


def complete(**usage):
    return {"type": "complete", "stop_reason": "completed", "usage": usage}


class PiAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.logs = self.root / "logs"
        self.original_test = "import unittest\nfrom calc import add\nclass AddTest(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2,3),5)\n"
        (self.repo / "calc.py").write_text("def add(a,b):\n    return a-b\n")
        (self.repo / "test_calc.py").write_text(self.original_test)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial")
        self.context = AgentContext()
        self.env = LocalEnvironment()
        self.batches = []
        self.requests = []
        self.lose_first_result = False
        self.result_lost = False
        self.delete_hook = None

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, capture_output=True, check=True).stdout

    def instruction(self):
        return ('<issue_description>Fix add(a,b) to sum its arguments.</issue_description>\n'
                'Make minimal changes to non-test files.\n## Submission\n'
                'Write your complete change as a unified diff to `change.diff`.')

    def agent(self):
        return PiAgent(logs_dir=self.logs, model_name="openai/adapted", workdir=str(self.repo),
            extra_env={"REACT_AGENT_BASE_URL": "http://service.invalid/v1", "REACT_AGENT_API_KEY": "dummy"},
            time_budget_sec=300)

    def write(self, identifier, filename, content):
        return operation(identifier, "write_file", {"path": str(self.repo / filename),
                                "data_base64": base64.b64encode(content.encode()).decode()})

    async def handler(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path.endswith("/sessions"):
            self.assertEqual(body["workdir"], "/testbed")
            self.assertEqual(body["model"], "adapted")
            self.assertIn("Fix add(a,b)", body["instruction"])
            self.assertIn("do not modify or add tests", body["append_system_prompt"])
            return httpx.Response(200, json={"session_id": "session-1", "token": "server-session-private", "model": "adapted"})
        self.assertEqual(request.headers["X-Mera-Session-Token"], "server-session-private")
        if request.method == "DELETE":
            if self.delete_hook:
                await self.delete_hook()
            return httpx.Response(204)
        if request.url.path.endswith("/events"):
            self.assertTrue(self.batches, "Unexpected extra event poll")
            return httpx.Response(200, json={"events": self.batches.pop(0)})
        if request.url.path.endswith("/result"):
            if self.lose_first_result and not self.result_lost:
                self.result_lost = True
                raise httpx.ReadError("Simulated lost acknowledgement", request=request)
            return httpx.Response(200, json={"accepted": True})
        if request.url.path.endswith("/continue"):
            return httpx.Response(200, json={"accepted": True})
        raise AssertionError("Unexpected request: " + str(request.url))

    async def run_agent(self, env=None):
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        with patch("custom_agent.pi_agent.httpx.AsyncClient", return_value=client):
            await self.agent().run(self.instruction(), env or self.env, self.context)

    async def test_real_edit_test_export_and_duplicate_delivery_execute_once(self):
        edit = self.write("edit-1", "calc.py", "def add(a,b):\n    return a+b\n")
        self.lose_first_result = True
        self.batches = [[edit, edit], [operation("test-1", "exec", {"command": "python3 -B -m unittest -q test_calc"})],
                        [complete(input=90, output=25, cacheRead=30, cacheWrite=10)]]
        await self.run_agent()
        self.assertFalse(self.batches)
        self.assertEqual((self.repo / "test_calc.py").read_text(), self.original_test)
        audit = json.loads((self.logs / "tool-audit.json").read_text())
        self.assertEqual(len(audit["operations"]), 2)
        self.assertIn("Ran 1 test", audit["operations"][1]["result"]["stderr"])
        self.assertEqual(audit["operations"][1]["result"]["returncode"], 0)
        edit_results = [body for method, path, body in self.requests if path.endswith("/result") and body["id"] == "edit-1"]
        self.assertEqual(len(edit_results), 3)
        self.assertTrue(all(body == edit_results[0] for body in edit_results))
        self.assertTrue(audit["patch_export"]["apply_check_passed"])
        clean = self.root / "clean"
        subprocess.run(["git", "clone", "-q", str(self.repo), str(clean)], check=True)
        subprocess.run(["git", "apply", "--check", str(self.repo / "change.diff")], cwd=clean, check=True)
        self.assertEqual(self.context.metadata["stop_reason"], "completed")
        self.assertTrue(self.context.metadata["file_constraints_valid"])
        self.assertEqual(self.context.n_input_tokens, 130)
        self.assertEqual(self.context.n_output_tokens, 25)
        self.assertEqual(self.context.n_cache_tokens, 30)
        logtext = (self.logs / "pi-trajectory.json").read_text() + (self.logs / "tool-audit.json").read_text()
        self.assertNotIn("server-session-private", logtext)

    async def test_forbidden_test_edit_is_reported_then_repaired_before_patch_export(self):
        self.batches = [[self.write("source", "calc.py", "def add(a,b):\n    return a+b\n")],
                        [self.write("bad-test", "test_calc.py", "assert True\n")], [complete(input=40, output=10)],
                        [self.write("repair-test", "test_calc.py", self.original_test)], [complete(input=70, output=20)]]
        await self.run_agent()
        continuations = [body for method, path, body in self.requests if path.endswith("/continue")]
        self.assertEqual(len(continuations), 1)
        self.assertIn("test_calc.py", continuations[0]["message"])
        self.assertEqual(self.context.metadata["contract_repairs"], 1)
        self.assertEqual(self.context.n_input_tokens, 70)
        self.assertEqual(self.context.n_output_tokens, 20)
        self.assertEqual((self.repo / "test_calc.py").read_text(), self.original_test)
        self.assertNotIn("test_calc.py", (self.repo / "change.diff").read_text())

    async def test_cancelled_environment_operation_saves_logs_and_stops_remote_session(self):
        entered = asyncio.Event()

        class BlockingEnvironment:
            async def exec(self, **kwargs):
                entered.set()
                await asyncio.Event().wait()

        self.batches = [[
            {"type": "event", "event": {"type": "message_end", "message": {"role": "assistant",
                "usage": {"input": 12, "output": 4, "cacheRead": 8, "cacheWrite": 2}}}},
            {"type": "event", "event": {"type": "usage_totals", "usage":
                {"input": 32, "output": 10, "cacheRead": 20, "cacheWrite": 4}, "turns": 1}},
            operation("blocking", "exec", {"command": "sleep 300"})]]
        task = asyncio.create_task(self.run_agent(BlockingEnvironment()))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.context.metadata["stop_reason"], "cancelled")
        self.assertEqual(self.context.n_input_tokens, 56)
        self.assertEqual(self.context.n_output_tokens, 10)
        self.assertEqual(self.context.n_cache_tokens, 20)
        self.assertTrue((self.logs / "pi-trajectory.json").is_file())
        self.assertTrue(any(method == "DELETE" for method, _, _ in self.requests))
        audit = json.loads((self.logs / "tool-audit.json").read_text())
        self.assertTrue(audit["operations"][0]["cancelled"])

    async def test_cancellation_during_remote_session_cleanup_still_saves_usage_and_logs(self):
        entered = asyncio.Event()

        async def wait_on_delete():
            entered.set()
            await asyncio.Event().wait()

        self.delete_hook = wait_on_delete
        self.batches = [[complete(input=90, output=25, cacheRead=30, cacheWrite=10)]]
        task = asyncio.create_task(self.run_agent())
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue((self.logs / "pi-trajectory.json").is_file())
        self.assertTrue((self.logs / "tool-audit.json").is_file())
        self.assertEqual(self.context.n_output_tokens, 25)
        self.assertEqual(self.context.metadata["stop_reason"], "cancelled")

    async def test_failed_test_observation_keeps_actual_error_text(self):
        self.batches = [[operation("test-fails", "exec", {"command": "python3 -B -m unittest -q test_calc"})],
                        [complete(input=20, output=10)]]
        await self.run_agent()
        evidence = json.loads((self.logs / "evidence.json").read_text())
        self.assertEqual(evidence["checks"][0]["returncode"], 1)
        self.assertIn("AssertionError", evidence["checks"][0]["output"])


if __name__ == "__main__":
    unittest.main()
