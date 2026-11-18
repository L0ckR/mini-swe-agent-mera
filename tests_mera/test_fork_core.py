from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_stage2"))

from custom_agent._vendor.mini_swe_agent.agents.default import DefaultAgent
from custom_agent._vendor.mini_swe_agent.exceptions import FormatError, Submitted


class ModelPort:
    config = {}

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    async def query(self, messages, **kwargs):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [{"role": "tool", "content": output["output"]} for output in outputs]

    def get_template_vars(self, **kwargs):
        return {"model_name": "local-test-model"}

    def serialize(self):
        return {"info": {"model_adapter": "test"}}


class EnvironmentPort:
    config = {}

    def __init__(self):
        self.actions = []

    async def execute(self, action, cwd=""):
        self.actions.append(action)
        if action.get("command") == "submit":
            raise Submitted({"role": "exit", "content": "done", "extra": {"exit_status": "Submitted", "submission": "done"}})
        return {"output": "observed", "returncode": 0}

    def get_template_vars(self, **kwargs):
        return {"cwd": "/testbed"}

    def serialize(self):
        return {"info": {"environment_adapter": "test"}}


def action(command):
    return {"role": "assistant", "content": None, "extra": {"cost": 0.0, "actions": [{"command": command}]}}


class ForkCoreTests(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, replies, **kwargs):
        self.model = ModelPort(replies)
        self.env = EnvironmentPort()
        return DefaultAgent(self.model, self.env, system_template="Work in {{cwd}}", instance_template="Fix {{task}}", cost_limit=0, **kwargs)

    async def test_native_async_query_execution_and_submission(self):
        agent = self.make_agent([action("inspect"), action("submit")])
        result = await agent.run("a defect")
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(self.env.actions, [{"command": "inspect"}, {"command": "submit"}])
        self.assertEqual(agent.n_calls, 2)
        self.assertEqual(self.model.seen[0][0]["content"], "Work in /testbed")
        self.assertEqual(self.model.seen[0][1]["content"], "Fix a defect")
        self.assertEqual(self.model.seen[1][-1]["role"], "tool")

    async def test_step_limit_stops_before_another_model_call(self):
        agent = self.make_agent([action("inspect")], step_limit=1)
        result = await agent.run("a defect")
        self.assertEqual(result["exit_status"], "LimitsExceeded")
        self.assertEqual(len(self.model.seen), 1)

    async def test_wall_time_uses_monotonic_and_stops_before_request(self):
        agent = self.make_agent([], wall_time_limit_seconds=10)
        with patch("custom_agent._vendor.mini_swe_agent.agents.default.time.monotonic", return_value=agent._start_time + 11):
            result = await agent.run("a defect")
        self.assertEqual(result["exit_status"], "TimeExceeded")
        self.assertEqual(self.model.seen, [])

    async def test_format_error_recovery_and_cost_are_preserved(self):
        correction = {"role": "user", "content": "Use a tool", "extra": {"cost": 0.25}}
        agent = self.make_agent([FormatError(correction), action("inspect"), action("submit")])
        result = await agent.run("a defect")
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(agent.cost, 0.25)
        self.assertEqual(agent.n_consecutive_format_errors, 0)
        self.assertIn(correction, self.model.seen[1])

    async def test_repeated_format_errors_stop_as_upstream(self):
        error = lambda: FormatError({"role": "user", "content": "Use a tool", "extra": {"cost": 0.1}})
        agent = self.make_agent([error(), error()], max_consecutive_format_errors=2)
        result = await agent.run("a defect")
        self.assertEqual(result["exit_status"], "RepeatedFormatError")
        self.assertEqual(len(self.model.seen), 2)

    async def test_unexpected_failure_saves_audit_then_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "native.json"
            agent = self.make_agent([RuntimeError("transport failed")], output_path=log)
            with self.assertRaisesRegex(RuntimeError, "transport failed"):
                await agent.run("a defect")
            data = json.loads(log.read_text())
            self.assertEqual(data["info"]["exit_status"], "RuntimeError")
            self.assertIn("transport failed", data["messages"][-1]["extra"]["exception_str"])
            self.assertFalse(log.with_suffix(".json.tmp").exists())

    async def test_cancellation_flushes_native_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "native.json"
            agent = self.make_agent([asyncio.CancelledError()], output_path=log)
            with self.assertRaises(asyncio.CancelledError):
                await agent.run("a defect")
            self.assertTrue(log.is_file())
            self.assertEqual(json.loads(log.read_text())["info"]["model_stats"]["api_calls"], 1)

    async def test_serialization_retains_upstream_format_and_adapter_metadata(self):
        agent = self.make_agent([action("submit")])
        await agent.run("a defect")
        result = agent.serialize({"extra_review": "passed"})
        self.assertEqual(result["trajectory_format"], "mini-swe-agent-1.1")
        self.assertEqual(result["info"]["mini_version"], "2.4.6")
        self.assertEqual(result["info"]["model_adapter"], "test")
        self.assertEqual(result["info"]["environment_adapter"], "test")
        self.assertEqual(result["extra_review"], "passed")


class ForkProvenanceTests(unittest.TestCase):
    def test_pristine_sources_match_recorded_hashes_and_license_is_preserved(self):
        pristine = ROOT / "vendor/mini_swe_agent"
        source = json.loads((pristine / "SOURCE.json").read_text())
        package = ROOT / "submission_stage2/custom_agent/_vendor/mini_swe_agent"
        manifest = json.loads((package / "UPSTREAM.json").read_text())
        self.assertEqual(source["commit"], manifest["commit"])
        self.assertEqual(source["commit"], "04d809ceab9df28f9adaed044884180159172930")
        for name, record in source["files"].items():
            self.assertEqual(hashlib.sha256((pristine / name).read_bytes()).hexdigest(), record["sha256"], name)
        self.assertEqual((pristine / "LICENSE.md").read_bytes(), (package / "LICENSE.md").read_bytes())
        self.assertIn("Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez", (package / "LICENSE.md").read_text())

    def test_import_does_not_load_host_global_config(self):
        module = importlib.import_module("custom_agent._vendor.mini_swe_agent")
        self.assertEqual(module.__version__, "2.4.6")
        self.assertFalse(hasattr(module, "global_config_file"))
        self.assertFalse(hasattr(module, "global_config_dir"))
