"""Render the exact pinned model template offline, without model weights or vLLM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from jinja2 import Environment, nodes
from jinja2.ext import Extension


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name("fixtures")


class GenerationExtension(Extension):
    """HF's generation tag marks assistant token spans; rendering is unchanged."""

    tags = {"generation"}

    def parse(self, parser):
        token = next(parser.stream)
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(token.lineno)

    def _render(self, caller):
        return caller()


def reject_template(message):
    raise ValueError(message)


class PinnedTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "submission_stage2/inference/model.json").read_text())
        cls.metadata = json.loads((FIXTURES / "qwen38_chat_template.source.json").read_text())
        cls.source = (FIXTURES / "qwen38_chat_template.jinja").read_text()
        env = Environment(extensions=[GenerationExtension])
        env.globals["raise_exception"] = reject_template
        cls.template = env.from_string(cls.source)
        cls.tools = json.loads((ROOT / "submission_stage2/inference/pi/tool-schema.json").read_text())

    def render(self, messages, **kwargs):
        defaults = self.manifest["default_chat_template_kwargs"] | kwargs
        return self.template.render(
            messages=messages, tools=self.tools, add_generation_prompt=True, **defaults,
        )

    def test_fixture_is_the_manifest_checkpoint_and_has_not_changed(self):
        self.assertEqual(self.metadata["revision"], self.manifest["revision"])
        self.assertEqual(self.metadata["model_id"], self.manifest["model_id"])
        self.assertEqual(hashlib.sha256(self.source.encode()).hexdigest(), self.metadata["sha256"])
        self.assertEqual(self.metadata["adapter"]["model_id"], self.manifest["adapter"]["model_id"])
        self.assertEqual(self.metadata["adapter"]["revision"], self.manifest["adapter"]["revision"])
        self.assertEqual(self.metadata["adapter"]["sha256"], self.metadata["sha256"])

    def test_server_default_leaves_smoke_check_output_budget_for_tool_call(self):
        prompt = self.render([{"role": "user", "content": "Use bash to run pwd."}])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertIn('"name": "bash"', prompt)

    def test_pi_replays_native_multiline_command_and_normal_text_completion_without_thinking(self):
        command = "python3 - <<'PY'\nif True:\n    print('indented')\nPY\n"
        messages = [
            {"role": "system", "content": "Solve the repository issue."},
            {"role": "user", "content": "Fix the defect and test the change."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": {"command": command, "timeout": 120}}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "exit_code: 0\nindented\n"},
            {"role": "assistant", "content": "Fixed the defect; focused check passed."},
        ]
        prompt = self.render(messages)
        self.assertIn(command, prompt)
        self.assertIn("<function=bash>\n<parameter=command>", prompt)
        self.assertIn("<parameter=timeout>\n120", prompt)
        self.assertIn("Fixed the defect; focused check passed.", prompt)
        self.assertNotIn("<function=finish>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_supported_efforts_and_invalid_high_effort(self):
        messages = [{"role": "user", "content": "Inspect the repository."}]
        for effort in ("low", "medium", "xhigh"):
            with self.subTest(effort=effort):
                self.render(messages, enable_thinking=True, reasoning_effort=effort)
        with self.assertRaisesRegex(ValueError, "Unexpected reasoning effort high"):
            self.render(messages, enable_thinking=True, reasoning_effort="high")


if __name__ == "__main__":
    unittest.main()
