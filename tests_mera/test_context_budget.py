from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission_stage2"))
from custom_agent.model_client import ForkModel
from custom_agent.policy import RunBudget


class ContextBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_identical_reasoning_aliases_are_counted_once(self):
        reasoning = "Detailed useful reasoning. " * 300
        single = [{"role": "assistant", "content": None, "reasoning": reasoning}]
        duplicate = [{**single[0], "reasoning_content": reasoning}]
        self.assertEqual(ForkModel._estimate(single), ForkModel._estimate(duplicate))

    async def test_long_recent_turns_are_selected_by_budget_instead_of_fixed_twelve(self):
        messages = [{"role": "system", "content": "Implement the issue."}, {"role": "user", "content": "Fix behavior and preserve API."}]
        for number in range(15):
            reasoning = (f"Confirmed fact {number}. " * 700)
            messages.extend([{"role": "assistant", "content": None, "reasoning": reasoning, "reasoning_content": reasoning,
                              "tool_calls": [{"id": str(number), "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"cat source_{number}.py"})}}]},
                             {"role": "tool", "tool_call_id": str(number), "content": "source evidence\n" * 200}])
        original = deepcopy(messages)
        requests = []

        async def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if "tools" not in payload:
                result = {"content": "Confirmed cause and edits retained. Focused check returned 0. Review remaining edge cases."}
            else:
                result = {"content": None, "tool_calls": [{"id": "next", "type": "function", "function": {"name": "bash", "arguments": '{"command":"git diff"}'}}]}
            return httpx.Response(200, json={"choices": [{"message": result, "finish_reason": "tool_calls" if "tools" in payload else "stop"}]})

        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                model = ForkModel("http://local.invalid/v1", "dummy", "model", Path(directory), RunBudget(), client=client, context_budget_tokens=42000)
                await model.query(messages)
                self.assertLessEqual(model._estimate(requests[-1]["messages"]), 42000)
                self.assertLess(sum(message["role"] == "assistant" for message in requests[-1]["messages"]), 12)
                self.assertEqual(messages, original)
                self.assertEqual(len(requests), 2)
