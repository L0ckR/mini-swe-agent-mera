from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_stage2"))
from custom_agent.model_client import ForkModel
from custom_agent.policy import RunBudget
from custom_agent.mini_agent import HarborTools


class PatchReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_review_uses_actual_issue_and_diff_without_tools_and_counts_usage(self):
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content":
                "Potential defect in merge: (1, 10), (2, 3) must preserve end=10."}}],
                "usage": {"prompt_tokens": 90, "completion_tokens": 20}})

        with tempfile.TemporaryDirectory() as temporary:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                model = ForkModel("http://local.invalid/v1", "dummy", "model", Path(temporary),
                                  RunBudget(), client=client)
                advice = await model.review_patch("Preserve nested intervals.", "ACTUAL diff: end = right")
            self.assertIn("advice", advice)
            self.assertIn("end=10", advice)
            self.assertEqual(len(requests), 1)
            payload = requests[0]
            self.assertNotIn("tools", payload)
            data = json.loads(payload["messages"][1]["content"])
            self.assertEqual(data["original_issue"], "Preserve nested intervals.")
            self.assertEqual(data["observed_patch_and_checks"], "ACTUAL diff: end = right")
            self.assertEqual(model.usage["completion_tokens"], 20)
            self.assertEqual(model._audit["requests"][0]["kind"], "patch_review")

    async def test_review_respects_remaining_time_and_short_shared_deadline(self):
        requests = []

        async def handler(request):
            requests.append(request)
            await asyncio.sleep(1)
            return httpx.Response(503)

        with tempfile.TemporaryDirectory() as temporary:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                budget = RunBudget()
                model = ForkModel("http://local.invalid/v1", "dummy", "model", Path(temporary),
                                  budget, client=client, request_timeout_sec=0.02)
                started = time.monotonic()
                advice = await model.review_patch("issue", "diff")
                self.assertIn("unavailable", advice)
                self.assertLess(time.monotonic() - started, 0.3)
                self.assertEqual(len(requests), 1)
                budget.deadline = time.monotonic() + 30
                self.assertIn("skipped", await model.review_patch("issue", "diff"))
                self.assertEqual(len(requests), 1)

    async def test_review_cancellation_is_not_swallowed(self):
        started = asyncio.Event()

        async def handler(request):
            started.set()
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as temporary:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                model = ForkModel("http://local.invalid/v1", "dummy", "model", Path(temporary),
                                  RunBudget(), client=client)
                task = asyncio.create_task(model.review_patch("issue", "diff"))
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

    async def test_first_finish_returns_independent_advice_for_agent_to_verify(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = HarborTools(None, RunBudget(), Path(temporary))

            async def snapshot():
                env.snapshots.append({"repository": {"fingerprint": "same", "has_changes": True}})
                return True, "Actual diff: end = right\nObserved command returned 0."

            seen = []

            async def reviewer(issue, evidence):
                seen.append((issue, evidence))
                return "Independent model review: verify nested intervals."

            env.final_snapshot = snapshot
            env.patch_reviewer = reviewer
            first = await env.execute({"name": "finish", "arguments": {"summary": "Everything is correct."}})
            self.assertFalse(first.get("submitted"))
            self.assertIn("verify nested intervals", first["output"])
            self.assertEqual(len(seen), 1)
            self.assertIn("Actual diff: end = right", seen[0][1])
            second = await env.execute({"name": "finish", "arguments": {"summary": "Reviewed."}})
            self.assertTrue(second.get("submitted"))
            self.assertEqual(len(seen), 1)
