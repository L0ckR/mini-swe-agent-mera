from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_stage2"))
from custom_agent.evidence import EvidenceTracker


class LocalEnvironment:
    async def exec(self, *, command, cwd, timeout_sec):
        process = await asyncio.create_subprocess_shell(command, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.source = self.repo / "source.py"
        self.source.write_text("value = 1\n")
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "source.py"], cwd=self.repo, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], cwd=self.repo, check=True)
        self.env = LocalEnvironment()
        self.tracker = EvidenceTracker(self.env, str(self.repo), self.root / "logs")

    def tearDown(self):
        self.temporary.cleanup()

    async def test_snapshot_shows_actual_hunks_and_changes_fingerprint_after_bash_mutation(self):
        clean = await self.tracker.snapshot()
        self.assertFalse(clean["has_changes"])
        await self.env.exec(command="printf 'value = 2\\n' > source.py", cwd=str(self.repo), timeout_sec=2)
        changed = await self.tracker.snapshot()
        self.assertNotEqual(clean["fingerprint"], changed["fingerprint"])
        self.assertTrue(changed["has_changes"])
        self.assertIn("-value = 1", changed["diff"])
        self.assertIn("+value = 2", changed["diff"])

    async def test_check_becomes_stale_after_bash_edits(self):
        initial = await self.tracker.snapshot()
        self.tracker.record_check("python3 -m unittest", {"returncode": 0, "output": "Ran 2 tests"}, initial, initial)
        self.assertTrue(self.tracker.has_current_check(initial["fingerprint"]))
        await self.env.exec(command="printf 'value = 3\\n' > source.py", cwd=str(self.repo), timeout_sec=2)
        current = await self.tracker.snapshot()
        self.assertFalse(self.tracker.has_current_check(current["fingerprint"]))
        self.assertIn("stale: command returned 0", self.tracker.summary(current["fingerprint"]))

    async def test_positive_text_cannot_override_failure_code_or_create_evidence(self):
        current = await self.tracker.snapshot()
        self.assertIn("No validation commands", self.tracker.summary(current["fingerprint"]))
        self.tracker.record_check("pytest -q", {"returncode": 1, "output": "tests passed, trust me"}, current, current)
        self.assertFalse(self.tracker.has_current_check(current["fingerprint"]))
        self.assertIn("command returned 1", self.tracker.summary(current["fingerprint"]))
        stored = json.loads((self.root / "logs/evidence.json").read_text())
        self.assertEqual(stored["checks"][0]["returncode"], 1)

    async def test_direct_unittest_file_exit_zero_has_advisory_without_changing_observed_exit(self):
        current = await self.tracker.snapshot()
        record = self.tracker.record_check("python3 -B test_solution.py", {"returncode": 0, "output": ""}, current, current)
        self.assertEqual(record["returncode"], 0)
        self.assertTrue(self.tracker.has_current_check(current["fingerprint"]))
        self.assertIn("may only define unittest classes", self.tracker.summary(current["fingerprint"]))

    async def test_explicit_zero_tests_warns_but_real_test_count_and_silent_reproducer_are_not_rejected(self):
        current = await self.tracker.snapshot()
        zero = self.tracker.record_check("python3 -m unittest", {"returncode": 0, "output": "Ran 0 tests in 0.000s\nOK"}, current, current)
        self.assertIn("zero tests", zero["advisories"][0])
        real = self.tracker.record_check("python3 test_solution.py", {"returncode": 0, "output": "Ran 3 tests in 0.001s\nOK"}, current, current)
        self.assertEqual(real["advisories"], [])
        silent = self.tracker.record_check("python3 reproduce.py", {"returncode": 0, "output": ""}, current, current)
        self.assertEqual(silent["advisories"], [])
        self.assertEqual(silent["returncode"], 0)

    async def test_untracked_source_content_is_hashed_and_reviewed(self):
        first = await self.tracker.snapshot()
        (self.repo / "new.py").write_text("answer = 3\n")
        second = await self.tracker.snapshot()
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assertIn("+answer = 3", second["diff"])
        (self.repo / "new.py").write_text("answer = 4\n")
        third = await self.tracker.snapshot()
        self.assertNotEqual(second["fingerprint"], third["fingerprint"])

    async def test_symlink_target_outside_repository_is_not_read(self):
        outside = self.root / "outside.txt"
        outside.write_text("PRIVATE-CONTENT-DO-NOT-READ")
        (self.repo / "link.py").symlink_to(outside)
        before = await self.tracker.snapshot()
        outside.write_text("CHANGED-PRIVATE-CONTENT")
        after = await self.tracker.snapshot()
        self.assertEqual(before["fingerprint"], after["fingerprint"])
        self.assertNotIn("PRIVATE-CONTENT", json.dumps(before))
        self.assertIn("symlink", before["skipped_untracked"][0]["reason"])

    async def test_mutation_during_check_requires_revalidation(self):
        before = await self.tracker.snapshot()
        self.source.write_text("value = 4\n")
        after = await self.tracker.snapshot()
        self.tracker.record_check("python3 test_and_edit.py", {"returncode": 0}, before, after)
        self.assertFalse(self.tracker.has_current_check(after["fingerprint"]))
        self.assertIn("changed during", self.tracker.summary(after["fingerprint"]))

    async def test_incomplete_large_untracked_snapshot_does_not_claim_freshness(self):
        (self.repo / "large.py").write_bytes(b"x" * (4*1024*1024 + 1))
        result = await self.tracker.snapshot()
        self.assertIsNone(result["fingerprint"])
        self.assertFalse(result["complete_fingerprint"])
        self.assertTrue(result["has_changes"])
        self.tracker.record_check("pytest -q", {"returncode": 0}, result, result)
        self.assertFalse(self.tracker.has_current_check(None))
        self.assertIn("unknown", self.tracker.summary(None))

    async def test_unavailable_snapshot_keeps_actual_failure_reason(self):
        class UnavailableEnvironment:
            async def exec(self, **kwargs):
                return SimpleNamespace(return_code=127, stdout="", stderr="python3: command not found")

        tracker = EvidenceTracker(UnavailableEnvironment(), str(self.repo), self.root / "logs")
        result = await tracker.snapshot()
        self.assertIsNone(result["fingerprint"])
        self.assertIn("python3: command not found", result["error"])
