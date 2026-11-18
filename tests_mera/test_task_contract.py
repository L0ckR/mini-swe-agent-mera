from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_stage2"))
from custom_agent.task_contract import TaskContract, validate_changes, write_requested_patch


class LocalEnvironment:
    def __init__(self):
        self.commands = []

    async def exec(self, *, command, cwd, timeout_sec):
        self.commands.append(command)
        process = await asyncio.create_subprocess_shell(command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())


class ContractTests(unittest.TestCase):
    def test_complete_public_inventory_detects_twenty_two_artifacts_without_ids(self):
        paths = sorted((ROOT / "data/public_statements").glob("*.md"))
        if not paths:
            self.skipTest("public statements are analysis data, excluded from standalone fork")
        contracts = [TaskContract.from_instruction(path.read_text()) for path in paths]
        self.assertEqual(len(contracts), 49)
        self.assertEqual(sum(contract.patch_path == "/testbed/change.diff" for contract in contracts), 22)
        self.assertTrue(all(contract.non_test_only and contract.dependencies_preinstalled for contract in contracts))

    def test_preserves_issue_body_and_reads_deliverable_outside_issue_tags(self):
        instruction = '<issue_description>Fix this exact code:\n```py\n    x = 1\n```</issue_description>\nMinimal changes to non-test files. All dependencies already installed.\n## Submission\nWrite your complete change as a unified diff to `/testbed/change.diff`.'
        contract = TaskContract.from_instruction(instruction)
        self.assertEqual(contract.issue_text, 'Fix this exact code:\n```py\n    x = 1\n```')
        self.assertEqual(contract.patch_path, "/testbed/change.diff")
        self.assertIn("do not modify or add tests", contract.guidance())
        self.assertIn("already installed", contract.guidance())

    def test_issue_example_patch_path_is_not_a_submission_directive(self):
        contract = TaskContract.from_instruction('<issue_description>Tool should write a patch to /tmp/example.patch.</issue_description>\nMake a source fix.')
        self.assertIsNone(contract.patch_path)

    def test_relative_requested_artifact_and_unsafe_targets(self):
        good = TaskContract.from_instruction("## Submission\nSave a patch to `out/fix.patch`.", "/repo")
        self.assertEqual(good.patch_path, "/repo/out/fix.patch")
        for path in ("/tmp/fix.patch", "../fix.patch", ".git/fix.patch"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                TaskContract.from_instruction("## Submission\nWrite a unified diff to `" + path + "`.", "/repo")


class PatchExporterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        (self.repo / "app.py").write_text("value = 1\n")
        (self.repo / "tests").mkdir()
        (self.repo / "tests/test_app.py").write_text("assert True\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.env = LocalEnvironment()
        self.contract = TaskContract.from_instruction("Minimal changes to non-test files.\n## Submission\nWrite your complete change as a unified diff to `change.diff`.", str(self.repo))

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, capture_output=True, check=True).stdout

    async def test_staged_unstaged_and_untracked_patch_applies_and_real_index_is_unchanged(self):
        (self.repo / "app.py").write_text("value = 2\n")
        self.git("add", "app.py")
        (self.repo / "app.py").write_text("value = 3\n")
        (self.repo / "new.py").write_text("new_value = 4\n")
        (self.repo / "junit.xml").write_text("generated\n")
        (self.repo / "change.diff").write_text("stale output\n")
        before = (self.repo / ".git/index").read_bytes()
        result = await write_requested_patch(self.env, self.contract)
        self.assertTrue(result["written"], result)
        self.assertTrue(result["apply_check_passed"])
        self.assertEqual((self.repo / ".git/index").read_bytes(), before)
        patch = (self.repo / "change.diff").read_text()
        self.assertIn("+value = 3", patch)
        self.assertIn("new.py", patch)
        self.assertNotIn("stale output", patch)
        self.assertNotIn("junit.xml", patch)
        self.assertFalse(list(self.repo.glob(".mera-export-*")))
        clean = self.root / "clean"
        subprocess.run(["git", "clone", "-q", str(self.repo), str(clean)], check=True)
        subprocess.run(["git", "apply", "--check", str(self.repo / "change.diff")], cwd=clean, check=True)
        subprocess.run(["git", "apply", str(self.repo / "change.diff")], cwd=clean, check=True)
        self.assertEqual((clean / "app.py").read_text(), "value = 3\n")
        self.assertEqual((clean / "new.py").read_text(), "new_value = 4\n")

    async def test_forbidden_test_changes_are_rejected_without_silent_filtering(self):
        (self.repo / "app.py").write_text("value = 3\n")
        (self.repo / "tests/test_app.py").write_text("assert False\n")
        before = (self.repo / ".git/index").read_bytes()
        result = await write_requested_patch(self.env, self.contract)
        self.assertFalse(result["written"])
        self.assertEqual(result["rejected_test_paths"], ["tests/test_app.py"])
        self.assertFalse((self.repo / "change.diff").exists())
        self.assertEqual((self.repo / ".git/index").read_bytes(), before)
        self.assertEqual((self.repo / "tests/test_app.py").read_text(), "assert False\n")

    async def test_binary_and_empty_changes_are_rejected(self):
        empty = await write_requested_patch(self.env, self.contract)
        self.assertFalse(empty["written"])
        (self.repo / "picture.bin").write_bytes(b"\0binary")
        binary = await write_requested_patch(self.env, self.contract)
        self.assertFalse(binary["written"])
        self.assertTrue(binary["binary_entries"])

    async def test_no_requested_artifact_never_executes_or_writes(self):
        result = await write_requested_patch(self.env, TaskContract.from_instruction("Fix the implementation.", str(self.repo)))
        self.assertEqual(self.env.commands, [])
        self.assertTrue(result["skipped"])

    async def test_non_test_validation_without_patch_artifact_rejects_tests_and_preserves_index(self):
        contract = TaskContract.from_instruction("Make minimal changes to non-test files.", str(self.repo))
        (self.repo / "test_extra.py").write_text("assert False\n")
        before = (self.repo / ".git/index").read_bytes()
        result = await validate_changes(self.env, contract)
        self.assertFalse(result["valid"])
        self.assertIn("test_extra.py", result["rejected_test_paths"])
        self.assertFalse((self.repo / "change.diff").exists())
        self.assertEqual((self.repo / ".git/index").read_bytes(), before)

    async def test_non_test_validation_accepts_source_change_without_writing_diff(self):
        contract = TaskContract.from_instruction("Make minimal changes to non-test files.", str(self.repo))
        (self.repo / "app.py").write_text("value = 2\n")
        result = await validate_changes(self.env, contract)
        self.assertTrue(result["valid"])
        self.assertEqual(result["changed_paths"], ["app.py"])
        self.assertFalse(result["written"])

    async def test_output_symlink_escape_is_rejected(self):
        target = self.root / "outside.diff"
        target.write_text("preserve")
        (self.repo / "change.diff").symlink_to(target)
        result = await write_requested_patch(self.env, self.contract)
        self.assertFalse(result["written"])
        self.assertEqual(target.read_text(), "preserve")

    async def test_cancellation_propagates(self):
        class Blocking:
            async def exec(self, **kwargs):
                raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await write_requested_patch(Blocking(), self.contract)
