"""Go/PHP repository bridges with a PATH containing no Python interpreter."""
from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_stage2"))
from custom_agent.evidence import EvidenceTracker
from custom_agent.pi_operations import PiOperations
from custom_agent.pi_portable import has_python
from custom_agent.policy import RunBudget
from custom_agent.task_contract import TaskContract, validate_changes, write_requested_patch


class NoPythonEnvironment:
    def __init__(self, path):
        self.path, self.commands = str(path), []

    async def exec(self, *, command, cwd, timeout_sec):
        self.commands.append(command)
        process = await asyncio.create_subprocess_exec(
            "/bin/bash", "-c", command, cwd=cwd, env={**os.environ, "PATH": self.path},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(), timeout_sec)
            return SimpleNamespace(return_code=process.returncode, stdout=out.decode(), stderr=err.decode())
        finally:
            if process.returncode is None:
                process.kill(); await process.wait()


class PortableTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"; self.repo.mkdir()
        self.bin = self.root / "bin"; self.bin.mkdir()
        for name in ("bash", "git", "readlink", "stat", "base64", "mktemp", "rm", "mv", "chmod", "mkdir", "wc", "head", "tail", "sha256sum"):
            path = shutil.which(name)
            self.assertIsNotNone(path)
            (self.bin / name).symlink_to(path)
        self.env = NoPythonEnvironment(self.bin)
        self.ops = PiOperations(self.env, RunBudget(), workdir=str(self.repo))
        (self.repo / "value.go").write_text("package value\nfunc Add(a,b int) int { return a-b }\n")
        (self.repo / "value.php").write_text("<?php function add($a,$b) { return $a-$b; }\n")
        (self.repo / "value_test.go").write_text("package value\n")
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "initial"], cwd=self.repo, check=True)
        self.contract = TaskContract("fix", non_test_only=True, patch_path=str(self.repo / "change.diff"), workdir=str(self.repo))

    def tearDown(self):
        self.temp.cleanup()

    async def write(self, name, data):
        return await self.ops.execute("write_file", {"path": name, "data_base64": base64.b64encode(data).decode()})

    async def test_go_php_file_ports_and_required_patch_apply_without_python(self):
        self.assertFalse(await has_python(self.env, str(self.repo)))
        for name in ("value.go", "value.php"):
            original = (self.repo / name).read_bytes()
            await self.ops.execute("access", {"path": name, "write": True})
            read = await self.ops.execute("read_file", {"path": name})
            self.assertEqual(base64.b64decode(read["data_base64"]), original)
            await self.write(name, original.replace(b"a-b", b"a+b").replace(b"$a-$b", b"$a+$b"))
        await self.ops.execute("mkdir", {"path": "src/new"})
        await self.write("src/new/info.php", b"<?php // new implementation file\n")
        before_index = (self.repo / ".git/index").read_bytes()
        validation = await validate_changes(self.env, self.contract)
        self.assertEqual(validation["returncode"], 0, validation)
        result = await write_requested_patch(self.env, self.contract)
        self.assertEqual(result["returncode"], 0, result)
        self.assertTrue(result["written"])
        self.assertEqual((self.repo / ".git/index").read_bytes(), before_index)
        clean = self.root / "clean"
        subprocess.run(["git", "clone", "-q", str(self.repo), str(clean)], check=True)
        subprocess.run(["git", "apply", "--check", str(self.repo / "change.diff")], cwd=clean, check=True)
        subprocess.run(["git", "apply", str(self.repo / "change.diff")], cwd=clean, check=True)
        self.assertIn("a+b", (clean / "value.go").read_text())
        self.assertIn("$a+$b", (clean / "value.php").read_text())
        self.assertTrue((clean / "src/new/info.php").is_file())
        probes = [command for command in self.env.commands if command == "command -v python3 >/dev/null 2>&1"]
        self.assertEqual(len(probes), 1)
        self.assertFalse(any("python3 -c" in command for command in self.env.commands))

    async def test_test_edits_are_rejected_without_changing_index_or_deliverable(self):
        await self.write("value.go", b"package value\nfunc Add(a,b int) int { return a+b }\n")
        await self.write("value_test.go", b"package value\n// forbidden edit\n")
        (self.repo / "change.diff").write_text("previous artifact")
        index = (self.repo / ".git/index").read_bytes()
        result = await write_requested_patch(self.env, self.contract)
        self.assertEqual(result["returncode"], 1)
        self.assertIn("value_test.go", result["rejected_test_paths"])
        self.assertEqual((self.repo / ".git/index").read_bytes(), index)
        self.assertEqual((self.repo / "change.diff").read_text(), "previous artifact")
        self.assertFalse(list(self.repo.glob(".mera-export-*")))

    async def test_evidence_tracks_real_edits_and_ignores_exported_patch(self):
        tracker = EvidenceTracker(self.env, str(self.repo), ignored_paths=("change.diff",))
        before = await tracker.snapshot()
        self.assertEqual(before["returncode"], 0, before)
        self.assertFalse(before["has_changes"])
        await self.write("value.php", b"<?php function add($a,$b) { return $a+$b; }\n")
        changed = await tracker.snapshot()
        self.assertNotEqual(before["fingerprint"], changed["fingerprint"])
        self.assertIn("$a+$b", changed["diff"])
        exported = await write_requested_patch(self.env, self.contract)
        self.assertEqual(exported["returncode"], 0, exported)
        after = await tracker.snapshot(include_diff=False)
        self.assertEqual(changed["fingerprint"], after["fingerprint"])
        self.assertEqual(after["returncode"], 0, after)

    async def test_large_write_preserves_mode_and_outside_symlink_is_not_read(self):
        target = self.repo / "value.go"; target.chmod(0o751)
        content = b"// large source\n" * 10000
        await self.write("value.go", content)
        self.assertEqual(target.stat().st_mode & 0o777, 0o751)
        outside = self.root / "outside"; outside.write_text("untouched")
        (self.repo / "link.php").symlink_to(outside)
        with self.assertRaises(RuntimeError):
            await self.ops.execute("read_file", {"path": "link.php"})
        with self.assertRaises(RuntimeError):
            await self.write("link.php", b"changed")
        self.assertEqual(outside.read_text(), "untouched")
        snap = await EvidenceTracker(self.env, str(self.repo)).snapshot()
        self.assertTrue(snap["complete_fingerprint"], snap)
        self.assertIn("symlink", snap["skipped_untracked"][0]["reason"])
