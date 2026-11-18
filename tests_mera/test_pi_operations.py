"""Real-file tests of the Harbor operation bridge, using trusted temp fixtures."""
from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_stage2"))
from custom_agent.pi_operations import PiOperations
from custom_agent.policy import RunBudget


class TrustedFixtureEnvironment:
    def __init__(self):
        self.commands = []

    async def exec(self, *, command, cwd, timeout_sec):
        self.commands.append(command)
        process = await asyncio.create_subprocess_shell(
            command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
            return SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()


class PiOperationsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.env = TrustedFixtureEnvironment()
        self.ops = PiOperations(self.env, RunBudget(), workdir=str(self.repo))
        (self.repo / "source.py").write_bytes(b"def add(a, b):\r\n    return a - b\r\n")
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                        "commit", "-qm", "initial"], cwd=self.repo, check=True)

    def tearDown(self):
        self.temporary.cleanup()

    async def write(self, path, data):
        return await self.ops.execute("write_file", {"path": path, "data_base64": base64.b64encode(data).decode()})

    async def test_native_file_ports_and_real_exec_preserve_bytes_and_track_new_files(self):
        original = (self.repo / "source.py").read_bytes()
        await self.ops.execute("access", {"path": "source.py", "write": True})
        result = await self.ops.execute("read_file", {"path": "source.py"})
        self.assertEqual(base64.b64decode(result["data_base64"]), original)
        await self.write("source.py", original.replace(b"a - b", b"a + b"))
        observed = await self.ops.execute("exec", {"command": "python3 -B -c 'from source import add; assert add(2,3)==5'",
                                                  "cwd": str(self.repo), "timeout_sec": 10})
        self.assertEqual(observed["returncode"], 0)
        await self.ops.execute("mkdir", {"path": "new/subdir"})
        await self.write("new/subdir/value.txt", "caf\u00e9\n".encode())
        patch = subprocess.run(["git", "diff", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True).stdout
        self.assertIn("new/subdir/value.txt", patch)
        self.assertIn("return a + b", patch)
        self.assertEqual((self.repo / "new/subdir/value.txt").read_text(), "caf\u00e9\n")

    async def test_large_native_edit_is_chunked_atomic_and_retains_executable_mode(self):
        target = self.repo / "source.py"
        target.chmod(0o751)
        before_inode = target.stat().st_ino
        data = ("# long source with Unicode caf\u00e9\n" * 9000).encode()
        self.assertGreater(len(data), 128 * 1024)
        await self.write("source.py", data)
        self.assertEqual(target.read_bytes(), data)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o751)
        self.assertNotEqual(target.stat().st_ino, before_inode)
        self.assertFalse(list(self.repo.glob(".mera-pi-transfer-*")))
        self.assertFalse(list(self.repo.glob(".mera-pi-write-*")))
        self.assertTrue(all(len(command.encode()) < 128 * 1024 for command in self.env.commands))
        self.assertNotIn(base64.b64encode(data).decode(), json.dumps(self.ops.records))

    async def test_interrupted_transfer_preserves_original_and_removes_its_artifact(self):
        original = (self.repo / "source.py").read_bytes()
        real_exec = self.env.exec
        calls = 0

        async def failing_exec(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return SimpleNamespace(return_code=1, stdout="", stderr="injected transport failure")
            return await real_exec(**kwargs)

        self.env.exec = failing_exec
        with self.assertRaisesRegex(RuntimeError, "Source transfer failed"):
            await self.write("source.py", b"x" * (200 * 1024))
        self.assertEqual((self.repo / "source.py").read_bytes(), original)
        self.assertFalse(list(self.repo.glob(".mera-pi-transfer-*")))

    async def test_traversal_external_paths_and_symlink_targets_are_rejected(self):
        outside = self.root / "outside.txt"
        outside.write_text("untouched")
        (self.repo / "link").symlink_to(outside)
        for path in ("../outside.txt", str(outside), "link"):
            with self.subTest(path=path):
                with self.assertRaises((ValueError, RuntimeError)):
                    await self.ops.execute("read_file", {"path": path})
                with self.assertRaises((ValueError, RuntimeError)):
                    await self.write(path, b"changed")
        self.assertEqual(outside.read_text(), "untouched")

    async def test_repository_erasure_is_rejected_before_harbor_exec(self):
        result = await self.ops.execute("exec", {"command": "rm -rf " + str(self.repo), "cwd": str(self.repo)})
        self.assertNotEqual(result["returncode"], 0)
        self.assertIn("repository", result["stderr"].lower())
        self.assertFalse(self.env.commands)
        self.assertTrue((self.repo / ".git").is_dir())

    async def test_timeout_is_explicit_and_does_not_claim_completion(self):
        cancelled = asyncio.Event()

        class WaitingEnvironment:
            async def exec(self, **kwargs):
                try:
                    await asyncio.sleep(10)
                finally:
                    cancelled.set()

        self.ops.environment = WaitingEnvironment()
        result = await self.ops.execute("exec", {"command": "python3 -m unittest", "timeout_sec": 0.02})
        self.assertEqual(result["returncode"], 124)
        self.assertTrue(result["timed_out"])
        self.assertIn("unconfirmed", result["stderr"])
        self.assertTrue(cancelled.is_set())

    async def test_invalid_arguments_never_start_a_process_or_modify_the_original(self):
        original = (self.repo / "source.py").read_bytes()
        for operation, args in [
            ("write_file", {"path": "source.py", "data_base64": "not valid base64!"}),
            ("exec", {"command": "echo ok", "timeout_sec": True}),
            ("exec", {"command": "echo ok", "timeout_sec": float("nan")}),
            ("unknown", {}),
            ("read_file", []),
        ]:
            with self.subTest(operation=operation, args=args), self.assertRaises(ValueError):
                await self.ops.execute(operation, args)
        self.assertFalse(self.env.commands)
        self.assertEqual((self.repo / "source.py").read_bytes(), original)
