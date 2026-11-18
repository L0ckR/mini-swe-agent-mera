"""Exercise submitted typed tools against real temporary files and Git indexes."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission_stage2"))
from custom_agent.tools import TaskTools, TOOLS


class LocalEnvironment:
    def __init__(self):
        self.calls = []

    async def exec(self, *, command, cwd, timeout_sec):
        self.calls.append({"command": command, "cwd": cwd, "timeout_sec": timeout_sec})
        process = await asyncio.create_subprocess_exec("bash", "-c", command, cwd=cwd,
                                                       stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE,
                                                       start_new_session=True)
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            os.killpg(process.pid, 9)
            await process.wait()
            raise
        return SimpleNamespace(stdout=stdout.decode(), stderr=stderr.decode(), return_code=process.returncode)


class TaskToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.outside = Path(self.temporary.name) / "outside"
        self.outside.mkdir()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.environment = LocalEnvironment()
        self.tools = TaskTools(self.environment, str(self.root))

    def tearDown(self):
        self.temporary.cleanup()

    async def call(self, name, **arguments):
        return await self.tools.execute({"name": name, "arguments": arguments}, timeout_sec=10)

    async def test_read_line_numbers_and_empty_or_past_eof_ranges(self):
        (self.root / "sample.py").write_bytes(b"one\r\ntwo\nthree")
        result = await self.call("read_file", path="sample.py", start_line=2, end_line=3)
        self.assertEqual(result["returncode"], 0)
        self.assertIn("2 | two\n3 | three", result["output"])
        self.assertNotIn("1 | one", result["output"])
        eof = await self.call("read_file", path="sample.py", start_line=9)
        self.assertEqual(eof["returncode"], 0)
        self.assertIn("beyond EOF", eof["output"])
        (self.root / "empty").write_text("")
        empty = await self.call("read_file", path="empty")
        self.assertIn("0 lines", empty["output"])
        self.assertEqual(empty["total_lines"], 0)

    async def test_exact_edit_preserves_unmodified_bytes_permissions_and_no_temporaries(self):
        path = self.root / "script.py"
        path.write_bytes(b"#!/usr/bin/python3\r\nvalue = 'old'\r\n  end\n")
        path.chmod(0o755)
        result = await self.call("edit_file", path="script.py", old_text="value = 'old'\r\n", new_text="value = 'new'\r\n")
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["changed"])
        self.assertEqual(path.read_bytes(), b"#!/usr/bin/python3\r\nvalue = 'new'\r\n  end\n")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)
        self.assertEqual(list(self.root.glob(".mera-*")), [])

    async def test_ambiguous_missing_and_overlapping_matches_do_not_mutate(self):
        path = self.root / "text"
        for original, old in (("abc abc", "abc"), ("abc", "missing"), ("aaa", "aa")):
            path.write_text(original)
            result = await self.call("edit_file", path="text", old_text=old, new_text="replacement")
            self.assertNotEqual(result["returncode"], 0)
            self.assertFalse(result["changed"])
            self.assertEqual(path.read_text(), original)

    async def test_shell_metacharacters_in_paths_and_content_remain_literal(self):
        path = "odd $(touch INJECTED) ' file.py"
        content = "literal `touch ALSO_INJECTED` $HOME\n'''quotes'''\n\"double\"\n"
        created = await self.call("create_file", path=path, content=content)
        self.assertEqual(created["returncode"], 0)
        self.assertEqual((self.root / path).read_text(), content)
        edited = await self.call("edit_file", path=path, old_text="$HOME", new_text="$(touch THIRD_INJECTED)")
        self.assertEqual(edited["returncode"], 0)
        self.assertIn("$(touch THIRD_INJECTED)", (self.root / path).read_text())
        for name in ("INJECTED", "ALSO_INJECTED", "THIRD_INJECTED"):
            self.assertFalse((self.root / name).exists())

    async def test_create_refuses_existing_files_and_only_adds_requested_path(self):
        (self.root / "unrelated").write_text("leave untracked")
        result = await self.call("create_file", path="pkg/new.py", content="answer = 42\n")
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["git_intent_to_add"])
        index = subprocess.run(["git", "ls-files"], cwd=self.root, check=True, text=True, capture_output=True).stdout
        self.assertEqual(index.strip(), "pkg/new.py")
        patch = subprocess.run(["git", "diff", "--", "pkg/new.py"], cwd=self.root, check=True, text=True, capture_output=True).stdout
        self.assertIn("+answer = 42", patch)
        duplicate = await self.call("create_file", path="pkg/new.py", content="overwrite")
        self.assertNotEqual(duplicate["returncode"], 0)
        self.assertEqual((self.root / "pkg/new.py").read_text(), "answer = 42\n")
        self.assertEqual(list((self.root / "pkg").glob(".mera-*")), [])

    async def test_traversal_absolute_escape_and_symlink_escape_rejected_for_all_file_tools(self):
        (self.outside / "secret").write_text("outside")
        (self.root / "escape").symlink_to(self.outside, target_is_directory=True)
        (self.root / "bad-link").symlink_to(self.outside / "secret")
        for path in ("../outside/secret", str(self.outside / "secret"), "escape/secret", "bad-link"):
            for name, extra in (("read_file", {}), ("edit_file", {"old_text": "outside", "new_text": "changed"}),
                                ("create_file", {"content": "created"})):
                result = await self.call(name, path=path, **extra)
                self.assertNotEqual(result["returncode"], 0, (path, name, result))
        self.assertEqual((self.outside / "secret").read_text(), "outside")

    async def test_binary_files_rejected_without_changes(self):
        for content in (b"hello\0world", b"\xff\xfeinvalid"):
            (self.root / "binary").write_bytes(content)
            for name, extra in (("read_file", {}), ("edit_file", {"old_text": "hello", "new_text": "changed"})):
                result = await self.call(name, path="binary", **extra)
                self.assertNotEqual(result["returncode"], 0)
            self.assertEqual((self.root / "binary").read_bytes(), content)

    async def test_invalid_arguments_never_start_an_environment_command(self):
        invalid = [
            {"name": "bash", "arguments": {"command": "pwd", "timeout_sec": True}},
            {"name": "bash", "arguments": {"command": "pwd", "timeout_sec": 301}},
            {"name": "read_file", "arguments": {"path": "a", "start_line": 0}},
            {"name": "read_file", "arguments": {"path": "a", "start_line": 9, "end_line": 8}},
            {"name": "read_file", "arguments": {"path": "a", "end_line": 401}},
            {"name": "edit_file", "arguments": {"path": "a", "old_text": "", "new_text": "x"}},
            {"name": "finish", "arguments": {"summary": ""}},
            {"name": "read_file", "arguments": {"path": "a", "surprise": True}},
            {"name": "read_file", "arguments": "not JSON"},
        ]
        for action in invalid:
            result = await self.tools.execute(action, 10)
            self.assertNotEqual(result["returncode"], 0)
        self.assertEqual(self.environment.calls, [])

    async def test_bash_deadline_and_output_bound_and_raw_audit(self):
        tools = TaskTools(self.environment, str(self.root), max_output_chars=100)
        result = await tools.execute({"name": "bash", "arguments": {"command": "printf '%0200d' 1", "timeout_sec": 200}}, 2.5)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(self.environment.calls[-1]["timeout_sec"], 2.5)
        self.assertIn("omitted", result["output"])
        self.assertGreater(len(result["raw_output"]), len(result["output"]))
        timed_out = await tools.execute({"name": "bash", "arguments": {"command": "sleep 5"}}, 0.02)
        self.assertEqual(timed_out["returncode"], 124)
        self.assertIn("unconfirmed", timed_out["output"])

    async def test_finish_and_expired_deadline_never_execute(self):
        result = await self.call("finish", summary="Changed code; focused tests passed.")
        self.assertTrue(result["submitted"])
        self.assertEqual(result["summary"], "Changed code; focused tests passed.")
        expired = await self.tools.execute({"name": "bash", "arguments": {"command": "touch unexpected"}}, 0)
        self.assertEqual(expired["returncode"], 124)
        self.assertFalse((self.root / "unexpected").exists())
        self.assertEqual(self.environment.calls, [])

    async def test_openai_action_shape_and_python_unavailable_are_reported(self):
        result = await self.tools.execute({"function": {"name": "bash", "arguments": json.dumps({"command": "printf success"})}}, 10)
        self.assertEqual(result["returncode"], 0)
        self.assertIn("success", result["output"])
        class MissingPython:
            async def exec(self, **kwargs):
                return SimpleNamespace(return_code=127, stdout="File tool requires python3 in the task container", stderr="")
        unavailable = await TaskTools(MissingPython()).execute({"name": "read_file", "arguments": {"path": "file"}}, 10)
        self.assertEqual(unavailable["returncode"], 127)
        self.assertIn("requires python3", unavailable["output"])

    async def test_cancellation_propagates_to_caller(self):
        started = asyncio.Event()
        class Cancellable:
            async def exec(self, **kwargs):
                started.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(TaskTools(Cancellable()).execute({"name": "bash", "arguments": {"command": "long command"}}, 20))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_parent_budget_clamps_timeout_and_expiry_never_starts_a_command(self):
        result = await self.tools.execute({"name": "bash", "arguments": {"command": "printf done", "timeout_sec": 300}}, 0.5)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(self.environment.calls[-1]["timeout_sec"], 0.5)
        calls = len(self.environment.calls)
        for remaining in (0, -1, float("nan")):
            expired = await self.tools.execute({"name": "bash", "arguments": {"command": "printf should-not-run"}}, remaining)
            self.assertEqual(expired["returncode"], 124)
        self.assertEqual(len(self.environment.calls), calls)

    async def test_actual_timeout_is_reported_as_unconfirmed_and_full_output_is_retained(self):
        timeout = await self.tools.execute({"name": "bash", "arguments": {"command": "sleep 5"}}, 0.03)
        self.assertEqual(timeout["returncode"], 124)
        self.assertIn("unconfirmed", timeout["output"])
        bounded = TaskTools(self.environment, str(self.root), max_output_chars=100)
        output = await bounded.execute({"name": "bash", "arguments": {"command": "python3 -c \"print('A'*500+'END')\""}}, 10)
        self.assertEqual(output["returncode"], 0)
        self.assertGreater(len(output["raw_output"]), 500)
        self.assertIn("characters omitted", output["output"])
        self.assertIn("END", output["output"])
        self.assertLess(len(output["output"]), 180)

    async def test_dangling_symlinks_and_binary_creation_are_not_overwritten(self):
        link = self.root / "dangling"
        link.symlink_to(self.root / "missing")
        result = await self.call("create_file", path="dangling", content="created")
        self.assertNotEqual(result["returncode"], 0)
        self.assertTrue(link.is_symlink())
        self.assertFalse((self.root / "missing").exists())
        binary = await self.call("create_file", path="new-binary", content="hello\0binary")
        self.assertNotEqual(binary["returncode"], 0)
        self.assertFalse((self.root / "new-binary").exists())


if __name__ == "__main__":
    unittest.main()
