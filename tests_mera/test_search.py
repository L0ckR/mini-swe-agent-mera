"""Literal source search runs inside the same bounded task helper as file edits."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission_stage2"))
from custom_agent.tools import TaskTools, TOOLS
from tests.test_fork_tools import LocalEnvironment


class SearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        self.environment = LocalEnvironment()
        self.tools = TaskTools(self.environment, str(self.root))

    def tearDown(self):
        self.temporary.cleanup()

    async def search(self, pattern, **arguments):
        return await self.tools.execute({"name": "search", "arguments": {"pattern": pattern, **arguments}}, 5)

    async def test_leading_dash_patterns_are_literal_and_line_numbered(self):
        (self.root / "core.py").write_text("before\nshort = '-h'\nlong = '--help'\nafter\n")
        for pattern, number in (("-h", 2), ("--help", 3)):
            with self.subTest(pattern=pattern):
                result = await self.search(pattern, context_lines=0)
                self.assertEqual(result["returncode"], 0)
                self.assertIn(f"core.py:{number}:", result["output"])
                self.assertIn(pattern, result["output"])
                self.assertNotIn("Usage:", result["output"])
                self.assertFalse(result["changed"])
        self.assertEqual(len(TOOLS), 7)

    async def test_glob_file_directory_and_context_windows(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "core.py").write_text("before\nneedle\nafter\n")
        (self.root / "README.md").write_text("needle docs\n")
        result = await self.search("needle", file_glob="src/*.py", context_lines=1)
        self.assertEqual(result["matches"], 1)
        self.assertIn("src/core.py:1- before\nsrc/core.py:2: needle\nsrc/core.py:3- after", result["output"])
        self.assertNotIn("needle docs", result["output"])
        direct = await self.search("needle", path="src/core.py", context_lines=0)
        self.assertEqual(direct["matches"], 1)
        directory = await self.search("needle", path=str(self.root / "src"), context_lines=0)
        self.assertEqual(directory["matches"], 1)

    async def test_result_and_large_line_bounds_are_explicit(self):
        (self.root / "source.py").write_text("needle\n" * 10)
        result = await self.search("needle", max_results=2, context_lines=0)
        self.assertEqual(result["matches"], 2)
        self.assertTrue(result["truncated"])
        self.assertIn("narrow path", result["output"])
        self.assertNotIn("source.py:3:", result["output"])
        (self.root / "source.py").write_text("x" * 10000 + "needle" + "z" * 10000)
        line = await self.search("needle", context_lines=0)
        self.assertEqual(line["matches"], 1)
        self.assertIn("needle", line["output"])
        self.assertLess(len(line["output"]), 900)

    async def test_traversal_and_outside_symlinks_rejected_and_not_followed(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "secret").write_text("PRIVATE needle")
        (self.root / "escape").symlink_to(outside, target_is_directory=True)
        (self.root / "link").symlink_to(outside / "secret")
        for path in ("../outside", str(outside), "escape", "escape/secret", "link"):
            with self.subTest(path=path):
                result = await self.search("needle", path=path)
                self.assertNotEqual(result["returncode"], 0)
                self.assertNotIn("PRIVATE", result["output"])
        recursive = await self.search("needle")
        self.assertEqual(recursive["matches"], 0)
        self.assertNotIn("PRIVATE", recursive["output"])

    async def test_binary_cache_and_oversized_files_are_skipped(self):
        (self.root / "binary").write_bytes(b"needle\0hidden")
        (self.root / "invalid").write_bytes(b"needle\xff")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("needle")
        (self.root / "big").write_text("needle" + "x" * 1500)
        small = TaskTools(self.environment, str(self.root), max_file_bytes=1000)
        result = await small.execute({"name": "search", "arguments": {"pattern": "needle"}}, 5)
        self.assertEqual(result["matches"], 0)
        self.assertEqual(result["skipped_files"], 3)
        self.assertFalse(result["truncated"])

    async def test_invalid_search_arguments_do_not_start_a_process(self):
        for arguments in ({"pattern": ""}, {"pattern": "a\nb"}, {"pattern": "\0"},
                          {"pattern": "x", "context_lines": 4}, {"pattern": "x", "max_results": 0},
                          {"pattern": "x", "path": ""}, {"pattern": "x", "regex": True}):
            result = await self.tools.execute({"name": "search", "arguments": arguments}, 5)
            self.assertNotEqual(result["returncode"], 0)
        self.assertEqual(self.environment.calls, [])


if __name__ == "__main__":
    unittest.main()
