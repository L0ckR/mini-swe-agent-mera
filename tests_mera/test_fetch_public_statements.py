from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from scripts.fetch_public_statements import COMMIT, REPOSITORY, fetch, statement_sources


class FetchPublicStatementsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.content = b"Public issue instruction only.\n"
        self.row = {"task_id": "py-001", "instruction_path": "data/public_statements/py-001.md",
                    "instruction_url": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/tasks/py-001/instruction.md",
                    "instruction_sha256": hashlib.sha256(self.content).hexdigest()}
        self.inventory = {"repository": REPOSITORY, "commit": COMMIT, "n_public_tasks": 1, "tasks": [self.row]}
        (self.root / "data").mkdir()
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        (self.root / "data/public_task_inventory.json").write_text(json.dumps(self.inventory))

    def test_only_pinned_instruction_is_downloaded_and_existing_hash_is_verified(self):
        urls = []
        def opener(request, timeout):
            urls.append(request.full_url)
            return BytesIO(self.content)
        self.assertEqual(fetch(self.root, opener=opener), (1, 0))
        self.assertEqual(fetch(self.root, opener=opener), (0, 1))
        self.assertEqual(urls, [f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/tasks/py-001/instruction.md"])

    def test_bad_download_hash_is_not_written(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            fetch(self.root, opener=lambda *args, **kwargs: BytesIO(b"wrong"))
        self.assertFalse((self.root / self.row["instruction_path"]).exists())

    def test_source_cannot_point_to_solution_or_unpinned_commit(self):
        self.row["instruction_url"] = self.row["instruction_url"].replace("instruction.md", "solution/solve.sh")
        self.save()
        with self.assertRaisesRegex(ValueError, "source or destination"):
            statement_sources(self.root)
        self.inventory["commit"] = "main"
        self.save()
        with self.assertRaisesRegex(ValueError, "pinned"):
            statement_sources(self.root)

    def test_destination_cannot_escape_or_follow_a_symlink(self):
        self.row["instruction_path"] = "../outside.md"
        self.save()
        with self.assertRaises(ValueError):
            statement_sources(self.root)
        self.row["instruction_path"] = "data/public_statements/py-001.md"
        self.save()
        (self.root / "data/public_statements").symlink_to(self.root / "outside", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            statement_sources(self.root)
