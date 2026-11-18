#!/usr/bin/env python3
"""Fetch only pinned public instruction.md files, verifying inventory SHA256.

No task tests, solutions, Dockerfiles, verifiers, or issue/PR pages are fetched.
External task statements remain outside the submission ZIP and published fork.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "MERA-Evaluation/SWE-MERA-CHALLENGE"
COMMIT = "070bcb675d96845745e28b40f3830c98139217ea"


def statement_sources(root: Path = ROOT) -> list[tuple[Path, str, str]]:
    """Validate every destination and source before any download or write."""
    root = Path(root).resolve()
    inventory = json.loads((root / "data/public_task_inventory.json").read_text())
    if inventory.get("repository") != REPOSITORY or inventory.get("commit") != COMMIT:
        raise ValueError("Inventory is not the pinned official public source")
    rows = inventory.get("tasks", [])
    if len(rows) != inventory.get("n_public_tasks"):
        raise ValueError("Public statement inventory count mismatch")
    result, seen = [], set()
    for row in rows:
        task = row.get("task_id", "")
        if not re.fullmatch(r"[a-z]+-\d{3}", task) or task in seen:
            raise ValueError("Invalid or duplicate public task identifier")
        seen.add(task)
        expected_url = f"https://github.com/{REPOSITORY}/blob/{COMMIT}/tasks/{task}/instruction.md"
        relative = f"data/public_statements/{task}.md"
        digest = row.get("instruction_sha256", "")
        if row.get("instruction_url") != expected_url or row.get("instruction_path") != relative:
            raise ValueError(f"Unexpected public instruction source or destination: {task}")
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Missing SHA256 for public instruction: {task}")
        target = root / relative
        if target.is_symlink() or not target.resolve().is_relative_to(root / "data/public_statements"):
            raise ValueError(f"Unsafe statement destination: {task}")
        raw_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/tasks/{task}/instruction.md"
        result.append((target, raw_url, digest))
    return result


def fetch(root: Path = ROOT, *, opener=urllib.request.urlopen) -> tuple[int, int]:
    downloaded, verified = 0, 0
    for target, url, digest in statement_sources(root):
        if target.is_file():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Existing statement fails SHA256: {target.name}")
            verified += 1
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "swe-mera-public-instruction-fetcher"})
        with opener(request, timeout=30) as response:
            content = response.read(2 * 1024 * 1024 + 1)
        if len(content) > 2 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Downloaded statement fails SHA256: {target.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".instruction-", delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        downloaded += 1
    return downloaded, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        downloaded, verified = fetch()
        print(f"Public instructions: downloaded={downloaded}, verified_existing={verified}")
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
