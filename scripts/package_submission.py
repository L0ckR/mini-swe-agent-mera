#!/usr/bin/env python3
"""Create a deterministic source-only Stage II ZIP and a local SHA256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    from .validate_submission import ValidationError, collect_directory, validate_files, validate_submission
except ImportError:
    from validate_submission import ValidationError, collect_directory, validate_files, validate_submission


def package_submission(source: Path | str, output: Path | str) -> dict:
    source, output = Path(source), Path(output)
    if output.suffix.lower() != ".zip":
        raise ValidationError("output must have the .zip suffix")
    try:
        output.resolve().relative_to(source.resolve())
    except ValueError:
        pass
    else:
        raise ValidationError("write the archive outside its source directory")
    files, warnings = collect_directory(source)
    report = validate_files(files, warnings)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for item in files:
                info = zipfile.ZipInfo(item.name, date_time=(2020, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if item.executable else 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                archive.writestr(info, item.data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        validate_submission(temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        **report.to_dict(), "archive": output.name,
        "sha256": digest, "archive_bytes": output.stat().st_size,
        "file_sha256": {item.name: hashlib.sha256(item.data).hexdigest() for item in files},
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=Path("submission_stage2"))
    parser.add_argument("--output", type=Path, default=Path("dist/stage2-submission.zip"))
    args = parser.parse_args(argv)
    try:
        manifest = package_submission(args.source, args.output)
    except (ValidationError, OSError) as exc:
        print(f"Packaging failed: {exc}", file=sys.stderr)
        return 1
    print(f"Created {args.output.resolve()}")
    print(f"SHA256: {manifest['sha256']}")
    print("Validation: static contract only; this archive has no claimed GPU or benchmark result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
