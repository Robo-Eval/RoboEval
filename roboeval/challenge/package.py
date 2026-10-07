"""Validate a submission and pack it into one archive for upload.

Writes MANIFEST.sha256 (one line per file) into the submission directory, then
creates <team>_<method>.tar.gz next to it and prints the archive's SHA-256 for
the submission form, so organizers can confirm they downloaded what you built.

Usage:
    python -m roboeval.challenge.package submission/
    python -m roboeval.challenge.package submission/ --exclude-checkpoint   # weights hosted via checkpoint_url
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
from pathlib import Path
from typing import Optional, Sequence

from roboeval.challenge import results as res
from roboeval.challenge.determinism import ensure_pinned
from roboeval.challenge.submission import load_metadata
from roboeval.challenge.validate import validate

MANIFEST = "MANIFEST.sha256"
# TBC: confirmed at launch.
MAX_BUNDLE_GB: Optional[float] = None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", text).strip("-") or "submission"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and package a challenge submission.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("submission", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None, help="Archive path (default: next to submission/).")
    parser.add_argument("--exclude-checkpoint", action="store_true",
                        help="Leave checkpoint/ out of the archive; metadata.yaml must set checkpoint_url.")
    parser.add_argument("--replay", type=int, default=10, metavar="N",
                        help="Episodes to replay during validation (default: 10; 0 to skip).")
    args = parser.parse_args(argv)
    sub = args.submission.resolve()

    report = validate(sub, replay=args.replay)
    meta = load_metadata(sub) if (sub / "metadata.yaml").exists() else {}
    if args.exclude_checkpoint and not meta.get("checkpoint_url"):
        report.fail("package", "--exclude-checkpoint requires checkpoint_url in metadata.yaml")
    report.print()
    if report.failed:
        print("\nNot packaging an invalid submission.")
        return 1

    skip_top = {MANIFEST} | ({"checkpoint"} if args.exclude_checkpoint else set())
    files = sorted(
        p for p in sub.rglob("*")
        if p.is_file() and p.relative_to(sub).parts[0] not in skip_top and "__pycache__" not in p.parts
    )
    total = sum(p.stat().st_size for p in files)
    if MAX_BUNDLE_GB is not None and total > MAX_BUNDLE_GB * 1e9:
        print(f"\nBundle is {total / 1e9:.1f} GB, over the {MAX_BUNDLE_GB} GB limit. "
              "Host the checkpoint separately and pass --exclude-checkpoint.")
        return 1

    lines = [f"{res.file_sha256(p)}  {p.relative_to(sub).as_posix()}" for p in files]
    (sub / MANIFEST).write_text("\n".join(lines) + "\n")

    out = args.output or sub.parent / f"{_slug(meta['team_name'])}_{_slug(meta['method_name'])}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for p in [sub / MANIFEST, *files]:
            tar.add(p, arcname=f"submission/{p.relative_to(sub).as_posix()}")

    print(f"\nWrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
    print(f"SHA-256: {res.file_sha256(out)}")
    print("Upload the archive, then paste its link and this SHA-256 into the submission form.")
    return 0


if __name__ == "__main__":
    ensure_pinned("roboeval.challenge.package", sys.argv[1:])
    sys.exit(main())
