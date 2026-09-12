#!/usr/bin/env python3
"""Build a complete release inventory after intentional release preparation edits."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version", default="1.1.0")
    args = parser.parse_args()
    root = args.root.resolve()
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or relative.as_posix() == "manifest.json":
            continue
        if any(part in {".git", "__pycache__", ".venv", "outputs"} for part in relative.parts) or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("Release inventories require regular files: " + str(relative))
        entries.append(dict(path=relative.as_posix(), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(schema=1, version=args.version, files=entries)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Inventory written: {len(entries):,} files, version {args.version}")

if __name__ == "__main__":
    main()
