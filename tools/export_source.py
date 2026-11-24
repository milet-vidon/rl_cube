#!/usr/bin/env python3
"""Build a history-free, source-only review archive from the current checkout."""
import argparse
import gzip
import io
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("tools/source-files.txt")
PACKAGE = "rl-cube-source"
MAX_FILE = 1024 * 1024
MAX_TOTAL = 10 * MAX_FILE
CREDENTIAL = re.compile(
    rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|(?:AKIA|ASIA)[A-Z0-9]{16}"
    rb"|github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}"
    rb"|https?://[^/\s:@]+:[^/\s@]+@"
)


def load_files():
    """Read only explicitly listed local text files; never walk a research tree."""
    files = {}
    total = 0
    for name in (ROOT / MANIFEST).read_text().splitlines():
        if not name or name.startswith("#"):
            continue
        rel = Path(name)
        path = ROOT / rel
        if (rel.is_absolute() or ".." in rel.parts or name != rel.as_posix()
                or path.resolve() != path or not path.is_file()):
            raise ValueError(f"Missing, linked or invalid source path: {name}")
        if path.stat().st_size > MAX_FILE:
            raise ValueError(f"Oversized source file: {name}")
        data = path.read_bytes()
        data.decode("utf-8")
        if b"\x00" in data or CREDENTIAL.search(data) or data.startswith(b"version https://git-lfs."):
            raise ValueError(f"Binary, credential marker or LFS pointer: {name}")
        if name in files:
            raise ValueError(f"Duplicate release entry: {name}")
        files[name] = data
        total += len(data)
        if total > MAX_TOTAL:
            raise ValueError("Review source inventory: package exceeds 10 MiB")
    files["README.md"] = files["docs/SOURCE_PACKAGE.md"]
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="new .tar.gz file (never overwritten)")
    out = parser.parse_args().archive
    files = load_files()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as raw:
        try:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                    for name, data in sorted(files.items()):
                        info = tarfile.TarInfo(f"{PACKAGE}/{name}")
                        info.size = len(data)
                        info.mode = 0o755 if name.endswith(".sh") else 0o644
                        tar.addfile(info, io.BytesIO(data))
        except BaseException:
            out.unlink()
            raise
    print(f"{out}: {len(files)} files; {out.stat().st_size:,} bytes compressed")


if __name__ == "__main__":
    main()
