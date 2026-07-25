"""Download and extract the FoR for-2sec dataset."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

FOR_2SEC_URL = "https://bil.eecs.yorku.ca/share/for-2sec.tar.gz"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {pct:5.1f}%  ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Archive already exists: {dest}")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"Already extracted under: {out_dir}")
        return
    print(f"Extracting {archive.name} -> {out_dir}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=out_dir)
    marker.write_text(archive.name, encoding="utf-8")
    print("Done.")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FoR for-2sec dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--url",
        default=FOR_2SEC_URL,
        help="Dataset archive URL",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only download, do not extract",
    )
    args = parser.parse_args()

    archive = args.data_dir / "for-2sec.tar.gz"
    extract_dir = args.data_dir / "for-2sec"

    download(args.url, archive)
    print(f"SHA256: {sha256_file(archive)}")
    if not args.skip_extract:
        extract(archive, extract_dir)
        print(f"Dataset ready at: {extract_dir.resolve()}")


if __name__ == "__main__":
    main()
