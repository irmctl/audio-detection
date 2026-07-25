"""Download Qwen2-Audio-7B via curl (more reliable than HF hub Xet on this network)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "Qwen/Qwen2-Audio-7B"
OUT = Path("data/models/qwen2-audio-7b")


def curl_download(filename: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / filename
    url = f"https://huggingface.co/{REPO}/resolve/main/{filename}"
    cmd = [
        "curl.exe",
        "-L",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "3",
        "-C",
        "-",
        "--connect-timeout",
        "30",
        "-o",
        str(dest),
        url,
    ]
    print(f"Downloading {filename} -> {dest}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"curl failed for {filename}: {proc.returncode}")
    print(f"OK {filename} size={dest.stat().st_size} sec={time.time()-t0:.1f}", flush=True)
    return dest


def main() -> None:
    files = list_repo_files(REPO)
    small = [f for f in files if not f.endswith(".safetensors") or f.endswith("index.json")]
    shards = sorted(f for f in files if f.startswith("model-") and f.endswith(".safetensors"))

    # small files via hub (fast)
    for f in small:
        if f.startswith("."):
            continue
        print("hub", f, flush=True)
        p = hf_hub_download(REPO, f)
        target = OUT / Path(f).name
        if not target.exists():
            target.write_bytes(Path(p).read_bytes())

    for f in shards:
        dest = OUT / f
        # skip if looks complete (>100MB and not tiny)
        if dest.exists() and dest.stat().st_size > 100_000_000:
            print(f"skip existing {f} ({dest.stat().st_size})", flush=True)
            continue
        curl_download(f)

    # marker for transformers local path
    meta = {"repo": REPO, "local_dir": str(OUT.resolve()), "files": sorted(p.name for p in OUT.iterdir())}
    (OUT / "download_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("ALL DONE", OUT.resolve(), flush=True)


if __name__ == "__main__":
    main()
