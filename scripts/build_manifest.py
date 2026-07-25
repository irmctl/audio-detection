"""Build a JSON manifest from FoR for-2sec folder layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
SPLIT_ALIASES = {
    "training": "train",
    "train": "train",
    "validation": "val",
    "val": "val",
    "valid": "val",
    "testing": "test",
    "test": "test",
}
LABEL_ALIASES = {
    "real": "real",
    "genuine": "real",
    "human": "real",
    "fake": "fake",
    "spoof": "fake",
    "synthetic": "fake",
}


def discover_root(data_dir: Path) -> Path:
    """Find directory that contains train/val/test style splits."""
    candidates = [data_dir]
    if (data_dir / "for-2sec").exists():
        candidates.append(data_dir / "for-2sec")
    # after extract, sometimes nested once more
    for c in list(candidates):
        for child in c.iterdir() if c.exists() else []:
            if child.is_dir():
                candidates.append(child)

    for c in candidates:
        names = {p.name.lower() for p in c.iterdir()} if c.exists() else set()
        if names & set(SPLIT_ALIASES):
            return c
        if {"real", "fake"} <= names:
            return c
    raise FileNotFoundError(
        f"Could not locate FoR splits under {data_dir}. "
        "Expected training/validation/testing with real/fake."
    )


def iter_labeled_files(root: Path) -> list[dict]:
    items: list[dict] = []

    # Case A: root/{split}/{label}/*.wav
    split_dirs = [
        p for p in root.iterdir() if p.is_dir() and p.name.lower() in SPLIT_ALIASES
    ]
    if split_dirs:
        for split_dir in sorted(split_dirs):
            split = SPLIT_ALIASES[split_dir.name.lower()]
            for label_dir in sorted(split_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                key = label_dir.name.lower()
                if key not in LABEL_ALIASES:
                    continue
                label = LABEL_ALIASES[key]
                for audio in sorted(label_dir.rglob("*")):
                    if audio.suffix.lower() in AUDIO_EXTS:
                        items.append(
                            {
                                "path": str(audio.resolve()),
                                "label": label,
                                "label_id": 0 if label == "real" else 1,
                                "split": split,
                            }
                        )
        return items

    # Case B: root/{label}/*.wav — create random-free deterministic split later
    label_dirs = [
        p for p in root.iterdir() if p.is_dir() and p.name.lower() in LABEL_ALIASES
    ]
    if label_dirs:
        for label_dir in sorted(label_dirs):
            label = LABEL_ALIASES[label_dir.name.lower()]
            for audio in sorted(label_dir.rglob("*")):
                if audio.suffix.lower() in AUDIO_EXTS:
                    items.append(
                        {
                            "path": str(audio.resolve()),
                            "label": label,
                            "label_id": 0 if label == "real" else 1,
                            "split": "unassigned",
                        }
                    )
        return items

    raise FileNotFoundError(f"No recognizable label/split layout in {root}")


def assign_splits(
    items: list[dict],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> list[dict]:
    import random

    if not items or items[0]["split"] != "unassigned":
        return items

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {"real": [], "fake": []}
    for it in items:
        by_label[it["label"]].append(it)

    out: list[dict] = []
    for label, group in by_label.items():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, it in enumerate(group):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            it = dict(it)
            it["split"] = split
            out.append(it)
    return out


def summarize(items: list[dict]) -> dict:
    summary: dict = {}
    for it in items:
        key = f"{it['split']}/{it['label']}"
        summary[key] = summary.get(key, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FoR JSON manifest")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/for-2sec"),
        help="Extracted FoR directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/manifest.json"),
        help="Output manifest path",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = discover_root(args.data_dir)
    print(f"Using dataset root: {root}")
    items = assign_splits(iter_labeled_files(root), seed=args.seed)
    if not items:
        raise SystemExit("No audio files found.")

    payload = {
        "dataset": "for-2sec",
        "root": str(root.resolve()),
        "n_items": len(items),
        "counts": summarize(items),
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(items)} items -> {args.out}")
    for k, v in sorted(payload["counts"].items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
