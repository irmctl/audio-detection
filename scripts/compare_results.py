"""Merge Whisper+MFCC and Wav2Vec2 probe results into one comparison JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare probe JSON results")
    parser.add_argument(
        "--whisper",
        type=Path,
        default=Path("results/whisper_mfcc_linear.json"),
    )
    parser.add_argument(
        "--wav2vec2",
        type=Path,
        default=Path("results/wav2vec2_linear.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/comparison.json"),
    )
    args = parser.parse_args()

    w = load(args.whisper)
    v = load(args.wav2vec2)

    def pick(exp: dict) -> dict:
        return {
            "experiment": exp["experiment"],
            "model": exp["model"],
            "val": exp["metrics"]["val"],
            "test": exp["metrics"]["test"],
        }

    rows = [pick(w), pick(v)]
    best = max(
        rows,
        key=lambda r: (
            r["test"].get("roc_auc") or 0.0,
            r["test"].get("f1") or 0.0,
        ),
    )

    payload = {
        "dataset": "for-2sec",
        "selection_rule": "highest test roc_auc, then f1",
        "experiments": rows,
        "recommended": {
            "experiment": best["experiment"],
            "model": best["model"],
            "test_roc_auc": best["test"].get("roc_auc"),
            "test_f1": best["test"].get("f1"),
            "test_accuracy": best["test"].get("accuracy"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Recommended: {best['experiment']} ({best['model']})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
