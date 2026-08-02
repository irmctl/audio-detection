"""Merge Whisper+MFCC and Wav2Vec2 probe results into one comparison JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pick_scores(exp: dict) -> dict:
    """Normalize CV or hold-out metrics for comparison."""
    metrics = exp.get("metrics", {})
    if "cv" in metrics:
        cv = metrics["cv"]
        return {
            "experiment": exp["experiment"],
            "model": exp["model"],
            "protocol": exp.get("protocol", "cv"),
            "cv": {
                "mean": cv["mean"],
                "std": cv["std"],
                "n_splits": cv.get("n_splits"),
            },
            "score_roc_auc": cv["mean"].get("roc_auc"),
            "score_f1": cv["mean"].get("f1"),
            "score_accuracy": cv["mean"].get("accuracy"),
        }
    return {
        "experiment": exp["experiment"],
        "model": exp["model"],
        "protocol": exp.get("protocol", "holdout"),
        "val": metrics.get("val"),
        "test": metrics.get("test"),
        "score_roc_auc": (metrics.get("test") or {}).get("roc_auc"),
        "score_f1": (metrics.get("test") or {}).get("f1"),
        "score_accuracy": (metrics.get("test") or {}).get("accuracy"),
    }


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

    rows = [pick_scores(w), pick_scores(v)]
    best = max(
        rows,
        key=lambda r: (
            r.get("score_roc_auc") or 0.0,
            r.get("score_f1") or 0.0,
        ),
    )

    payload = {
        "dataset": "for-2sec",
        "selection_rule": "highest CV mean (or test) roc_auc, then f1",
        "experiments": rows,
        "recommended": {
            "experiment": best["experiment"],
            "model": best["model"],
            "protocol": best.get("protocol"),
            "roc_auc": best.get("score_roc_auc"),
            "f1": best.get("score_f1"),
            "accuracy": best.get("score_accuracy"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Recommended: {best['experiment']} ({best['model']})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
