"""Run Whisper / Wav2Vec2 / WavLM / AASIST / Qwen2-Audio probes on eval manifests.

Outputs per-dataset and summary JSON under results/multi/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_probe.extractors import build_extractor  # noqa: E402
from audio_probe.io_utils import load_audio, load_manifest, save_json, set_seed  # noqa: E402
from audio_probe.metrics import compute_metrics  # noqa: E402
from audio_probe.probe import fit_linear_probe, predict_probe  # noqa: E402

# Slowest model (Qwen2-Audio) last
MODELS = ["whisper", "wav2vec2", "wavlm", "aasist", "qwen2_audio"]
DATASETS = ["asvspoof", "wavefake", "fakeavceleb"]
N_SAMPLES = 500


def extract_matrix(extractor, items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for it in tqdm(items, desc=f"{extractor.name}"):
        audio = load_audio(it["path"], target_sr=16000)
        if audio.size == 0:
            continue
        feat = extractor.extract(audio, sr=16000)
        xs.append(feat.astype(np.float32))
        ys.append(int(it["label_id"]))
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def run_one(
    dataset: str,
    model: str,
    manifest_path: Path,
    out_dir: Path,
    device: str | None,
    seed: int,
) -> dict:
    out_path = out_dir / f"{dataset}__{model}.json"
    if out_path.exists():
        print(f"Skip existing {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    items = load_manifest(manifest_path)
    train = [x for x in items if x["split"] == "train"]
    val = [x for x in items if x["split"] == "val"]
    test = [x for x in items if x["split"] == "test"]

    t0 = time.time()
    try:
        extractor = build_extractor(model, device=device)
        x_train, y_train = extract_matrix(extractor, train)
        x_val, y_val = extract_matrix(extractor, val)
        x_test, y_test = extract_matrix(extractor, test)
        clf = fit_linear_probe(x_train, y_train, seed=seed)
        val_pred, val_prob = predict_probe(clf, x_val)
        test_pred, test_prob = predict_probe(clf, x_test)
        payload = {
            "dataset": dataset,
            "model": model,
            "status": "ok",
            "seed": seed,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "feature_dim": int(x_train.shape[1]),
            "elapsed_sec": round(time.time() - t0, 1),
            "metrics": {
                "val": compute_metrics(y_val, val_pred, val_prob),
                "test": compute_metrics(y_test, test_pred, test_prob),
            },
        }
    except Exception as e:
        payload = {
            "dataset": dataset,
            "model": model,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(f"ERROR {dataset}/{model}: {e}")

    save_json(out_path, payload)
    return payload


def summarize(results: list[dict], out_path: Path) -> dict:
    rows = []
    for r in results:
        if r.get("status") != "ok":
            rows.append(
                {
                    "dataset": r["dataset"],
                    "model": r["model"],
                    "status": "error",
                    "error": r.get("error"),
                }
            )
            continue
        t = r["metrics"]["test"]
        rows.append(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "status": "ok",
                "test_accuracy": t["accuracy"],
                "test_f1": t["f1"],
                "test_roc_auc": t["roc_auc"],
                "elapsed_sec": r.get("elapsed_sec"),
            }
        )

    # best model per dataset by test roc_auc then f1
    best = {}
    for ds in DATASETS:
        ok = [x for x in rows if x["dataset"] == ds and x.get("status") == "ok"]
        if not ok:
            continue
        best[ds] = max(
            ok,
            key=lambda x: (x.get("test_roc_auc") or 0.0, x.get("test_f1") or 0.0),
        )

    payload = {
        "protocol": "linear_probe",
        "n_per_dataset": N_SAMPLES,
        "models": MODELS,
        "datasets": DATASETS,
        "results": rows,
        "best_per_dataset": best,
    }
    save_json(out_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/eval/manifests"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/multi"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ds in args.datasets:
        manifest = args.manifest_dir / f"{ds}_{N_SAMPLES}.json"
        if not manifest.exists():
            print(f"Missing manifest: {manifest}")
            results.append(
                {
                    "dataset": ds,
                    "model": "*",
                    "status": "error",
                    "error": f"missing manifest {manifest}",
                }
            )
            continue
        for model in args.models:
            print(f"\n=== {ds} / {model} ===")
            results.append(
                run_one(ds, model, manifest, args.out_dir, args.device, args.seed)
            )

    summary = summarize(
        [r for r in results if r.get("model") != "*"],
        args.out_dir / "summary.json",
    )
    print("\nSUMMARY")
    for row in summary["results"]:
        if row.get("status") != "ok":
            print(f"  {row['dataset']:12} {row['model']:12} ERROR")
        else:
            print(
                f"  {row['dataset']:12} {row['model']:12} "
                f"acc={row['test_accuracy']:.3f} f1={row['test_f1']:.3f} "
                f"auc={row['test_roc_auc']:.3f}"
            )


if __name__ == "__main__":
    main()
