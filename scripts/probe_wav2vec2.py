"""Wav2Vec2 frozen encoder with logistic regression linear probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_probe.io_utils import (  # noqa: E402
    load_audio,
    load_manifest,
    save_json,
    set_seed,
    split_items,
    subsample_stratified,
)
from audio_probe.metrics import compute_metrics  # noqa: E402
from audio_probe.probe import fit_linear_probe, predict_probe, run_stratified_kfold  # noqa: E402


@torch.inference_mode()
def extract_wav2vec2(
    audio: np.ndarray,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    outputs = model(input_values)
    hidden = outputs.last_hidden_state.squeeze(0)  # [T, D]
    return hidden.mean(dim=0).float().cpu().numpy()


def extract_features(
    items: list[dict],
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for it in tqdm(items, desc="features"):
        audio = load_audio(it["path"], target_sr=16000)
        if audio.size == 0:
            continue
        feat = extract_wav2vec2(audio, processor, model, device)
        xs.append(feat)
        ys.append(int(it["label_id"]))
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wav2Vec2 linear probe")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Stratified k-fold CV (default 5). Use 0 for hold-out train/val/test.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="CV mode: stratified subsample of all pooled items",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/wav2vec2_linear.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    items = load_manifest(args.manifest)
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2Model.from_pretrained(args.model).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if args.cv and args.cv >= 2:
        pool = subsample_stratified(items, args.max_samples, seed=args.seed)
        print(f"CV n_splits={args.cv} n_samples={len(pool)}")
        x, y = extract_features(pool, processor, model, device)
        cv = run_stratified_kfold(x, y, n_splits=args.cv, seed=args.seed)
        payload = {
            "experiment": "wav2vec2_linear",
            "dataset": "for-2sec",
            "model": args.model,
            "probe": "logistic_regression",
            "protocol": f"{args.cv}-fold_stratified_cv",
            "seed": args.seed,
            "device": str(device),
            "feature_dim": int(x.shape[1]),
            "n_samples": int(len(y)),
            "metrics": {"cv": cv},
        }
        save_json(args.out, payload)
        m, s = cv["mean"], cv["std"]
        auc = (
            f"{m['roc_auc']:.4f}±{s['roc_auc']:.4f}"
            if m["roc_auc"] is not None
            else "None"
        )
        print(
            f"CV  acc={m['accuracy']:.4f}±{s['accuracy']:.4f} "
            f"f1={m['f1']:.4f}±{s['f1']:.4f} auc={auc}"
        )
        print(f"Wrote {args.out}")
        return

    train_items = subsample_stratified(
        split_items(items, "train"), args.max_train, seed=args.seed
    )
    val_items = subsample_stratified(
        split_items(items, "val"), args.max_val, seed=args.seed + 1
    )
    test_items = subsample_stratified(
        split_items(items, "test"), args.max_test, seed=args.seed + 2
    )
    print(f"hold-out train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    x_train, y_train = extract_features(train_items, processor, model, device)
    x_val, y_val = extract_features(val_items, processor, model, device)
    x_test, y_test = extract_features(test_items, processor, model, device)

    clf = fit_linear_probe(x_train, y_train, seed=args.seed)

    val_pred, val_prob = predict_probe(clf, x_val)
    test_pred, test_prob = predict_probe(clf, x_test)
    val_metrics = compute_metrics(y_val, val_pred, val_prob)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)

    payload = {
        "experiment": "wav2vec2_linear",
        "dataset": "for-2sec",
        "model": args.model,
        "probe": "logistic_regression",
        "protocol": "holdout",
        "seed": args.seed,
        "device": str(device),
        "feature_dim": int(x_train.shape[1]),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
    }
    save_json(args.out, payload)
    print(f"Val  acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics['roc_auc']}")
    print(f"Test acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} auc={test_metrics['roc_auc']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
