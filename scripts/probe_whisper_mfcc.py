"""Whisper encoder + MFCC features with logistic regression linear probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

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
from audio_probe.probe import fit_linear_probe, predict_probe  # noqa: E402


def extract_mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.concatenate([mfcc, delta, delta2], axis=0)
    return np.concatenate([feats.mean(axis=1), feats.std(axis=1)]).astype(np.float32)


@torch.inference_mode()
def extract_whisper(
    audio: np.ndarray,
    processor: WhisperProcessor,
    model: WhisperModel,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)
    outputs = model.encoder(input_features)
    hidden = outputs.last_hidden_state.squeeze(0)  # [T, D]
    return hidden.mean(dim=0).float().cpu().numpy()


def extract_features(
    items: list[dict],
    processor: WhisperProcessor,
    model: WhisperModel,
    device: torch.device,
    n_mfcc: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for it in tqdm(items, desc="features"):
        audio = load_audio(it["path"], target_sr=16000)
        if audio.size == 0:
            continue
        w = extract_whisper(audio, processor, model, device)
        m = extract_mfcc(audio, sr=16000, n_mfcc=n_mfcc)
        xs.append(np.concatenate([w, m]))
        ys.append(int(it["label_id"]))
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper + MFCC linear probe")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--model", default="openai/whisper-base")
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/whisper_mfcc_linear.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    items = load_manifest(args.manifest)
    train_items = subsample_stratified(
        split_items(items, "train"), args.max_train, seed=args.seed
    )
    val_items = subsample_stratified(
        split_items(items, "val"), args.max_val, seed=args.seed + 1
    )
    test_items = subsample_stratified(
        split_items(items, "test"), args.max_test, seed=args.seed + 2
    )
    print(f"train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperModel.from_pretrained(args.model).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    x_train, y_train = extract_features(
        train_items, processor, model, device, args.n_mfcc
    )
    x_val, y_val = extract_features(
        val_items, processor, model, device, args.n_mfcc
    )
    x_test, y_test = extract_features(
        test_items, processor, model, device, args.n_mfcc
    )

    clf = fit_linear_probe(x_train, y_train, seed=args.seed)

    val_pred, val_prob = predict_probe(clf, x_val)
    test_pred, test_prob = predict_probe(clf, x_test)
    val_metrics = compute_metrics(y_val, val_pred, val_prob)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)

    payload = {
        "experiment": "whisper_mfcc_linear",
        "dataset": "for-2sec",
        "model": args.model,
        "n_mfcc": args.n_mfcc,
        "probe": "logistic_regression",
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
