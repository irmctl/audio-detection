from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


LABEL_MAP = {"real": 0, "fake": 1}
INV_LABEL_MAP = {0: "real", 1: "fake"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_audio(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Load mono float32 audio resampled to target_sr."""
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def split_items(
    items: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    return [x for x in items if x["split"] == split]


def subsample_stratified(
    items: list[dict[str, Any]],
    max_items: int | None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Keep class balance when limiting the number of examples."""
    if max_items is None or max_items >= len(items):
        return items

    rng = random.Random(seed)
    by_label: dict[Any, list[dict[str, Any]]] = {}
    for it in items:
        by_label.setdefault(it["label_id"], []).append(it)

    labels = sorted(by_label)
    per_class = max(1, max_items // max(1, len(labels)))
    out: list[dict[str, Any]] = []
    for label in labels:
        group = list(by_label[label])
        rng.shuffle(group)
        out.extend(group[:per_class])

    rng.shuffle(out)
    return out[:max_items]
