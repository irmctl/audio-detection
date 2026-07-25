"""Prepare 500-sample (250 real / 250 fake) manifests for ASVspoof, WaveFake, FakeAVCeleb."""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

TARGET_N = 500  # 250 real + 250 fake
SR = 16000


def decode_hf_audio(audio_obj) -> tuple[np.ndarray, int]:
    """Decode HF Audio feature with or without auto-decode."""
    if audio_obj is None:
        raise ValueError("empty audio")
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            arr = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj.get("sampling_rate", SR))
            return arr, sr
        raw = audio_obj.get("bytes")
        if raw is not None:
            import io

            arr, sr = sf.read(io.BytesIO(raw), always_2d=False)
            return np.asarray(arr, dtype=np.float32), int(sr)
        path = audio_obj.get("path")
        if path and Path(path).exists():
            arr, sr = sf.read(path, always_2d=False)
            return np.asarray(arr, dtype=np.float32), int(sr)
    raise ValueError(f"unsupported audio object: {type(audio_obj)}")


def save_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        if audio.shape[0] <= 8 and audio.shape[1] > 8:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        sr = SR
    max_len = SR * 4
    if audio.shape[0] > max_len:
        audio = audio[:max_len]
    sf.write(str(path), audio, sr)


def write_manifest(out: Path, dataset: str, items: list[dict], seed: int) -> None:
    rng = random.Random(seed)
    by = {"real": [], "fake": []}
    for it in items:
        by[it["label"]].append(it)
    for k in by:
        rng.shuffle(by[k])

    n_each = TARGET_N // 2
    selected = by["real"][:n_each] + by["fake"][:n_each]
    if len(by["real"]) < n_each or len(by["fake"]) < n_each:
        raise RuntimeError(
            f"{dataset}: need {n_each} per class, got real={len(by['real'])} fake={len(by['fake'])}"
        )

    rng.shuffle(selected)
    # 70/15/15 stratified
    splits = {"train": [], "val": [], "test": []}
    for label in ("real", "fake"):
        group = [x for x in selected if x["label"] == label]
        n = len(group)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        for i, it in enumerate(group):
            it = dict(it)
            if i < n_train:
                it["split"] = "train"
            elif i < n_train + n_val:
                it["split"] = "val"
            else:
                it["split"] = "test"
            splits[it["split"]].append(it)

    all_items = splits["train"] + splits["val"] + splits["test"]
    rng.shuffle(all_items)
    payload = {
        "dataset": dataset,
        "n_items": len(all_items),
        "target_n": TARGET_N,
        "seed": seed,
        "counts": {
            f"{it['split']}/{it['label']}": sum(
                1 for x in all_items if x["split"] == it["split"] and x["label"] == it["label"]
            )
            for it in all_items
        },
        "items": all_items,
    }
    # compact counts
    counts: dict[str, int] = {}
    for it in all_items:
        key = f"{it['split']}/{it['label']}"
        counts[key] = counts.get(key, 0) + 1
    payload["counts"] = counts

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({len(all_items)} items) {counts}")


def prepare_asvspoof(root: Path, seed: int) -> Path:
    out_audio = root / "asvspoof" / "wav"
    manifest = root / "manifests" / "asvspoof_500.json"
    if manifest.exists():
        print(f"Skip ASVspoof (exists): {manifest}")
        return manifest

    print("Loading ASVspoof2019_LA from HuggingFace (streaming sample)...")
    ds = load_dataset(
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
        split="test",
        streaming=True,
    )
    # Prefer path/bytes + soundfile to avoid torchcodec issues on Windows
    ds = ds.cast_column("audio", Audio(sampling_rate=SR, decode=False))

    items: list[dict] = []
    n_real = n_fake = 0
    need = TARGET_N // 2
    for i, row in enumerate(tqdm(ds, desc="asvspoof", total=need * 4)):
        lab = row["label"]
        # ClassLabel: bonafide=0, spoof=1 (or string)
        if isinstance(lab, str):
            is_fake = lab.lower() in {"spoof", "fake", "1"}
        else:
            is_fake = int(lab) == 1
        label = "fake" if is_fake else "real"
        if label == "real" and n_real >= need:
            continue
        if label == "fake" and n_fake >= need:
            continue
        try:
            audio, sr = decode_hf_audio(row["audio"])
        except Exception:
            continue
        fname = f"{label}_{len(items):04d}.wav"
        path = out_audio / fname
        save_wav(path, audio, sr)
        items.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "label_id": 1 if label == "fake" else 0,
                "source": "ASVspoof2019_LA",
            }
        )
        if label == "real":
            n_real += 1
        else:
            n_fake += 1
        if n_real >= need and n_fake >= need:
            break

    write_manifest(manifest, "asvspoof", items, seed)
    return manifest


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already downloaded: {dest}")
        return
    print(f"Downloading {url} -> {dest}")

    def hook(bn, bs, ts):
        if ts > 0:
            print(f"\r  {bn * bs / ts * 100:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def prepare_wavefake(root: Path, seed: int) -> Path:
    """WaveFake-style 500 set.

    Official WaveFake (~29GB) / LJSpeech script datasets are unreliable here.
    Fallback: stratified sample from local FoR for-2sec (real+fake), tagged as proxy.
    """
    out_audio = root / "wavefake" / "wav"
    manifest = root / "manifests" / "wavefake_500.json"
    if manifest.exists():
        print(f"Skip WaveFake (exists): {manifest}")
        return manifest

    for_root = Path("data/for-2sec/for-2seconds")
    if not for_root.exists():
        raise RuntimeError("FoR not found for WaveFake proxy. Run scripts/download_for.py first.")

    need = TARGET_N // 2
    items: list[dict] = []
    rng = random.Random(seed + 11)

    for label, split_name in (("real", "training"), ("fake", "training")):
        src = for_root / split_name / label
        files = sorted(src.glob("*.wav"))
        rng.shuffle(files)
        files = files[:need]
        if len(files) < need:
            raise RuntimeError(f"WaveFake proxy: need {need} {label}, found {len(files)}")
        for i, src_path in enumerate(tqdm(files, desc=f"wavefake-{label}")):
            audio, sr = sf.read(str(src_path), always_2d=False)
            path = out_audio / f"{label}_{i:04d}.wav"
            save_wav(path, audio, sr)
            items.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "label_id": 1 if label == "fake" else 0,
                    "source": "FoR-for-2sec-proxy-for-WaveFake",
                }
            )

    write_manifest(manifest, "wavefake", items, seed)
    # annotate proxy
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["proxy"] = True
    payload["proxy_note"] = (
        "Official WaveFake/LJSpeech download failed; used FoR for-2sec subset as proxy."
    )
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def prepare_fakeavceleb(root: Path, seed: int) -> Path:
    """FakeAVCeleb requires a request form; use FoR proxy if mirrors fail."""
    out_audio = root / "fakeavceleb" / "wav"
    manifest = root / "manifests" / "fakeavceleb_500.json"
    if manifest.exists():
        print(f"Skip FakeAVCeleb (exists): {manifest}")
        return manifest

    # try HF mirrors first (best effort)
    need = TARGET_N // 2
    items: list[dict] = []
    candidates = [
        "thenewsupercell/fixed-fakeavceleb",
    ]
    for name in candidates:
        print(f"Trying FakeAVCeleb HF source: {name}")
        try:
            ds = load_dataset(name, split="train", streaming=True)
            try:
                ds = ds.cast_column("audio", Audio(decode=False))
            except Exception:
                pass
            n_real = n_fake = 0
            items = []
            for row in tqdm(ds, desc=f"fav-{name.split('/')[-1]}", total=need * 2):
                label = None
                for k in ("label", "labels", "class", "is_fake", "category"):
                    if k not in row:
                        continue
                    v = row[k]
                    if isinstance(v, bool):
                        label = "fake" if v else "real"
                    elif isinstance(v, (int, np.integer)):
                        label = "fake" if int(v) == 1 else "real"
                    elif isinstance(v, str):
                        s = v.lower()
                        if any(x in s for x in ("fake", "spoof", "deepfake")):
                            label = "fake"
                        elif any(x in s for x in ("real", "bona", "genuine", "original")):
                            label = "real"
                    break
                if label is None:
                    continue
                if label == "real" and n_real >= need:
                    continue
                if label == "fake" and n_fake >= need:
                    continue
                try:
                    audio, sr = decode_hf_audio(row.get("audio"))
                except Exception:
                    continue
                path = out_audio / f"{label}_{len(items):04d}.wav"
                save_wav(path, audio, sr)
                items.append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "label_id": 1 if label == "fake" else 0,
                        "source": name,
                    }
                )
                if label == "real":
                    n_real += 1
                else:
                    n_fake += 1
                if n_real >= need and n_fake >= need:
                    write_manifest(manifest, "fakeavceleb", items, seed)
                    return manifest
        except Exception as e:
            print("  failed:", e)

    print("Falling back to FoR proxy for FakeAVCeleb...")
    for_root = Path("data/for-2sec/for-2seconds")
    if not for_root.exists():
        raise RuntimeError("FoR not found for FakeAVCeleb proxy.")

    rng = random.Random(seed + 29)
    items = []
    for label in ("real", "fake"):
        src = for_root / "testing" / label
        files = sorted(src.glob("*.wav"))
        rng.shuffle(files)
        files = files[:need]
        for i, src_path in enumerate(tqdm(files, desc=f"fav-proxy-{label}")):
            audio, sr = sf.read(str(src_path), always_2d=False)
            path = out_audio / f"{label}_{i:04d}.wav"
            save_wav(path, audio, sr)
            items.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "label_id": 1 if label == "fake" else 0,
                    "source": "FoR-for-2sec-proxy-for-FakeAVCeleb",
                }
            )
    write_manifest(manifest, "fakeavceleb", items, seed)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["proxy"] = True
    payload["proxy_note"] = (
        "Official FakeAVCeleb requires access form; used FoR for-2sec testing split as proxy."
    )
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def prepare_fakeavceleb_from_folders(root: Path, seed: int) -> Path | None:
    """If user already placed real/fake wav folders, build manifest."""
    base = root / "fakeavceleb" / "wav"
    manifest = root / "manifests" / "fakeavceleb_500.json"
    if manifest.exists():
        return manifest
    real_dir, fake_dir = base / "real", base / "fake"
    if not (real_dir.is_dir() and fake_dir.is_dir()):
        return None
    items = []
    for label, d in (("real", real_dir), ("fake", fake_dir)):
        for p in sorted(d.rglob("*.wav"))[: TARGET_N // 2]:
            items.append(
                {
                    "path": str(p.resolve()),
                    "label": label,
                    "label_id": 1 if label == "fake" else 0,
                    "source": "local",
                }
            )
    if len(items) < TARGET_N:
        return None
    write_manifest(manifest, "fakeavceleb", items, seed)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/eval"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        choices=["asvspoof", "wavefake", "fakeavceleb", "all"],
        default="all",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    if args.only in ("asvspoof", "all"):
        try:
            prepare_asvspoof(args.root, args.seed)
        except Exception as e:
            errors.append(f"asvspoof: {e}")
            print("ASVSPOOF FAILED:", e)
    if args.only in ("wavefake", "all"):
        try:
            prepare_wavefake(args.root, args.seed)
        except Exception as e:
            errors.append(f"wavefake: {e}")
            print("WAVEFAKE FAILED:", e)
    if args.only in ("fakeavceleb", "all"):
        try:
            m = prepare_fakeavceleb_from_folders(args.root, args.seed)
            if m is None:
                prepare_fakeavceleb(args.root, args.seed)
        except Exception as e:
            errors.append(f"fakeavceleb: {e}")
            print("FAKEAVCELEB FAILED:", e)

    if errors:
        print("Completed with errors:")
        for e in errors:
            print(" -", e)
        # exit non-zero only if nothing usable
        manifests = list((args.root / "manifests").glob("*_500.json"))
        if not manifests:
            raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()
