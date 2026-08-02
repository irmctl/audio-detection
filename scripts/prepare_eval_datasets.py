"""Prepare 500-sample (250 real / 250 fake) manifests for original datasets.

Supported:
  - asvspoof: HuggingFace ASVspoof2019_LA
  - wavefake: HuggingFace ajaykarthick/wavefake-audio (R vs WFx)
  - for: local FoR for-2sec
  - fakeavceleb: local official FakeAVCeleb tree (form-gated; no FoR proxy)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

TARGET_N = 500  # 250 real + 250 fake
SR = 16000
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


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


def write_manifest(
    out: Path,
    dataset: str,
    items: list[dict],
    seed: int,
    *,
    source_note: str | None = None,
    proxy: bool = False,
) -> None:
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

    # Keep split tags for optional hold-out; CV pools all items.
    all_items: list[dict] = []
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
            all_items.append(it)

    rng.shuffle(all_items)
    counts: dict[str, int] = {}
    for it in all_items:
        key = f"{it['split']}/{it['label']}"
        counts[key] = counts.get(key, 0) + 1

    payload: dict = {
        "dataset": dataset,
        "n_items": len(all_items),
        "target_n": TARGET_N,
        "seed": seed,
        "proxy": proxy,
        "counts": counts,
        "items": all_items,
    }
    if source_note:
        payload["source_note"] = source_note

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({len(all_items)} items) {counts}")


def prepare_asvspoof(root: Path, seed: int, force: bool = False) -> Path:
    out_audio = root / "asvspoof" / "wav"
    manifest = root / "manifests" / "asvspoof_500.json"
    if manifest.exists() and not force:
        print(f"Skip ASVspoof (exists): {manifest}")
        return manifest

    print("Loading ASVspoof2019_LA from HuggingFace (streaming sample)...")
    ds = load_dataset(
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
        split="test",
        streaming=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=SR, decode=False))

    items: list[dict] = []
    n_real = n_fake = 0
    need = TARGET_N // 2
    for row in tqdm(ds, desc="asvspoof", total=need * 4):
        lab = row["label"]
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
        path = out_audio / f"{label}_{len(items):04d}.wav"
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

    write_manifest(
        manifest,
        "asvspoof",
        items,
        seed,
        source_note="Original ASVspoof2019 LA (HF SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA)",
        proxy=False,
    )
    return manifest


def prepare_wavefake(root: Path, seed: int, force: bool = False) -> Path:
    """Original WaveFake via HF mirror (R=real LJSpeech refs, WFx=generated)."""
    out_audio = root / "wavefake" / "wav"
    manifest = root / "manifests" / "wavefake_500.json"
    if manifest.exists() and not force:
        print(f"Skip WaveFake (exists): {manifest}")
        return manifest

    print("Loading original WaveFake from HuggingFace (streaming)...")
    ds = load_dataset("ajaykarthick/wavefake-audio", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR, decode=False))

    need = TARGET_N // 2
    items: list[dict] = []
    n_real = n_fake = 0
    for row in tqdm(ds, desc="wavefake", total=need * 8):
        tag = str(row.get("real_or_fake", "")).upper()
        if tag == "R":
            label = "real"
        elif tag.startswith("WF"):
            label = "fake"
        else:
            continue
        if label == "real" and n_real >= need:
            continue
        if label == "fake" and n_fake >= need:
            continue
        try:
            audio, sr = decode_hf_audio(row["audio"])
        except Exception:
            continue
        path = out_audio / f"{label}_{len(items):04d}.wav"
        save_wav(path, audio, sr)
        items.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "label_id": 1 if label == "fake" else 0,
                "source": "ajaykarthick/wavefake-audio",
                "wavefake_tag": tag,
            }
        )
        if label == "real":
            n_real += 1
        else:
            n_fake += 1
        if n_real >= need and n_fake >= need:
            break

    write_manifest(
        manifest,
        "wavefake",
        items,
        seed,
        source_note=(
            "Original WaveFake HF mirror ajaykarthick/wavefake-audio "
            "(R=real, WFx=synthetic generators)"
        ),
        proxy=False,
    )
    return manifest


def prepare_for(root: Path, seed: int, force: bool = False) -> Path:
    """Original FoR for-2sec local archive."""
    out_audio = root / "for" / "wav"
    manifest = root / "manifests" / "for_500.json"
    if manifest.exists() and not force:
        print(f"Skip FoR (exists): {manifest}")
        return manifest

    for_root = Path("data/for-2sec/for-2seconds")
    if not for_root.exists():
        raise RuntimeError("FoR not found. Run: uv run python scripts/download_for.py")

    need = TARGET_N // 2
    rng = random.Random(seed + 7)
    items: list[dict] = []
    # Prefer training split for larger pool; still original FoR.
    for label in ("real", "fake"):
        src = for_root / "training" / label
        files = sorted(src.glob("*.wav"))
        if len(files) < need:
            # fallback: pool all splits
            files = sorted((for_root).rglob(f"*/{label}/*.wav"))
        rng.shuffle(files)
        files = files[:need]
        if len(files) < need:
            raise RuntimeError(f"FoR: need {need} {label}, found {len(files)}")
        for i, src_path in enumerate(tqdm(files, desc=f"for-{label}")):
            audio, sr = sf.read(str(src_path), always_2d=False)
            path = out_audio / f"{label}_{i:04d}.wav"
            save_wav(path, audio, sr)
            items.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "label_id": 1 if label == "fake" else 0,
                    "source": "FoR-for-2sec",
                }
            )

    write_manifest(
        manifest,
        "for",
        items,
        seed,
        source_note="Original FoR for-2sec (training split sample)",
        proxy=False,
    )
    return manifest


def _collect_audio_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def _extract_audio_from_video(src: Path, dest: Path) -> bool:
    """Best-effort ffmpeg extract; returns True on success."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-f",
        "wav",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False


def prepare_fakeavceleb(
    root: Path,
    seed: int,
    force: bool = False,
    fakeav_root: Path | None = None,
) -> Path:
    """Original FakeAVCeleb from a local download (request-form gated)."""
    out_audio = root / "fakeavceleb" / "wav"
    manifest = root / "manifests" / "fakeavceleb_500.json"
    if manifest.exists() and not force:
        print(f"Skip FakeAVCeleb (exists): {manifest}")
        return manifest

    candidates = []
    if fakeav_root is not None:
        candidates.append(fakeav_root)
    candidates.extend(
        [
            root / "fakeavceleb" / "raw",
            root / "fakeavceleb" / "FakeAVCeleb",
            Path("data/FakeAVCeleb"),
            Path("data/fakeavceleb_raw"),
        ]
    )

    base = next((p for p in candidates if p.exists()), None)
    if base is None:
        raise RuntimeError(
            "Original FakeAVCeleb not found locally. "
            "Fill https://bit.ly/38prlVO then place the dataset under "
            "data/eval/fakeavceleb/raw (RealVideo-RealAudio / FakeVideo-FakeAudio) "
            "and re-run with --only fakeavceleb --force."
        )

    # Discover real/fake folders by common FakeAVCeleb naming.
    real_dirs = [
        p
        for p in base.rglob("*")
        if p.is_dir()
        and p.name.lower() in {"realvideo-realaudio", "real", "genuine", "bonafide"}
    ]
    fake_dirs = [
        p
        for p in base.rglob("*")
        if p.is_dir()
        and (
            p.name.lower() in {"fakevideo-fakeaudio", "fake", "spoof", "synthetic"}
            or "fake" in p.name.lower()
        )
    ]
    # Prefer exact category names when present.
    exact_real = [p for p in real_dirs if p.name == "RealVideo-RealAudio"]
    exact_fake = [p for p in fake_dirs if p.name == "FakeVideo-FakeAudio"]
    if exact_real:
        real_dirs = exact_real
    if exact_fake:
        fake_dirs = exact_fake

    if not real_dirs or not fake_dirs:
        # Flat real/fake wav layout
        real_flat = base / "real"
        fake_flat = base / "fake"
        if real_flat.is_dir() and fake_flat.is_dir():
            real_dirs, fake_dirs = [real_flat], [fake_flat]
        else:
            raise RuntimeError(
                f"Could not find Real/Fake folders under {base}. "
                "Expected RealVideo-RealAudio and FakeVideo-FakeAudio."
            )

    need = TARGET_N // 2
    rng = random.Random(seed + 29)

    def gather(dirs: list[Path]) -> list[Path]:
        files: list[Path] = []
        for d in dirs:
            files.extend(_collect_audio_files(d))
            # also videos if no audio yet
        if len(files) < need:
            vids = []
            for d in dirs:
                for p in d.rglob("*"):
                    if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                        vids.append(p)
            rng.shuffle(vids)
            for v in vids:
                if len(files) >= need * 2:
                    break
                dest = out_audio / "_from_video" / f"{v.stem}.wav"
                if dest.exists() or _extract_audio_from_video(v, dest):
                    files.append(dest)
        rng.shuffle(files)
        return files

    real_files = gather(real_dirs)[:need]
    fake_files = gather(fake_dirs)[:need]
    if len(real_files) < need or len(fake_files) < need:
        raise RuntimeError(
            f"FakeAVCeleb under {base}: need {need} per class, "
            f"got real={len(real_files)} fake={len(fake_files)}"
        )

    items: list[dict] = []
    for label, files in (("real", real_files), ("fake", fake_files)):
        for i, src_path in enumerate(tqdm(files, desc=f"fakeavceleb-{label}")):
            audio, sr = sf.read(str(src_path), always_2d=False)
            path = out_audio / f"{label}_{i:04d}.wav"
            save_wav(path, audio, sr)
            items.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "label_id": 1 if label == "fake" else 0,
                    "source": "FakeAVCeleb-official-local",
                }
            )

    write_manifest(
        manifest,
        "fakeavceleb",
        items,
        seed,
        source_note=f"Original FakeAVCeleb local tree: {base.resolve()}",
        proxy=False,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/eval"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Rebuild even if manifest exists")
    parser.add_argument(
        "--fakeav-root",
        type=Path,
        default=None,
        help="Path to official FakeAVCeleb download root",
    )
    parser.add_argument(
        "--only",
        choices=["asvspoof", "wavefake", "fakeavceleb", "for", "all"],
        default="all",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    if args.only in ("asvspoof", "all"):
        try:
            prepare_asvspoof(args.root, args.seed, force=args.force)
        except Exception as e:
            errors.append(f"asvspoof: {e}")
            print("ASVSPOOF FAILED:", e)
    if args.only in ("wavefake", "all"):
        try:
            prepare_wavefake(args.root, args.seed, force=args.force)
        except Exception as e:
            errors.append(f"wavefake: {e}")
            print("WAVEFAKE FAILED:", e)
    if args.only in ("for", "all"):
        try:
            prepare_for(args.root, args.seed, force=args.force)
        except Exception as e:
            errors.append(f"for: {e}")
            print("FOR FAILED:", e)
    if args.only in ("fakeavceleb", "all"):
        try:
            prepare_fakeavceleb(
                args.root,
                args.seed,
                force=args.force,
                fakeav_root=args.fakeav_root,
            )
        except Exception as e:
            errors.append(f"fakeavceleb: {e}")
            print("FAKEAVCELEB FAILED:", e)

    if errors:
        print("Completed with errors:")
        for e in errors:
            print(" -", e)
        manifests = list((args.root / "manifests").glob("*_500.json"))
        if not manifests:
            raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()
