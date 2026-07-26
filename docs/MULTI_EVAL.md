# Multi-dataset / multi-model audio deepfake linear-probe eval

## Süre tahmini (CPU, ~1000 örnek/dataset × 3 dataset)

| Aşama | Tahmini süre |
|-------|----------------|
| Veri hazırlama (ASVspoof + LJSpeech + WaveFake + FakeAVCeleb) | **1–3 saat** (indirme hızına bağlı) |
| Whisper | ~15–25 dk |
| Wav2Vec2 | ~8–15 dk |
| WavLM | ~8–15 dk |
| AASIST (ONNX) | ~5–10 dk |
| **Qwen2-Audio-7B** | **3–8+ saat** (en yavaş / RAM riski) |
| **Toplam** | **~5–12 saat** |

GPU (CUDA) varsa Qwen hariç modeller ~3–5× hızlanır; Qwen özellikle GPU ister.

## Çalıştırma

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
cd "c:\Users\fatih\OneDrive\Masaüstü\Tez"

# 1) Veri (1000 örnek/set)
uv run python scripts/prepare_eval_datasets.py

# 2) Tüm modeller
uv run python scripts/run_multi_eval.py

# Qwen'i sona bırakmak için:
uv run python scripts/run_multi_eval.py --models whisper wav2vec2 wavlm aasist
uv run python scripts/run_multi_eval.py --models qwen2_audio
```

Çıktı: `results/multi/{dataset}__{model}.json` + `results/multi/summary.json`

##
