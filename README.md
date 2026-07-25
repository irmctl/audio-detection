# Audio deepfake probe — FoR linear probing

Gerçek vs AI-üretilmiş ses tespiti için baseline linear probe deneyleri.
Paket yönetimi: **uv**.

## Kurulum

```powershell
# PATH'e uv ekle (yeni terminalde gerekebilir)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

cd "c:\Users\fatih\OneDrive\Masaüstü\Tez"
uv sync
```

## Çalıştırma sırası

```powershell
# 1) FoR for-2sec indir (~1GB) ve aç
uv run python scripts/download_for.py

# 2) Manifest (JSON) oluştur
uv run python scripts/build_manifest.py

# 3) Whisper + MFCC → linear probe
uv run python scripts/probe_whisper_mfcc.py

# 4) Wav2Vec2 → linear probe
uv run python scripts/probe_wav2vec2.py

# 5) Karşılaştırma JSON
uv run python scripts/compare_results.py
```

Hızlı duman testi (az örnek):

```powershell
uv run python scripts/probe_whisper_mfcc.py --max-train 200 --max-val 50 --max-test 50
uv run python scripts/probe_wav2vec2.py --max-train 200 --max-val 50 --max-test 50
```

## Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `data/manifest.json` | path / label / split listesi |
| `results/whisper_mfcc_linear.json` | Whisper+MFCC metrikleri |
| `results/wav2vec2_linear.json` | Wav2Vec2 metrikleri |
| `results/comparison.json` | önerilen model |

## Notlar

- Encoder ağırlıkları dondurulur; yalnızca Logistic Regression eğitilir.
- CUDA varsa otomatik kullanılır (`--device cpu` ile zorlanabilir).
- Veri seti: [FoR for-2sec](https://bil.eecs.yorku.ca/share/for-2sec.tar.gz)
