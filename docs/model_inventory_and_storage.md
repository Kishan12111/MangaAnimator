# Model Inventory + Colab 50GB Storage Plan

This project supports multiple download profiles. The default Colab profile is now **`max_quality_50gb`** to fit within typical Colab disk limits.

## Profiles

- `max_quality` — highest quality, can exceed 50GB depending on cache and revisions.
- `max_quality_50gb` — quality-first but curated to stay under ~50GB.
- `light` — faster setup and lower storage.

## Model list (used by downloader)

| Model | Purpose | Approx Size (GB) |
|---|---|---:|
| `Qwen/Qwen2-VL-7B-Instruct` | Vision-language scene understanding | 15.0 |
| `llava-hf/llava-v1.6-vicuna-13b-hf` | Strong VLM (max profile only) | 26.0 |
| `facebook/sam2-hiera-large` | Character/region segmentation | 2.5 |
| `LiheYoung/depth-anything-large-hf` | Depth estimation/parallax | 1.5 |
| `stabilityai/stable-diffusion-2-inpainting` | Background inpainting | 7.0 |
| `runwayml/stable-diffusion-v1-5` | Base generation support | 4.5 |
| `openai/whisper-large-v3` | Speech/text alignment support | 3.0 |
| `kha-white/manga-ocr-base` | OCR for manga text | 0.6 |

## Default first-run recommendation (Colab)

```bash
DOWNLOAD_PROFILE=max_quality_50gb DOWNLOAD_REPOS=1 DOWNLOAD_STRICT=1 bash scripts/install_colab.sh
```

## If you have extra storage or mounted Drive

```bash
DOWNLOAD_PROFILE=max_quality DOWNLOAD_REPOS=1 DOWNLOAD_STRICT=1 bash scripts/install_colab.sh
```

## Check what profile is being used

- Installer reads `DOWNLOAD_PROFILE` (default is `max_quality_50gb`).
- Downloader prints selected profile and estimated total size.

## Tips to stay under 50GB

1. Use `max_quality_50gb` profile.
2. Keep HF cache in one place and clean stale revisions periodically.
3. Avoid downloading both huge VLMs unless needed.
4. If needed, mount Google Drive and set model/checkpoint paths there.


- Tip: keep Colab output responsive by using `DOWNLOAD_QUIET=1` (default in installer).
