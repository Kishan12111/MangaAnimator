# MangaAnimator Colab Production Guide (Single Source of Truth)

This is the only explanation guide for running the project in Colab with high quality and controlled storage.

## 1) Recommended Colab setup

1. Open Colab and select **GPU** runtime.
2. Clone repo:

```bash
!git clone <YOUR_REPO_URL>
%cd MangaAnimator
```

3. (Optional) set HF token for gated models:

```python
import os
os.environ["HF_TOKEN"] = "hf_xxx"
```

4. Install dependencies and download models with a hard storage cap:

```bash
!DOWNLOAD_PROFILE=max_quality_50gb DOWNLOAD_MAX_TOTAL_GB=60 DOWNLOAD_REPOS=1 DOWNLOAD_STRICT=1 DOWNLOAD_QUIET=1 bash scripts/install_colab.sh
```

## 2) Model names used by downloader

- `Qwen/Qwen2-VL-7B-Instruct`
- `llava-hf/llava-v1.6-vicuna-13b-hf` (only when profile/cap allows)
- `facebook/sam2-hiera-large`
- `LiheYoung/depth-anything-large-hf`
- `stabilityai/stable-diffusion-2-inpainting`
- `runwayml/stable-diffusion-v1-5`
- `openai/whisper-large-v3`
- `kha-white/manga-ocr-base`

## 3) Storage control (important)

The downloader enforces `--max-total-gb` (default `60`).
If selected models exceed the cap, it automatically drops the largest models first and reports exactly which were removed.

You can override:

```bash
!DOWNLOAD_MAX_TOTAL_GB=55 bash scripts/install_colab.sh
```

## 4) Run pipeline and verify compute logs

```bash
!python -m src.orchestration.run_all --input /content/panel.png --workdir outputs/full_pipeline --config configs/default.yaml --log-level INFO
```

Look for:
- `Compute detected: device=cuda ...`
- `Running stage ... used_vram=... free_vram=...`
- `GPU warmup result: {'warmup': True ...}`

## 5) Optional Flask + ngrok exposure

```bash
!pip install -q pyngrok
```

```python
import os, subprocess, time
from pyngrok import ngrok

os.environ["FLASK_HOST"] = "0.0.0.0"
os.environ["FLASK_PORT"] = "5000"

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
proc = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(5)
url = ngrok.connect(5000, "http")
print(url)
```

## 6) Output lag prevention in Colab

Model downloader quiet mode is enabled by default via `DOWNLOAD_QUIET=1`, which disables noisy progress bars and keeps cells responsive.
