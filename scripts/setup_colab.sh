#!/usr/bin/env bash
set -euo pipefail

# MangaVID setup helper for Google Colab runtimes.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# FFmpeg is required for video encode/mux
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y ffmpeg
fi

# Optional: install CUDA torch + diffusion stack for anime generation.
# Safe to skip if using non-anime mode or CPU-only fallbacks.
if [[ "${INSTALL_DIFFUSION:-0}" == "1" ]]; then
  python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
  python -m pip install --upgrade diffusers transformers accelerate safetensors controlnet-aux
fi

echo "Colab setup complete. You can run: python app.py"
