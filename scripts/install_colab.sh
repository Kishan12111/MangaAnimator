#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/8] Upgrading pip tooling"
python -m pip install -U pip setuptools wheel

echo "[2/8] Installing quality runtime deps"
python -m pip install -U \
  numpy scipy pillow pyyaml tqdm imageio imageio-ffmpeg \
  ffmpeg-python shapely trimesh scikit-image

echo "[3/8] Installing torch (CUDA when available)"
python - <<'PY'
import subprocess, sys
try:
    import torch
    print("torch already installed:", torch.__version__)
except Exception:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ])
PY

echo "[4/8] Installing transformers + acceleration stack"
python -m pip install -U transformers accelerate bitsandbytes sentencepiece safetensors peft diffusers huggingface_hub

echo "[5/8] Installing vision + OCR + quality modules"
python -m pip install -U ultralytics manga-ocr paddleocr timm opencv-python-headless

echo "[6/8] Installing optional animation/audio extras"
python -m pip install -U onnxruntime-gpu phonemizer librosa

echo "[7/8] Downloading model weights from configs/model_registry.yaml"
# Set DOWNLOAD_MODELS=0 to skip automatic downloads
if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
  python scripts/download_models.py --registry configs/model_registry.yaml --models-dir models/checkpoints --repos-dir models/repos
else
  echo "[INFO] Skipping model downloads because DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-0}"
fi

echo "[8/8] Done"
echo "Run quality pipeline: python -m src.orchestration.run_all --input /content/panel.png --workdir outputs/full_pipeline --config configs/default.yaml --resume"
