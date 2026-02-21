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
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
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

echo "[7/8] Downloading heavyweight model weights (50GB-safe default profile)"
# Controls:
# - DOWNLOAD_MODELS=0 : skip downloads
# - DOWNLOAD_PROFILE=light|max_quality|max_quality_50gb
# - DOWNLOAD_STRICT=1 : fail install if any model fails
# - DOWNLOAD_REPOS=1 : also clone source repos in registry
if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
  PROFILE="${DOWNLOAD_PROFILE:-max_quality_50gb}"
  STRICT_FLAG=""
  REPO_FLAG=""
  QUIET_FLAG=""
  if [[ "${DOWNLOAD_QUIET:-1}" == "1" ]]; then
    QUIET_FLAG="--quiet"
  fi
  if [[ "${DOWNLOAD_STRICT:-0}" == "1" ]]; then
    STRICT_FLAG="--strict"
  fi
  if [[ "${DOWNLOAD_REPOS:-1}" == "1" ]]; then
    REPO_FLAG="--include-repos"
  fi
  MAX_TOTAL_GB="${DOWNLOAD_MAX_TOTAL_GB:-60}"
  python scripts/download_models.py \
    --registry configs/model_registry.yaml \
    --models-dir models/checkpoints \
    --repos-dir models/repos \
    --profile "${PROFILE}" \
    --max-total-gb "${MAX_TOTAL_GB}" \
    ${STRICT_FLAG} ${REPO_FLAG} ${QUIET_FLAG}
else
  echo "[INFO] Skipping model downloads because DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-0}"
fi

echo "[8/8] Done"
echo "Run quality pipeline: python -m src.orchestration.run_all --input /content/panel.png --workdir outputs/full_pipeline --config configs/default.yaml --resume"
