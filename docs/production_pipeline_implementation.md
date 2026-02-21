# Production Implementation: Manga-to-Animation 2D Puppet Pipeline (Quality-First)

## A. System Architecture Diagram

```mermaid
flowchart LR
    I[Manga Panel Image(s)] --> P[Panel Understanding (VLM+OCR)]
    P --> S[Scene JSON]
    S --> C[Character Extraction (SAM2 + Pose)]
    C --> R[Auto Rigging (IK + Weights)]
    S --> F[Face + Lip Sync (Visemes + Emotion)]
    I --> B[Background Engine (Depth + Parallax)]
    R --> A[Animation Engine (Template / Pose-transfer / Procedural)]
    F --> A
    B --> V[Renderer (Cinematic Composite + Encode)]
    A --> V
    V --> O[24 FPS+ Output]
```

## B. Complete File Structure

```text
src/
  common/                  # checkpoints, stage runtime, model backends
  panel_understanding/
  character_extraction/
  rigging/
  face_lipsync/
  background/
  animation/
  renderer/
  orchestration/
configs/
scripts/
tests/
docs/
```

## C. Installation Script (with auto model downloads)

```bash
# Optional for gated models
export HF_TOKEN=your_huggingface_token

# Automatic dependency + model setup
DOWNLOAD_PROFILE=max_quality_50gb DOWNLOAD_REPOS=1 DOWNLOAD_STRICT=1 bash scripts/install_colab.sh
```

Notes:
- `scripts/install_colab.sh` now automatically downloads models listed in `configs/model_registry.yaml`.
- Set `DOWNLOAD_MODELS=0` if you only want to install dependencies.

## D. All Code Modules

- `src/common/stage.py`
- `src/common/model_backends.py`
- `src/panel_understanding/pipeline.py`
- `src/character_extraction/pipeline.py`
- `src/rigging/pipeline.py`
- `src/animation/pipeline.py`
- `src/face_lipsync/pipeline.py`
- `src/background/pipeline.py`
- `src/renderer/pipeline.py`
- `src/orchestration/run_stage.py`
- `src/orchestration/run_all.py`
- `scripts/download_models.py`

## E. Execution Pipeline

```bash
python -m src.orchestration.run_all \
  --input /path/to/panel.png \
  --workdir outputs/full_pipeline \
  --config configs/default.yaml \
  --resume
```

## F. Test Example

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## G. Quality Optimization Tips

- Use `quality.profile=max_quality` and GPU runtime.
- Keep `target_fps=24` with longer shots for smoother motion (`shot_duration_sec`).
- Increase `layer_canvas` for cleaner puppet edges.
- Keep ffmpeg available for MP4 export (GIF fallback is debug-oriented).

## H. Failure Cases + Fixes

- Missing heavyweight model libs/checkpoints in Colab: installer now attempts automatic download from registry; rerun install with `HF_TOKEN` for gated repos.
- OOM at max-quality: lower batch/tile settings or switch to lower-quality runtime profile.
- Weak OCR/dialogue alignment: ensure MangaOCR weights are installed and input resolution is high.

## Powerful Open-Source Model Targets

Configured in `configs/model_registry.yaml` (quality-first references):
- Florence-2 Large / Qwen2-VL 7B for scene understanding
- SAM2-Hiera-Large for segmentation
- DWPose/OpenPose for body guidance
- Depth-Anything V2 Large for depth/parallax
- SD 2 Inpainting for background completion



## GPU/VRAM Utilization

- `run_all` now logs GPU device name and VRAM (used/free/total) at startup and per stage.
- A CUDA warmup matrix multiply runs by default (`quality.force_gpu_warmup=true`) to ensure kernels are active.
- Tune warmup size via `quality.gpu_warmup_matrix`.


## Model Inventory and 50GB Budget

- See `docs/model_inventory_and_storage.md` for exact model list and size estimates.
- Default installer profile is now `max_quality_50gb` to fit Colab storage limits.
