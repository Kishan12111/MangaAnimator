from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from src.common.io_utils import read_json
from src.common.stage import StageContext, build_stage_parser, load_config, run_stage


def _load_image(path: str, default_shape: tuple[int, int] = (1080, 1920)) -> np.ndarray:
    if path and Path(path).exists():
        return np.array(Image.open(path).convert("RGB"))
    h, w = default_shape
    return np.zeros((h, w, 3), dtype=np.uint8)


def _encode_with_ffmpeg(frame_dir: Path, out_path: Path, fps: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-vf",
        "format=yuv420p",
        str(out_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    return completed.returncode == 0


def _cinematic_grade(frame: Image.Image) -> Image.Image:
    frame = ImageEnhance.Contrast(frame).enhance(1.18)
    frame = ImageEnhance.Color(frame).enhance(1.12)
    return frame.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))


def compute(context: StageContext, config: dict) -> dict:
    manifest = read_json(context.input_path)
    fps = int(manifest.get("fps", 24))
    seconds = float(manifest.get("duration_sec", 3.0))
    frame_count = max(1, int(fps * seconds))

    quality = config.get("quality", {})
    motion_amplitude = float(quality.get("camera_motion_amplitude", 0.14))

    bg = _load_image(manifest.get("background", ""))
    h, w, _ = bg.shape

    frame_dir = context.output_path.parent / f"{context.output_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        bg_img = Image.fromarray(bg.copy())

        zoom = 1.0 + (motion_amplitude * t)
        crop_w, crop_h = int(w / zoom), int(h / zoom)
        x0 = max(0, int((w - crop_w) * t * 0.5))
        y0 = max(0, int((h - crop_h) * t * 0.4))
        bg_img = bg_img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((w, h), Image.Resampling.LANCZOS)

        draw = ImageDraw.Draw(bg_img)
        glow = int(80 + 60 * np.sin(t * np.pi))
        draw.ellipse((int(w * 0.08), int(h * 0.10), int(w * 0.16), int(h * 0.22)), fill=(255, 230, glow))
        draw.text((30, h - 60), f"shot {idx:03d}", fill=(240, 240, 240))

        graded = _cinematic_grade(bg_img)
        frame_path = frame_dir / f"frame_{idx:04d}.png"
        graded.save(frame_path)
        frames.append(graded)

    video_path = context.output_path.with_suffix(".mp4")
    encoded = _encode_with_ffmpeg(frame_dir, video_path, fps)
    if not encoded:
        fallback_path = context.output_path.with_suffix(".gif")
        frames[0].save(fallback_path, save_all=True, append_images=frames[1:], duration=int(1000 / max(1, fps)), loop=0)
        video_path = fallback_path

    return {
        "render_manifest": str(context.input_path),
        "video_path": str(video_path),
        "fps": fps,
        "frames": frame_count,
        "metadata": {
            "module": "renderer",
            "camera_motion": True,
            "grading": "cinematic",
            "quality_profile": quality.get("profile", "max_quality"),
            "ffmpeg_used": encoded,
        },
    }


def main() -> int:
    parser = build_stage_parser("renderer", "Rendering stage")
    args = parser.parse_args()
    context = StageContext(
        stage_name=args.stage_name,
        input_path=Path(args.input),
        output_path=Path(args.output),
        config_path=Path(args.config),
        checkpoint_dir=Path(args.checkpoints),
        debug=args.debug,
        resume=args.resume,
    )
    config = load_config(context.config_path)
    run_stage(context, config, compute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
