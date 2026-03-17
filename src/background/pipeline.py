from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.common.stage import StageContext, build_stage_parser, load_config, run_stage


def _blur(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(arr)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out += padded[1 + dy : 1 + dy + arr.shape[0], 1 + dx : 1 + dx + arr.shape[1]] // 9
    return out


def compute(context: StageContext, config: dict) -> dict:
    if context.input_path.exists():
        image = Image.open(context.input_path).convert("RGB")
    else:
        image = Image.new("RGB", (1280, 720), "black")

    arr = np.array(image)
    h, w, _ = arr.shape
    gray = arr.mean(axis=2).astype(np.uint8)
    depth = _blur(gray)

    near_mask = depth < 85
    mid_mask = (depth >= 85) & (depth < 170)
    far_mask = depth >= 170

    layer_dir = context.output_path.parent / f"{context.output_path.stem}_layers"
    layer_dir.mkdir(parents=True, exist_ok=True)

    def save_masked(mask: np.ndarray, name: str) -> str:
        out = arr.copy()
        out[~mask] = 0
        path = layer_dir / name
        Image.fromarray(out).save(path)
        return str(path)

    near_path = save_masked(near_mask, "bg_near.png")
    mid_path = save_masked(mid_mask, "bg_mid.png")
    far_path = save_masked(far_mask, "bg_far.png")
    depth_path = layer_dir / "depth.png"
    Image.fromarray(depth).save(depth_path)

    return {
        "image_ref": str(context.input_path),
        "layers": {"near": near_path, "mid": mid_path, "far": far_path, "depth": str(depth_path)},
        "camera_path": [{"t": 0.0, "x": 0, "y": 0, "zoom": 1.0}, {"t": 3.0, "x": int(0.03 * w), "y": int(0.02 * h), "zoom": 1.08}],
        "metadata": {"module": "background", "parallax": True},
    }


def main() -> int:
    parser = build_stage_parser("background", "Background processing stage")
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
