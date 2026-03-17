from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from src.common.debug_dump import dump_debug_json
from src.common.io_utils import read_json
from src.common.model_backends import BACKENDS
from src.common.stage import StageContext, build_stage_parser, load_config, run_stage


BODY_PARTS = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]


def _soft_part_layer(size: tuple[int, int], bbox: list[int]) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(tuple(bbox), radius=8, fill=(255, 255, 255, 255))
    return layer.filter(ImageFilter.GaussianBlur(radius=0.6))


def compute(context: StageContext, config: dict) -> dict:
    scene = read_json(context.input_path)
    output_dir = context.output_path.parent / f"{context.output_path.stem}_layers"
    output_dir.mkdir(parents=True, exist_ok=True)

    quality = config.get("quality", {})
    canvas_size = tuple(quality.get("layer_canvas", [1536, 1536]))
    part_scale = float(quality.get("part_scale", 1.0))

    assets = []
    for character in scene.get("characters", []):
        parts = BACKENDS.character_parts(character["bbox"], quality_scale=part_scale)
        char_assets = {}
        for part_name in BODY_PARTS:
            part_img = _soft_part_layer(canvas_size, parts[part_name])
            out_path = output_dir / f"{character['char_id']}_{part_name}.png"
            part_img.save(out_path)
            char_assets[part_name] = str(out_path)
        assets.append({"char_id": character["char_id"], "parts": parts, "layer_paths": char_assets})

    result = {
        "scene_ref": str(context.input_path),
        "assets": assets,
        "metadata": {
            "module": "character_extraction",
            "batching": config.get("runtime", {}).get("batch_size", "auto"),
            "canvas": canvas_size,
            "quality_profile": quality.get("profile", "max_quality"),
        },
    }
    dump_debug_json(context.debug, context.output_path.with_name(context.output_path.stem + "_debug.json"), result)
    return result


def main() -> int:
    parser = build_stage_parser("character_extraction", "Character extraction stage")
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
