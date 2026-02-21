from __future__ import annotations

from pathlib import Path

from src.common.io_utils import read_json
from src.common.stage import StageContext, build_stage_parser, load_config, run_stage


TEMPLATE_LIBRARY = {
    "idle_talk": [
        {"frame": 0, "root": [0, 0], "head_rot": 0.0},
        {"frame": 6, "root": [0, -2], "head_rot": -2.0},
        {"frame": 12, "root": [0, 0], "head_rot": 2.0},
    ],
    "dramatic_pose": [
        {"frame": 0, "root": [0, 0], "head_rot": 0.0},
        {"frame": 8, "root": [0, -12], "head_rot": 8.0},
        {"frame": 16, "root": [0, -4], "head_rot": 0.0},
    ],
}


def _procedural_frames(length: int = 24) -> list[dict]:
    return [{"frame": i, "root": [0, (-1) ** i], "head_rot": ((i % 6) - 3) * 0.8} for i in range(length)]


def compute(context: StageContext, config: dict) -> dict:
    payload = read_json(context.input_path)
    mode = payload.get("motion_mode", "template")

    if mode == "template":
        clip = TEMPLATE_LIBRARY["idle_talk"]
    elif mode == "pose_transfer":
        clip = payload.get("pose_keyframes", TEMPLATE_LIBRARY["dramatic_pose"])
    else:
        clip = _procedural_frames(24)

    tracks = []
    for char_id in payload.get("char_ids", ["char_main"]):
        tracks.append({"char_id": char_id, "mode": mode, "keyframes": clip})

    return {
        "input_ref": str(context.input_path),
        "motion_mode": mode,
        "animation_tracks": tracks,
        "fps": 24,
        "metadata": {"module": "animation", "supports": ["template", "pose_transfer", "procedural"]},
    }


def main() -> int:
    parser = build_stage_parser("animation", "Animation engine stage")
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
