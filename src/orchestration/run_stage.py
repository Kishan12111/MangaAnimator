from __future__ import annotations

import argparse
from pathlib import Path

from src.common.stage import StageContext, load_config, run_stage

from src.panel_understanding.pipeline import compute as panel_compute
from src.character_extraction.pipeline import compute as character_compute
from src.rigging.pipeline import compute as rig_compute
from src.animation.pipeline import compute as animation_compute
from src.face_lipsync.pipeline import compute as face_compute
from src.background.pipeline import compute as background_compute
from src.renderer.pipeline import compute as renderer_compute

STAGE_REGISTRY = {
    "panel_understanding": panel_compute,
    "character_extraction": character_compute,
    "rigging": rig_compute,
    "animation": animation_compute,
    "face_lipsync": face_compute,
    "background": background_compute,
    "renderer": renderer_compute,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one pipeline stage")
    parser.add_argument("stage", choices=sorted(STAGE_REGISTRY.keys()), help="Stage name")
    parser.add_argument("--input", required=True, help="Input path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path")
    parser.add_argument("--checkpoints", default="outputs/checkpoints", help="Checkpoint dir")
    parser.add_argument("--resume", action="store_true", help="Skip stage if unchanged")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = StageContext(
        stage_name=args.stage,
        input_path=Path(args.input),
        output_path=Path(args.output),
        config_path=Path(args.config),
        checkpoint_dir=Path(args.checkpoints),
        debug=args.debug,
        resume=args.resume,
    )
    config = load_config(context.config_path)
    config["log_level"] = args.log_level
    run_stage(context, config, STAGE_REGISTRY[args.stage])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
