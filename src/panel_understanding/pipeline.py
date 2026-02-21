from __future__ import annotations

from pathlib import Path

from src.common.debug_dump import dump_debug_json
from src.common.model_backends import BACKENDS
from src.common.schemas import validate_scene_payload
from src.common.stage import StageContext, build_stage_parser, load_config, run_stage


def compute(context: StageContext, config: dict) -> dict:
    quality_profile = config.get("quality", {}).get("profile", "max_quality")
    backend = BACKENDS.panel_understanding(context.input_path, quality_profile)
    payload = backend.payload

    scene = {
        "page_id": context.input_path.stem,
        "panel_id": f"{context.input_path.stem}_0001",
        "reading_order": 1,
        "scene_type": "dramatic_dialogue",
        "characters": payload["characters"],
        "dialogue": payload["dialogue"],
        "panels": payload["panels"],
        "metadata": {
            "module": "panel_understanding",
            "model_profile": config.get("runtime", {}),
            "backend": backend.used_backend,
            "quality_profile": quality_profile,
        },
    }
    ok, errors = validate_scene_payload(scene)
    scene["valid"] = ok
    scene["validation_errors"] = errors

    dump_debug_json(context.debug, context.output_path.with_name(context.output_path.stem + "_debug.json"), scene)
    return scene


def main() -> int:
    parser = build_stage_parser("panel_understanding", "Panel understanding stage")
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
