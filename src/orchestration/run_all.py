from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.animation.pipeline import compute as anim_compute
from src.background.pipeline import compute as bg_compute
from src.character_extraction.pipeline import compute as char_compute
from src.common.compute_monitor import get_compute_snapshot, gpu_warmup
from src.common.io_utils import write_json
from src.common.logger import configure_logging, get_logger
from src.common.stage import StageContext, load_config, run_stage
from src.face_lipsync.pipeline import compute as face_compute
from src.panel_understanding.pipeline import compute as panel_compute
from src.renderer.pipeline import compute as render_compute
from src.rigging.pipeline import compute as rig_compute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full manga-to-animation pipeline")
    parser.add_argument("--input", required=True, help="Input manga panel image")
    parser.add_argument("--workdir", default="outputs/full_pipeline", help="Working directory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _ctx(stage: str, in_path: Path, out_path: Path, args: argparse.Namespace) -> StageContext:
    return StageContext(
        stage_name=stage,
        input_path=in_path,
        output_path=out_path,
        config_path=Path(args.config),
        checkpoint_dir=Path(args.workdir) / "checkpoints",
        debug=args.debug,
        resume=args.resume,
    )


def _run(log, stage_name: str, fn, context: StageContext, config: dict) -> dict:
    t0 = time.perf_counter()
    result = run_stage(context, config, fn)
    elapsed = time.perf_counter() - t0
    log.info("Stage '%s' finished in %.2fs", stage_name, elapsed)
    return result


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level)
    log = get_logger("run_all")
    workdir = Path(args.workdir)
    artifacts = workdir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    config["log_level"] = args.log_level

    snap = get_compute_snapshot()
    log.info(
        "Compute detected: device=%s gpu=%s total_vram=%sGB used=%sGB free=%sGB",
        snap.device,
        snap.gpu_name,
        snap.total_vram_gb,
        snap.used_vram_gb,
        snap.free_vram_gb,
    )

    if config.get("quality", {}).get("force_gpu_warmup", True):
        warm = gpu_warmup(int(config.get("quality", {}).get("gpu_warmup_matrix", 2048)))
        log.info("GPU warmup result: %s", warm)

    panel_out = artifacts / "01_scene.json"
    scene = _run(log, "panel_understanding", panel_compute, _ctx("panel_understanding", Path(args.input), panel_out, args), config)

    char_out = artifacts / "02_chars.json"
    chars = _run(log, "character_extraction", char_compute, _ctx("character_extraction", panel_out, char_out, args), config)

    rig_out = artifacts / "03_rig.json"
    rigs = _run(log, "rigging", rig_compute, _ctx("rigging", char_out, rig_out, args), config)

    face_out = artifacts / "04_face.json"
    face = _run(log, "face_lipsync", face_compute, _ctx("face_lipsync", panel_out, face_out, args), config)

    bg_out = artifacts / "05_bg.json"
    bg = _run(log, "background", bg_compute, _ctx("background", Path(args.input), bg_out, args), config)

    motion_mode = config.get("quality", {}).get("default_motion_mode", "procedural")
    anim_request = artifacts / "06_anim_request.json"
    write_json(
        anim_request,
        {
            "motion_mode": motion_mode,
            "char_ids": [r["char_id"] for r in rigs.get("rigs", [])] or ["char_main"],
            "fps": int(config.get("quality", {}).get("target_fps", 24)),
        },
    )

    anim_out = artifacts / "06_anim.json"
    anim = _run(log, "animation", anim_compute, _ctx("animation", anim_request, anim_out, args), config)

    render_request = artifacts / "07_render_request.json"
    write_json(
        render_request,
        {
            "background": bg["layers"]["mid"],
            "fps": int(config.get("quality", {}).get("target_fps", 24)),
            "duration_sec": float(config.get("quality", {}).get("shot_duration_sec", 3.0)),
        },
    )

    render_out = artifacts / "07_render.json"
    render = _run(log, "renderer", render_compute, _ctx("renderer", render_request, render_out, args), config)

    summary = {
        "workdir": str(workdir),
        "video": render["video_path"],
        "quality_profile": config.get("quality", {}).get("profile", "max_quality"),
        "compute": snap.__dict__,
        "stages": ["panel", "character", "rigging", "face", "background", "animation", "renderer"],
        "artifacts": {
            "scene": str(panel_out),
            "characters": str(char_out),
            "rig": str(rig_out),
            "face": str(face_out),
            "background": str(bg_out),
            "animation": str(anim_out),
            "render": str(render_out),
        },
    }
    write_json(workdir / "pipeline_summary.json", summary)
    log.info("Pipeline completed. Video: %s", render["video_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
