from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointManager, StageCheckpoint
from .io_utils import read_json, sha256_file, sha256_text, write_json
from .logger import configure_logging, get_logger
from .compute_monitor import get_compute_snapshot


@dataclass
class StageContext:
    stage_name: str
    input_path: Path
    output_path: Path
    config_path: Path
    checkpoint_dir: Path
    debug: bool
    resume: bool


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return read_json(path)

    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        # fallback for test envs without pyyaml
        return {}


def build_stage_parser(stage_name: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="Input artifact path")
    parser.add_argument("--output", required=True, help="Output artifact path")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path")
    parser.add_argument("--checkpoints", default="outputs/checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Enable checkpoint resume")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.set_defaults(stage_name=stage_name)
    return parser


def run_stage(
    context: StageContext,
    config: dict[str, Any],
    compute_fn,
) -> dict[str, Any]:
    configure_logging(level=str(config.get("log_level", "INFO")))
    log = get_logger(context.stage_name)
    manager = CheckpointManager(context.checkpoint_dir)

    input_hash = sha256_file(context.input_path) if context.input_path.exists() else sha256_text(str(context.input_path))
    config_hash = sha256_file(context.config_path) if context.config_path.exists() else sha256_text(str(config))

    if context.resume and manager.should_skip(context.stage_name, input_hash, config_hash):
        log.info("Checkpoint hit, skipping '%s'", context.stage_name)
        if context.output_path.exists():
            return read_json(context.output_path)
        return {"stage": context.stage_name, "skipped": True}

    snap = get_compute_snapshot()
    log.info(
        "Running stage '%s' on %s (gpu=%s, used_vram=%sGB, free_vram=%sGB)",
        context.stage_name,
        snap.device,
        snap.gpu_name,
        snap.used_vram_gb,
        snap.free_vram_gb,
    )

    context.output_path.parent.mkdir(parents=True, exist_ok=True)
    result = compute_fn(context, config)
    write_json(context.output_path, result)

    manager.save(
        StageCheckpoint(
            stage_name=context.stage_name,
            input_hash=input_hash,
            config_hash=config_hash,
            status="done",
            metadata={"debug": context.debug, "output": str(context.output_path)},
        )
    )
    log.info("Completed stage '%s'", context.stage_name)
    return result
