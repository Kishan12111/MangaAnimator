from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


@dataclass
class StageCheckpoint:
    stage_name: str
    input_hash: str
    config_hash: str
    status: str
    metadata: dict[str, Any]


class CheckpointManager:
    """File-based checkpoints for resumable stages."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def stage_path(self, stage_name: str) -> Path:
        return self.root_dir / f"{stage_name}.json"

    def load(self, stage_name: str) -> StageCheckpoint | None:
        path = self.stage_path(stage_name)
        if not path.exists():
            return None
        data = read_json(path)
        return StageCheckpoint(**data)

    def save(self, checkpoint: StageCheckpoint) -> None:
        write_json(self.stage_path(checkpoint.stage_name), checkpoint.__dict__)

    def should_skip(self, stage_name: str, input_hash: str, config_hash: str) -> bool:
        checkpoint = self.load(stage_name)
        if checkpoint is None:
            return False
        return (
            checkpoint.status == "done"
            and checkpoint.input_hash == input_hash
            and checkpoint.config_hash == config_hash
        )
