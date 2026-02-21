from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import write_json


def dump_debug_json(enabled: bool, path: Path, payload: Any) -> None:
    if not enabled:
        return
    write_json(path, payload)


def dump_debug_text(enabled: bool, path: Path, content: str) -> None:
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
