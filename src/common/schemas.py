from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DialogueLine:
    speaker: str
    text: str
    timing_sec: list[float] = field(default_factory=list)


@dataclass
class CharacterState:
    char_id: str
    bbox: list[int]
    emotion: str
    action: str


@dataclass
class SceneDescription:
    page_id: str
    panel_id: str
    reading_order: int
    scene_type: str
    characters: list[CharacterState]
    dialogue: list[DialogueLine]
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_scene_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    required_keys = ["page_id", "panel_id", "reading_order", "scene_type", "characters", "dialogue"]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")

    if "characters" in payload and not isinstance(payload["characters"], list):
        errors.append("characters must be a list")
    if "dialogue" in payload and not isinstance(payload["dialogue"], list):
        errors.append("dialogue must be a list")

    return len(errors) == 0, errors
