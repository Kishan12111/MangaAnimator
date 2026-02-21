from __future__ import annotations

from pathlib import Path

from src.common.io_utils import read_json
from src.common.stage import StageContext, build_stage_parser, load_config, run_stage

VISEME_TABLE = {
    "a": "A", "e": "E", "i": "I", "o": "O", "u": "U",
}


def _text_to_visemes(text: str) -> list[str]:
    visemes: list[str] = []
    for char in text.lower():
        if char in VISEME_TABLE:
            visemes.append(VISEME_TABLE[char])
    return visemes or ["REST"]


def compute(context: StageContext, config: dict) -> dict:
    scene = read_json(context.input_path)
    tracks = []
    for line in scene.get("dialogue", []):
        text = line.get("text", "")
        visemes = _text_to_visemes(text)
        timing = line.get("timing_sec", [0.0, 1.0])
        duration = max(0.1, timing[1] - timing[0]) if len(timing) >= 2 else 1.0
        step = duration / max(1, len(visemes))
        keyframes = [{"t": timing[0] + idx * step if len(timing) >= 2 else idx * step, "viseme": viseme} for idx, viseme in enumerate(visemes)]
        tracks.append({"speaker": line.get("speaker", "unknown"), "visemes": keyframes, "emotion": "neutral"})

    return {
        "scene_ref": str(context.input_path),
        "facial_tracks": tracks,
        "metadata": {"module": "face_lipsync", "emotion_mode": "rule_based"},
    }


def main() -> int:
    parser = build_stage_parser("face_lipsync", "Face and lip-sync stage")
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
