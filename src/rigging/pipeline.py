from __future__ import annotations

from pathlib import Path

from src.common.io_utils import read_json
from src.common.stage import StageContext, build_stage_parser, load_config, run_stage


SKELETON_TEMPLATE = {
    "joints": [
        "root",
        "spine",
        "neck",
        "head",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_hip",
        "right_knee",
        "right_ankle",
    ],
    "constraints": {"knee": [0, 160], "elbow": [0, 170], "neck": [-45, 45]},
}


def _joint_positions(parts: dict[str, list[int]]) -> dict[str, list[int]]:
    torso = parts.get("torso", [0, 0, 1, 1])
    head = parts.get("head", torso)
    left_arm = parts.get("left_arm", torso)
    right_arm = parts.get("right_arm", torso)
    left_leg = parts.get("left_leg", torso)
    right_leg = parts.get("right_leg", torso)

    def center(bbox: list[int]) -> list[int]:
        return [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

    return {
        "root": center(torso),
        "spine": [center(torso)[0], torso[1] + (torso[3] - torso[1]) // 3],
        "neck": [center(head)[0], head[3]],
        "head": center(head),
        "left_shoulder": [left_arm[2], left_arm[1]],
        "left_elbow": center(left_arm),
        "left_wrist": [left_arm[0], left_arm[3]],
        "right_shoulder": [right_arm[0], right_arm[1]],
        "right_elbow": center(right_arm),
        "right_wrist": [right_arm[2], right_arm[3]],
        "left_hip": [left_leg[0], left_leg[1]],
        "left_knee": center(left_leg),
        "left_ankle": [left_leg[2], left_leg[3]],
        "right_hip": [right_leg[2], right_leg[1]],
        "right_knee": center(right_leg),
        "right_ankle": [right_leg[0], right_leg[3]],
    }


def compute(context: StageContext, config: dict) -> dict:
    assets = read_json(context.input_path)
    rigs = []
    for item in assets.get("assets", []):
        joints = _joint_positions(item.get("parts", {}))
        weights = {part: {joint: 1.0 if part.split("_")[0] in joint else 0.15 for joint in SKELETON_TEMPLATE["joints"]} for part in item.get("parts", {}).keys()}
        rigs.append(
            {
                "char_id": item["char_id"],
                "skeleton": SKELETON_TEMPLATE,
                "joint_positions": joints,
                "weights": weights,
            }
        )

    return {
        "asset_ref": str(context.input_path),
        "rigs": rigs,
        "export": {"format": "rig_json", "animation_ready": True},
        "metadata": {"module": "rigging", "device": config.get("runtime", {}).get("device", "auto")},
    }


def main() -> int:
    parser = build_stage_parser("rigging", "Auto rigging stage")
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
