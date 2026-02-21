from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _is_hf_repo(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("http://") or value.startswith("https://"):
        return False
    return "/" in value and " " not in value


def _flatten_strings(data: Any) -> list[str]:
    values: list[str] = []
    if isinstance(data, dict):
        for v in data.values():
            values.extend(_flatten_strings(v))
    elif isinstance(data, list):
        for v in data:
            values.extend(_flatten_strings(v))
    elif isinstance(data, str):
        values.append(data)
    return values


def _download_hf(repo_id: str, target_dir: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir / repo_id.replace("/", "__")),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return True
    except Exception as exc:
        print(f"[WARN] HF download failed for {repo_id}: {exc}")
        return False


def _clone_git(url: str, target_dir: Path) -> bool:
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = target_dir / name
    if dest.exists():
        print(f"[INFO] Repo already exists: {dest}")
        return True
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])
        return True
    except Exception as exc:
        print(f"[WARN] git clone failed for {url}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights listed in configs/model_registry.yaml")
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--models-dir", default="models/checkpoints")
    parser.add_argument("--repos-dir", default="models/repos")
    parser.add_argument("--include-repos", action="store_true", help="Also clone known source repos")
    args = parser.parse_args()

    registry = _load_yaml(Path(args.registry))
    models_dir = Path(args.models_dir)
    repos_dir = Path(args.repos_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    strings = _flatten_strings(registry)
    repo_ids = sorted({s for s in strings if _is_hf_repo(s)})

    print(f"[INFO] Found {len(repo_ids)} model ids in registry")
    ok = 0
    for repo_id in repo_ids:
        print(f"[INFO] Downloading {repo_id}")
        if _download_hf(repo_id, models_dir):
            ok += 1

    if args.include_repos:
        git_repos = [
            "https://github.com/facebookresearch/sam2.git",
            "https://github.com/IDEA-Research/DWPose.git",
            "https://github.com/LiheYoung/Depth-Anything.git",
        ]
        for url in git_repos:
            _clone_git(url, repos_dir)

    print(f"[INFO] Downloaded {ok}/{len(repo_ids)} registry models")
    if token := os.environ.get("HF_TOKEN"):
        print("[INFO] HF_TOKEN detected and used by huggingface_hub.")
    else:
        print("[INFO] No HF_TOKEN set. Public models should still download; gated models will fail.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
