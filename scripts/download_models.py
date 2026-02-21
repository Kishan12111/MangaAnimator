from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any


EXTRA_MAX_QUALITY_MODELS = [
    "llava-hf/llava-v1.6-vicuna-13b-hf",
    "Qwen/Qwen2-VL-7B-Instruct",
    "facebook/sam2-hiera-large",
    "LiheYoung/depth-anything-large-hf",
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-v1-5",
    "openai/whisper-large-v3",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _is_hf_repo(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("http://") or value.startswith("https://"):
        return False
    return "/" in value and " " not in value and len(value.split("/")) == 2


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
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir / repo_id.replace("/", "__")),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=8,
        )
        print(f"[OK] Downloaded {repo_id}")
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
        print(f"[OK] Cloned {url}")
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
    parser.add_argument("--profile", default="max_quality", choices=["max_quality", "light"], help="Download profile")
    parser.add_argument("--strict", action="store_true", help="Fail if any model download fails")
    args = parser.parse_args()

    registry = _load_yaml(Path(args.registry))
    models_dir = Path(args.models_dir)
    repos_dir = Path(args.repos_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    strings = _flatten_strings(registry)
    repo_ids = {s for s in strings if _is_hf_repo(s)}
    if args.profile == "max_quality":
        repo_ids.update(EXTRA_MAX_QUALITY_MODELS)

    repo_ids = sorted(repo_ids)
    print(f"[INFO] Targeting {len(repo_ids)} model repos (profile={args.profile})")

    ok = 0
    failed: list[str] = []
    for repo_id in repo_ids:
        if _download_hf(repo_id, models_dir):
            ok += 1
        else:
            failed.append(repo_id)

    if args.include_repos:
        repo_urls = registry.get("repos", [])
        for url in repo_urls:
            _clone_git(url, repos_dir)

    print(f"[INFO] Downloaded {ok}/{len(repo_ids)} model repos")
    if failed:
        print("[WARN] Failed repos:")
        for item in failed:
            print(f"  - {item}")

    if token := os.environ.get("HF_TOKEN"):
        print("[INFO] HF_TOKEN detected and used by huggingface_hub.")
    else:
        print("[INFO] No HF_TOKEN set. Public models should still download; gated models will fail.")

    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
