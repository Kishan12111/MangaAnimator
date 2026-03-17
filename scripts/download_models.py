from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any


MODEL_CATALOG = {
    # repo_id: estimated download size in GB (approx, varies by files selected)
    "llava-hf/llava-v1.6-vicuna-13b-hf": 26.0,
    "Qwen/Qwen2-VL-7B-Instruct": 15.0,
    "facebook/sam2-hiera-large": 2.5,
    "LiheYoung/depth-anything-large-hf": 1.5,
    "stabilityai/stable-diffusion-2-inpainting": 7.0,
    "runwayml/stable-diffusion-v1-5": 4.5,
    "openai/whisper-large-v3": 3.0,
    "kha-white/manga-ocr-base": 0.6,
}

PROFILE_MODELS = {
    "max_quality": [
        "llava-hf/llava-v1.6-vicuna-13b-hf",
        "Qwen/Qwen2-VL-7B-Instruct",
        "facebook/sam2-hiera-large",
        "LiheYoung/depth-anything-large-hf",
        "stabilityai/stable-diffusion-2-inpainting",
        "runwayml/stable-diffusion-v1-5",
        "openai/whisper-large-v3",
        "kha-white/manga-ocr-base",
    ],
    "max_quality_50gb": [
        # curated to stay around ~34-38GB total
        "Qwen/Qwen2-VL-7B-Instruct",
        "facebook/sam2-hiera-large",
        "LiheYoung/depth-anything-large-hf",
        "stabilityai/stable-diffusion-2-inpainting",
        "runwayml/stable-diffusion-v1-5",
        "openai/whisper-large-v3",
        "kha-white/manga-ocr-base",
    ],
    "light": [
        "facebook/sam2-hiera-large",
        "LiheYoung/depth-anything-large-hf",
        "runwayml/stable-diffusion-v1-5",
        "kha-white/manga-ocr-base",
    ],
}


def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


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


def _download_hf(repo_id: str, target_dir: Path, quiet: bool) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir / repo_id.replace("/", "__")),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=8,
        )
        _log(f"[OK] Downloaded {repo_id}", quiet)
        return True
    except Exception as exc:
        print(f"[WARN] HF download failed for {repo_id}: {exc}")
        return False


def _clone_git(url: str, target_dir: Path, quiet: bool) -> bool:
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = target_dir / name
    if dest.exists():
        _log(f"[INFO] Repo already exists: {dest}", quiet)
        return True
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])
        _log(f"[OK] Cloned {url}", quiet)
        return True
    except Exception as exc:
        print(f"[WARN] git clone failed for {url}: {exc}")
        return False


def _estimate_size(repo_ids: list[str]) -> float:
    return round(sum(MODEL_CATALOG.get(repo_id, 2.0) for repo_id in repo_ids), 2)


def _cap_models_by_size(repo_ids: list[str], max_total_gb: float) -> tuple[list[str], list[str], float]:
    selected = list(repo_ids)
    dropped: list[str] = []
    while _estimate_size(selected) > max_total_gb and selected:
        largest = max(selected, key=lambda rid: MODEL_CATALOG.get(rid, 2.0))
        selected.remove(largest)
        dropped.append(largest)
    return selected, dropped, _estimate_size(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights listed in configs/model_registry.yaml")
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--models-dir", default="models/checkpoints")
    parser.add_argument("--repos-dir", default="models/repos")
    parser.add_argument("--include-repos", action="store_true", help="Also clone known source repos")
    parser.add_argument(
        "--profile",
        default="max_quality_50gb",
        choices=["max_quality", "max_quality_50gb", "light"],
        help="Download profile",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any model download fails")
    parser.add_argument("--max-total-gb", type=float, default=60.0, help="Hard storage cap for selected models")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-model logs to avoid Colab output lag")
    args = parser.parse_args()

    if args.quiet:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    registry = _load_yaml(Path(args.registry))
    models_dir = Path(args.models_dir)
    repos_dir = Path(args.repos_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    strings = _flatten_strings(registry)
    registry_ids = {s for s in strings if _is_hf_repo(s)}

    profile_models = PROFILE_MODELS.get(args.profile, PROFILE_MODELS["max_quality_50gb"])
    repo_ids = sorted(set(profile_models).union(registry_ids.intersection(set(profile_models))))

    initial_est = _estimate_size(repo_ids)
    repo_ids, dropped, est = _cap_models_by_size(repo_ids, args.max_total_gb)
    print(f"[INFO] Targeting {len(repo_ids)} model repos (profile={args.profile}, est_size_gb~{est}, cap_gb={args.max_total_gb}, quiet={args.quiet})")
    if dropped:
        print(f"[INFO] Trimmed from est_size_gb~{initial_est} to ~{est} by dropping {len(dropped)} model(s):")
        for rid in dropped:
            print(f"  - {rid}")

    ok = 0
    failed: list[str] = []
    for idx, repo_id in enumerate(repo_ids, start=1):
        if args.quiet:
            print(f"[INFO] ({idx}/{len(repo_ids)}) downloading {repo_id}")
        if _download_hf(repo_id, models_dir, quiet=args.quiet):
            ok += 1
        else:
            failed.append(repo_id)

    if args.include_repos:
        repo_urls = registry.get("repos", [])
        for url in repo_urls:
            _clone_git(url, repos_dir, quiet=args.quiet)

    print(f"[INFO] Downloaded {ok}/{len(repo_ids)} model repos")
    if failed:
        print("[WARN] Failed repos:")
        for item in failed:
            print(f"  - {item}")

    if token := os.environ.get("HF_TOKEN"):
        _log("[INFO] HF_TOKEN detected and used by huggingface_hub.", quiet=args.quiet)
    else:
        _log("[INFO] No HF_TOKEN set. Public models should still download; gated models will fail.", quiet=args.quiet)

    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
