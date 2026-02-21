from __future__ import annotations

import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab demo runner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--workdir", default="outputs/full_pipeline")
    args = parser.parse_args()

    cmd = [
        "python",
        "-m",
        "src.orchestration.run_all",
        "--input",
        args.input,
        "--workdir",
        args.workdir,
        "--resume",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
