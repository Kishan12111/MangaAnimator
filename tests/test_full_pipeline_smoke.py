import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from src.common.io_utils import read_json


class FullPipelineSmokeTest(unittest.TestCase):
    def test_full_pipeline_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_path = temp / "panel.png"
            canvas = Image.new("RGB", (640, 640), "white")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((24, 24, 616, 616), outline="black", width=5)
            draw.rectangle((180, 190, 460, 590), outline="black", width=3)
            canvas.save(image_path)

            workdir = temp / "work"
            cmd = [
                "python",
                "-m",
                "src.orchestration.run_all",
                "--input",
                str(image_path),
                "--workdir",
                str(workdir),
                "--config",
                "configs/default.yaml",
            ]
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0)

            summary = read_json(workdir / "pipeline_summary.json")
            self.assertEqual(summary["quality_profile"], "max_quality")
            self.assertTrue(Path(summary["video"]).exists())


if __name__ == "__main__":
    unittest.main()
