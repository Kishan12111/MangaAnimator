import subprocess
import tempfile
import unittest
from pathlib import Path


class StageIndependenceTest(unittest.TestCase):
    def test_run_stage_scaffold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "result.txt"
            cmd = [
                "python",
                "-m",
                "src.orchestration.run_stage",
                "panel_understanding",
                "--input",
                "input/a.png",
                "--output",
                str(output_file),
            ]
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0)
            self.assertTrue(output_file.exists())


if __name__ == "__main__":
    unittest.main()
