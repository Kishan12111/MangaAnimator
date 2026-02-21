import tempfile
import unittest
from pathlib import Path

from src.common.checkpoint import CheckpointManager, StageCheckpoint


class ResumeCheckpointTest(unittest.TestCase):
    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir))
            manager.save(
                StageCheckpoint(
                    stage_name="panel",
                    input_hash="a",
                    config_hash="b",
                    status="done",
                    metadata={"profile": "cpu"},
                )
            )
            self.assertTrue(manager.should_skip("panel", "a", "b"))
            self.assertFalse(manager.should_skip("panel", "x", "b"))


if __name__ == "__main__":
    unittest.main()
