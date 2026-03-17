import unittest

from src.common.schemas import validate_scene_payload


class SceneSchemaTest(unittest.TestCase):
    def test_validate_scene_payload_success(self):
        payload = {
            "page_id": "p1",
            "panel_id": "p1_1",
            "reading_order": 1,
            "scene_type": "dialogue",
            "characters": [],
            "dialogue": [],
        }
        ok, errors = validate_scene_payload(payload)
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_validate_scene_payload_missing_fields(self):
        ok, errors = validate_scene_payload({"page_id": "p1"})
        self.assertFalse(ok)
        self.assertGreaterEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
