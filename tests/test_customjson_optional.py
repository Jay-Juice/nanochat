import os
import unittest
from unittest.mock import patch

from tasks.customjson import load_optional_custom_json


class OptionalCustomJsonTests(unittest.TestCase):
    def test_empty_path_disables_optional_dataset(self):
        self.assertIsNone(load_optional_custom_json(""))

    def test_missing_path_fails_instead_of_silently_loading_zero_rows(self):
        with patch("tasks.customjson.os.path.isfile", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Optional JSONL dataset not found"):
                load_optional_custom_json("missing.jsonl")

    def test_nonempty_dataset_is_returned(self):
        fake_task = unittest.mock.MagicMock()
        fake_task.__len__.return_value = 3
        with (
            patch("tasks.customjson.os.path.isfile", return_value=True),
            patch("tasks.customjson.CustomJSON", return_value=fake_task),
        ):
            self.assertIs(load_optional_custom_json("math.jsonl"), fake_task)

    @unittest.skipUnless(os.environ.get("NANOCHAT_MATH_SFT_DATA"), "integration dataset not configured")
    def test_real_math_dataset_loads_expected_rows(self):
        task = load_optional_custom_json(os.environ["NANOCHAT_MATH_SFT_DATA"])
        self.assertEqual(len(task), 190278)


if __name__ == "__main__":
    unittest.main()
