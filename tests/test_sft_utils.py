import unittest

from nanochat.sft_utils import resolve_output_model_tag


class ResolveOutputModelTagTests(unittest.TestCase):
    def test_explicit_output_tag_keeps_input_and_output_separate(self):
        self.assertEqual(
            resolve_output_model_tag("d32-base", "d32-math15", depth=32),
            "d32-math15",
        )

    def test_defaults_to_input_tag_for_backward_compatibility(self):
        self.assertEqual(resolve_output_model_tag("d32-base", None, depth=32), "d32-base")

    def test_defaults_to_depth_when_no_tags_are_given(self):
        self.assertEqual(resolve_output_model_tag(None, None, depth=12), "d12")


if __name__ == "__main__":
    unittest.main()
