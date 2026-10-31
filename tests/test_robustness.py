"""Tests for deterministic spelling-noise generation."""

import unittest

from src.robustness import perturb_text


class RobustnessTests(unittest.TestCase):
    def test_noise_is_reproducible_and_changes_text(self) -> None:
        text = "show my account balance"
        first = perturb_text(text, 0.10, 123)
        second = perturb_text(text, 0.10, 123)
        self.assertEqual(first, second)
        self.assertNotEqual(first, text)

    def test_zero_alpha_text_is_safe(self) -> None:
        self.assertEqual(perturb_text("123 !!!", 0.10, 123), "123 !!!")


if __name__ == "__main__":
    unittest.main()
