"""Numerical tests for the M1 cross-entropy and M2 R-Drop objectives."""

import unittest

import torch

from src.losses import symmetric_kl_divergence, training_objective


class LossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.logits = torch.randn(4, 5, requires_grad=True)
        self.labels = torch.tensor([0, 1, 2, 4])

    def test_m1_has_no_consistency_term_and_backpropagates(self) -> None:
        total, classification, consistency = training_objective(self.logits, self.labels)
        self.assertAlmostEqual(total.item(), classification.item(), places=6)
        self.assertEqual(consistency.item(), 0.0)
        total.backward()
        self.assertIsNotNone(self.logits.grad)

    def test_identical_rdrop_predictions_have_zero_kl(self) -> None:
        total, classification, consistency = training_objective(
            self.logits, self.labels, second_logits=self.logits, rdrop_alpha=4.0
        )
        self.assertAlmostEqual(consistency.item(), 0.0, places=6)
        self.assertAlmostEqual(total.item(), classification.item(), places=6)

    def test_rdrop_total_is_ce_plus_weighted_symmetric_kl(self) -> None:
        second = torch.randn_like(self.logits)
        total, classification, consistency = training_objective(
            self.logits, self.labels, second_logits=second, rdrop_alpha=4.0
        )
        expected_kl = symmetric_kl_divergence(self.logits, second)
        self.assertGreater(consistency.item(), 0.0)
        self.assertAlmostEqual(consistency.item(), expected_kl.item(), places=6)
        self.assertAlmostEqual(total.item(), classification.item() + 4.0 * consistency.item(), places=5)


if __name__ == "__main__":
    unittest.main()
