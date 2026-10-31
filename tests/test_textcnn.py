"""Unit tests for the offline TextCNN data and model contracts."""

import unittest

import pandas as pd
import torch

from src.textcnn_common import TextCNNClassifier, TextIntentDataset, Vocabulary, tokenize


class TextCNNTests(unittest.TestCase):
    def test_vocabulary_and_padding_are_deterministic(self) -> None:
        texts = pd.Series(["Show my balance", "show balance now"])
        first = Vocabulary.build(texts, max_size=20, min_frequency=1)
        second = Vocabulary.build(texts, max_size=20, min_frequency=1)
        self.assertEqual(first.tokens_to_indices, second.tokens_to_indices)
        ids, mask = first.encode("Show unknown balance", max_length=5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(sum(mask), 3)
        self.assertEqual(ids[1], first.tokens_to_indices["<unk>"])

    def test_model_output_shape_and_eval_mode(self) -> None:
        model = TextCNNClassifier(30, 151, 12, [3, 4, 5], 4, 0.5)
        frame = pd.DataFrame({"text": ["short text", "a much longer piece of text"], "label": ["short", "long"]})
        vocabulary = Vocabulary.build(frame["text"], 30, 1)
        dataset = TextIntentDataset(frame, vocabulary, {"short": 0, "long": 1}, 12)
        batch = {key: torch.stack([dataset[0][key], dataset[1][key]]) for key in ["input_ids", "attention_mask"]}
        model.eval()
        with torch.no_grad():
            first = model(**batch).logits
            second = model(**batch).logits
        self.assertEqual(first.shape, (2, 151))
        self.assertTrue(torch.equal(first, second))

    def test_tokenizer_preserves_contractions(self) -> None:
        self.assertIn("what's", tokenize("What's my balance?"))


if __name__ == "__main__":
    unittest.main()
