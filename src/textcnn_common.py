"""Vocabulary, dataset, and model utilities for CPU-reproducible TextCNN runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Apply a fixed, lowercase tokenizer without external resources."""
    return TOKEN_PATTERN.findall(text.lower())


class Vocabulary:
    """A training-only vocabulary with deterministic frequency/alphabetical ties."""

    def __init__(self, tokens_to_indices: dict[str, int]):
        self.tokens_to_indices = tokens_to_indices
        self.indices_to_tokens = [None] * len(tokens_to_indices)
        for token, index in tokens_to_indices.items():
            self.indices_to_tokens[index] = token

    @classmethod
    def build(cls, texts: pd.Series, max_size: int, min_frequency: int) -> "Vocabulary":
        counts = Counter(token for text in texts for token in tokenize(text))
        candidates = [
            (token, count)
            for token, count in counts.items()
            if count >= min_frequency and token not in {"<pad>", "<unk>"}
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        tokens = ["<pad>", "<unk>"] + [token for token, _ in candidates[: max_size - 2]]
        return cls({token: index for index, token in enumerate(tokens)})

    def __len__(self) -> int:
        return len(self.tokens_to_indices)

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        unknown = self.tokens_to_indices["<unk>"]
        ids = [self.tokens_to_indices.get(token, unknown) for token in tokenize(text)][:max_length]
        attention = [1] * len(ids)
        padding = max_length - len(ids)
        return ids + [0] * padding, attention + [0] * padding

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.tokens_to_indices, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))


class TextIntentDataset(Dataset):
    """Encode fixed text and labels into padded TextCNN tensors."""

    def __init__(self, frame: pd.DataFrame, vocabulary: Vocabulary, label_to_id: dict[str, int], max_length: int):
        self.examples = [vocabulary.encode(text, max_length) for text in frame["text"]]
        self.labels = torch.tensor([label_to_id[label] for label in frame["label"]], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, attention_mask = self.examples[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": self.labels[index],
        }


class TextCNNClassifier(nn.Module):
    """A compact convolutional sentence classifier with trainable embeddings."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        filter_sizes: list[int],
        num_filters: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=size) for size in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            activated = torch.relu(conv(embedded))
            if attention_mask is not None:
                mask = attention_mask[:, : activated.shape[2]].unsqueeze(1)
                activated = activated.masked_fill(mask == 0, -1e4)
            pooled.append(activated.max(dim=2).values)
        features = self.dropout(torch.cat(pooled, dim=1))
        return SimpleNamespace(logits=self.classifier(features))
