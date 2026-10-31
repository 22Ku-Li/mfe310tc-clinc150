"""Generate a real M2 inference demonstration for the recorded presentation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data import PROJECT_ROOT
from src.textcnn_common import TextCNNClassifier, TextIntentDataset, Vocabulary
from src.transformer_common import make_data_loader, predict_probabilities, resolve_device


def main() -> None:
    checkpoint = PROJECT_ROOT / "checkpoints/m2_textcnn_rdrop/seed_42/best"
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    config = state["config"]
    labels = state["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    vocabulary = Vocabulary.load(checkpoint / "vocabulary.json")
    device = resolve_device("cpu")
    model = TextCNNClassifier(
        len(vocabulary),
        len(labels),
        config["model"]["embedding_dim"],
        config["model"]["filter_sizes"],
        config["model"]["num_filters"],
        config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location=device, weights_only=True))
    model.eval()

    predictions = pd.read_csv(PROJECT_ROOT / "results/m2_textcnn_rdrop/seed_42/test/predictions.csv")
    oos_example = predictions.loc[
        (predictions["true_label"] == "oos")
        & (predictions["predicted_label"] == "oos")
        & (predictions["text"].str.len() < 90)
    ].sort_values("predicted_probability", ascending=False).iloc[0]
    examples = [
        "what does post rush hour traffic generally look like heading into the lincoln tunnel in manhattan from new jersey",
        oos_example["text"],
        "this song needs to be added to my country music playlist",
    ]
    frame = pd.DataFrame({"sample_id": ["demo_1", "demo_2", "demo_3"], "text": examples, "label": ["traffic", "oos", "update_playlist"]})
    dataset = TextIntentDataset(frame, vocabulary, label_to_id, config["model"]["max_length"])
    loader = make_data_loader(dataset, 3, False, 42, 0, False)
    _, probabilities, _, _ = predict_probabilities(model, loader, device, False)
    predicted_ids = probabilities.argmax(axis=1)
    rows = []
    for index, text in enumerate(examples):
        rows.append({
            "input": text,
            "reference_label": frame.iloc[index]["label"],
            "predicted_label": labels[int(predicted_ids[index])],
            "confidence": float(probabilities[index, predicted_ids[index]]),
            "oos_probability": float(probabilities[index, label_to_id["oos"]]),
            "correct": bool(labels[int(predicted_ids[index])] == frame.iloc[index]["label"]),
        })

    output_dir = PROJECT_ROOT / "results/report_assets"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "inference_demo.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(11.2, 5.5))
    axis.axis("off")
    colors = ["#355C52", "#6B5B3E", "#8C3B36"]
    headings = ["SUPPORTED INTENT", "OUT-OF-SCOPE", "ERROR TO DISCUSS"]
    for index, (row, color, heading) in enumerate(zip(rows, colors, headings)):
        top = 0.96 - index * 0.33
        axis.text(0.02, top, heading, transform=axis.transAxes, fontsize=12, fontweight="bold", color=color, va="top")
        input_text = row["input"] if len(row["input"]) <= 90 else row["input"][:87] + "..."
        axis.text(0.02, top - 0.065, f"Input: {input_text}", transform=axis.transAxes, fontsize=10.5, color="#27231F", va="top", wrap=True)
        axis.text(0.02, top - 0.15, f"Prediction: {row['predicted_label']}  |  confidence {row['confidence']:.3f}", transform=axis.transAxes, fontsize=11.5, fontweight="bold", color=color, va="top")
        axis.text(0.02, top - 0.215, f"Reference: {row['reference_label']}  |  OOS probability {row['oos_probability']:.3f}", transform=axis.transAxes, fontsize=9.5, color="#6B665F", va="top")
        if index < 2:
            axis.plot([0.02, 0.98], [top - 0.27, top - 0.27], transform=axis.transAxes, color="#D6CEC1", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_demo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
