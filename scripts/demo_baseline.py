"""Run an independently retrained M0 classifier on illustrative text inputs.

This is the TF-IDF/logistic-regression baseline, not the TextCNN model.
The model was retrained on the supplied training split in a new environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np

if __package__:
    from .restore_demo_model import restore_model, sha256_file
else:
    from restore_demo_model import restore_model, sha256_file

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'artifacts' / 'm0_tfidf_logreg_retrained.joblib'
EXAMPLES = [
    'what is the weather tomorrow',
    'how do i repair an industrial robot controller',
]


def predict_one(model, text: str) -> None:
    """Print the highest-scoring class and its uncalibrated model score."""
    probs = model.predict_proba([text])[0]
    top = int(np.argmax(probs))
    label = str(model.classes_[top])
    print(f'Input: {text}')
    print(f'Predicted label: {label}')
    print(f'Model score: {float(probs[top]):.4f}')
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ARTIFACT)
    parser.add_argument('--sha256', help='Expected SHA-256 required for a custom model file')
    parser.add_argument('--examples', action='store_true', help='Run two illustrative inputs')
    parser.add_argument('--text', help='Classify a single input')
    args = parser.parse_args()
    if not args.examples and not args.text:
        parser.error('Specify --examples or --text "your input".')
    try:
        if args.model.resolve() == ARTIFACT.resolve():
            args.model = restore_model(output_path=args.model)
        elif not args.model.is_file():
            parser.error(f'Model file not found: {args.model}. See the reproduction notes.')
        elif not args.sha256 or sha256_file(args.model) != args.sha256.lower():
            parser.error('A custom model requires a matching --sha256 from a trusted source.')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    model = joblib.load(args.model)
    print('M0: TF-IDF + logistic regression (independent CPU retraining)')
    print('Scores are not calibrated guarantees of correctness.\n')
    if args.examples:
        for example in EXAMPLES:
            predict_one(model, example)
    if args.text:
        predict_one(model, args.text)


if __name__ == '__main__':
    main()
