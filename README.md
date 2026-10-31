# Robust Intent Classification and OOS Detection on CLINC150

> **Repository privacy and scope.** Student-identifying details in the report-building and historical audit materials have been replaced with explicit REDACTED placeholders. This copy is for code review; it is not the personalised coursework report. Replace the placeholders only in a private local copy when producing a personal report. Historical metrics and predictions were not changed. The independently retrained M0 demo model is included as verified binary parts; the original TextCNN checkpoints remain unavailable.

A clean clone does not contain the raw or processed dataset. Run `python -m src.download_data` followed by `python -m src.data` before scripts requiring them. The dataset attribution and license notice are preserved under `LICENSES/`. The Windows absolute paths and student identifiers were removed from this repository copy, so historical command metadata and report-audit text are privacy-redacted, not verbatim records. The coursework submission ZIP and personalised report remain separate.

MFE310TC coursework project: compare a TF-IDF baseline with a TextCNN model and one controlled R-Drop modification for 150-intent classification plus out-of-scope (OOS) detection.

## Research question

How accurately can a manufacturing-assistant text interface classify supported user commands and reject unsupported requests, and how robust are the models to spelling noise?

## Current status

Repository: https://github.com/22Ku-Li/mfe310tc-clinc150. It is private; marker access must be granted before assessment. The accompanying submission ZIP is the fixed data, code and result snapshot.


- [x] Pin and verify the official UCI `data_full` dataset.
- [x] Preserve the original train/validation/test splits.
- [x] Audit missing values, duplicates, cross-split overlap, label conflicts, class balance, and text lengths.
- [x] Train and evaluate M0: word/character TF-IDF + logistic regression on validation and frozen test data.
- [x] Complete M1 TextCNN and M2 TextCNN + R-Drop with seeds 13, 42, and 87 on CPU.
- [x] Run final test and matched 5%/10% spelling-noise evaluation after configuration freeze.
- [x] Add controlled-configuration gates, multi-seed aggregation, paired M2−M1 differences, and report-ready error transitions.
- The prior package recorded thirteen passing unit/integration tests and tiny-model smoke tests; those historical claims have not been reverified after this package revision.
- [x] Implement an optional DistilBERT GPU workflow for later use if an approved terminal becomes available.
- [ ] Approved remote Linux GPU practice: **not completed and not claimed**.

Final test results are present. Archived records state that they were generated after model/configuration freeze; saved metadata does not independently prove historical test ordering. All formal TextCNN runs used CPU; synthetic smoke-test outputs were deleted and are not coursework evidence.

## Reproduction

Python 3.12 was recorded for the baseline. `requirements-lock.txt` contains baseline dependencies and does not include PyTorch; it is not a complete deep-learning environment lock. For the TextCNN runs, metadata records Windows 11, Python 3.12.14, NumPy 2.5.4, pandas 3.0.2 and PyTorch 2.14.1+cpu, whereas the baseline lock specifies NumPy 2.4.4. Exact environment reconstruction and package availability have not been independently verified. Install a compatible PyTorch build separately for the neural-model code; the versions in `requirements.txt` are ranges, not a verified exact lock.

The delivered archive does **not** include the original trained TextCNN checkpoints or the original baseline model artifact. A separately retrained baseline is included for an illustrative live demonstration (see below). Stored TextCNN predictions and metrics can be inspected, but TextCNN inference and checkpoint-based evaluation require recovering the original artifacts or retraining. Retraining must be checked against the saved outcomes; matching results are not guaranteed. Run the following commands in a separate working copy if you need to preserve the delivered results.

For baseline preparation, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt  # includes PyTorch; range-based, not an exact historical lock

# The repository excludes the raw dataset; download and verify it first.
python -m src.download_data

# Validate the raw data, create processed splits, and run quality checks.
python -m src.data

# Train M0 and evaluate only on validation data.
python -m src.train_baseline --config configs/baseline.yaml --split val

# Run unit tests.
python -m unittest discover -s tests -v  # complete suite requires PyTorch
```

On a Linux server, the baseline script uses the included raw data when present. In a separate copy, `ALLOW_OVERWRITE_RESULTS=1 bash scripts/run_baseline.sh` explicitly permits replacement of saved baseline results.

## Controlled M1/M2 experiment

The submitted results compare the following models; DistilBERT is a separate optional workflow and was not used for the reported experiments.

| Model | Training objective | Role |
|---|---|---|
| M0 | Word/character TF-IDF + logistic regression | Reference baseline |
| M1 | TextCNN + cross-entropy | Principal deep-learning model |
| M2 | Same TextCNN + cross-entropy and R-Drop | Controlled modification, `alpha=1.0` |

M1 and M2 use the same TextCNN architecture, shared hyperparameters and evaluation protocol, with seeds 13, 42 and 87. The configurations are `configs/textcnn.yaml` and `configs/textcnn_rdrop.yaml`; `python -m src.validate_textcnn_design` checks the controlled comparison. A separate validation-only alpha-4 trial is retained in `results/hyperparameter_tuning/`; it was not the final M2 setting.

### Optional DistilBERT workflow

The optional `scripts/run_transformer_experiments.sh` and `scripts/evaluate_test.sh` target DistilBERT checkpoints, not the reported TextCNN experiments. This workflow was not executed for the presented report and does not provide evidence of approved remote GPU use. Do not run its final-test script for the reported TextCNN comparison.

### Frozen test evaluation

The supplied results state that final tests were run after the model configurations were frozen. Do not use test outcomes to select models, thresholds or hyperparameters. The TextCNN evaluation command below requires recovered or retrained checkpoints; their absence in this ZIP prevents an immediate rerun.

## Dataset provenance and governance

- Dataset: CLINC150, UCI `data_full` version.
- UCI DOI: https://doi.org/10.24432/C5MP58
- Original authors' repository: https://github.com/clinc/oos-eval
- License metadata: the UCI webpage states CC BY 4.0, while the downloaded `data/raw/LICENSE` file states Attribution 3.0 Unported. The discrepancy is preserved and documented; the dataset is used with attribution.
- Download URL: https://archive.ics.uci.edu/static/public/570/clinc150.zip
- Archive SHA-256: `0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2`
- Raw JSON SHA-256: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`

The raw and processed datasets are excluded from Git. They can be restored exactly with `python -m src.download_data` and `python -m src.data`.

### Fixed split sizes

| Split | In-scope | OOS | Total |
|---|---:|---:|---:|
| Train | 15,000 | 100 | 15,100 |
| Validation | 3,000 | 100 | 3,100 |
| Test | 4,500 | 1,000 | 5,500 |

The training set contains 100 examples for each of the 150 in-scope intents and 100 OOS examples. OOS is treated as the 151st class.

### Data-quality findings

- No missing labels, missing text, empty utterances, or within-split duplicate text were found.
- Three exact/normalized texts are shared between train and validation; two are shared between train and test. No validation-test exact overlap was found.
- Four normalized texts have conflicting labels across splits. These are retained to preserve the official benchmark and are documented as a leakage/annotation limitation rather than silently removed.
- Training utterances contain 8.33 words on average (median 8, 95th percentile 14, maximum 28).

Detailed evidence is in `results/data_checks/`, including the dataset manifest, split summary, label distribution, overlap audit, conflicting-label examples, and figures.

## Completed CPU TextCNN experiment

The assessed deep-learning comparison uses a compact TextCNN with trainable embeddings, not DistilBERT. M1 uses cross-entropy; M2 changes only the training objective by adding R-Drop with weight 1.0. Both use seeds 13, 42, and 87. All six formal runs were executed on CPU, and `approved_remote_linux_gpu_used=false` is recorded in their metadata.

```bash
# In a separate working copy: rerun six training runs and the validation summary.
# This command overwrites result files; compare new output with archived evidence.
ALLOW_OVERWRITE_RESULTS=1 bash scripts/run_textcnn_experiments.sh

# After all models are frozen, evaluate a checkpoint explicitly.
python -m src.evaluate_textcnn \
  --checkpoint checkpoints/m2_textcnn_rdrop/seed_42/best \
  --split test --device cpu --allow-test

# Aggregate clean and noisy results and rebuild figures/report.
python -m src.aggregate_results --require-all-seeds
python -m src.aggregate_robustness
python -m src.analyze_textcnn
python -m report.build_cpu_report
```

### Final test results

M1/M2 values are mean ± seed standard deviation.

| Metric | M0 TF-IDF | M1 TextCNN | M2 R-Drop |
|---|---:|---:|---:|
| Accuracy | 0.7964 | 0.7942 ± 0.0021 | 0.7972 ± 0.0062 |
| Macro-F1 | 0.8508 | 0.8293 ± 0.0018 | 0.8343 ± 0.0044 |
| In-scope Macro-F1 | 0.9148 | 0.8778 ± 0.0013 | 0.8842 ± 0.0032 |
| OOS recall | 0.2710 | 0.4373 ± 0.0042 | 0.4227 ± 0.0185 |
| OOS F1 | 0.4176 | 0.5862 ± 0.0066 | 0.5746 ± 0.0203 |
| OOS AUROC | 0.9240 | 0.9410 ± 0.0053 | 0.9377 ± 0.0037 |

R-Drop improves clean Macro-F1 for every seed and improves noisy Macro-F1 for every seed at both 5% and 10% noise. M0 remains stronger on in-scope language and absolute noise robustness; TextCNN provides better default OOS detection.

| Model | Clean Macro-F1 | 5% noise | 10% noise |
|---|---:|---:|---:|
| M0 TF-IDF | 0.8508 | 0.7898 | 0.7246 |
| M1 TextCNN | 0.8293 ± 0.0018 | 0.7131 ± 0.0045 | 0.5921 ± 0.0060 |
| M2 R-Drop | 0.8343 ± 0.0044 | 0.7210 ± 0.0036 | 0.6034 ± 0.0027 |

Complete histories, final predictions, paired summaries, robustness results, error transitions, and report figures are under `results/`.

### Independent verification

The current result audit recomputes saved metrics independently of the production evaluation helpers:

```bash
python audit/audit_results.py  # current audit; missing checkpoints reported separately
```

The result audit covers 30 prediction files, 23,700 raw rows, all clean/noisy metrics, seed summaries, paired deltas, split identity, and byte-identical noise inputs. It writes fresh output to results/audit/current_recheck/ and does not overwrite historical audits. Use --require-checkpoints to fail if any checkpoint is absent. Saved metadata is not independent proof of past training or test ordering. `audit/audit_report.py` is retained as a historical audit of an earlier delivery layout; it expects obsolete filenames and statements and is not the validation command for this final report. The final report review is recorded in `results/audit/finalization/report_final_audit.json`. Except for `results/audit/current_recheck/` and `results/audit/finalization/`, audit outputs are historical and do not establish that the modified package has passed all checks.

## M0 baseline

The baseline fits preprocessing on training data only:

- Word TF-IDF: 1–2 grams, `min_df=2`, at most 50,000 features.
- Character TF-IDF: 3–5 grams within word boundaries, `min_df=2`, at most 50,000 features.
- Logistic regression: `C=1.0`, `lbfgs`, at most 300 iterations.
- Seed: 42. The recorded run produced 32,755 combined features.

### Validation results

| Metric | Value |
|---|---:|
| Accuracy | 0.893871 |
| Macro-F1 | 0.900307 |
| In-scope accuracy | 0.912000 |
| In-scope Macro-F1 | 0.912153 |
| OOS precision | 0.833333 |
| OOS recall | 0.350000 |
| OOS F1 | 0.492958 |
| OOS AUROC | 0.950907 |

The high OOS AUROC but low OOS recall shows that ranking OOS scores and making a good default classification decision are different questions. This will be examined in the model comparison and error analysis; no test-driven threshold adjustment is permitted.

The complete configuration, metrics, per-intent scores, top confusions, predictions, and run metadata are stored under `results/m0_tfidf_logreg/val/`. The original baseline artifact is not included. A separately retrained M0 model is included as described below.

### Deterministic spelling-noise results

Run `python -m src.robustness --split val` and `python -m src.analyze_baseline` to regenerate the challenge predictions, figures, and representative error tables.

| Noise rate | Accuracy | Macro-F1 | OOS F1 | OOS AUROC |
|---:|---:|---:|---:|---:|
| 0% | 0.893871 | 0.900307 | 0.492958 | 0.950907 |
| 5% | 0.820000 | 0.829812 | 0.400000 | 0.917750 |
| 10% | 0.756774 | 0.768500 | 0.378378 | 0.881213 |

The challenge uses fixed per-row seeds and identical noisy inputs for every model. Results and figures are under `results/robustness/m0_tfidf_logreg/val/`. These are validation measurements, not final test results.

## Repository layout

```text
configs/                       Immutable M0/M1/M2 experiment configurations
data/                          Raw and processed data are excluded; download and prepare locally
results/                       Data audits, metrics, logs, and predictions
audit/                         Independent result and report audit scripts
scripts/                       GPU execution, frozen-test, and smoke-test entry points
src/data.py                    Split-preserving loader and quality audit
src/download_data.py           Pinned download and SHA-256 verification
src/prepare_model.py           Shared pretrained-model snapshot and provenance
src/losses.py                  Cross-entropy and R-Drop objectives
src/train_baseline.py          M0 training and evaluation
src/train_transformer.py       Optional DistilBERT GPU training workflow
src/evaluate_transformer.py    Frozen-checkpoint evaluation for optional DistilBERT
src/textcnn_common.py          TextCNN vocabulary, tensors, and model
src/train_textcnn.py           Completed M1/M2 CPU training workflow
src/evaluate_textcnn.py        Frozen TextCNN clean/noise evaluation
src/aggregate_results.py       Multi-seed and paired-difference summaries
src/aggregate_robustness.py    Multi-seed noise and paired-difference summaries
src/robustness.py              Deterministic spelling-noise evaluation
src/analyze_baseline.py        Baseline figures and representative errors
src/analyze_textcnn.py         Final TextCNN figures and error transitions
src/validate_experiment_design.py Controlled-comparison gate
report/build_cpu_report.py     Final-evidence two-column report builder
tests/                         Unit and integration tests
artifacts/                     Verified binary parts for the retrained M0 demo model; no original model
checkpoints/                   Expected local trained checkpoints (not included)
```

## References

Larson, S., Mahendran, A., Peper, J. J., Clarke, C., Lee, A., Hill, P., Kummerfeld, J. K., Leach, K., Laurenzano, M. A., Tang, L., & Mars, J. (2019). *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*. Proceedings of EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1131/

Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. Proceedings of EMNLP 2014, 1746–1751. https://aclanthology.org/D14-1181/

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. https://arxiv.org/abs/1910.01108

Liang, X., Wu, L., Li, J., Wang, Y., Meng, Q., Qin, T., Chen, W., Zhang, M., & Liu, T.-Y. (2021). *R-Drop: Regularized Dropout for Neural Networks*. Advances in Neural Information Processing Systems 34, 10890–10905. https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html

## Packaging revision and privacy

ChatGPT assisted in reviewing and revising the packaging, code, audit, report, and presentation and in independently retraining an M0 baseline for demonstration after the originally disclosed Kimi Work assistance. The student must verify the final work and declare the actual tools and uses.

This copy redacts the original private project-root prefix from saved run-metadata command strings, preserving the script name and arguments. These entries document commands from the prior environment, rather than verbatim private absolute paths. Data, predictions, metrics and training logs have not been changed. Historical audit files predate these revisions; the separately saved current recheck is described below. The updated report includes the repository URL and retains the CPU-only and missing-checkpoint disclosures. Marker access, the individual recording and student verification remain necessary.


### Verification of the revised package

A fresh metric-only audit on the supplied data and predictions passed 1,256 checks across 30 prediction files and 23,700 source rows. Six required neural checkpoints were absent. The audit therefore reports `passed_with_missing_checkpoints`; this does not establish training provenance, original test ordering, or independent model reproducibility. The full neural test suite remains unverified in an environment lacking PyTorch.


### Run the independently retrained M0 demonstration

`artifacts/m0_tfidf_logreg_retrained.joblib` is restored automatically from the tracked binary parts under `artifacts/m0_tfidf_logreg_retrained.parts/` when you run the demo. Each part and the complete file are checked against the manifest SHA-256; splitting changes no model bytes. The assembled file is ignored by Git. To restore it explicitly, run `python scripts/restore_demo_model.py`. A missing/corrupt part or a mismatched existing model stops with an error rather than loading or overwriting it.

This is a new CPU-trained baseline, not the original historical checkpoint or a TextCNN model. It was trained from the supplied training split and saved/reloaded in the recorded environment. Its 3,100 validation predicted labels match the historical validation predictions; exact probability equality was not established and the test set was not used for this reproduction. See `results/m0_independent_reproduction/` for provenance and the validation results.

For a fresh Python 3.12 environment, `requirements-demo.txt` records the packages and versions used in the retraining environment; installation into a fresh environment has not been verified. Then run:

```bash
python -m pip install -r requirements-demo.txt
python scripts/demo_baseline.py --examples
python scripts/demo_baseline.py --text "what is the weather tomorrow"
```

The command outputs model predictions and uncalibrated model scores, not guaranteed truth. Explain in any recording that this is M0 baseline inference. The original TextCNN checkpoints remain unavailable. The separately retrained M0 demonstration model is tracked as binary parts in this repository. For a custom `--model` path, also pass its trusted `--sha256`; checksum verification does not make an untrusted pickle safe to load. The original TextCNN checkpoints are absent.
