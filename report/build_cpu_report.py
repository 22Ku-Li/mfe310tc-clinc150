"""Build the completed CPU TextCNN technical report with final test evidence."""

from __future__ import annotations

import json
import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from report.build_report import (
    ACCENT,
    DELIVERY_DIR,
    DRAFT,
    MUTED,
    PROJECT_ROOT,
    citation,
    figure,
    make_styles,
    three_line_table,
)

OUTPUT_PATH = DELIVERY_DIR / "MFE310TC_Technical_Report.pdf"


def build_report() -> dict:
    DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((PROJECT_ROOT / "results/report_assets/final_analysis_summary.json").read_text(encoding="utf-8"))
    quality = json.loads((PROJECT_ROOT / "results/data_checks/data_quality_summary.json").read_text(encoding="utf-8"))
    clean = pd.read_csv(PROJECT_ROOT / "results/summary/seed_summary.csv")
    validation = clean.loc[clean["split"] == "val"].set_index("experiment")
    noise = pd.read_csv(PROJECT_ROOT / "results/summary/robustness_seed_summary.csv")
    baseline_noise = pd.read_csv(PROJECT_ROOT / "results/robustness/m0_tfidf_logreg/test/noise_metrics.csv")
    m0 = analysis["m0_test"]
    m1 = analysis["m1_test_mean"]
    m1sd = analysis["m1_test_std"]
    m2 = analysis["m2_test_mean"]
    m2sd = analysis["m2_test_std"]

    document = BaseDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.6 * cm,
        title="Robust Intent Classification and OOS Detection with TextCNN and R-Drop",
        author="STUDENT_NAME_REDACTED (Student ID STUDENT_ID_REDACTED), individual project",
        subject="MFE310TC Natural Language Processing and Deep Learning coursework",
    )
    cover_frame = Frame(2.0 * cm, 1.8 * cm, A4[0] - 4.0 * cm, A4[1] - 3.6 * cm, id="cover")
    gutter = 0.6 * cm
    column_width = (A4[0] - 3.2 * cm - gutter) / 2
    left = Frame(1.6 * cm, 1.6 * cm, column_width, A4[1] - 3.4 * cm, id="left")
    right = Frame(1.6 * cm + column_width + gutter, 1.6 * cm, column_width, A4[1] - 3.4 * cm, id="right")

    def later_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("LiberationSerif", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.6 * cm, A4[1] - 1.0 * cm, "MFE310TC | TextCNN intent classification and OOS detection")
        canvas.setFillColor(ACCENT)
        canvas.drawRightString(A4[0] - 1.6 * cm, A4[1] - 1.0 * cm, "CPU STUDY | REPRODUCIBILITY LIMITS DISCLOSED")
        canvas.setStrokeColor(colors.HexColor("#aaaaaa"))
        canvas.setLineWidth(0.4)
        canvas.line(1.6 * cm, A4[1] - 1.15 * cm, A4[0] - 1.6 * cm, A4[1] - 1.15 * cm)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(A4[0] / 2, 0.85 * cm, f"Page {doc.page}")
        canvas.restoreState()

    document.addPageTemplates([
        PageTemplate(id="Cover", frames=[cover_frame]),
        PageTemplate(id="TwoCol", frames=[left, right], onPage=later_page),
    ])
    styles = make_styles()
    story = []

    # Cover sheet.
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Xi’an Jiaotong-Liverpool University", ParagraphStyle("UniversityCPU", fontName="LiberationSerif-Bold", fontSize=14, leading=18, alignment=TA_CENTER, textColor=ACCENT)))
    story.append(Spacer(1, 0.5 * cm))
    module_data = [
        ["Paper CODE", "Module Leader", "Department", "Email"],
        ["MFE310TC", "MODULE_LEADER_NAME_REDACTED", Paragraph("School of Intelligent Manufacturing Ecosystem", ParagraphStyle("DeptCover", fontName="LiberationSerif", fontSize=8, leading=9, alignment=TA_CENTER)), "MODULE_LEADER_EMAIL_REDACTED"],
    ]
    module_table = Table(module_data, colWidths=[2.5 * cm, 3.0 * cm, 6.0 * cm, 4.8 * cm])
    module_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "LiberationSerif-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "LiberationSerif"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#777777")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(module_table)
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph("2026-27 Semester 1 - Coursework 1", ParagraphStyle("CoverLineCPU", fontName="LiberationSerif-Bold", fontSize=12, leading=16, alignment=TA_CENTER)))
    story.append(Paragraph("Postgraduate - Year 1", ParagraphStyle("CoverLine2CPU", fontName="LiberationSerif", fontSize=11, leading=15, alignment=TA_CENTER)))
    story.append(Paragraph("Natural Language Processing and Deep Learning", ParagraphStyle("CoverModuleCPU", fontName="LiberationSerif", fontSize=12, leading=16, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.7 * cm))
    story.append(Paragraph("Robust Intent Classification and Out-of-Scope Detection for an Intelligent Manufacturing Assistant", ParagraphStyle("CoverTitleCPU", fontName="LiberationSerif-Bold", fontSize=21, leading=26, alignment=TA_CENTER, textColor=ACCENT)))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("A Controlled TextCNN and R-Drop Study on CLINC150", ParagraphStyle("CoverSubtitleCPU", fontName="LiberationSerif-Italic", fontSize=12, leading=16, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("CPU EXPERIMENTS COMPLETE - APPROVED GPU REQUIREMENT NOT MET", ParagraphStyle("DraftWarningCPU", fontName="LiberationSerif-Bold", fontSize=10.5, leading=15, alignment=TA_CENTER, textColor=DRAFT)))
    story.append(Spacer(1, 0.7 * cm))
    student_data = [
        ["Student Name", "Student ID"],
        ["STUDENT_NAME_REDACTED", "STUDENT_ID_REDACTED"],
        ["School Email", "Group"],
        ["STUDENT_EMAIL_REDACTED", "Individual project (instructor approved)"],
    ]
    student_table = Table(student_data, colWidths=[7.0 * cm, 7.0 * cm])
    student_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "LiberationSerif"),
        ("FONTNAME", (0, 0), (-1, 0), "LiberationSerif-Bold"),
        ("FONTNAME", (0, 2), (-1, 2), "LiberationSerif-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(student_table)
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Submission deadline: 1 November 2026", ParagraphStyle("DeadlineCPU", fontName="LiberationSerif-Bold", fontSize=11, leading=15, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("All M0/M1/M2 validation, final test, and spelling-noise results in this document were generated by the accompanying code. Approved remote Linux GPU execution is explicitly not claimed.", ParagraphStyle("CoverNoteCPU", fontName="LiberationSerif", fontSize=9, leading=13, alignment=TA_CENTER, textColor=MUTED)))
    story.append(NextPageTemplate("TwoCol"))
    story.append(PageBreak())

    counted_text = []

    def add_body(text: str):
        counted_text.append(text)
        story.append(Paragraph(text, styles["ReportBody"]))

    def add_heading(text: str, level: int = 1):
        counted_text.append(text)
        story.append(Paragraph(text, styles[f"ReportHeading{level}"]))

    abstract = (
        f"Abstract-We compare TF-IDF logistic regression (M0), a compact TextCNN (M1), and a controlled R-Drop variant (M2) for 150-intent classification and out-of-scope (OOS) detection on CLINC150. M0 gives the highest test Macro-F1 ({m0['macro_f1']:.4f}) and in-scope Macro-F1 ({m0['in_scope_macro_f1']:.4f}), but catches only {m0['oos_recall']:.1%} of unsupported requests. M1 and M2 raise mean OOS recall to {m1['oos_recall']:.1%} and {m2['oos_recall']:.1%}, with OOS F1 of {m1['oos_f1']:.4f} and {m2['oos_f1']:.4f}. R-Drop raises mean test Macro-F1 from {m1['macro_f1']:.4f} to {m2['macro_f1']:.4f}; the paired difference is positive for all three seeds. It also improves Macro-F1 under 5% and 10% spelling noise, although M0 remains more robust overall. All reported experiments ran on CPU. The approved remote Linux GPU requirement is not met."
    )
    keywords = "Keywords-intent classification, out-of-scope detection, CLINC150, TextCNN, R-Drop, robustness"
    counted_text.extend([abstract, keywords])
    story.append(Paragraph(abstract, styles["ReportBody"]))
    story.append(Paragraph(keywords, styles["ReportBody"]))

    add_heading("I. INTRODUCTION")
    add_body(
        "A reliable assistant should do more than route supported commands to the right function. It should also recognize when a request falls outside its scope. For a factory worker asking about maintenance records, machine status, or an operating instruction, either kind of failure matters. A wrong label can lead to an irrelevant function, while accepting an unsupported command can suggest that the assistant can do something it cannot. We therefore evaluate intent classification and OOS behaviour together."
    )
    add_body(
        f"We use CLINC150 as a controlled benchmark for this study{citation(1)}. It is not a factory corpus, and we do not treat benchmark performance as evidence of industrial readiness. The manufacturing setting simply gives us a concrete way to examine reliability. Our experiments ask three questions: how well a transparent baseline works, whether a small convolutional network offers a different accuracy-OOS trade-off, and whether one controlled R-Drop change improves clean and noisy performance."
    )
    add_body(
        "We initially planned to use DistilBERT, but could not confirm access to an approved GPU terminal. We therefore changed the principal model to a TextCNN that can be trained and checked on CPU. This is a deviation from the specified environment, not a workaround that satisfies it. The functional and evaluation work is complete, but we make no claim of having completed the assessed remote Linux GPU practice."
    )

    add_heading("II. PROBLEM DEFINITION AND OBJECTIVES")
    add_body(
        "Each utterance is treated as a 151-way single-label problem: one of 150 supported intents, or OOS. The intended users are operators and engineers working with a constrained assistant. The system returns a routing decision only; it does not generate an answer or control equipment. An unsupported or ambiguous request should be clarified or passed to a person, not forced into the closest available intent."
    )
    add_body(
        "We had five objectives: fit a credible baseline without using validation or test data to learn features; compare M0, M1, and M2 on the same splits and metrics; measure overall and in-scope performance together with OOS precision, recall, F1, and AUROC; test deterministic spelling noise; and inspect representative errors, including cases where R-Drop changes the prediction. Macro-F1 is used for validation selection. The recorded protocol states that all six M1/M2 runs and model configurations were frozen before test evaluation."
    )

    add_heading("III. DATASET DESCRIPTION AND GOVERNANCE")
    split_counts = {row["split"]: row for row in quality["split_summary"]}
    add_body(
        f"CLINC150 contains {quality['total_rows']:,} English utterances covering 150 supported intents and OOS{citation(1)}. We use the UCI <i>data_full</i> release and keep the original partitions: {split_counts['train']['rows']:,} training, {split_counts['val']['rows']:,} validation, and {split_counts['test']['rows']:,} test examples{citation(2)}. Each supported intent has 100 training examples, with a further 100 OOS examples. Validation has 100 OOS cases, but the test set has 1,000. OOS estimates are therefore less stable on validation data and should not be expected to transfer exactly to test."
    )
    add_body(
        "The two official sources do not report the same license version. UCI's metadata lists CC BY 4.0, whereas the LICENSE file inside the downloaded bundle states Attribution 3.0 Unported. The package retains that file and records the discrepancy rather than silently selecting one version. The dataset and paper are cited in all distributed materials."
    )
    split_table = [
        ["Split", "In-scope", "OOS", "Total"],
        ["Train", "15,000", "100", "15,100"],
        ["Validation", "3,000", "100", "3,100"],
        ["Test", "4,500", "1,000", "5,500"],
    ]
    story.append(KeepTogether([
        Paragraph('<a name="table1cpu"/>TABLE I. FIXED CLINC150 SPLIT SIZES', styles["ReportCaption"]),
        three_line_table(split_table, [2.3 * cm, 2.1 * cm, 1.5 * cm, 1.6 * cm], aligns=["LEFT", "RIGHT", "RIGHT", "RIGHT"]),
    ]))
    story.append(Spacer(1, 5))
    add_body(
        "Quality checks found no missing labels, missing text, empty utterances, or within-split duplicate text. Training utterances average 8.33 whitespace-delimited words, with median 8, a 95th percentile of 14, and maximum 28. Source files and the archive are SHA-256 verified. Processed CSV files retain stable sample identifiers and source indices. Vocabulary construction uses training utterances only, with minimum token frequency two and a maximum nominal size of 20,000; the realized vocabulary contains 2,875 tokens including padding and unknown symbols."
    )
    add_body(
        f"All queries are English and single-intent. Crowd workers paraphrased seed phrases or responded to scenarios{citation(1)}. The benchmark therefore captures useful lexical variation but not multi-intent commands, dialogue context, accented speech transcripts, or specialized maintenance vocabulary. OOS is heterogeneous rather than a coherent semantic class, so a model may learn the observed unsupported style without covering every possible unsupported request."
    )
    add_body(
        "The audit found three exact or normalized texts shared between train and validation and two shared between train and test. Four normalized texts have conflicting labels across splits. For example, “what is on my to do list” is associated with <i>reminder</i> in validation but <i>todo_list</i> in training. These examples are retained to preserve the official benchmark and are reported as leakage and annotation limitations rather than silently removed."
    )
    story.append(figure(PROJECT_ROOT / "results/data_checks/figures/split_counts.png", "Fig. 1. Fixed split sizes with OOS shown separately.", "fig1cpu", styles))

    add_heading("IV. METHODOLOGY")
    add_heading("A. M0: TF-IDF logistic regression", 2)
    add_body(
        f"M0 combines word 1-2 gram and character 3-5 gram TF-IDF features with multinomial logistic regression{citation(3)}. Both vectorizers use sublinear term-frequency scaling, lowercase text, Unicode accent stripping, minimum document frequency two, and at most 50,000 features each. Their union produces 32,755 features. Logistic regression uses C=1.0, lbfgs, at most 300 iterations, and seed 42. Feature fitting occurs on training data only."
    )
    add_heading("B. M1: compact TextCNN", 2)
    add_body(
        f"M1 applies convolutional filters of widths 3, 4, and 5 to 100-dimensional trainable word embeddings, with 100 filters per width. ReLU activation and max-over-time pooling produce a 300-dimensional sentence representation, followed by dropout 0.5 and a 151-way linear classifier. This design models local word-order patterns while remaining small enough for CPU experimentation{citation(4)}. Utterances are truncated or padded to 32 tokens, and padding is masked during pooling. The realized model has 453,251 parameters."
    )
    add_body(
        "M1 is trained for at most 12 epochs with batch size 64, AdamW, learning rate 0.001, weight decay 0.0001, 5% linear warmup, and gradient clipping 5.0. Early stopping uses validation Macro-F1 with patience three. M1 and M2 use the same vocabulary, initialization policy, data order, optimizer, scheduler, and evaluation code. Each is run with seeds 13, 42, and 87. No pretrained model or external API is required."
    )
    add_heading("C. M2: one controlled R-Drop modification", 2)
    add_body(
        f"M2 changes only the training objective. Each training utterance passes through the dropout-enabled network twice. The objective is the mean cross-entropy over the two predictions plus a symmetric Kullback-Leibler consistency term{citation(5)}. An initial weight of 4.0 reduced validation Macro-F1 to 0.8304 and OOS F1 to 0.2247 for seed 13; it was therefore rejected using validation evidence only. Weight 1.0 gave a small clean-score trade-off but improved 5% and 10% validation-noise Macro-F1, so it was frozen for all remaining seeds. No test result informed this choice."
    )
    add_heading("D. Evaluation and reproducibility", 2)
    add_body(
        "Overall accuracy measures the total decision rate, while Macro-F1 treats each label equally. Separate in-scope metrics prevent OOS behaviour from being hidden. OOS precision, recall, F1, and AUROC are reported together because ranking quality and default decision quality differ. Three paired seeds provide mean and standard deviation, not a claim of statistical significance. The archive contains configuration snapshots, data hashes, epoch histories, environment records and predictions. The neural training code saves best checkpoints, but those original weights are absent from the submitted package. The scripted final-test workflow requires explicit confirmation and frozen checkpoints."
    )

    add_heading("V. FINAL TEST RESULTS")
    add_heading("A. Clean-set model comparison", 2)
    add_body(
        "Table II gives the frozen test results. For M1 and M2, we report the mean and standard deviation across three seeds. M0 was trained once with seed 42. The two neural models do not dominate the baseline: their main advantage lies in OOS detection."
    )
    result_rows = [["Metric", "M0 TF-IDF", "M1 TextCNN", "M2 R-Drop"]]
    display_metrics = [
        ("Accuracy", "accuracy"),
        ("Macro-F1", "macro_f1"),
        ("In-scope Macro-F1", "in_scope_macro_f1"),
        ("OOS recall", "oos_recall"),
        ("OOS F1", "oos_f1"),
        ("OOS AUROC", "oos_auroc"),
    ]
    for label, metric in display_metrics:
        result_rows.append([
            label,
            f"{m0[metric]:.4f}",
            f"{m1[metric]:.4f} ± {m1sd[metric]:.4f}",
            f"{m2[metric]:.4f} ± {m2sd[metric]:.4f}",
        ])
    story.append(KeepTogether([
        Paragraph('<a name="table2cpu"/>TABLE II. FINAL TEST RESULTS (MEAN ± SEED SD)', styles["ReportCaption"]),
        three_line_table(result_rows, [2.6 * cm, 1.65 * cm, 2.0 * cm, 2.0 * cm], font_size=7.2, aligns=["LEFT", "RIGHT", "RIGHT", "RIGHT"]),
    ]))
    story.append(Spacer(1, 5))
    add_body(
        f"M0 remains the better ordinary classifier. Its Macro-F1 is {m0['macro_f1']:.4f} and its in-scope Macro-F1 is {m0['in_scope_macro_f1']:.4f}, compared with {m2['macro_f1']:.4f} and {m2['in_scope_macro_f1']:.4f} for M2. The advantage reverses for unsupported requests. Mean OOS recall rises from {m0['oos_recall']:.1%} with M0 to {m1['oos_recall']:.1%} with M1 and {m2['oos_recall']:.1%} with M2. OOS F1 also rises from {m0['oos_f1']:.4f} to {m1['oos_f1']:.4f} and {m2['oos_f1']:.4f}. Which model is preferable therefore depends on whether the application prioritizes ordinary intent accuracy or rejection of out-of-scope input."
    )
    add_body(
        f"M2 improves mean clean Macro-F1 by {m2['macro_f1'] - m1['macro_f1']:.4f} over M1. The paired gains are {analysis['clean_macro_f1_delta_by_seed']['13']:.4f}, {analysis['clean_macro_f1_delta_by_seed']['42']:.4f}, and {analysis['clean_macro_f1_delta_by_seed']['87']:.4f} for seeds 13, 42, and 87, respectively. The effect is small but consistent on the final test set. It is not uniform across all metrics: M2 reduces mean OOS recall by {m1['oos_recall'] - m2['oos_recall']:.4f} and OOS F1 by {m1['oos_f1'] - m2['oos_f1']:.4f}."
    )
    story.append(figure(PROJECT_ROOT / "results/report_assets/final_model_comparison.png", "Fig. 2. Final test comparison; error bars show seed standard deviations.", "fig2cpu", styles))

    add_heading("B. Validation evidence and test discipline", 2)
    add_body(
        f"Mean validation Macro-F1 was {validation.loc['m1_textcnn_ce', 'macro_f1_mean']:.4f} ± {validation.loc['m1_textcnn_ce', 'macro_f1_std']:.4f} for M1 and {validation.loc['m2_textcnn_rdrop', 'macro_f1_mean']:.4f} ± {validation.loc['m2_textcnn_rdrop', 'macro_f1_std']:.4f} for M2. R-Drop was slightly lower on average, with improvements in some seeds and declines in others. We therefore report the positive final-test differences as observations, not as an outcome guaranteed by validation. No R-Drop weight or model choice was changed after seeing the 5,500-row test results."
    )

    add_heading("C. Spelling-noise robustness", 2)
    add_body(
        "Noise uses deletion, adjacent swaps and substitutions with fixed per-row seeds. The 5% and 10% rates determine the rounded number of attempted edits on alphabetic characters, with at least one attempt for an input containing alphabetic characters. Attempts can overlap or leave a character unchanged, so these are nominal rates. Every model receives the same corrupted utterance. We evaluate 5% and 10% noise on the frozen test set without further training or threshold adjustment. This is a test of surface corruption only; it says little about paraphrases, domain shift, or deliberately adversarial input."
    )
    noise_rows = [["Model", "Clean", "5% noise", "10% noise"]]
    for model_name, experiment in [("M1 TextCNN", "m1_textcnn_ce"), ("M2 R-Drop", "m2_textcnn_rdrop")]:
        values = []
        for rate in [0.0, 0.05, 0.10]:
            if rate == 0.0:
                mean = m1["macro_f1"] if experiment == "m1_textcnn_ce" else m2["macro_f1"]
                std = m1sd["macro_f1"] if experiment == "m1_textcnn_ce" else m2sd["macro_f1"]
            else:
                row = noise.loc[(noise["split"] == "test") & (noise["experiment"] == experiment) & (noise["noise_rate"] == rate)].iloc[0]
                mean, std = row["macro_f1_mean"], row["macro_f1_std"]
            values.append(f"{mean:.4f} ± {std:.4f}")
        noise_rows.append([model_name, *values])
    noise_rows.append(["M0 TF-IDF", *[f"{value:.4f}" for value in baseline_noise["macro_f1"]]])
    story.append(KeepTogether([
        Paragraph('<a name="table3cpu"/>TABLE III. FINAL TEST MACRO-F1 UNDER SPELLING NOISE', styles["ReportCaption"]),
        three_line_table(noise_rows, [2.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm], font_size=7.2, aligns=["LEFT", "RIGHT", "RIGHT", "RIGHT"]),
    ]))
    story.append(Spacer(1, 5))
    add_body(
        f"At 5% noise, mean Macro-F1 rises from {analysis['noise_macro_f1_mean']['m1_textcnn_ce']['0.05']:.4f} with M1 to {analysis['noise_macro_f1_mean']['m2_textcnn_rdrop']['0.05']:.4f} with M2. At 10%, it rises from {analysis['noise_macro_f1_mean']['m1_textcnn_ce']['0.1']:.4f} to {analysis['noise_macro_f1_mean']['m2_textcnn_rdrop']['0.1']:.4f}. Every seed shows the same direction at both rates. Even so, M0 remains well ahead in absolute noise Macro-F1 ({baseline_noise.iloc[1]['macro_f1']:.4f} and {baseline_noise.iloc[2]['macro_f1']:.4f}), and TextCNN loses more accuracy as corruption increases. R-Drop reduces this weakness; it does not solve it."
    )
    story.append(figure(PROJECT_ROOT / "results/report_assets/final_noise_comparison.png", "Fig. 3. Final test Macro-F1 under deterministic spelling noise.", "fig3cpu", styles))
    story.append(figure(PROJECT_ROOT / "results/report_assets/rdrop_paired_effects.png", "Fig. 4. Paired M2−M1 Macro-F1 differences for every seed and noise rate.", "fig4cpu", styles))

    add_heading("D. Systematic error analysis", 2)
    add_body(
        "Most large confusion groups in the predeclared seed-42 analysis are OOS utterances accepted as supported intents. The most frequent are OOS→<i>recipe</i> (33 cases), OOS→<i>definition</i> (18), OOS→<i>smart_home</i> (17), and OOS→<i>weather</i> (17). Unsupported requests can therefore share strong lexical cues with valid functions. Errors within the supported classes look different: <i>calendar</i>→<i>calendar_update</i>, for example, hinges on whether the user wants information or a change."
    )
    add_body(
        "The paired examples help explain R-Drop's mixed effect. For “what does post rush hour traffic generally look like heading into the lincoln tunnel in manhattan from new jersey”, M1 predicts OOS with probability 0.717, but M2 correctly predicts <i>traffic</i> with probability 0.939. For “this song needs to be added to my country music playlist”, M1 correctly predicts <i>update_playlist</i>, whereas M2 predicts <i>play_music</i> with probability 0.778. R-Drop fixes some errors and introduces others, sometimes with high confidence. A confidence score by itself is not a safety guarantee."
    )
    add_body(
        "Some disagreements also come from the data. “what is on my to do list” has different labels in different splits, so it should not be counted as an ordinary modelling mistake without comment. We include row-level predictions, shared-error tables, and lists of errors corrected or introduced by M2 so that readers can inspect these cases directly."
    )

    add_heading("VI. DISCUSSION AND RESPONSIBLE APPLICATION")
    add_body(
        "There is no single best model in these results. M0 is inexpensive, interpretable, better on supported intents, and less sensitive to character noise. TextCNN catches more unsupported requests. R-Drop gives small, consistent gains in clean and noisy Macro-F1, at some cost to mean OOS performance. M0 is the sensible choice when ordinary command accuracy matters most. M1 or M2 may be preferable when rejecting unsupported input is more important, provided the lower in-scope score and remaining OOS errors are acceptable."
    )
    add_body(
        "These results do not justify deployment in a factory. CLINC150 contains general assistant queries, not maintenance procedures, equipment alarms, safety interlocks, or engineering abbreviations. A real deployment would need representative approved data, another bias and leakage audit, and tests of multilingual input and speech-recognition errors. It would also need visible scope limits, calibrated rejection, and a safe human fallback for high-impact actions."
    )
    add_body(
        "We retain the dataset paper and both license statements in the package. Before publication, check the code package for credentials, private paths, restricted data, and redistribution restrictions. The original TextCNN checkpoints are not included; a separately retrained M0 baseline is included. Examples should be screened before they appear in screenshots or videos. Future security tests should include malformed and deliberately misleading requests, not just random spelling noise. No external API or closed hosted model is used."
    )
    add_body(
        "Good results do not remove the compliance issue. The course requires principal deep-learning experiments on an approved remote Linux GPU server, whereas every M1/M2 run used local CPU. The metadata records <i>approved_remote_linux_gpu_used=false</i>. We retain the optional DistilBERT GPU workflow in case an approved terminal becomes available, but it has not been executed."
    )

    add_heading("VII. CONCLUSIONS")
    add_body(
        f"M0 is the strongest ordinary classifier, with test Macro-F1 {m0['macro_f1']:.4f} and in-scope Macro-F1 {m0['in_scope_macro_f1']:.4f}. M1 and M2 offer better unsupported-request detection, raising OOS F1 from {m0['oos_f1']:.4f} to {m1['oos_f1']:.4f} and {m2['oos_f1']:.4f}. R-Drop gives small, seed-consistent gains in clean and noisy Macro-F1, but slightly lowers mean OOS performance and does not close the robustness gap to M0. The practical lesson is to select models against the operating objective rather than a single score. The recorded experiments used CPU; reproducing the neural models requires retraining because their original checkpoints are not included. The approved Linux GPU practice requirement remains unmet."
    )

    story.append(FrameBreak())
    story.append(Paragraph("REFERENCES", styles["ReportHeading1"]))
    references = [
        (1, "S. Larson, A. Mahendran, J. J. Peper, C. Clarke, A. Lee, P. Hill, J. K. Kummerfeld, K. Leach, M. A. Laurenzano, L. Tang, and J. Mars, “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” <i>Proceedings of EMNLP-IJCNLP</i>, pp. 1311-1316, 2019. <a href='https://aclanthology.org/D19-1131/' color='blue'>https://aclanthology.org/D19-1131/</a>"),
        (2, "CLINC150 [Dataset]. UCI Machine Learning Repository, 2020. CC BY 4.0. <a href='https://doi.org/10.24432/C5MP58' color='blue'>https://doi.org/10.24432/C5MP58</a>"),
        (3, "F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825-2830, 2011. <a href='https://jmlr.org/papers/v12/pedregosa11a.html' color='blue'>https://jmlr.org/papers/v12/pedregosa11a.html</a>"),
        (4, "Y. Kim, “Convolutional Neural Networks for Sentence Classification,” <i>Proceedings of EMNLP 2014</i>, pp. 1746-1751. <a href='https://aclanthology.org/D14-1181/' color='blue'>https://aclanthology.org/D14-1181/</a>"),
        (5, "X. Liang, L. Wu, J. Li, Y. Wang, Q. Meng, T. Qin, W. Chen, M. Zhang, and T.-Y. Liu, “R-Drop: Regularized Dropout for Neural Networks,” <i>Advances in Neural Information Processing Systems 34</i>, pp. 10890-10905, 2021. <a href='https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html' color='blue'>NeurIPS 2021</a>"),
    ]
    for number, text in references:
        story.append(KeepTogether([Paragraph(f'<a name="ref{number}"/>[{number}] {text}', styles["ReportReference"])]))

    story.append(Paragraph("APPENDIX A - REPRODUCTION AND AI-USE DECLARATION", styles["ReportHeading1"]))
    story.append(Paragraph(
        "Repository: <a href='https://github.com/22Ku-Li/mfe310tc-clinc150' color='blue'>https://github.com/22Ku-Li/mfe310tc-clinc150</a>. This private repository requires marker access to be granted. The submitted ZIP provides the fixed data, code and evidence snapshot; README.md documents reproduction commands. Original TextCNN checkpoints are absent. A separately retrained M0 reproduced all 3,100 archived validation labels; probability equivalence was not checked. An independent audit passed 1,256 data and metric checks; historical neural training and the full test suite were not independently rerun.",
        styles["ReportSmall"],
    ))
    story.append(Paragraph(
        "AI-use declaration: Kimi Work assisted with planning, code, CPU experiments, figures and drafting. ChatGPT assisted with subsequent review, revisions, metric verification, M0 retraining and packaging. AI assistance does not replace the student's responsibility to verify and understand every submission. The student reports instructor approval for an individual group. The required individual recording is a separate submission. Approved remote Linux GPU execution is not claimed.",
        styles["ReportSmall"],
    ))

    document.build(story)
    plain = re.sub(r"<[^>]+>", " ", " ".join(counted_text))
    words = re.findall(r"\b[\w’'-]+\b", plain)
    build_info = {
        "output": OUTPUT_PATH.name,
        "approximate_body_words_excluding_references_and_appendices": len(words),
        "status": "cpu_report_prepared_private_repository_access_required",
        "model_variant": "TextCNN",
        "m1_m2_formal_cpu_results_present": True,
        "official_test_results_present": True,
        "approved_remote_linux_gpu_used": False,
    }
    (DELIVERY_DIR / "report_build_info.json").write_text(json.dumps(build_info, indent=2), encoding="utf-8")
    return build_info


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
