"""Build the evidence-gated MFE310TC working report as a two-column PDF."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont, TTFError
import os


def register_report_fonts() -> str:
    """Keep stable report font names across machines, without extra dependencies.

    MFE_REPORT_FONT_DIR may point to an installed serif font family. If none
    of the supported TrueType families is available, use ReportLab's built-in
    Times family. The fallback can change line wrapping and should be reviewed.
    """
    aliases = (
        "LiberationSerif", "LiberationSerif-Bold",
        "LiberationSerif-Italic", "LiberationSerif-BoldItalic",
    )
    families = (
        ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
         "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf"),
        ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
         "DejaVuSerif-Italic.ttf", "DejaVuSerif-BoldItalic.ttf"),
        ("Times New Roman.ttf", "Times New Roman Bold.ttf",
         "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf"),
        ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
    )
    directories = []
    if os.environ.get("MFE_REPORT_FONT_DIR"):
        directories.append(Path(os.environ["MFE_REPORT_FONT_DIR"]).expanduser())
    if os.environ.get("CODEX_PRIMARY_RUNTIME_ROOT"):
        directories.append(Path(os.environ["CODEX_PRIMARY_RUNTIME_ROOT"]) /
                           "dependencies/native/libreoffice-headless/libreoffice/share/fonts/truetype")
    directories.extend([
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/TTF"),
        Path("/usr/local/share/fonts"),
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path.home() / "Library/Fonts",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    ])
    source = "built-in Times"
    fonts = None
    for directory in directories:
        for files in families:
            paths = [directory / filename for filename in files]
            if not all(path.is_file() for path in paths):
                continue
            try:
                candidate = [TTFont(alias, str(path)) for alias, path in zip(aliases, paths)]
            except (OSError, ValueError, TTFError):
                continue
            fonts = candidate
            source = str(paths[0])
            break
        if fonts is not None:
            break
    if fonts is None:
        faces = ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic")
        fonts = [pdfmetrics.Font(alias, face, "WinAnsiEncoding")
                 for alias, face in zip(aliases, faces)]
    for font in fonts:
        pdfmetrics.registerFont(font)
    pdfmetrics.registerFontFamily(
        aliases[0], normal=aliases[0], bold=aliases[1],
        italic=aliases[2], boldItalic=aliases[3],
    )
    return source


REPORT_FONT_SOURCE = register_report_fonts()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELIVERY_DIR = PROJECT_ROOT.parent / "delivery"
OUTPUT_PATH = DELIVERY_DIR / "MFE310TC_Technical_Report_WORKING_DRAFT.pdf"
ACCENT = colors.HexColor("#263f5f")
MUTED = colors.HexColor("#666666")
DRAFT = colors.HexColor("#8b3038")


def citation(number: int) -> str:
    return f'<super><a href="#ref{number}" color="black">[{number}]</a></super>'


def crossref(anchor: str, label: str) -> str:
    return f'<a href="#{anchor}" color="black"><u>{label}</u></a>'


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["Normal"],
            fontName="LiberationSerif",
            fontSize=9.4,
            leading=12.0,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#222222"),
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading1",
            parent=styles["Heading1"],
            fontName="LiberationSerif-Bold",
            fontSize=10.2,
            leading=12.5,
            alignment=TA_LEFT,
            textColor=ACCENT,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading2",
            parent=styles["Heading2"],
            fontName="LiberationSerif-Italic",
            fontSize=9.6,
            leading=12,
            textColor=ACCENT,
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportCaption",
            parent=styles["Normal"],
            fontName="LiberationSerif",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceBefore=3,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportReference",
            parent=styles["Normal"],
            fontName="LiberationSerif",
            fontSize=8,
            leading=10,
            leftIndent=15,
            firstLineIndent=-15,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSmall",
            parent=styles["Normal"],
            fontName="LiberationSerif",
            fontSize=8,
            leading=10,
            spaceAfter=3,
        )
    )
    return styles


def three_line_table(data, widths, font_size=8.2, aligns=None):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "LiberationSerif-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "LiberationSerif"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.65, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if aligns:
        for column, alignment in enumerate(aligns):
            commands.append(("ALIGN", (column, 0), (column, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def figure(path: Path, caption: str, anchor: str, styles, width=8.0 * cm):
    from PIL import Image as PILImage

    with PILImage.open(path) as source:
        pixel_width, pixel_height = source.size
    image = Image(str(path), width=width, height=width * pixel_height / pixel_width)
    caption_paragraph = Paragraph(f'<a name="{anchor}"/>{caption}', styles["ReportCaption"])
    return KeepTogether([image, caption_paragraph])


def body_paragraph(text: str, styles):
    return Paragraph(text, styles["ReportBody"])


def build_report() -> dict:
    DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((PROJECT_ROOT / "results/m0_tfidf_logreg/val/metrics.json").read_text(encoding="utf-8"))
    quality = json.loads((PROJECT_ROOT / "results/data_checks/data_quality_summary.json").read_text(encoding="utf-8"))
    noise = pd.read_csv(PROJECT_ROOT / "results/robustness/m0_tfidf_logreg/val/noise_metrics.csv")
    errors = pd.read_csv(PROJECT_ROOT / "results/m0_tfidf_logreg/val/errors_in_scope.csv")

    document = BaseDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.6 * cm,
        title="Robust Intent Classification and Out-of-Scope Detection — Working Draft",
        author="MFE310TC project group; student details not supplied",
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
        canvas.drawString(1.6 * cm, A4[1] - 1.0 * cm, "MFE310TC | CLINC150 intent classification and OOS detection")
        canvas.setFillColor(DRAFT)
        canvas.drawRightString(A4[0] - 1.6 * cm, A4[1] - 1.0 * cm, "WORKING DRAFT — NOT FOR SUBMISSION")
        canvas.setStrokeColor(colors.HexColor("#aaaaaa"))
        canvas.setLineWidth(0.4)
        canvas.line(1.6 * cm, A4[1] - 1.15 * cm, A4[0] - 1.6 * cm, A4[1] - 1.15 * cm)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(A4[0] / 2, 0.85 * cm, f"Page {doc.page}")
        canvas.restoreState()

    document.addPageTemplates(
        [
            PageTemplate(id="Cover", frames=[cover_frame]),
            PageTemplate(id="TwoCol", frames=[left, right], onPage=later_page),
        ]
    )
    styles = make_styles()
    story = []

    # Cover sheet.
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Xi’an Jiaotong-Liverpool University", ParagraphStyle("University", fontName="LiberationSerif-Bold", fontSize=14, leading=18, alignment=TA_CENTER, textColor=ACCENT)))
    story.append(Spacer(1, 0.5 * cm))
    module_data = [
        ["Paper CODE", "Module Leader", "Department", "Email"],
        ["MFE310TC", "MODULE_LEADER_NAME_REDACTED", "School of Intelligent Manufacturing Ecosystem", "MODULE_LEADER_EMAIL_REDACTED"],
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
    story.append(Spacer(1, 1.0 * cm))
    story.append(Paragraph("2026–27 Semester 1 — Coursework 1", ParagraphStyle("CoverLine", fontName="LiberationSerif-Bold", fontSize=12, leading=16, alignment=TA_CENTER)))
    story.append(Paragraph("Postgraduate — Year 1", ParagraphStyle("CoverLine2", fontName="LiberationSerif", fontSize=11, leading=15, alignment=TA_CENTER)))
    story.append(Paragraph("Natural Language Processing and Deep Learning", ParagraphStyle("CoverModule", fontName="LiberationSerif", fontSize=12, leading=16, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph("Robust Intent Classification and Out-of-Scope Detection for an Intelligent Manufacturing Assistant", ParagraphStyle("CoverTitle", fontName="LiberationSerif-Bold", fontSize=21, leading=26, alignment=TA_CENTER, textColor=ACCENT)))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("DistilBERT with a Controlled R-Drop Improvement on CLINC150", ParagraphStyle("CoverSubtitle", fontName="LiberationSerif-Italic", fontSize=12, leading=16, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph("WORKING DRAFT — M0 RESULTS COMPLETE; M1/M2 GPU RESULTS PENDING", ParagraphStyle("DraftWarning", fontName="LiberationSerif-Bold", fontSize=11, leading=15, alignment=TA_CENTER, textColor=DRAFT)))
    story.append(Spacer(1, 0.9 * cm))
    student_data = [
        ["Student Name", "Student ID"],
        ["________________________________", "________________________________"],
        ["School Email", "Group Number"],
        ["________________________________", "________________________________"],
    ]
    student_table = Table(student_data, colWidths=[7.0 * cm, 7.0 * cm])
    student_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "LiberationSerif-Bold"),
        ("FONTNAME", (0, 2), (-1, 2), "LiberationSerif-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "LiberationSerif"),
        ("FONTNAME", (0, 3), (-1, 3), "LiberationSerif"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(student_table)
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph("Submission deadline: 1 November 2026", ParagraphStyle("Deadline", fontName="LiberationSerif-Bold", fontSize=11, leading=15, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("This draft contains only evidence generated in the project workspace. No DistilBERT training, test-set result, log, or citation has been fabricated.", ParagraphStyle("CoverNote", fontName="LiberationSerif", fontSize=9, leading=13, alignment=TA_CENTER, textColor=MUTED)))
    story.append(NextPageTemplate("TwoCol"))
    story.append(PageBreak())

    counted_text = []

    def add_body(text: str):
        counted_text.append(text)
        story.append(body_paragraph(text, styles))

    def add_heading(text: str, level: int = 1):
        counted_text.append(text)
        story.append(Paragraph(text, styles[f"ReportHeading{level}"]))

    abstract_text = "Abstract—This project investigates robust intent classification and out-of-scope (OOS) rejection for a text-based intelligent manufacturing assistant. Using the fixed CLINC150 benchmark, the study compares a TF-IDF logistic-regression baseline with a planned DistilBERT model and one controlled R-Drop modification. The completed baseline reaches 0.8939 accuracy and 0.9003 Macro-F1 on validation data, but only 0.3500 OOS recall despite 0.9509 OOS AUROC. Deterministic spelling noise reduces Macro-F1 to 0.8298 at a 5% edit rate and 0.7685 at 10%. These results show that strong in-scope classification and OOS score ranking do not guarantee reliable rejection or robustness. The Transformer comparison has been implemented and smoke-tested but has not yet been run on the approved Linux GPU server; consequently, this document is a working draft and not a completed submission."
    keywords_text = "Keywords—intent classification, out-of-scope detection, CLINC150, DistilBERT, R-Drop, robustness"
    counted_text.extend([abstract_text, keywords_text])
    story.append(Paragraph(abstract_text, styles["ReportBody"]))
    story.append(Paragraph(keywords_text, styles["ReportBody"]))

    add_heading("I. INTRODUCTION")
    add_body(
        "Interactive assistants are useful only when they can both understand supported requests and recognize the boundary of their competence. In a manufacturing setting, a worker may ask for a maintenance record, a machine status, or a standard operating instruction. A confident but wrong intent label can trigger an irrelevant retrieval step, while accepting an unsupported command can create the impression that the system is capable of an action it was never designed to perform. Intent classification should therefore be evaluated together with out-of-scope (OOS) behaviour rather than as an isolated accuracy problem."
    )
    add_body(
        f"This project uses CLINC150, a public English benchmark designed for intent classification and OOS prediction{citation(1)}. The dataset provides a reproducible way to study short user utterances, semantically adjacent intents, and unsupported requests. Its intended role here is methodological: it is not claimed to be a factory corpus. The manufacturing-assistant scenario motivates the reliability requirements, while the report explicitly treats the difference between general-domain CLINC150 language and real industrial terminology as a deployment limitation."
    )
    add_body(
        "Three questions guide the work. RQ1 asks how well a reproducible TF-IDF logistic-regression baseline performs on 150 intents plus OOS. RQ2 asks whether a DistilBERT classifier can improve on that baseline under the same split and metrics. RQ3 asks whether one controlled R-Drop modification changes clean performance and spelling-noise robustness. The first question is answered with validation evidence in this draft. The second and third require the approved remote Linux GPU experiments and remain open."
    )

    add_heading("II. PROBLEM DEFINITION AND OBJECTIVES")
    add_body(
        "The task is formulated as 151-way single-label classification: each English utterance is assigned to one of 150 supported intents or to the OOS class. The intended users are operators and engineers interacting with a constrained assistant front end. The intended output is a routing decision, not a generated answer and not an autonomous control action. This boundary matters because a low-confidence or unsupported request should be escalated to a human or a clearer interface rather than silently mapped to an unrelated function."
    )
    add_body(
        "The measurable objectives are: (1) establish a credible baseline using training data only for feature fitting; (2) compare baseline, principal, and improved models with identical splits and evaluation code; (3) assess overall and in-scope accuracy, Macro-F1, OOS precision, recall, F1 and AUROC; (4) measure deterministic spelling-noise degradation; and (5) diagnose representative errors. Macro-F1 is the model-selection metric because it gives each intent equal weight. No test result is used for model, hyperparameter, or threshold selection."
    )

    add_heading("III. DATASET DESCRIPTION AND GOVERNANCE")
    split_counts = {row["split"]: row for row in quality["split_summary"]}
    add_body(
        f"The UCI CLINC150 <i>data_full</i> metadata lists CC BY 4.0, while the bundled LICENSE states Attribution 3.0 Unported; the discrepancy is retained and documented{citation(2)}. The dataset contains {quality['total_rows']:,} utterances across 150 in-scope labels and OOS. The original partitions are retained: {split_counts['train']['rows']:,} training, {split_counts['val']['rows']:,} validation, and {split_counts['test']['rows']:,} test examples. The training partition contains 100 examples for every supported intent and 100 OOS examples. Validation contains 100 OOS examples, whereas test contains 1,000; this difference makes validation OOS estimates noisier and prevents a direct assumption that validation OOS performance will transfer exactly to test."
    )
    split_table = [
        ["Split", "In-scope", "OOS", "Total"],
        ["Train", "15,000", "100", "15,100"],
        ["Validation", "3,000", "100", "3,100"],
        ["Test", "4,500", "1,000", "5,500"],
    ]
    story.append(Paragraph('<a name="table1"/>TABLE I. FIXED CLINC150 SPLIT SIZES', styles["ReportCaption"]))
    story.append(three_line_table(split_table, [2.3 * cm, 2.1 * cm, 1.5 * cm, 1.6 * cm], aligns=["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 5))
    add_body(
        "Quality checks found no missing labels, missing text, empty utterances, or within-split duplicate text. Training utterances average 8.33 whitespace-delimited words, with median 8, a 95th percentile of 14, and maximum 28. Short texts make character n-grams useful but also mean that a single corrupted token can remove much of the available evidence. The source files and archive are SHA-256 verified, and the processed CSV files retain stable sample identifiers and source indices."
    )
    add_body(
        f"All queries are English and single-intent. Crowd workers produced utterances by paraphrasing seed phrases or responding to scenarios{citation(1)}. This construction provides useful lexical variation but does not represent every form of real user language. In particular, the benchmark cannot evaluate multi-intent commands, prolonged dialogue context, accented speech transcripts, or the specialized vocabulary of maintenance and safety procedures. OOS is also a heterogeneous collection rather than a single coherent semantic domain. Treating it as one class is practical for this study, but a model may learn to recognize only the observed OOS style and remain overconfident on unfamiliar unsupported requests. These linguistic and sampling limitations motivate cautious subgroup and error analysis rather than a single aggregate claim of generalization."
    )
    add_body(
        "The audit also found three exact or normalized texts shared between train and validation and two shared between train and test. Four normalized texts have conflicting labels across splits. For example, “what is on my to do list” is associated with <i>reminder</i> in validation but <i>todo_list</i> in training. These examples are retained to preserve the official benchmark; they are not silently cleaned. Their presence is recorded as a leakage and annotation limitation when interpreting both validation errors and any later test score."
    )
    story.append(figure(PROJECT_ROOT / "results/data_checks/figures/split_counts.png", "Fig. 1. Fixed split sizes with OOS shown separately.", "fig1", styles))

    add_heading("IV. METHODOLOGY")
    add_heading("A. Reference baseline", 2)
    add_body(
        f"M0 combines word and character TF-IDF features with multinomial logistic regression{citation(3)}. Word 1–2 grams capture lexical intent cues, while character 3–5 grams within word boundaries provide partial robustness to inflection and spelling variation. Both vectorizers use sublinear term-frequency scaling, lowercase text, Unicode accent stripping, minimum document frequency two, and at most 50,000 features each. Their union produces {metrics['feature_count']:,} features. Logistic regression uses C=1.0, the lbfgs solver, at most 300 iterations, and seed 42. Feature fitting occurs on the training partition only."
    )
    add_heading("B. Principal Transformer model", 2)
    add_body(
        f"M1 fine-tunes DistilBERT-base-uncased, a smaller distilled Transformer intended to retain useful contextual language representations with lower inference cost than its teacher{citation(4)}. Utterances are tokenized to at most 64 tokens. The classification head contains 151 outputs and is trained with cross-entropy. Planned training uses five epochs at most, batch size 16, learning rate 2×10<super>-5</super>, weight decay 0.01, 10% warmup, gradient clipping, mixed precision on CUDA, and early stopping after two non-improving validation epochs. The best checkpoint is selected by validation Macro-F1."
    )
    add_heading("C. One controlled R-Drop modification", 2)
    add_body(
        f"M2 changes only the training objective. Each training utterance passes through the same dropout-enabled model twice, producing two predictive distributions. The objective is the mean of the two cross-entropy losses plus 4.0 times their symmetric Kullback–Leibler divergence. R-Drop is designed to reduce inconsistency between dropout submodels{citation(5)}. This is a plausible robustness modification, not a guaranteed improvement: the project will report a negative or negligible result if the evidence does not support the modification."
    )
    add_body(
        "M1 and M2 share the pretrained snapshot, tokenization, optimizer, scheduler, batch size, epoch limit, early-stopping rule, data order policy, seeds, and evaluation code. An automated configuration gate rejects any difference other than experiment name, loss type, and R-Drop weight. Each model is run with seeds 13, 42, and 87; mean, standard deviation, and paired M2−M1 differences will be reported. Offline tiny-model tests verify both training loops and checkpoint reloads but are not experimental evidence."
    )
    add_heading("D. Evaluation and reproducibility protocol", 2)
    add_body(
        "Overall accuracy describes the total decision rate, whereas Macro-F1 averages precision-and-recall performance across labels and reduces the influence of frequent classes. Separate in-scope measurements prevent OOS behaviour from being hidden inside a strong intent score. For OOS, precision measures the reliability of an OOS decision, recall measures the fraction of unsupported requests caught, and AUROC measures ranking quality across possible thresholds. Reporting them together is necessary because a model can rank well while making poor decisions at its default operating point. Per-intent reports, row-level predictions, and top confusion pairs provide additional diagnostic evidence."
    )
    add_body(
        "For the Transformer comparison, three paired seeds provide mean and standard deviation estimates of run-to-run variability. They are not treated as a substitute for a large-sample confidence interval or a claim of statistical significance. Every run stores an immutable configuration snapshot, data-file hashes, dependency and hardware information, epoch history, best-epoch metrics, and predictions. The test command requires explicit confirmation and the presence of all six frozen checkpoints. This design keeps model development on validation data and makes any later test evaluation a distinct, auditable event rather than another tuning iteration."
    )

    add_heading("V. EXPERIMENTS AND RESULTS")
    add_heading("A. Baseline validation performance", 2)
    add_body(
        "The recorded M0 run used Python 3.12 on the local CPU environment and completed 25 solver iterations. Evaluation covered all 3,100 validation utterances with the same metrics planned for M1 and M2. No official test predictions or metrics have been produced. The main results are shown in Table II and Fig. 2."
    )
    result_table = [
        ["Metric", "Validation value"],
        ["Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Macro-F1", f"{metrics['macro_f1']:.4f}"],
        ["In-scope accuracy", f"{metrics['in_scope_accuracy']:.4f}"],
        ["In-scope Macro-F1", f"{metrics['in_scope_macro_f1']:.4f}"],
        ["OOS precision", f"{metrics['oos_precision']:.4f}"],
        ["OOS recall", f"{metrics['oos_recall']:.4f}"],
        ["OOS F1", f"{metrics['oos_f1']:.4f}"],
        ["OOS AUROC", f"{metrics['oos_auroc']:.4f}"],
    ]
    story.append(KeepTogether([
        Paragraph('<a name="table2"/>TABLE II. M0 VALIDATION RESULTS', styles["ReportCaption"]),
        three_line_table(result_table, [5.0 * cm, 2.3 * cm], aligns=["LEFT", "RIGHT"]),
    ]))
    story.append(Spacer(1, 5))
    add_body(
        f"In-scope performance is strong for a simple baseline: accuracy is {metrics['in_scope_accuracy']:.4f} and Macro-F1 is {metrics['in_scope_macro_f1']:.4f}. OOS behaviour is more problematic. The model obtains {metrics['oos_auroc']:.4f} AUROC, meaning that its OOS probability usually ranks OOS examples above in-scope examples. Yet the default 151-way decision recalls only {metrics['oos_recall']:.2%} of OOS examples. A high AUROC therefore does not imply a usable default rejection rule. The validation set contains only 100 OOS cases, so the recall estimate should be interpreted cautiously rather than optimized through repeated threshold experiments."
    )
    story.append(figure(PROJECT_ROOT / "results/m0_tfidf_logreg/val/figures/baseline_metrics.png", "Fig. 2. M0 validation metrics; OOS F1 contrasts sharply with OOS AUROC.", "fig2", styles))

    add_heading("B. Spelling-noise robustness", 2)
    add_body(
        "A deterministic challenge applies deletion, adjacent-letter swapping, and substitution to alphabetic characters. The noise seed is fixed per row before evaluation, and identical noisy inputs are reserved for every model. At least one edit is applied to every non-empty utterance. Rates 0%, 5%, and 10% are evaluated on validation data without changing the fitted baseline. This protocol measures sensitivity to surface corruption; it does not cover every form of paraphrase, domain shift, or adversarial input."
    )
    noise_rows = [["Noise", "Acc.", "Macro-F1", "OOS F1", "AUROC"]]
    for _, row in noise.iterrows():
        noise_rows.append([
            f"{row['noise_rate']:.0%}",
            f"{row['accuracy']:.4f}",
            f"{row['macro_f1']:.4f}",
            f"{row['oos_f1']:.4f}",
            f"{row['oos_auroc']:.4f}",
        ])
    story.append(KeepTogether([
        Paragraph('<a name="table3"/>TABLE III. M0 ROBUSTNESS TO DETERMINISTIC SPELLING NOISE', styles["ReportCaption"]),
        three_line_table(noise_rows, [1.25 * cm, 1.55 * cm, 1.65 * cm, 1.45 * cm, 1.65 * cm], font_size=7.6, aligns=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]),
    ]))
    story.append(Spacer(1, 5))
    add_body(
        f"At 5% noise, Macro-F1 falls by {abs(noise.iloc[1]['macro_f1_delta_from_clean']):.4f} to {noise.iloc[1]['macro_f1']:.4f}; at 10%, it falls by {abs(noise.iloc[2]['macro_f1_delta_from_clean']):.4f} to {noise.iloc[2]['macro_f1']:.4f}. OOS F1 declines from {noise.iloc[0]['oos_f1']:.4f} to {noise.iloc[2]['oos_f1']:.4f}. The monotonic degradation shows that surface robustness is a genuine weakness of the baseline. Character n-grams do not eliminate it. These validation results motivate the controlled R-Drop comparison but cannot demonstrate that R-Drop will help."
    )
    story.append(figure(PROJECT_ROOT / "results/robustness/m0_tfidf_logreg/val/figures/noise_robustness.png", "Fig. 3. M0 performance under deterministic spelling noise.", "fig3", styles))

    add_heading("C. Systematic error analysis", 2)
    add_body(
        "The most frequent individual confusion is <i>insurance</i> to <i>insurance_change</i>, with nine validation errors. Other recurring pairs include <i>change_user_name</i> to <i>user_name</i>, <i>improve_credit_score</i> to <i>credit_score</i>, and <i>shopping_list</i> to <i>shopping_list_update</i>. These errors are linguistically plausible: the label boundary often depends on whether an utterance asks for information or requests a state change. Bag-of-ngram features can capture the shared topic while missing the pragmatic distinction."
    )
    example = errors.iloc[0]
    add_body(
        f"One informative example is “{example['text']}”. The validation label is <i>{example['true_label']}</i>, but M0 predicts <i>{example['predicted_label']}</i> with probability {example['predicted_probability']:.3f}. As noted in the data audit, the same normalized text has a different training label. This example should not be interpreted as an ordinary model failure alone; it also exposes annotation inconsistency. By contrast, “show me a plan to help my credit score rating” is labelled <i>improve_credit_score</i> but predicted as <i>credit_score</i>, illustrating a genuine request-versus-information boundary error."
    )
    add_body(
        "Missed OOS cases show a related risk. “what's the definition of nuclear engineering” is OOS but is classified as <i>definition</i>, while “where do eggs come from” becomes <i>where_are_you_from</i>. Surface overlap with a supported intent can therefore override the fact that the requested content lies outside the assistant's scope. A manufacturing deployment should not treat the predicted intent as sufficient evidence for action; it needs calibrated rejection, clear scope presentation, and human escalation for uncertain cases."
    )
    story.append(figure(PROJECT_ROOT / "results/m0_tfidf_logreg/val/figures/oos_score_distribution.png", "Fig. 4. Validation OOS-score distributions for in-scope and OOS utterances.", "fig4", styles))

    add_heading("VI. DISCUSSION AND RESPONSIBLE APPLICATION")
    add_body(
        "The baseline demonstrates that a transparent, inexpensive pipeline can provide a strong reference point. It also identifies two limitations that aggregate in-scope accuracy would hide: unreliable default OOS rejection and substantial spelling-noise degradation. For a real assistant, these are different failure modes requiring different mitigations. Better representations may improve semantic discrimination, while consistency regularization may improve stability; neither automatically establishes calibration or safe deployment."
    )
    add_body(
        "The largest external-validity limitation is domain mismatch. CLINC150 covers general assistant intents rather than maintenance procedures, equipment alarms, safety interlocks, or engineering abbreviations. A model that performs well on this benchmark may fail on factory language, multilingual code switching, or speech-recognition errors. Any industrial extension therefore needs separately approved, representative evaluation data and a renewed bias and leakage audit. The present study should be read as a controlled NLP methods exercise, not evidence of readiness for machine control."
    )
    add_body(
        "Responsible use also requires data and model governance. The dataset and pretrained model licenses must be retained with their citations. The repository excludes credentials, private absolute paths, restricted data, and large locally generated checkpoints. Public examples should be screened before screenshots or videos are shared. At inference time, the system should expose uncertainty, avoid autonomous high-impact actions, and provide human review for unsupported or ambiguous commands. Security testing should include malformed and deliberately misleading inputs, not only random spelling noise."
    )
    add_body(
        "Generative AI was used as an implementation and drafting assistant, not as a source of experimental evidence. The mitigation is procedural: dataset hashes and metrics are generated by scripts, configurations are snapshotted, model training is separated from final test evaluation, and unexecuted GPU experiments are explicitly marked pending. These controls reduce the risk of fabricated evidence but do not replace the group members' responsibility to inspect the code, rerun the pipeline, and verify every reported claim."
    )

    add_heading("VII. CONCLUSIONS AND NEXT STEPS")
    add_body(
        "The completed baseline establishes a reproducible reference for 151-way intent and OOS classification. It achieves strong in-scope validation performance, but low OOS recall and substantial spelling-noise degradation show that overall accuracy is insufficient for a reliable assistant. The main DistilBERT and R-Drop pipeline is implemented, configuration-controlled, and smoke-tested. The next mandatory step is to run all six M1/M2 seed jobs on the approved Linux GPU server, inspect validation evidence, and only then freeze every configuration. Final test, paired seed summaries, noise evaluation, and report conclusions must follow that freeze; no such result is claimed in this working draft."
    )

    story.append(Paragraph("REFERENCES", styles["ReportHeading1"]))
    references = [
        (1, "S. Larson, A. Mahendran, J. J. Peper, C. Clarke, A. Lee, P. Hill, J. K. Kummerfeld, K. Leach, M. A. Laurenzano, L. Tang, and J. Mars, “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” <i>Proceedings of EMNLP-IJCNLP</i>, 2019. <a href='https://aclanthology.org/D19-1131/' color='blue'>https://aclanthology.org/D19-1131/</a>"),
        (2, "CLINC150 [Dataset]. UCI Machine Learning Repository, 2020. CC BY 4.0. <a href='https://doi.org/10.24432/C5MP58' color='blue'>https://doi.org/10.24432/C5MP58</a>"),
        (3, "F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011."),
        (4, "V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” <i>arXiv:1910.01108</i>, 2019. <a href='https://arxiv.org/abs/1910.01108' color='blue'>https://arxiv.org/abs/1910.01108</a>"),
        (5, "X. Liang, L. Wu, J. Li, Y. Wang, Q. Meng, T. Qin, W. Chen, M. Zhang, and T.-Y. Liu, “R-Drop: Regularized Dropout for Neural Networks,” <i>Advances in Neural Information Processing Systems 34</i>, pp. 10890–10905, 2021. <a href='https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html' color='blue'>NeurIPS 2021</a>"),
    ]
    for number, text in references:
        story.append(Paragraph(f'<a name="ref{number}"/>[{number}] {text}', styles["ReportReference"]))

    story.append(Paragraph("APPENDIX A — REPRODUCTION AND AI-USE DECLARATION", styles["ReportHeading1"]))
    story.append(Paragraph(
        "The accompanying local code package contains the pinned data downloader, split-preserving quality audit, M0 training and evaluation, deterministic noise evaluation, M1/M2 training, controlled-configuration checks, frozen-test guards, and aggregation scripts. The GitHub publication link has not yet been created; the local ZIP is the current fixed snapshot. Full commands and environment records are in README.md. Raw data and model binaries are excluded and restored using documented public sources.",
        styles["ReportSmall"],
    ))
    story.append(Paragraph(
        "AI-use declaration: Kimi Work was used to interpret the assignment, select the project design, generate and debug Python code, create figures and this report draft, and organize reproducibility evidence. Verification consisted of SHA-256 data checks, direct metric-file reads, ten unit/integration tests, tiny-model M1/M2 train-and-reload smoke tests, and rendered-PDF inspection. No GPU experiment, test metric, log, or citation was invented. Group members must still perform an independent technical and academic review before submission.",
        styles["ReportSmall"],
    ))
    story.append(Paragraph("APPENDIX B — EVIDENCE STILL REQUIRED BEFORE SUBMISSION", styles["ReportHeading1"]))
    story.append(Paragraph(
        "(1) Six completed M1/M2 Linux GPU runs with seeds 13, 42, and 87; (2) base-model resolved commit and environment lock; (3) best-epoch validation summaries; (4) frozen final test results for M0/M1/M2; (5) matched robustness results for M1/M2; (6) paired M2−M1 analysis and revised conclusions; (7) student names, IDs, emails and group number; (8) an accessible GitHub repository and final ZIP; and (9) individual presentations of at most five minutes and eight slides per student.",
        styles["ReportSmall"],
    ))

    document.build(story)
    plain = re.sub(r"<[^>]+>", " ", " ".join(counted_text))
    words = re.findall(r"\b[\w’'-]+\b", plain)
    build_info = {
        "output": OUTPUT_PATH.name,
        "approximate_body_words_excluding_references_and_appendices": len(words),
        "status": "working_draft_not_for_submission",
        "m1_m2_gpu_results_present": False,
        "official_test_results_present": False,
    }
    (DELIVERY_DIR / "report_build_info.json").write_text(json.dumps(build_info, indent=2), encoding="utf-8")
    return build_info


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
