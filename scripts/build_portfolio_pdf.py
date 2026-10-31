"""Build the portfolio PDF from audited project results without running models.

Install requirements-docs.txt, then run python scripts/build_portfolio_pdf.py.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/PulmoTrack_Portfolio_Project_Details.pdf"
WEBSITE = "https://pulmotrack-farahsayari-contribution-sef2wgqdo5bzl9wmgfwxry.streamlit.app/"
REPO = "https://github.com/FarahSayari/pulmoTrack-farahsayari-contribution"
WHO = "https://www.who.int/news-room/fact-sheets/detail/lung-cancer"
LUNA = "https://luna16.grand-challenge.org/Data/"
CRISP = "https://www.ibm.com/docs/en/SS3RA7_18.6.0/pdf/ModelerCRISPDM.pdf"
W, H = A4
MARGIN = 44
CW = W - 2 * MARGIN
NAVY = colors.HexColor("#0B1726")
TEAL = colors.HexColor("#087F78")
MINT = colors.HexColor("#77DDC7")
INK = colors.HexColor("#203448")
MUTED = colors.HexColor("#596D7E")
PALE = colors.HexColor("#EDF6F5")
LINE = colors.HexColor("#D7E4E7")
WHITE = colors.white
STORY = []
MARKDOWN = []


def register_fonts():
    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Portfolio", str(regular)))
            pdfmetrics.registerFont(TTFont("PortfolioBold", str(bold)))
            pdfmetrics.registerFontFamily("Portfolio", normal="Portfolio", bold="PortfolioBold",
                                         italic="Portfolio", boldItalic="PortfolioBold")
            return
    raise RuntimeError("Install Arial or DejaVu Sans to build the embedded-font PDF.")


register_fonts()
STYLE = {
    "body": ParagraphStyle("Body", fontName="Portfolio", fontSize=10, leading=15.1,
                           textColor=INK, spaceAfter=9),
    "small": ParagraphStyle("Small", fontName="Portfolio", fontSize=8, leading=11.4,
                            textColor=MUTED, spaceAfter=8),
    "h1": ParagraphStyle("Heading", fontName="PortfolioBold", fontSize=24, leading=28,
                         textColor=NAVY, spaceBefore=4, spaceAfter=15, keepWithNext=True),
    "h2": ParagraphStyle("Subheading", fontName="PortfolioBold", fontSize=13.5, leading=18,
                         textColor=TEAL, spaceBefore=9, spaceAfter=7, keepWithNext=True),
    "eyebrow": ParagraphStyle("Eyebrow", fontName="PortfolioBold", fontSize=8, leading=11,
                              textColor=TEAL, spaceAfter=8, keepWithNext=True),
    "cover": ParagraphStyle("Cover", fontName="PortfolioBold", fontSize=42, leading=46,
                            textColor=WHITE, spaceAfter=5, keepWithNext=True),
    "cover_subtitle": ParagraphStyle("CoverSubtitle", fontName="PortfolioBold", fontSize=21, leading=26,
                                     textColor=MINT, spaceAfter=12, keepWithNext=True),
    "cover_body": ParagraphStyle("CoverBody", fontName="Portfolio", fontSize=10, leading=14.5,
                                 textColor=colors.HexColor("#D4E0EA"), spaceAfter=12),
    "cover_small": ParagraphStyle("CoverSmall", fontName="Portfolio", fontSize=7.5, leading=10.5,
                                  textColor=colors.HexColor("#A5BACD"), spaceAfter=8),
    "cover_h2": ParagraphStyle("CoverH2", fontName="PortfolioBold", fontSize=12, leading=16,
                               textColor=WHITE, spaceBefore=8, spaceAfter=8),
    "cell": ParagraphStyle("Cell", fontName="Portfolio", fontSize=8.6, leading=12,
                           textColor=INK, spaceAfter=0),
    "cell_header": ParagraphStyle("CellHeader", fontName="PortfolioBold", fontSize=8.5, leading=12,
                                  textColor=WHITE, spaceAfter=0),
    "cover_cell": ParagraphStyle("CoverCell", fontName="Portfolio", fontSize=9, leading=12,
                                 textColor=WHITE, spaceAfter=0),
}


def link(label, url, cover=False):
    return f'<link href="{html.escape(url, quote=True)}" color="{"#8BE2D2" if cover else "#087F78"}">{html.escape(label)}</link>'


def markdown(text):
    text = re.sub(r'<link href="([^"]+)"[^>]*>(.*?)</link>', r'[\2](\1)', text)
    text = text.replace("<b>", "**").replace("</b>", "**").replace("<br/>", " / ")
    return html.unescape(re.sub(r"<(?!https?)[^>]+>", "", text))


def p(text, kind="body"):
    prefix = "# " if kind in ("h1", "cover") else "## " if kind in ("h2", "cover_h2") else ""
    MARKDOWN.extend([prefix + markdown(text), ""])
    return Paragraph(text, STYLE[kind])


def add(text, kind="body"):
    STORY.append(p(text, kind))


def page(title, eyebrow):
    STORY.append(PageBreak())
    add(eyebrow.upper(), "eyebrow")
    add(title, "h1")


def table(headers, rows, widths):
    values = [[Paragraph(html.escape(str(item)), STYLE["cell_header"]) for item in headers]]
    values += [[Paragraph(html.escape(str(item)), STYLE["cell"]) for item in row] for row in rows]
    result = Table(values, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, WHITE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), 1, MINT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.35, LINE),
    ]))
    STORY.extend([result, Spacer(1, 10)])
    MARKDOWN.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
    MARKDOWN.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    MARKDOWN.append("")


def callout(text):
    result = Table([[p(text)]], colWidths=[CW], hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 11), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    STORY.extend([result, Spacer(1, 10)])


def illustration(path, max_height):
    image = Image(str(path), lazy=0)
    ratio = min(CW / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth, image.drawHeight = image.imageWidth * ratio, image.imageHeight * ratio
    image.hAlign = "CENTER"
    STORY.extend([image, Spacer(1, 7)])


def header_footer(canvas, doc):
    canvas.saveState()
    if doc.page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        canvas.setFillColor(MINT)
        canvas.rect(MARGIN, H - 37, 43, 3, fill=1, stroke=0)
        canvas.setFont("Portfolio", 8)
        canvas.setFillColor(colors.HexColor("#A5BACD"))
        canvas.drawRightString(W - MARGIN, H - 36, "FARAH SAYARI / PORTFOLIO")
        footer_color = colors.HexColor("#A5BACD")
    else:
        canvas.setFillColor(WHITE)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN, H - 38, W - MARGIN, H - 38)
        canvas.setFont("PortfolioBold", 8)
        canvas.setFillColor(TEAL)
        canvas.drawString(MARGIN, H - 27, "PulmoTrack")
        canvas.setFont("Portfolio", 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(W - MARGIN, H - 27, "RESEARCH / DEEP LEARNING / MEDICAL IMAGING")
        footer_color = MUTED
    canvas.setFillColor(footer_color)
    canvas.setFont("Portfolio", 8)
    canvas.drawString(MARGIN, 24, "Farah Sayari · Project details")
    canvas.drawRightString(W - MARGIN, 24, f"{doc.page:02d}")
    canvas.restoreState()


class PortfolioDocument(BaseDocTemplate):
    """Expose the document's requested sections as PDF navigation bookmarks."""

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        name = flowable.style.name
        if name not in ("Cover", "Heading", "Subheading", "CoverH2"):
            return
        self._outline_index = getattr(self, "_outline_index", 0) + 1
        key = f"section_{self._outline_index}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(flowable.getPlainText(), key,
                                 level=0 if name in ("Cover", "Heading") else 1)


def build():
    STORY.clear()
    MARKDOWN.clear()
    archive = json.loads((ROOT / "assets/archive/manifest.json").read_text(encoding="utf-8"))
    experiments = {item["id"]: item for item in archive["experiments"]}
    paper = json.loads((ROOT / "assets/paper/manifest.json").read_text(encoding="utf-8"))

    # Page 1: project identity, the requested information card, and technologies.
    STORY.append(Spacer(1, 12))
    add("PulmoTrack", "cover")
    add("Deep Learning for<br/>Pulmonary Nodule Analysis", "cover_subtitle")
    add("An academic medical-imaging project using LUNA16 CT scans to study nodule candidate detection, binary classification and exploratory 3D segmentation, presented through an interactive research website.", "cover_body")
    illustration(ROOT / "assets/overview/CT Multiplanar Views Dashboard.png", 160)
    add("Axial, coronal and sagittal CT concept illustration; the highlighted findings are illustrative.", "cover_small")
    add("Project information", "cover_h2")
    info = [
        ("Category", "Research, DL, Medical Imaging, Computer Vision"),
        ("Role", "Data Scientist, Project Manager"),
        ("Methodology", "CRISP-DM"), ("Dataset", "LUNA16"),
        ("Status", "Completed"), ("Scientific paper", "Available · Team research manuscript"),
        ("Website", link("Open the live Streamlit website", WEBSITE, cover=True)),
    ]
    card = Table([[Paragraph(f"<b>{label}</b>", STYLE["cover_cell"]),
                   Paragraph(value, STYLE["cover_cell"])] for label, value in info],
                 colWidths=[106, CW - 106], hAlign="LEFT")
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#152B3C")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#294354")),
    ]))
    STORY.append(card)
    MARKDOWN.extend(f"- **{label}:** {markdown(value)}" for label, value in info)
    MARKDOWN.append("")
    add("Technologies", "cover_h2")
    add("<b>Research:</b> Python, PyTorch, torchvision, NumPy, pandas, SciPy, scikit-learn, SimpleITK.<br/>"
        "<b>Analysis &amp; delivery:</b> Jupyter Notebook, Google Colab, Matplotlib, Plotly, Pillow, Streamlit, Git, GitHub and Streamlit Community Cloud.", "cover_body")

    # Page 2: motivation and personal contribution, within the team-project scope.
    page("Project Overview", "Motivation and contribution")
    add("Lung cancer is the leading cause of cancer-related death worldwide and is often diagnosed at an advanced stage, when treatment options are more limited. Earlier identification and appropriate follow-up can improve the opportunity for effective treatment.")
    add("PulmoTrack explores how deep learning can analyze pulmonary nodules in volumetric CT data. The project connects medical-image preprocessing, three research objectives, model evaluation and a deployed interface that makes the experiments understandable to a portfolio audience.")
    add("The work is an academic research study. Its recorded models evaluate nodule candidates and masks; they do not establish a clinically validated cancer diagnosis system.")
    add("Scientific context: " + link("World Health Organization — Lung cancer", WHO) + ".", "small")
    add("My Contributions", "h1")
    contributions = [
        ("Research organization", "Used CRISP-DM to structure the CT workstream, organize experiments and communicate the scope of the three objectives."),
        ("LUNA16 preprocessing", "Worked on CT resampling, annotation and candidate inspection, world-to-voxel coordinate conversion, intensity normalization and 3D patch preparation."),
        ("Detection and classification", "Developed and studied 3D candidate-screening and binary classification approaches, comparing their saved training histories, confusion matrices and ROC evaluations."),
        ("Exploratory segmentation", "Investigated 3D U-Net, attention and residual-attention variants, and tiny-nodule oversampling, documenting both promising summaries and difficult positive-only cases."),
        ("Results and documentation", "Organized notebook outputs, kept evaluation cohorts and checkpoint results distinct, and contributed to the team research manuscript."),
        ("Portfolio deployment", "Created a Streamlit research showcase with CT visualizations, model overviews, result interpretation, source resources and access to the research paper."),
    ]
    for label, detail in contributions:
        add(f"<b>{label}.</b> {detail}")

    # Page 3: the dataset and the three separate objectives.
    page("LUNA16 Dataset", "Data and research objectives")
    add("LUNA16 (Lung Nodule Analysis 2016) is derived from LIDC-IDRI and contains <b>888 volumetric chest CT scans</b>. It provides CT volumes, candidate locations and nodule annotations for pulmonary nodule research.")
    add("The official reference standard includes nodules at least 3 mm in diameter accepted by at least three of four radiologists. Scans with slice thickness greater than 2.5 mm were excluded. The CT collection is approximately 90 GB in scale; the deployed showcase uses saved outputs rather than distributing the dataset.")
    add("The project prepares different patch collections for different experiments: <b>64³ voxels</b> for detection/classification and <b>48³ voxels</b> for the archived segmentation studies. Candidate labels, nodule reference masks and lung-field masks serve different purposes.")
    add("Dataset source and access: " + link("LUNA16 official data page", LUNA) + ".", "small")
    add("Explore the Three DSOs", "h1")
    callout("<b>DSO1 — Nodule Detection.</b> Screen supplied candidate locations using a U-Net-style 3D encoder with global pooling and a two-class head. The selected run uses 9,215 patches: 1,215 positives and 8,000 negatives. Weighted focal loss addresses the class imbalance. This is candidate screening, rather than full-volume localization.")
    callout("<b>DSO2 — Nodule Classification.</b> Compare Improved 3D CNN, ResNet3D Lite and a separate U-Net encoder classifier. Recorded folder counts are 1,243 true-nodule and 761 false-nodule patches. The evaluated task distinguishes true nodules from false nodules; benign/malignant classification is a broader project goal.")
    callout("<b>DSO3 — Exploratory 3D Segmentation.</b> Learn voxel-level nodule masks with U-Net variants and study the challenge of tiny nodules. This is a parallel research exploration, not a prerequisite for detection or classification. The intended later pipeline uses the provided LUNA16 lung masks.")
    add("Naming: exploratory segmentation is presented as DSO3 in the portfolio overview and sidebar; it is named DSO5 in the original research material.", "small")

    # Page 4: the complete six-stage methodology, including its deployment phase.
    page("CRISP-DM Methodology", "An iterative research process")
    add("CRISP-DM structures the work into six connected phases. The process is iterative: findings from data preparation and evaluation inform the next modeling decisions.")
    phases = [
        ("01 / Business understanding", "Translate the early-identification motivation into research questions: can a model screen nodule candidates, distinguish true and false nodules, and delineate candidate nodule regions? Define the academic scope, separate the three objectives and set measurable evaluation criteria."),
        ("02 / Data understanding", "Inspect 3D CT volumes, voxel spacing, scan identifiers, annotation coordinates, nodule sizes and candidate labels. Identify class imbalance, limited tiny-nodule visibility, differences between patch collections, and the distinction between lung masks and nodule masks."),
        ("03 / Data processing", "Resample CT volumes, convert physical coordinates to voxel coordinates, normalize intensities and extract fixed-size patches. Prepare labels and spherical nodule references. Apply the documented augmentation and sampling strategies, with separate dataset splits for each study."),
        ("04 / Modeling", "Develop 3D encoder-based candidate screening, three binary classification architectures and encoder–decoder segmentation variants. Use Adam optimization and the losses actually recorded in each study: weighted focal loss for detection, cross-entropy for classification and binary cross-entropy for segmentation."),
        ("05 / Evaluation", "Review the appropriate metrics for each task: accuracy, ROC-AUC and confusion matrices for candidate classifiers; Dice, precision, recall and F1 for segmentation. Keep test and validation cohorts distinct, distinguish best-checkpoint results from final-epoch logs, and inspect original predictions."),
        ("06 / Deployment", "Publish an interactive Streamlit showcase of the completed research. Package saved figures, results and source references so visitors can explore the work without downloading LUNA16 or rerunning the models. This phase communicates the experiments; it does not deploy clinical inference."),
    ]
    for heading, description in phases:
        STORY.append(KeepTogether([p(heading, "h2"), p(description)]))
    add("Methodology reference: " + link("IBM CRISP-DM guide", CRISP) + ".", "small")

    # Page 5: precise classifier evaluations, read directly from the archive.
    page("Results and Performance", "DSO1 and DSO2 / Notebook evaluations")
    add("The values below belong to the saved notebook evaluations. Accuracy is calculated from the corresponding original confusion matrix; ROC-AUC comes from that checkpoint’s recorded output.")
    add("DSO1 / Nodule Detection", "h2")
    ids = ["det_validation", "det_test"]
    table(["Cohort", "Patches", "Accuracy", "ROC-AUC"],
          [[experiments[i]["stage"].title(), f'{experiments[i]["sample_count"]:,}',
            f'{experiments[i]["metrics"]["accuracy"]:.2%}', f'{experiments[i]["metrics"]["roc_auc"]:.4f}'] for i in ids],
          [CW * .36, CW * .18, CW * .23, CW * .23])
    (tn, fp), (fn, tp) = experiments["det_test"]["matrix"]
    add(f"On the 921-patch test cohort, the saved model correctly identifies <b>{tp} of {tp + fn} nodule patches</b> "
        f"({tp / (tp + fn):.2%} sensitivity), with <b>{fp} false positives and {fn} false negatives</b>. "
        "These are candidate-level outcomes; they do not establish false positives per full CT scan.")
    add("DSO2 / Nodule Classification", "h2")
    ids = ["cls_cnn_validation", "cls_cnn_test", "cls_resnet_validation", "cls_unet_validation"]
    labels = ["Improved 3D CNN · validation", "Improved 3D CNN · test", "ResNet3D Lite · validation", "U-Net encoder · validation"]
    table(["Study / cohort", "Patches", "Accuracy", "ROC-AUC"],
          [[label, str(experiments[i]["sample_count"]), f'{experiments[i]["metrics"]["accuracy"]:.2%}',
            f'{experiments[i]["metrics"]["roc_auc"]:.4f}'] for label, i in zip(labels, ids)],
          [CW * .43, CW * .15, CW * .21, CW * .21])
    (tn, fp), (fn, tp) = experiments["cls_cnn_test"]["matrix"]
    add(f"The Improved 3D CNN test evaluation identifies <b>{tp} of {tp + fn} true-nodule patches</b>, "
        f"with {fp} false positives and {fn} false negatives. Its accuracy and AUC describe binary candidate labels, not malignancy diagnosis.")
    callout("<b>How to compare these results.</b> Each model retains its own recorded cohort. The ResNet matrix contains 401 patches despite an earlier 200-patch validation log. Splitting is by patch, and a patient-disjoint benchmark is not established. These studies support analysis of the recorded runs, not a controlled architecture ranking.")
    add("Primary notebooks: 11_nodule_detection.ipynb and 11_nodule_classification.ipynb. Best-checkpoint evaluations are separate from last-epoch accuracies.", "small")

    # Page 6: segmentation results, retaining positive/mixed-cohort distinctions.
    page("Results and Performance", "DSO3 / Exploratory segmentation")
    add("Segmentation results depend strongly on the selected patches and the handling of empty masks. The table preserves these separate experimental contexts; all values are notebook-reported summaries.")
    seg_rows = [
        ("3D U-Net", "Five positives · A", "unet_positive_a", "dice"),
        ("3D U-Net", "Five positives · B", "unet_positive_b", "dice"),
        ("3D U-Net", "Full positive-patch folder", "unet_full_positive", "f1"),
        ("3D U-Net", "Mixed validation loader", "unet_validation", "dice"),
        ("Attention U-Net", "Five positive patches", "attention_positive", "dice"),
        ("Attention U-Net", "Ten mixed patches", "attention_mixed", "dice"),
        ("Residual attention", "Ten positives · initial", "residual_initial", "dice"),
        ("Residual attention", "Ten positives · later", "residual_repeat", "dice"),
    ]
    table(["Model study", "Evaluation context", "Patches", "Reported score"],
          [[name, cohort, str(experiments[i]["sample_count"] or "Not recorded"),
            f'{field.title()} {experiments[i]["metrics"][field]:.4f}'] for name, cohort, i, field in seg_rows],
          [CW * .23, CW * .34, CW * .16, CW * .27])
    add("The <b>0.9504 validation Dice</b> comes from a mixed, augmented loader, where smoothing assigns empty prediction/reference pairs a Dice near 1. The attention model’s mixed-patch evaluator similarly assigns perfect scores to empty pairs. Positive-only samples show the remaining difficulty of recovering small nodule regions.")
    add("Tiny-nodule augmentation retains saved prediction examples and a final validation binary cross-entropy loss of <b>0.0446</b>. No Dice, precision, recall or F1 evaluation was recorded for that run; a lower loss is not an overlap score.")
    illustration(ROOT / "assets/archive/modeling_13_03.png", 165)
    add("Original saved baseline prediction — positive-patch experiment A, notebook cell 13; displayed case Dice: 0.2556. Partial mask recovery illustrates the positive-case challenge, rather than the mixed validation summary.", "small")

    # Page 7: actual delivery and the available team manuscript.
    page("Deployment", "Research delivery and supporting paper")
    add("PulmoTrack is hosted on <b>Streamlit Community Cloud</b>, with the GitHub repository as its deployment source. The interface preserves the dark navy and teal medical-imaging identity, sidebar navigation and CT-focused visual presentation.")
    add("Visitors can explore model architectures, recorded metrics, training histories, original segmentation predictions, pilot Grad-CAM, research resources and the manuscript viewer. The app presents saved evidence; it requires no CT dataset download or live model execution.")
    callout("<b>Delivery workflow:</b> research notebooks → exported figures and result summaries → Streamlit interface → GitHub → Streamlit Community Cloud.")
    add(link("Open the live PulmoTrack research showcase", WEBSITE))
    add(link(WEBSITE, WEBSITE), "small")
    add("The website is a completed research showcase, rather than a patient-upload or diagnostic inference service. Repository changes can be reflected in the deployed app through Community Cloud’s GitHub integration.")
    add("Deployment reference: " + link("Streamlit Community Cloud documentation", "https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app") + ".", "small")
    add("Research Paper", "h1")
    add(html.escape(paper["title"]), "h2")
    add("<b>Availability:</b> Available through the website’s Research Paper section, with View PDF and Download PDF options.<br/>"
        "<b>Document:</b> Team research manuscript · May 2025 · 11 pages.<br/>"
        "<b>Authors:</b> " + html.escape(", ".join(paper["authors"])) + ".")
    add("The manuscript covers the broader team project, including CT and chest X-ray work. This portfolio document focuses on the LUNA16 CT contribution.")
    highlights = paper["highlights"]
    add(f'<b>Paper-reported LUNA16 highlights (pp. 4–5):</b> segmentation Dice {highlights["segmentation_dice"]:.2f}; '
        f'detection sensitivity {highlights["detection_sensitivity"]:.0%} at {highlights["false_positives_per_scan"]:g} false positives per scan; '
        f'classification AUC {highlights["classification_auc"]:.2f}. '
        'These manuscript summaries are separate from the notebook evaluations on pages 5–6; the supplied classification notebooks evaluate true/false nodule labels.')
    add("Project source: " + link("GitHub repository", REPO) + ". The original manuscript is stored as assets/paper/PulmoTrackPaper.pdf and is accessible through the live app.", "small")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = PortfolioDocument(str(OUTPUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                               topMargin=55, bottomMargin=43, title="PulmoTrack — Portfolio Project Details",
                               author="Farah Sayari", subject="LUNA16 deep learning research portfolio documentation")
    document.addPageTemplates(PageTemplate(id="Portfolio", frames=[Frame(MARGIN, 43, CW, H - 98,
                                                                          leftPadding=0, rightPadding=0,
                                                                          topPadding=0, bottomPadding=0)],
                                          onPage=header_footer))
    document.build(STORY)
    OUTPUT.with_suffix(".md").write_text("\n".join(MARKDOWN), encoding="utf-8")
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build()
