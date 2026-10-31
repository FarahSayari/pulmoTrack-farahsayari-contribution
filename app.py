"""pulmoTrack: three research objectives, using stored evidence only."""
from __future__ import annotations

import base64
import html
import json
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from pulmotrack.archive import ROOT, evidence_csv, figure_path, load_archive, source_label
from pulmotrack.detection import confusion_chart, outcome_summary, roc_display
from pulmotrack.paper import load_paper, paper_path
from pulmotrack.segmentation import METRIC_LABELS, interpretation as segmentation_interpretation, metric_chart

st.set_page_config(page_title="pulmoTrack | CT Research Workspace", page_icon="🫁", layout="wide")
st.html(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>")


@st.cache_data
def get_archive(manifest_mtime):
    return load_archive()


archive = get_archive((ROOT / "assets/archive/manifest.json").stat().st_mtime_ns)
models = {model["id"]: model for model in archive["models"]}
experiments = {experiment["id"]: experiment for experiment in archive["experiments"]}
figures = {figure["id"]: figure for figure in archive["figures"]}
dsos = {dso["id"]: dso for dso in archive["dsos"]}
VIEWS = {"dso1": "DSO1 · Detection", "dso2": "DSO2 · Classification", "dso5": "DSO5 · Segmentation"}


def stat(label, value, detail=""):
    st.html(f'<div class="stat"><div class="stat-label">{html.escape(label)}</div>'
            f'<div class="stat-value">{html.escape(str(value))}</div>'
            f'<div class="stat-detail">{html.escape(detail)}</div></div>')


def title(eyebrow, heading, description):
    st.html(f'<div class="eyebrow">{html.escape(eyebrow)}</div>')
    st.title(heading)
    st.html(f'<div class="section-intro">{html.escape(description)}</div>')


def source(record):
    url = "https://github.com/FarahSayari/pulmoTrack-farahsayari-contribution/blob/main/notebooks/" + quote(record["notebook"])
    st.caption(f"Source: [{source_label(record)}]({url})")


def show_figure(figure, download=False):
    st.image(str(figure_path(figure)), width=min(figure["width"], 1460))
    source(figure)
    if download:
        st.download_button("Download original PNG", figure_path(figure).read_bytes(),
                           file_name=figure["file"], mime="image/png", key="download_" + figure["id"])


def chart_theme(figure, height=350):
    figure.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#b5c5d5", family="Arial", size=12),
                         margin=dict(l=15, r=20, t=40, b=35), legend=dict(orientation="h", y=1.16, x=0),
                         xaxis=dict(gridcolor="#243245", zeroline=False),
                         yaxis=dict(gridcolor="#243245", zeroline=False))
    return figure


def navigate(view):
    st.session_state["navigation"] = view


@st.dialog("Research paper", width="large")
def view_research_paper():
    paper = load_paper()
    st.caption(paper["title"])
    page_number = st.selectbox("PDF page", range(1, len(paper["pages"]) + 1),
                               format_func=lambda number: f"Page {number} of {len(paper['pages'])}", key="paper_page")
    page = paper["pages"][page_number - 1]
    st.image(str(paper_path(page["file"])), width=page["width"])
    st.caption("Page rendering from the original project manuscript.")
    st.download_button("Download original PDF", paper_path(paper["file"]).read_bytes(),
                       file_name=paper["file"], mime="application/pdf", key="paper_dialog_download")


def overview():
    paper = load_paper()
    highlights = paper["highlights"]
    left, right = st.columns([1, 1.28], gap="large")
    with left:
        st.html('<div class="eyebrow">PulmoTrack / Academic research</div>'
                '<h1 class="overview-title showcase-title">Deep learning for<br><span>pulmonary nodules.</span></h1>'
                '<div class="hero-copy showcase-intro">Exploring detection, classification and independent '
                '3D segmentation on <strong>LUNA16 chest CT scans</strong>.</div>')
        st.link_button("Explore LUNA16 ↗", "https://luna16.grand-challenge.org/Home/", type="primary")
        st.caption("888 volumetric CT scans from LIDC-IDRI, with pulmonary nodule annotations.")
    with right:
        with st.container(border=True):
            st.html('<div class="viewer-label"><strong>CT / MULTIPLANAR VIEWS</strong><span>Concept illustration</span></div>')
            st.image(str(ROOT / "assets/overview/CT Multiplanar Views Dashboard.png"), width=1401)
            st.caption("Axial · Coronal · Sagittal")

    st.html('<div class="showcase-section"><div class="eyebrow">LUNA16 / Paper-reported summary</div><h2>Research highlights</h2></div>')
    for column, (label, value, detail) in zip(st.columns(4), [
        ("3D CT scans", f'{highlights["ct_scans"]:,}', "LUNA16 research dataset"),
        ("Segmentation Dice", f'{highlights["segmentation_dice"]:.2f}', "3D nodule segmentation"),
        ("Detection sensitivity", f'{highlights["detection_sensitivity"]:.0%}', f'At {highlights["false_positives_per_scan"]:g} false positives per scan'),
        ("Classification AUC", f'{highlights["classification_auc"]:.2f}', "LUNA16 classification summary"),
    ]):
        with column:
            st.html(f'<div class="showcase-highlight"><div class="stat-label">{html.escape(label)}</div>'
                    f'<div class="showcase-highlight-value">{value}</div><div class="stat-detail">{html.escape(detail)}</div></div>')

    with st.container(border=True):
        context_title, context_copy = st.columns([1, 2.1], gap="large")
        with context_title:
            st.html('<div class="eyebrow">The research motivation</div>')
            st.subheader("Why early detection matters")
        with context_copy:
            st.write("Lung cancer is the leading cause of cancer-related death worldwide and is often diagnosed at an advanced stage. Earlier diagnosis can expand treatment options and improve the chance of successful treatment.")
            st.caption("[Scientific context · World Health Organization](https://www.who.int/news-room/fact-sheets/detail/lung-cancer)")

    st.html('<div class="showcase-section"><div class="eyebrow">Three objectives / Distinct questions</div><h2>Explore the study</h2></div>')
    objective_cards = [
        dict(id="dso1", label="DSO1 / Candidate screening", title="Nodule Detection", image="detection.png",
             caption="Detection concept illustration", description="Assess whether a CT candidate patch contains a nodule, and inspect the trade-off between missed findings and false positives.", action="Explore detection"),
        dict(id="dso2", label="DSO2 / Classification models", title="Nodule Classification", image="classification.png",
             caption="Classification concept · illustrative score", description="Study 3D CNN, residual and U-Net encoders. The archived notebooks evaluate true-nodule versus false-nodule classification.", action="Explore classification"),
        dict(id="dso5", label="DSO5 / Independent challenge", title="Exploratory 3D Segmentation", image="segmentation.png",
             caption="Segmentation concept illustration", description="Explore voxel-level nodule delineation with 3D U-Net variants, with particular attention to the limitations of tiny-nodule segmentation.", action="Explore segmentation"),
    ]
    for column, card in zip(st.columns(3, gap="medium"), objective_cards):
        with column:
            with st.container(border=True, key="objective_card_" + card["id"]):
                st.image(str(ROOT / "assets/overview" / card["image"]), width=1448)
                st.caption(card["caption"])
                st.html(f'<div class="eyebrow">{card["label"]}</div><h3 class="showcase-objective-title">{card["title"]}</h3>'
                        f'<p class="showcase-objective-copy">{card["description"]}</p>')
                st.button(card["action"] + " →", key="open_" + card["id"],
                          on_click=navigate, args=(VIEWS[card["id"]],), width="stretch")
    st.caption("Segmentation is an independent research challenge. Detection and classification do not depend on the learned segmentation models.")

    st.html('<div class="showcase-section"><div class="eyebrow">An iterative research process</div><h2>CRISP-DM methodology</h2></div>')
    st.write("The work is presented through six connected stages, from problem framing to communicating the findings.")
    phases = [
        ("01", "Business understanding", "Define the nodule-analysis objectives and the questions each task should answer."),
        ("02", "Data understanding", "Inspect CT volumes, annotation structure and class imbalance in LUNA16."),
        ("03", "Data preparation", "Resample volumes, normalize intensities and construct candidate patches and references."),
        ("04", "Modeling", "Explore task-specific 3D CNN, residual and U-Net architectures."),
        ("05", "Evaluation", "Examine overlap scores, ROC curves, confusion matrices and study limitations."),
        ("06", "Deployment & communication", "Share the research through a team manuscript and this interactive showcase."),
    ]
    for start in [0, 3]:
        for column, (number, heading, copy) in zip(st.columns(3), phases[start:start + 3]):
            with column:
                st.html(f'<div class="showcase-process-card"><div class="eyebrow">{number}</div><h3>{heading}</h3><p>{copy}</p></div>')
    st.caption("[CRISP-DM framework · IBM](https://www.ibm.com/docs/en/spss-modeler/18.6.0?topic=dm-crisp-help-overview)")

    st.html('<div class="showcase-section"><div class="eyebrow">Farah Sayari / LUNA16 workstream</div><h2>My Contribution</h2></div>')
    contributions = [
        ("CT preprocessing", "Prepared LUNA16 volumes through resampling, HU normalization, annotation matching and 3D patch extraction."),
        ("Nodule detection", "Explored candidate-screening models and examined detection results, false positives and model behavior."),
        ("Nodule classification", "Compared 3D classification architectures and reviewed their recorded training and evaluation results."),
        ("Exploratory segmentation", "Investigated 3D U-Net variants, augmentation and tiny-nodule oversampling as a separate scientific challenge."),
    ]
    for start in [0, 2]:
        for column, (heading, copy) in zip(st.columns(2), contributions[start:start + 2]):
            with column:
                st.html(f'<div class="showcase-contribution"><h3>{heading}</h3><p>{copy}</p></div>')

    st.html('<div class="showcase-section"><div class="eyebrow">The team manuscript / May 2025</div><h2>Research Paper</h2></div>')
    with st.container(border=True):
        paper_text, paper_actions = st.columns([2.2, 1], gap="large")
        with paper_text:
            st.markdown("**" + paper["title"] + "**")
            st.write("The team manuscript covers the wider CT and chest X-ray project. This portfolio focuses on my LUNA16 CT workstream.")
            st.caption(f'{len(paper["pages"])} pages · Farah Sayari and co-authors · Project research manuscript')
        with paper_actions:
            if st.button("View PDF", type="primary", width="stretch", key="view_paper"):
                view_research_paper()
            st.download_button("Download PDF", paper_path(paper["file"]).read_bytes(), file_name=paper["file"],
                               mime="application/pdf", width="stretch", key="paper_download")

    st.html('<div class="showcase-section"><div class="eyebrow">Inspect the supporting evidence</div><h2>Research Resources</h2></div>')
    for column, args in zip(st.columns(3), [
        ("Source notebooks", len(archive["sources"]), "Selected preprocessing and modeling notebooks"),
        ("Model studies", len(archive["models"]), "Eight main studies and two earlier pilots"),
        ("Original figures", len(archive["figures"]), "Plus a 48-frame Grad-CAM animation"),
    ]):
        with column:
            stat(*args)
    gallery_action, sources_action = st.columns(2)
    with gallery_action:
        st.button("Browse the CT gallery →", on_click=navigate, args=("CT gallery",), width="stretch")
    with sources_action:
        st.button("Explore models & sources →", on_click=navigate, args=("Models & sources",), width="stretch")


def binary_results(model_id, key):
    st.subheader("Evaluation results")
    detection_study = model_id == "det_unet"
    if detection_study:
        st.write("The best checkpoint is selected by validation loss, then evaluated on a separate test split. Choose a cohort to inspect its recorded results.")
    else:
        st.write("Review the selected model’s own evaluation cohort. The Improved 3D CNN includes both test and validation results; the other models retain validation evaluations.")
    available = [e for e in archive["experiments"] if e["model"] == model_id and e["tier"] == "main"]
    available.sort(key=lambda e: e["stage"] != "test")
    selected = st.radio("Evaluation cohort", [e["id"] for e in available], horizontal=True,
                        format_func=lambda value: experiments[value]["stage"].title() + " · best checkpoint", key=key + "_evaluation")
    record = experiments[selected]
    if model_id == "cls_resnet":
        st.info("This saved ResNet evaluation contains 401 patches; the earlier split log lists 200 validation patches. Treat it as a separate historical cohort when reviewing the results.")
    outcome = outcome_summary(record["matrix"])
    style_key = "detection" if detection_study else "classification"
    with st.container(key=style_key + "_metrics"):
        for column, (label, value, detail) in zip(st.columns(3), [
            (record["stage"].title() + " accuracy", f'{record["metrics"]["accuracy"]:.2%}', "Correct predictions / evaluated patches"),
            ("ROC-AUC", f'{record["metrics"]["roc_auc"]:.4f}', "Recorded discrimination across thresholds"),
            ("Evaluated patches", f'{record["sample_count"]:,}', f'{outcome["tn"] + outcome["fp"]:,} non-nodules · {outcome["tp"] + outcome["fn"]:,} nodules'),
        ]):
            with column:
                stat(label, value, detail)
    matrix_column, roc_column = st.columns(2, gap="large")
    with matrix_column:
        with st.container(border=True, key=style_key + "_matrix"):
            st.subheader("Confusion matrix")
            st.caption("Counts at the notebook’s argmax decision rule.")
            st.plotly_chart(confusion_chart(record["matrix"]), use_container_width=True, key=key + "_confusion")
        source(figures[record["matrix_figure_id"]])
    with roc_column:
        roc = figures[record["roc_figure_id"]]
        with st.container(border=True, key=style_key + "_roc"):
            st.subheader("ROC curve")
            st.caption("Sensitivity versus false positive rate across thresholds.")
            st.image(roc_display(figure_path(roc).read_bytes(), roc["width"], roc["height"],
                                 record["stage"].title() + " ROC curve from the original experiment"), width=roc["width"])
        source(roc)
    st.caption("The matrix uses the original counts. ROC display colors are adapted from the original PNG; its curve is unchanged. The AUC card retains four decimal places rather than the plot legend’s rounding.")
    st.subheader("Results interpretation")
    interpretation, errors = st.columns([1.15, 1], gap="large")
    with interpretation:
        st.markdown(f'**{outcome["tp"]:,} of {outcome["tp"] + outcome["fn"]:,} nodule patches were identified** '
                    f'({outcome["sensitivity"]:.2%} sensitivity). Of all patches predicted as nodules, '
                    f'**{outcome["precision"]:.2%} were true nodules** (precision).')
        if detection_study:
            st.write("The high ROC-AUC indicates strong separation between the two patch classes in this cohort. Accuracy includes the larger non-nodule class; sensitivity and the error counts provide complementary context.")
        else:
            negative_count = outcome["tn"] + outcome["fp"]
            st.markdown(f'It correctly rejected **{outcome["tn"]:,} of {negative_count:,} non-nodule patches** '
                        f'({outcome["tn"] / negative_count:.2%} specificity). ROC-AUC describes ranking across thresholds; '
                        'the confusion matrix shows errors at the saved decision rule.')
    with errors:
        st.html(f'<div class="detection-errors"><div><strong>{outcome["fp"]:,} false positives</strong>'
                '<p>Non-nodule patches predicted as nodules: extra candidates to review.</p></div>'
                f'<div><strong>{outcome["fn"]:,} false negatives</strong>'
                '<p>Nodule patches predicted as non-nodules: missed candidates in this cohort.</p></div></div>')
    if detection_study:
        st.caption("These are candidate-level results, not full-scan sensitivity or false positives per scan. The notebook splits patches randomly; it does not establish a patient-disjoint evaluation or clinical performance.")
    else:
        st.caption("These metrics evaluate true-nodule versus false-nodule classification, not benign/malignant status. Cohorts differ across models, so these results do not form a controlled model ranking.")
    with st.expander("Evaluation context & original figures"):
        st.write(record["note"])
        if "last_epoch_accuracy" in record:
            st.caption(f'Last-epoch validation accuracy: {record["last_epoch_accuracy"]:.2%}. The best-checkpoint matrix above gives {record["metrics"]["accuracy"]:.2%}.')
        st.write("Raw probability arrays were not preserved, so the ROC is displayed from its saved experimental figure rather than reconstructed.")
        source(record)
        for column, field, label in zip(st.columns(2), ["matrix_figure_id", "roc_figure_id"],
                                        ["Download original matrix", "Download original ROC"]):
            original = figures[record[field]]
            with column:
                st.download_button(label, figure_path(original).read_bytes(), file_name=original["file"],
                                   mime="image/png", key=key + "_original_" + field)
    st.download_button("Download DSO results CSV", evidence_csv(archive, record["dso"]),
                       file_name=record["dso"] + "_recorded_results.csv", mime="text/csv", key=key + "_csv")


def training_panel(runs, key):
    by_id = {run["id"]: run for run in runs}
    selected = st.selectbox("Training experiment", list(by_id), format_func=lambda value: by_id[value]["label"], key=key + "_run")
    run = by_id[selected]
    points = run["points"]
    columns = st.columns(3)
    with columns[0]:
        stat("Recorded epochs", len(points), run["loss"])
    with columns[1]:
        stat("Lowest logged validation loss", f'{min(p["validation_loss"] for p in points):.4f}', "Rounded values from the original logs")
    with columns[2]:
        if "validation_accuracy" in points[-1]:
            stat("Last-epoch validation accuracy", f'{points[-1]["validation_accuracy"]:.2%}', "Different from the best-checkpoint evaluation")
        else:
            stat("Train / validation patches", f'{run["training_patches"]:,} / {run["validation_patches"]:,}')
    log_scale = st.toggle("Logarithmic loss axis", value=False, key=key + "_log")
    figure = go.Figure()
    for field, label, color in [("train_loss", "Training", "#63d9c1"), ("validation_loss", "Validation", "#b2a2ef")]:
        figure.add_trace(go.Scatter(x=[p["epoch"] for p in points], y=[p[field] for p in points], name=label,
                                    mode="lines+markers", line=dict(color=color, width=2.5), marker=dict(size=4)))
    figure.update_xaxes(title="Epoch")
    figure.update_yaxes(title=run["loss"], type="log" if log_scale else "linear")
    st.plotly_chart(chart_theme(figure), use_container_width=True, key=key + "_loss")
    if "validation_accuracy" in points[-1]:
        accuracy = go.Figure()
        for field, label, color in [("train_accuracy", "Training", "#63d9c1"), ("validation_accuracy", "Validation", "#b2a2ef")]:
            values = [p for p in points if field in p]
            if values:
                accuracy.add_trace(go.Scatter(x=[p["epoch"] for p in values], y=[p[field] * 100 for p in values],
                                              name=label, mode="lines+markers", line=dict(color=color, width=2.5)))
        accuracy.update_xaxes(title="Epoch")
        accuracy.update_yaxes(title="Accuracy (%)", range=[0, 105])
        st.plotly_chart(chart_theme(accuracy, 300), use_container_width=True, key=key + "_accuracy")
    st.caption(run["note"])
    source(run)
    st.download_button("Download training log", pd.DataFrame(points).to_csv(index=False).encode(),
                       file_name=selected + ".csv", mime="text/csv", key=key + "_training_csv")


def preparation_panel():
    records = [f for f in archive["figures"] if f["dso"] == "shared" and f["tier"] != "pilot"]
    groups = list(dict.fromkeys(f["group"] for f in records))
    labels = {f["group"]: f["title"] for f in records}
    group = st.selectbox("Preparation collection", groups, format_func=labels.get, key="preparation_collection")
    candidates = [f for f in records if f["group"] == group]
    example = st.slider("Saved figure", 1, len(candidates), 1, key="prep_" + group) if len(candidates) > 1 else 1
    show_figure(candidates[example - 1])
    st.caption("Lung-mask figures here are historical threshold/morphology examples. They are distinct from the learned nodule-segmentation experiments and the provided LUNA16 lung masks.")
    st.caption("The initial candidate plots contain historical coordinate/label issues. A negative candidate is not evidence of a model false negative.")


def gradcam_panel():
    st.subheader("Where the pilot CNN focused")
    st.write("A saved Grad-CAM illustration from the earlier 196-patch experiment. It shows model attention, not a nodule boundary or diagnostic probability.")
    animation = archive["animations"][0]
    display = st.radio("Display", ["Animation", "Inspect a frame"], horizontal=True, key="cam_display")
    path = figure_path(animation)
    if display == "Animation":
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        st.html(f'<img class="cam-animation" src="data:image/gif;base64,{encoded}" alt="Saved pilot CNN Grad-CAM animation" style="width:616px;max-width:100%;border-radius:12px;"/>')
    else:
        slice_index = st.slider("Saved CT slice", animation["first_slice"], animation["first_slice"] + animation["frames"] - 1, 31, key="cam_slice")
        with Image.open(path) as image:
            image.seek(slice_index - animation["first_slice"])
            st.image(image.copy(), width=616)
    st.caption("48 saved frames · CT patch slices 16–63. Slice 31 matches the presentation heatmap.")
    source(animation)
    st.download_button("Download Grad-CAM GIF", path.read_bytes(), file_name="pulmotrack_gradcam.gif", mime="image/gif")
    with st.expander("Earlier model studies · small pilot"):
        st.write("196 patches: 137 training, 29 validation, 30 test. These experiments are separate from the larger candidate study.")
        st.write("**Simple 3D CNN:** recorded test accuracy 83.33%. **3D ResNet-18:** last validation accuracy 86.21%, peak validation accuracy 96.55%; no test accuracy or AUC was recorded.")
        training_panel([r for r in archive["training"] if r["tier"] == "pilot"], "pilot")


def detection():
    st.image(str(ROOT / "assets/DSO1/Nodule detection.png"), width=1460)
    st.caption("Nodule detection concept · illustrative markers and scores.")
    st.subheader("Model overview")
    cards = [
        ("Architecture", "3D U-Net encoder",
         "3D convolutions capture nodule shape and surrounding tissue across CT slices. The encoder learns features at multiple scales, making it a suitable approach for volumetric CT analysis.",
         "U-Net was developed for biomedical segmentation. Here, its encoder is adapted with global pooling and a two-class head for candidate screening, rather than generating segmentation masks."),
        ("Dataset", "LUNA16 candidates",
         "9,215 patches: 1,215 nodules and 8,000 non-nodules, approximately 13% / 87%. Preparation covers 888 CT scans.",
         "Why not 50 / 50? The preparation run retains all positive candidates and randomly samples 8,000 negatives. This preserves more negative examples than a 1:1 subset; weighted focal loss addresses the imbalance during training."),
        ("Input & output", "64 × 64 × 64 voxels",
         "A fixed 3D crop surrounds each candidate with nearby tissue context while using less memory than a full CT volume. The size also halves cleanly through three pooling stages: 64 → 32 → 16 → 8.",
         "One CT intensity channel enters the encoder; two class scores represent non-nodule and nodule. Fixed-size patches also allow consistent batching during training."),
    ]
    st.html('<div class="detection-model-grid">' + ''.join(
        f'<div class="detection-model-card"><div class="eyebrow">{html.escape(label)}</div>'
        f'<h3>{html.escape(heading)}</h3><p>{html.escape(description)}</p>'
        f'<p>{html.escape(rationale)}</p></div>'
        for label, heading, description, rationale in cards) + '</div>')
    st.html('<div class="detection-flow" aria-label="Detection architecture">'
            '<span>3D CT patch</span><i>→</i><span>Convolution &amp; pooling</span><i>→</i>'
            '<span>Global average pooling</span><i>→</i><span>Two-class head</span></div>')
    st.caption("The encoder learns volumetric features; a linear head produces class logits, with softmax scores used for ROC evaluation. It has no segmentation decoder or bounding-box output. Segmentation is a separate research challenge.")
    results_tab, training_tab, cam_tab, prep_tab = st.tabs(["Results", "Training history", "Grad-CAM & pilot studies", "Preparation & scope"])
    with results_tab:
        binary_results("det_unet", "detection")
    with training_tab:
        st.subheader("Learning from 3D candidate patches")
        st.write("Weighted focal loss (γ = 2) addresses the unequal class frequencies. The run uses Adam at a learning rate of 0.0001, batch size 16, and 20 recorded epochs; checkpoints are selected by the lowest validation loss.")
        st.caption("Training: 6,451 patches · Validation: 1,843 · Test: 921. The best saved validation checkpoint corresponds to epoch 15; last-epoch values describe epoch 20.")
        training_panel([r for r in archive["training"] if r["model"] == "det_unet"], "detection")
    with cam_tab:
        gradcam_panel()
    with prep_tab:
        st.subheader("From CT scans to candidate patches")
        st.write("The preparation notebooks resample CT volumes, match supplied candidate coordinates to nodule annotations, and extract labeled 3D patches. The selected dataset contains all 1,215 positive candidates and a subset of 8,000 negative candidates from the larger 754,975-candidate collection.")
        st.info(dsos["dso1"]["scope"])
        st.caption("The notebook uses a seeded random patch split of approximately 70% / 20% / 10%. Preparation images show annotations and historical preprocessing, rather than predictions from the main detection model.")
        preparation_panel()


def classification():
    st.image(str(ROOT / "assets/DSO2/Nodule Classification.png"), width=1460)
    st.caption("Classification concept · diagnostic label and confidence are illustrative.")
    st.subheader("Model overview")
    options = [m["id"] for m in archive["models"] if m["dso"] == "dso2"]
    model = st.selectbox("Classification model", options, format_func=lambda value: models[value]["name"], key="classification_model")
    architecture = {
        "cls_cnn": (
            "Improved 3D CNN",
            "Three convolution blocks learn volumetric patterns with 32 → 64 → 128 channels. Adaptive pooling feeds a 256-unit dense layer, dropout and a two-class output.",
            "This compact architecture provides a baseline for learning local CT features. Dropout regularizes its classification head.",
            "3D convolution blocks", "Adaptive pooling & dense head"),
        "cls_resnet": (
            "ResNet3D Lite",
            "A 16-channel stem feeds residual blocks with 32 → 64 → 128 channels. Global average pooling and a linear head produce two class scores.",
            "Residual shortcuts combine each block’s transformed features with a shortcut path, allowing the model to retain earlier information as features deepen.",
            "3D residual blocks", "Global average pooling"),
        "cls_unet": (
            "3D U-Net encoder",
            "Four double-convolution stages use 32 → 64 → 128 → 256 channels, pooling and dropout. Global pooling feeds a 128-unit dense classifier and a two-class output.",
            "The encoder captures CT features at several spatial scales. Its classification head replaces a segmentation decoder, adapting the architecture to patch labels.",
            "Multiscale 3D encoder", "Global pooling & dense head"),
    }[model]
    heading, description, rationale, feature_stage, pooling_stage = architecture
    cards = [
        ("Architecture", heading, description, rationale),
        ("Dataset", "LUNA16-derived patches",
         "Recorded folder counts: 1,243 true-nodule and 761 false-nodule patches, totaling 2,004 examples (approximately 62% / 38%). This dataset is separate from DSO1.",
         "The notebook retains this class mix rather than resampling to 50 / 50. The initial split log records 1,603 training, 200 validation and 201 test patches; each saved evaluation’s actual cohort is shown below."),
        ("Input & output", "64 × 64 × 64 voxels",
         "Each candidate-centered CT crop is normalized by its own mean and standard deviation. A fixed-size 3D patch retains nearby tissue context with a smaller input than a full scan.",
         "One intensity channel enters the network. Two scores represent false nodule and true nodule; argmax gives the predicted label and softmax scores support ROC evaluation."),
    ]
    st.html('<div class="detection-model-grid">' + ''.join(
        f'<div class="detection-model-card"><div class="eyebrow">{html.escape(label)}</div>'
        f'<h3>{html.escape(card_heading)}</h3><p>{html.escape(copy)}</p><p>{html.escape(detail)}</p></div>'
        for label, card_heading, copy, detail in cards) + '</div>')
    st.html('<div class="detection-flow" aria-label="Classification architecture">'
            '<span>Normalized 3D CT patch</span><i>→</i>'
            f'<span>{html.escape(feature_stage)}</span><i>→</i><span>{html.escape(pooling_stage)}</span><i>→</i>'
            '<span>True / false nodule</span></div>')
    results_tab, training_tab, scope_tab = st.tabs(["Results", "Training history", "Study context"])
    with results_tab:
        binary_results(model, "classification_" + model)
    with training_tab:
        st.subheader("Learning to classify CT patches")
        st.write("The selected model uses cross-entropy loss and Adam optimization at an initial learning rate of 0.0001. Training history and best-checkpoint evaluations are shown separately.")
        training_panel([r for r in archive["training"] if r["model"] == model], "classification_training")
    with scope_tab:
        st.write(models[model]["description"])
        st.write("The classification dataset is constructed from true-nodule and false-nodule folders. It does not contain evaluated benign/malignant labels. These studies should not be interpreted as cancer diagnosis.")
        st.write("Validation populations are not uniform: the CNN and U-Net matrices contain 200 patches, while the ResNet matrix contains 401. The source split configuration and some later outputs differ, so there is no shared model leaderboard.")
        st.caption("The CNN/U-Net runs shown here differ from the corresponding presentation slides. Unverified slide-only figures are not substituted for notebook evidence.")


def segmentation():
    st.image(str(ROOT / "assets/DSO5/Exploratory segmentation.png"), width=1460)
    st.caption("DSO5 · Independent research challenge · concept illustration.")
    st.info(dsos["dso5"]["scope"])
    st.subheader("Model overview")
    model = st.selectbox("Segmentation model", [m["id"] for m in archive["models"] if m["dso"] == "dso5"],
                         format_func=lambda value: models[value]["name"], key="seg_model")
    architecture = {
        "unet": ("3D U-Net", "An encoder–decoder uses 32 → 64 → 128 feature channels. Two upsampling stages combine decoder features with encoder skip connections to recover the patch’s spatial detail.",
                 "3D convolutions retain context across CT slices, while skip connections carry local detail into the predicted mask. This baseline explores voxel-level nodule delineation.", "Encoder & skip connections"),
        "attention": ("Attention 3D U-Net", "The baseline encoder–decoder is extended with attention gates on the skip connections. Each gate weights encoder features before they are combined with decoder features.",
                      "The gates explore selective use of spatial features during reconstruction. Their benefit must be assessed within each saved cohort; the notebook does not provide a common model benchmark.", "Attention-gated skips"),
        "residual": ("Residual Attention 3D U-Net", "Residual double-convolution blocks add shortcut paths within the encoder–decoder. Attention gates also weight the encoder features passed through skip connections.",
                     "This variant explores residual feature reuse alongside attention. The initial and later evaluations are preserved separately because the runs reused the same checkpoint path.", "Residual blocks & attention"),
        "tiny": ("U-Net · tiny-nodule training", "The baseline 3D U-Net is retained, with a modified training dataset that repeats patches identified as containing tiny nodules and applies augmentation.",
                 "Oversampling increases their exposure during training. Saved examples and loss curves document this experiment; no numerical overlap evaluation was recorded.", "Encoder & skip connections"),
    }[model]
    heading, description, rationale, feature_stage = architecture
    dataset_detail = ("The tiny-nodule run records 2,140 training and 536 validation entries after oversampling. Repeated entries are not additional unique CT scans."
                      if model == "tiny" else
                      "The baseline split records 2,036 training and 509 validation patches. Negative patches use empty reference masks; positive and negative evaluations are kept distinct.")
    cards = [
        ("Architecture", heading, description, rationale),
        ("Dataset & reference", "LUNA16 nodule patches",
         "Reference nodule masks are spheres generated from annotation centers and diameters. They approximate the nodule region and are distinct from lung-field masks.", dataset_detail),
        ("Input & output", "48 × 48 × 48 voxels",
         "One normalized CT intensity channel enters the model. A local 3D crop provides surrounding tissue context while keeping the input smaller than a full CT volume.",
         "The decoder returns a same-size probability map, one value per voxel. A threshold of 0.5 produces the binary nodule mask used in the recorded evaluations."),
    ]
    st.html('<div class="detection-model-grid">' + ''.join(
        f'<div class="detection-model-card"><div class="eyebrow">{html.escape(label)}</div>'
        f'<h3>{html.escape(card_heading)}</h3><p>{html.escape(copy)}</p><p>{html.escape(detail)}</p></div>'
        for label, card_heading, copy, detail in cards) + '</div>')
    st.html('<div class="detection-flow" aria-label="DSO5 segmentation architecture">'
            '<span>48³ CT patch</span><i>→</i>'
            f'<span>{html.escape(feature_stage)}</span><i>→</i><span>3D decoder</span><i>→</i>'
            '<span>Voxel-wise nodule mask</span></div>')
    scores, examples, training, scope = st.tabs(["Recorded scores", "Prediction examples", "Training history", "Study context"])
    with scores:
        st.subheader("Evaluation results")
        records = [e for e in archive["experiments"] if e["model"] == model]
        if records:
            selected = st.selectbox("Evaluation context", [e["id"] for e in records],
                                    index=next((i for i, e in enumerate(records) if e["id"] == "unet_validation"), 0),
                                    format_func=lambda value: experiments[value]["label"], key="seg_evaluation")
            record = experiments[selected]
            st.caption(record["note"])
            primary = "dice" if "dice" in record["metrics"] else "f1"
            cohort = "Mixed validation" if selected == "unet_validation" else ("Mixed patches" if selected == "attention_mixed" else "Positive patches")
            with st.container(key="segmentation_metrics"):
                for column, (label, value, detail) in zip(st.columns(3), [
                    ("Reported " + METRIC_LABELS[primary], f'{record["metrics"][primary]:.4f}', "Original notebook summary"),
                    ("Evaluated patches", str(record["sample_count"] or "Not recorded"), "Count for this evaluation only"),
                    ("Evaluation cohort", cohort, "Historical patch-level evaluation"),
                ]):
                    with column:
                        stat(label, value, detail)
            chart_column, explanation_column = st.columns([1.1, 1], gap="large")
            with chart_column:
                with st.container(border=True):
                    st.subheader("Recorded segmentation metrics")
                    st.plotly_chart(metric_chart(record["metrics"]), use_container_width=True, key="segmentation_scores")
                st.caption("Only recorded values are plotted. No voxel counts, confusion matrix or ROC data were preserved for these evaluations.")
            with explanation_column:
                with st.container(border=True):
                    st.subheader("Reading the metrics")
                    st.markdown("**Dice / F1** measure overlap between the predicted and reference nodule regions.")
                    st.markdown("**Precision** describes how much of the predicted nodule region matches the reference; extra voxels reduce precision.")
                    st.markdown("**Recall** describes how much of the reference region is recovered; missed voxels reduce recall.")
                    st.markdown("**Voxel accuracy** includes background. It can stay high even when a small nodule is poorly segmented.")
            st.subheader("Results interpretation")
            st.write(segmentation_interpretation(record))
            st.caption("Different samples, augmentation and empty-mask scoring rules prevent a controlled ranking of these model studies.")
            with st.expander("Evaluation context & source"):
                st.write(record["note"])
                source(record)
        else:
            run = next(r for r in archive["training"] if r["model"] == model)
            st.info("This tiny-nodule experiment has no recorded Dice, precision, recall, F1 or voxel-accuracy evaluation. Its original prediction examples and training history remain available in the adjacent tabs.")
            with st.container(key="segmentation_metrics"):
                for column, (label, value, detail) in zip(st.columns(3), [
                    ("Final validation BCE", f'{run["points"][-1]["validation_loss"]:.4f}', "Loss, not an overlap score"),
                    ("Training entries", f'{run["training_patches"]:,}', "After tiny-nodule oversampling"),
                    ("Validation entries", f'{run["validation_patches"]:,}', "Historical patch split"),
                ]):
                    with column:
                        stat(label, value, detail)
            st.subheader("Results interpretation")
            st.write("The saved loss documents optimization, but does not establish nodule overlap or an improvement over other models. The original figures support qualitative inspection of the predicted masks.")
        with st.expander("All recorded segmentation evaluations"):
            all_records = [e for e in archive["experiments"] if e["dso"] == "dso5"]
            rows = [{"Model": models[e["model"]]["name"], "Experiment": e["label"], "Samples": str(e["sample_count"] or "Not recorded"),
                     **{label: e["metrics"].get(field) for field, label in METRIC_LABELS.items()}} for e in all_records]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                label: st.column_config.NumberColumn(label, format="%.4f") for label in METRIC_LABELS.values()})
            st.caption("Historical evaluations use different cohorts. Missing metrics remain unavailable.")
        st.download_button("Download segmentation results CSV", evidence_csv(archive, "dso5"), file_name="dso5_segmentation_results.csv", mime="text/csv")
    with examples:
        st.subheader("Original segmentation predictions")
        st.write("Inspect the saved CT slices, spherical reference masks and model predictions for the selected study. The experiment selector identifies the source of each figure; these examples need not belong to the score cohort selected in the first tab.")
        records = [f for f in archive["figures"] if f["model"] == model and f["category"] == "prediction"]
        groups = list(dict.fromkeys(f["group"] for f in records))
        labels = {f["group"]: f["title"] for f in records}
        group = st.selectbox("Recorded experiment", groups, format_func=labels.get, key="seg_group_" + model)
        candidates = [f for f in records if f["group"] == group]
        index = st.slider("Saved example", 1, len(candidates), 1, key="seg_example_" + group)
        show_figure(candidates[index - 1], download=True)
        if model == "tiny":
            st.caption("This model has saved examples and a final validation BCE loss of 0.0446; no numerical segmentation evaluation was recorded.")
    with training:
        st.subheader("Learning voxel-level nodule masks")
        st.write("The recorded studies use binary cross-entropy loss with Adam optimization. These curves show the selected model’s saved training history; initial and later residual-attention runs remain separate.")
        training_panel([r for r in archive["training"] if r["model"] == model], "segmentation")
    with scope:
        st.subheader("DSO5 · An independent research challenge")
        st.write("This exploration studies nodule delineation and model development. Detection and classification do not depend on its generated masks. The intended later pipeline uses the provided LUNA16 lung masks; that integration is not demonstrated in this archive.")
        st.write("Reference nodule masks are spheres drawn from annotation centers and diameters. They are distinct from lung-field masks. Image and mask augmentation is randomized independently; validation inherits augmentation, and splitting is by patch rather than patient.")
        st.write("Repeated residual-attention runs overwrite the same checkpoint path. The presentation comparison averages 0.71–0.75 are not supported by matching numerical summaries in the supplied notebooks, so they are not included here.")


def gallery():
    title("CT review room", "The CT gallery", "Explore original CT preparation, segmentation, evaluation and explainability figures across the three objectives.")
    a, b = st.columns(2)
    with a:
        dso = st.selectbox("Objective", ["all", "dso1", "dso2", "dso5", "shared"],
                           format_func=lambda value: "All objectives" if value == "all" else ("Shared preparation" if value == "shared" else VIEWS[value]), key="gallery_dso")
    with b:
        tier = st.radio("Study scope", ["Main studies", "Earlier pilot studies", "All studies"], horizontal=True, key="gallery_tier")
    records = [f for f in archive["figures"] if dso == "all" or f["dso"] == dso or (dso in ["dso1", "dso2"] and f["dso"] == "shared")]
    records = [f for f in records if tier == "All studies" or (f["tier"] == "pilot") == (tier == "Earlier pilot studies")]
    categories = {"all": "All content", "preparation": "CT preparation", "prediction": "Segmentation predictions", "confusion_matrix": "Confusion matrices", "roc": "ROC curves", "training": "Training curves", "gradcam": "Grad-CAM", "chart": "Other charts", "distribution": "Score distributions"}
    category = st.selectbox("Content", list(categories), format_func=categories.get, key="gallery_category")
    records = [f for f in records if category == "all" or f["category"] == category]
    if not records:
        st.info("No saved figures match these filters. Choose another content type or study scope.")
        return
    groups = list(dict.fromkeys(f["group"] for f in records))
    labels = {f["group"]: f["title"] for f in records}
    group = st.selectbox("Figure collection", groups, format_func=labels.get, key="gallery_group")
    candidates = [f for f in records if f["group"] == group]
    index = st.slider("Saved figure", 1, len(candidates), 1, key="gallery_example_" + group) if len(candidates) > 1 else 1
    selected = candidates[index - 1]
    st.subheader(selected["title"])
    show_figure(selected, download=True)
    st.caption("Figure labels and colors are preserved from the source notebook. Controls select recorded figures, not missing CT volumes.")


def sources_and_models():
    title("Model studies & evidence", "The research collection", "Architectures, selected notebooks and the provenance behind each result.")
    model_tab, source_tab, context_tab = st.tabs(["Model studies", "Source notebooks", "Evidence notes"])
    with model_tab:
        dso = st.radio("Objective", list(dsos), format_func=lambda value: VIEWS[value], horizontal=True, key="models_dso")
        records = [m for m in archive["models"] if m["dso"] == dso]
        for start in range(0, len(records), 2):
            for column, model in zip(st.columns(2), records[start:start + 2]):
                with column:
                    with st.container(border=True):
                        st.html(f'<div class="eyebrow">{html.escape(model["tier"])} / {dso.upper()}</div>')
                        st.subheader(model["name"])
                        st.write(model["description"])
                        st.caption(f'{model["input_size"]}³ CT patch input')
                        with st.expander("Source & checkpoint details"):
                            st.code(model["notebook"], language=None)
                            if model["checkpoint"]:
                                st.write(f'**Stored checkpoint:** {model["checkpoint"]}')
                                st.write(f'**Parameters:** {model["parameters"]:,}')
                                st.code(model["sha256"], language=None, wrap_lines=True)
                            else:
                                st.caption("The experiment is documented in its notebook; its checkpoint is not included in this collection.")
    with source_tab:
        rows = [{"Notebook": row["notebook"], "Objective": row["dso"].upper(), "Scope": row["tier"], "Evidence": row["role"]} for row in archive["sources"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Only the selected, inspected notebooks feed this interface. Earlier incomplete drafts and the Markdown file with an .ipynb extension are excluded.")
        st.download_button("Download all results CSV", evidence_csv(archive), file_name="pulmotrack_all_results.csv", mime="text/csv")
        st.download_button("Download archive manifest", json.dumps(archive, indent=2).encode(), file_name="pulmotrack_archive_manifest.json", mime="application/json")
    with context_tab:
        st.write("**Study objectives.** DSO1 screens nodule candidates, DSO2 studies binary classification, and DSO5 explores nodule segmentation as an independent challenge.")
        st.write("**Reading the scores.** Each result belongs to the model, checkpoint and sample shown. Studies use different samples and scoring rules, so their scores are not a direct ranking.")
        st.write("**Classification labels.** The saved classification evaluations distinguish true nodules from false nodules.")
        st.write("**Segmentation masks.** The experiments use spherical nodule reference masks. Future stages are intended to use the original LUNA16 lung masks, which cover the lung region rather than individual nodules.")
        st.write("**Result sources.** Overview highlights come from the project manuscript. Detailed model pages use their saved notebook evaluations, without rerunning the models.")
        st.write("**Figures and illustrations.** Original experiment figures are available for download. Header images are concept illustrations; their labels and scores are not individual model predictions.")


with st.sidebar:
    st.html('<div class="brand"><span class="brand-mark" aria-hidden="true">+</span><div class="brand-name">pulmo<span>Track</span></div></div><div class="brand-sub">CT imaging · research workspace</div>')
    view = st.radio("Workspace", ["Overview", *VIEWS.values(), "CT gallery", "Models & sources"], key="navigation")
    st.divider()
    st.caption("COLLECTION")
    st.markdown("**LUNA16 · Pulmonary nodules**")
    st.caption("Detection / Classification / Segmentation")
    st.html('<div class="sidebar-note"><strong>Farah Sayari</strong><br>pulmoTrack-farahsayari contribution</div>')

st.html('<div class="topline"><div class="eyebrow">Pulmonary imaging / research workspace</div><div class="archive-tag"><span class="archive-dot"></span>Three objectives · one collection</div></div>')
{"Overview": overview, VIEWS["dso1"]: detection, VIEWS["dso2"]: classification, VIEWS["dso5"]: segmentation,
 "CT gallery": gallery, "Models & sources": sources_and_models}[view]()
st.html('<div class="footer"><span>pulmoTrack · Farah Sayari contribution</span><span>Research prototype · not validated for clinical diagnosis</span></div>')
