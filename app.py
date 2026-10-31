"""Streamlit explorer of existing pulmoTrack outputs. No inference or dataset."""
from __future__ import annotations

import html
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pulmotrack.archive import ROOT, evidence_csv, figure_path, load_archive, source_label

st.set_page_config(page_title="pulmoTrack | CT Research Archive", page_icon="🫁", layout="wide")
st.html(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>")


@st.cache_data
def get_archive(manifest_mtime):
    return load_archive()


archive = get_archive((ROOT / "assets" / "archive" / "manifest.json").stat().st_mtime_ns)
models = {model["id"]: model for model in archive["models"]}
experiments = {experiment["id"]: experiment for experiment in archive["experiments"]}


def stat(label, value, detail=""):
    st.html(f'<div class="stat"><div class="stat-label">{html.escape(label)}</div>'
            f'<div class="stat-value">{html.escape(str(value))}</div>'
            f'<div class="stat-detail">{html.escape(detail)}</div></div>')


def title(eyebrow, heading, description):
    st.html(f'<div class="eyebrow">{html.escape(eyebrow)}</div>')
    st.title(heading)
    st.html(f'<div class="section-intro">{html.escape(description)}</div>')


def source(record):
    st.caption("Source: " + source_label(record))


def show_figure(figure, caption=True):
    st.image(str(figure_path(figure)), width=min(figure["width"], 1460))
    if caption:
        source(figure)


def chart_theme(figure, height=380):
    figure.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b5c5d5", family="Arial", size=12),
        margin=dict(l=15, r=20, t=38, b=35), legend=dict(orientation="h", y=1.16, x=0),
        xaxis=dict(gridcolor="#243245", zeroline=False),
        yaxis=dict(gridcolor="#243245", zeroline=False),
        hoverlabel=dict(bgcolor="#172637", font_color="#eff6fc"),
    )
    return figure


def model_card(model, number):
    st.html(f'<div class="model-card"><div class="model-id">MODEL {number:02d} / 3D SEGMENTATION</div>'
            f'<h3>{html.escape(model["name"])}</h3><p>{html.escape(model["description"])}</p>'
            f'<div class="model-meta"><span>{model["parameters"] / 1e6:.2f}M parameters</span>'
            f'<span>{model["size_bytes"] / 1024**2:.2f} MiB checkpoint</span></div></div>')


def navigate(view):
    st.session_state["navigation"] = view


def overview():
    left, right = st.columns([1, 1.28], gap="large")
    with left:
        st.html('<div class="eyebrow">Pulmonary nodule segmentation</div>'
                '<div class="hero-title">A closer look.<br><span>A clearer study.</span></div>'
                '<div class="hero-copy">Explore the CT images, segmentation examples and model experiments '
                'behind pulmoTrack. Every result comes from the original LUNA16 notebooks.</div>'
                '<div class="hero-foot">A saved research collection by Farah Sayari</div>')
        st.write("")
        st.button("Explore CT examples  →", type="primary", on_click=navigate, args=("CT gallery",))
    with right:
        with st.container(border=True):
            st.html('<div class="viewer-label"><strong>CT / MULTIPLANAR VIEWS</strong><span>Saved notebook figure</span></div>')
            hero = next(f for f in archive["figures"] if f["group"] == "preparation_18")
            show_figure(hero, caption=False)
            st.caption("Axial · Coronal · Sagittal | normalized CT after lung masking")
            source(hero)
    columns = st.columns(4)
    for column, args in zip(columns, [
        ("Stored models", "04", "Three architectures · four checkpoints"),
        ("Original figures", f'{len(archive["figures"]):02d}', "Exported directly from notebook outputs"),
        ("Recorded evaluations", f'{len(archive["experiments"]):02d}', "Each retains its evaluation context"),
        ("Model input", "48³", "3D patches used in the notebooks"),
    ]):
        with column:
            stat(*args)
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        st.subheader("From CT to segmentation")
        st.caption("A visual record of the original research workflow.")
        for number, heading, text in [
            ("01", "Prepare the scan", "Resample to 1 mm spacing, apply lung masks, and normalize intensities."),
            ("02", "Study the models", "Review baseline, attention, residual attention and augmented U-Net experiments."),
            ("03", "Inspect the evidence", "Browse saved predictions and read scores alongside their evaluation method."),
        ]:
            st.html(f'<div class="step"><small>{number}</small><strong>{heading}</strong><p>{text}</p></div>')
    with b:
        with st.container(border=True):
            st.subheader("Read the results in context")
            st.write("Experiments use different samples and scoring rules. Mixed-patch scores and positive-only scores describe different evaluations.")
            st.caption("There is no shared held-out benchmark in this archive, so the app does not rank the models.")
            st.button("Review results & methods  →", on_click=navigate, args=("Results & methods",))


def gallery():
    title("CT review room", "The CT gallery", "Browse original CT figures and saved segmentation examples. The images below are recorded outputs, not new predictions.")
    collection = st.radio("Collection", ["Segmentation examples", "CT preparation", "Experiment charts"],
                          horizontal=True, key="gallery_collection")
    if collection in ["CT preparation", "Experiment charts"]:
        categories = ["preparation"] if collection == "CT preparation" else ["chart", "distribution"]
        figures = [f for f in archive["figures"] if f["category"] in categories]
        groups = list(dict.fromkeys(f["group"] for f in figures))
        group_names = {f["group"]: f["title"] + (" · score distribution" if f["category"] == "distribution" else "") for f in figures}
        group = st.selectbox("Preparation stage" if collection == "CT preparation" else "Saved chart", groups,
                             format_func=group_names.get, key="preparation_stage" if collection == "CT preparation" else "saved_chart")
        candidates = [f for f in figures if f["group"] == group]
        experiment = None
        model_name = "CT PREPARATION" if collection == "CT preparation" else "EXPERIMENT CHART"
    else:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            model_id = st.selectbox("Model", list(models), format_func=lambda value: models[value]["name"], key="gallery_model")
        figures = [f for f in archive["figures"] if f["category"] == "prediction" and f["model"] == model_id]
        groups = list(dict.fromkeys(f["group"] for f in figures))
        group_names = {f["group"]: f["title"] for f in figures}
        with c2:
            group = st.selectbox("Recorded experiment", groups, format_func=group_names.get, key=f"gallery_experiment_{model_id}")
        candidates = [f for f in figures if f["group"] == group]
        experiment = experiments.get(group)
        model_name = models[model_id]["name"]
    main, aside = st.columns([3.2, 1], gap="large")
    with aside:
        st.subheader("Saved example")
        if len(candidates) > 1:
            example = st.slider("Example number", 1, len(candidates), 1, key=f"example_{group}")
        else:
            example = 1
            st.caption("One figure in this collection.")
        selected = candidates[example - 1]
        st.caption(f"{example} of {len(candidates)} saved figures")
        st.divider()
        st.markdown("**About this view**")
        if collection == "CT preparation":
            st.write("An original figure from the data preparation notebook.")
        elif collection == "Experiment charts":
            st.write("An original chart from the modeling notebook. Values and labels are preserved as recorded.")
        elif model_id == "tiny":
            st.write("CT patch, reference mask and saved prediction from the augmented U-Net experiment.")
            st.caption("No numerical segmentation evaluation was recorded for this model.")
        else:
            st.write("Adjacent CT slices with their reference masks and model predictions, as plotted in the notebook.")
        st.caption("The original figure includes its own slice labels. Volume navigation is unavailable because CT arrays are not included.")
        st.download_button("Download original PNG", figure_path(selected).read_bytes(),
                           file_name=selected["file"], mime="image/png", width="stretch")
    with main:
        with st.container(border=True):
            st.html(f'<div class="viewer-label"><strong>{html.escape(model_name.upper())}</strong>'
                    f'<span>ARCHIVED EXAMPLE {example:02d}</span></div>')
            show_figure(selected)
            if collection == "Segmentation examples":
                st.html('<div class="legend"><span><i></i>CT image</span><span><i class="reference"></i>Reference mask</span>'
                        '<span><i class="prediction"></i>Saved prediction</span></div>')
                st.caption("Legend describes the figure panels; colors inside the original PNG are unchanged.")
    if experiment:
        st.subheader("Recorded experiment averages")
        st.caption("These are experiment-level scores, not measurements for the selected image.")
        metric_items = list(experiment["metrics"].items())
        if metric_items:
            columns = st.columns(len(metric_items))
            for column, (metric, value) in zip(columns, metric_items):
                with column:
                    stat(metric.upper() if metric == "f1" else metric.title(), f"{value:.4f}")
        st.info(experiment["note"])
    distributions = [f for f in archive["figures"] if f["group"] == group and f["category"] == "distribution"]
    if distributions and collection == "Segmentation examples":
        with st.expander("Original score-distribution chart"):
            show_figure(distributions[0])


def model_library():
    title("Model library", "Four approaches. One study.", "Inspect the architectures and checkpoint files preserved in the project. This dashboard reads saved results and never loads or runs these weights.")
    for start in [0, 2]:
        columns = st.columns(2, gap="large")
        for index, column in enumerate(columns, start=start):
            model = archive["models"][index]
            with column:
                model_card(model, index + 1)
                with st.expander("Checkpoint & experiment details"):
                    st.code("models/" + model["checkpoint"], language=None)
                    st.write(f"**Parameters:** {model['parameters']:,}")
                    st.write("**Input:** one-channel 48 × 48 × 48 CT patch")
                    st.write("**Output:** one-channel segmentation probability volume")
                    st.caption("Parameter counts come from the notebook architecture definitions.")
                    if model["id"] == "tiny":
                        st.write("Recorded training: 10 epochs; final validation BCE loss 0.0446. No saved Dice evaluation.")
                    elif model["id"] == "residual":
                        st.write("Two training runs write to the same checkpoint filename. Historical plots are associated with their notebook cells; the current file cannot establish which earlier results it reproduces.")
                    else:
                        st.write("See Results & methods for the recorded training curves and evaluation scopes.")
                    st.caption("Checkpoint SHA-256")
                    st.code(model["sha256"], language=None, wrap_lines=True)
    with st.container(border=True):
        st.subheader("The shared 3D segmentation pipeline")
        st.write("48³ CT patch → encoder → bottleneck → decoder with skip connections → probability mask")
        st.caption("Attention models add gates to the skip connections. The residual model also adds residual convolution blocks. Reference nodule masks were generated as spheres from annotation centers and diameters.")


def results():
    title("Evidence & evaluation", "Results, with the full picture.", "Recorded values, original training logs and evaluation context. Missing metrics stay unavailable; no benchmark is recomputed.")
    scores, training, methods = st.tabs(["Recorded scores", "Training curves", "Study notes"])
    with scores:
        st.info("These evaluations use different sample sets and empty-mask conventions. Read each experiment independently; the scores are not a model leaderboard.")
        rows = []
        for experiment in archive["experiments"]:
            rows.append({"Model": models[experiment["model"]]["name"], "Experiment": experiment["label"],
                         "Samples": str(experiment["sample_count"]) if experiment["sample_count"] else "Not recorded",
                         **{name: experiment["metrics"].get(name.lower()) for name in ["Dice", "Precision", "Recall", "F1", "Accuracy"]}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            name: st.column_config.NumberColumn(name, format="%.4f") for name in ["Dice", "Precision", "Recall", "F1", "Accuracy"]})
        st.caption("A dash means the metric was not recorded in that evaluation output. The augmented U-Net has saved figures and loss logs, but no numerical segmentation evaluation.")
        selected = st.selectbox("Inspect an evaluation", list(experiments),
                                format_func=lambda key: models[experiments[key]["model"]]["name"] + " / " + experiments[key]["label"], key="evaluation")
        experiment = experiments[selected]
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            metrics = experiment["metrics"]
            fig = go.Figure(go.Bar(x=[k.upper() if k == "f1" else k.title() for k in metrics],
                                   y=list(metrics.values()), marker_color="#63d9c1",
                                   text=[f"{v:.4f}" for v in metrics.values()], textposition="outside",
                                   hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
            fig.update_yaxes(range=[0, 1.15], title="Recorded score")
            st.plotly_chart(chart_theme(fig, 300), use_container_width=True, key="evaluation_chart")
        with right:
            st.markdown("**Evaluation context**")
            st.write(experiment["note"])
            source(experiment)
            st.caption("Decision threshold in these evaluation cells: 0.5. Saved values are rounded as printed in the notebook.")
        st.download_button("Download results CSV", evidence_csv(archive), file_name="pulmotrack_recorded_results.csv", mime="text/csv")
    with training:
        by_id = {run["id"]: run for run in archive["training"]}
        selected_run = st.selectbox("Training experiment", list(by_id), format_func=lambda key: by_id[key]["label"], key="training_run")
        run = by_id[selected_run]
        points = run["points"]
        columns = st.columns(3)
        with columns[0]:
            stat("Recorded epochs", len(points), "Extracted from printed training logs")
        with columns[1]:
            stat("Lowest logged validation loss", f'{min(p["validation_loss"] for p in points):.4f}', "Binary cross-entropy · lower is better")
        with columns[2]:
            stat("Train / validation patches", f'{run["training_patches"]:,} / {run["validation_patches"]:,}', "Historical patch split")
        log_scale = st.toggle("Logarithmic loss axis", value=False)
        fig = go.Figure()
        for field, label, color in [("train_loss", "Training loss", "#63d9c1"),
                                    ("validation_loss", "Validation loss", "#b2a2ef")]:
            fig.add_trace(go.Scatter(x=[p["epoch"] for p in points], y=[p[field] for p in points],
                                     mode="lines+markers", name=label, line=dict(color=color, width=2.5),
                                     marker=dict(size=4), hovertemplate="Epoch %{x}<br>Loss %{y:.4f}<extra>%{fullData.name}</extra>"))
        fig.update_xaxes(title="Epoch", dtick=5 if len(points) > 20 else 1)
        fig.update_yaxes(title="Binary cross-entropy", type="log" if log_scale else "linear")
        st.plotly_chart(chart_theme(fig), use_container_width=True, key="training_chart")
        source(run)
        st.caption("Curves use recorded four-decimal log values. A low training loss is not a Dice score or evidence of clinical accuracy.")
        st.download_button("Download this training log", pd.DataFrame(points).to_csv(index=False).encode(),
                           file_name=f"{selected_run}.csv", mime="text/csv")
    with methods:
        st.subheader("How to read this archive")
        for heading, text in [
            ("Saved evidence only", "Every CT image is an embedded notebook PNG. Every plotted loss and numerical evaluation is extracted from stored text outputs. The app does not require CT data, PyTorch, or model execution."),
            ("Reference masks", "The preparation notebook constructs spherical nodule masks from annotation centers and diameters. These are approximate reference regions, rather than manually delineated nodule boundaries."),
            ("Evaluation populations differ", "Positive-only samples, mixed random patches, and the validation loader are separate populations. Their scores do not establish a fair cross-model ranking. Tiny-nodule performance is not separately quantified by the saved evaluation outputs."),
            ("Training and validation limitations", "The dataset code randomizes image and mask augmentation separately. Validation inherits augmentation, and splitting is by patch rather than patient. The later residual training cell also splits the earlier dataset variable instead of its newly created oversampled dataset."),
            ("Empty-mask scoring", "The mixed attention evaluation assigns a perfect score to correctly empty masks. The U-Net validation Dice uses smoothing. These conventions can raise mixed averages relative to positive-only segmentation results."),
            ("What is unavailable", "The repository has no CT arrays, original patch data, stored prediction arrays, or persistent validation split. The app cannot browse arbitrary volume slices, recompute metrics, or verify checkpoint performance. Original figures and their source cells remain available."),
        ]:
            with st.expander(heading, expanded=heading == "Saved evidence only"):
                st.write(text)
        st.download_button("Download archive manifest", json.dumps(archive, indent=2).encode(),
                           file_name="pulmotrack_archive_manifest.json", mime="application/json")
        st.caption("Manifest contains figure and checkpoint checksums plus notebook/cell provenance. Cell numbers are 1-based.")


with st.sidebar:
    st.html('<div class="brand"><span class="brand-mark" aria-hidden="true">+</span>'
            '<div class="brand-name">pulmo<span>Track</span></div></div>'
            '<div class="brand-sub">CT segmentation · research archive</div>')
    view = st.radio("Workspace", ["Overview", "CT gallery", "Models", "Results & methods"], key="navigation")
    st.divider()
    st.caption("COLLECTION")
    st.markdown("**LUNA16 · Pulmonary nodules**")
    st.caption("3D CT patches / segmentation research")
    st.html('<div class="sidebar-note">Built from saved notebook outputs.<br>No dataset or model execution needed.<br><br>'
            '<strong>Farah Sayari</strong><br>pulmoTrack-farahsayari contribution</div>')

st.html('<div class="topline"><div class="eyebrow">Lung imaging / research workspace</div>'
        '<div class="archive-tag"><span class="archive-dot"></span>Saved results · no inference</div></div>')
{"Overview": overview, "CT gallery": gallery, "Models": model_library, "Results & methods": results}[view]()
st.html('<div class="footer"><span>pulmoTrack · Farah Sayari contribution</span>'
        '<span>Research prototype · not validated for clinical diagnosis</span></div>')
