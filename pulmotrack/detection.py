"""Presentation of preserved detection evaluations; no model execution."""
from __future__ import annotations

import base64
import html

import plotly.graph_objects as go


def confusion_chart(matrix):
    """Keep true labels on rows and predicted labels on columns."""
    outcomes = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"{count:,}<br>{outcomes[row][column]}" for column, count in enumerate(values)]
            for row, values in enumerate(matrix)]
    chart = go.Figure(go.Heatmap(
        z=matrix, x=["Non-nodule", "Nodule"], y=["Non-nodule", "Nodule"],
        text=text, texttemplate="%{text}", textfont=dict(size=16, color="#edf5f9"),
        colorscale=[[0, "#172a38"], [0.35, "#24534f"], [1, "#358675"]],
        zmin=0, zmax=max(max(row) for row in matrix), showscale=False,
        xgap=6, ygap=6,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Patches: %{z:,}<extra></extra>"))
    chart.update_layout(height=390, paper_bgcolor="#111d2a", plot_bgcolor="#111d2a",
                        margin=dict(l=10, r=10, t=25, b=45),
                        font=dict(color="#b5c5d5", family="Arial", size=12))
    chart.update_xaxes(title="Predicted label", side="bottom", fixedrange=True, showgrid=False)
    chart.update_yaxes(title="Actual label", autorange="reversed", fixedrange=True, showgrid=False)
    return chart


def roc_display(png_bytes, width, height, label):
    """Recolor the original raster in SVG without tracing or synthesizing ROC points.

    The affine sRGB palette maps white to navy, black to light gray, the original
    Matplotlib blue to teal, and its orange reference line to muted blue-gray.
    All image geometry and the archived PNG itself remain unchanged.
    """
    encoded = base64.b64encode(png_bytes).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{int(width)}" height="{int(height)}" '
        f'viewBox="0 0 {int(width)} {int(height)}" role="img" '
        f'aria-label="{html.escape(label, quote=True)}">'
        '<defs><filter id="detection-roc-palette" color-interpolation-filters="sRGB">'
        '<feColorMatrix type="matrix" values="'
        '0.26982632 -1.21260373 0.29964016 0 0.70980392 '
        '-5.51034920 11.18776713 -6.33624146 0 0.77254902 '
        '-3.62520930 7.20410333 -4.24948226 0 0.83529412 '
        '0 0 0 1 0"/></filter></defs>'
        f'<image width="{int(width)}" height="{int(height)}" '
        f'href="data:image/png;base64,{encoded}" filter="url(#detection-roc-palette)"/></svg>'
    )
    return svg


def outcome_summary(matrix):
    """Threshold metrics derived solely from the saved integer confusion matrix."""
    (tn, fp), (fn, tp) = matrix
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, sensitivity=tp / (tp + fn), precision=tp / (tp + fp))
