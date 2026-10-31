"""Present saved segmentation summaries without synthesizing evaluation data."""
from __future__ import annotations

import plotly.graph_objects as go


METRIC_LABELS = {"dice": "Dice", "precision": "Precision", "recall": "Recall", "f1": "F1", "accuracy": "Voxel accuracy"}


def metric_chart(metrics):
    """Plot only reported summaries; unavailable metrics are not zero scores."""
    fields = [field for field in METRIC_LABELS if field in metrics]
    chart = go.Figure(go.Bar(
        x=[metrics[field] for field in fields], y=[METRIC_LABELS[field] for field in fields],
        orientation="h", marker=dict(color="#63d9c1", cornerradius=5),
        text=[f"{metrics[field]:.4f}" for field in fields], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    chart.update_layout(height=320, paper_bgcolor="#111d2a", plot_bgcolor="#111d2a",
                        font=dict(color="#b5c5d5", family="Arial", size=12),
                        margin=dict(l=10, r=55, t=20, b=40), showlegend=False, bargap=0.45)
    chart.update_xaxes(title="Reported score (0–1)", range=[0, 1.12],
                      tickvals=[0, 0.25, 0.5, 0.75, 1], gridcolor="#243245", zeroline=False)
    chart.update_yaxes(autorange="reversed", showgrid=False, fixedrange=True)
    return chart


def interpretation(record):
    """Keep each historical result tied to its sample and scoring convention."""
    metrics = record["metrics"]
    if record["id"] == "unet_validation":
        return (f'The recorded Dice of {metrics["dice"]:.4f} comes from the mixed validation loader. '
                'Smoothing gives empty prediction/reference pairs a Dice near 1. Positive-only samples '
                'recorded Dice scores of 0.2115 and 0.3127 on different five-patch selections, '
                'so the validation summary alone does not establish strong nodule delineation.')
    if record["id"] in ("unet_positive_a", "unet_positive_b"):
        return (f'The recorded Dice of {metrics["dice"]:.4f} indicates limited overlap on this '
                'five-positive-patch sample. The two baseline samples contain different patches; '
                'their scores are not a controlled before-and-after comparison.')
    if record["id"] == "unet_full_positive":
        return (f'Precision of {metrics["precision"]:.2%} with recall of {metrics["recall"]:.2%} '
                'means predicted nodule voxels are more often correct than reference nodule voxels '
                'are recovered. This points to missed nodule regions. The evaluated folder’s '
                'held-out status and sample count are not established.')
    if record["id"] == "attention_positive":
        return (f'Dice of {metrics["dice"]:.4f}, precision of {metrics["precision"]:.2%} and '
                f'recall of {metrics["recall"]:.2%} show limited overlap on five positive patches. '
                'Both extra predicted voxels and missed reference voxels remain substantial in this sample.')
    if record["id"] == "attention_mixed":
        return (f'The reported Dice of {metrics["dice"]:.4f} covers ten mixed positive and negative patches. '
                'The evaluator explicitly assigns 1 to Dice, precision, recall and F1 when both masks '
                'are empty. These averages therefore cannot be read as nodule-only segmentation performance.')
    if record["id"] in ("residual_initial", "residual_repeat"):
        return (f'Dice of {metrics["dice"]:.4f} and recall of {metrics["recall"]:.2%} show poor recovery '
                'of reference nodule voxels on this ten-positive-patch sample. High voxel accuracy '
                'mainly reflects agreement on background and does not resolve the low nodule overlap. '
                'The initial and later runs are separate historical evaluations.')
    raise ValueError("Unrecognized segmentation evaluation context.")
