"""Segmentation summaries retain their metrics, cohorts and missing values."""
import pytest

from pulmotrack.archive import load_archive
from pulmotrack.segmentation import METRIC_LABELS, interpretation, metric_chart


RECORDS = [record for record in load_archive()["experiments"] if record["dso"] == "dso5"]


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_segmentation_charts_only_report_preserved_metrics(record):
    fields = [field for field in METRIC_LABELS if field in record["metrics"]]
    trace = metric_chart(record["metrics"]).data[0]
    assert list(trace.x) == [record["metrics"][field] for field in fields]
    assert list(trace.y) == [METRIC_LABELS[field] for field in fields]
    assert len(trace.x) == len(record["metrics"])
    assert interpretation(record)


def test_mixed_cohort_context_and_unavailable_tiny_scores_are_preserved():
    records = {record["id"]: record for record in RECORDS}
    assert "empty" in interpretation(records["unet_validation"])
    assert "empty" in interpretation(records["attention_mixed"])
    assert "background" in interpretation(records["residual_initial"])
    assert "dice" not in records["unet_full_positive"]["metrics"]
    assert not any(record["model"] == "tiny" for record in RECORDS)
    assert all(record["dso"] == "dso5" for record in RECORDS)
