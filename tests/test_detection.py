"""Binary charts preserve experimental evidence and its label orientation."""
import base64
import xml.etree.ElementTree as ET

import pytest

from pulmotrack.archive import figure_path, load_archive
from pulmotrack.detection import confusion_chart, outcome_summary, roc_display


@pytest.mark.parametrize("cohort", ["det_test", "det_validation", "cls_cnn_test", "cls_cnn_validation",
                                    "cls_resnet_validation", "cls_unet_validation"])
def test_binary_presentation_preserves_counts_and_original_roc(cohort):
    archive = load_archive()
    record = next(item for item in archive["experiments"] if item["id"] == cohort)
    roc = next(item for item in archive["figures"] if item["id"] == record["roc_figure_id"])
    chart = confusion_chart(record["matrix"])
    assert [list(row) for row in chart.data[0].z] == record["matrix"]
    assert chart.layout.yaxis.autorange == "reversed"
    assert "False positive" in chart.data[0].text[0][1]
    assert "False negative" in chart.data[0].text[1][0]
    summary = outcome_summary(record["matrix"])
    assert summary["tp"] + summary["tn"] == round(record["metrics"]["accuracy"] * record["sample_count"])
    assert summary["sensitivity"] == record["matrix"][1][1] / sum(record["matrix"][1])
    original = figure_path(roc).read_bytes()
    svg = ET.fromstring(roc_display(original, roc["width"], roc["height"], "Original ROC"))
    embedded = svg.find("{http://www.w3.org/2000/svg}image")
    assert base64.b64decode(embedded.attrib["href"].split(",", 1)[1]) == original
    assert not svg.findall(".//{http://www.w3.org/2000/svg}path"), "Do not invent ROC coordinates."
