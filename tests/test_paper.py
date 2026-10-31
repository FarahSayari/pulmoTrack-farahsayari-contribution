"""Protect document fidelity and attribution of the portfolio highlights."""
import hashlib

from PIL import Image
from streamlit.testing.v1 import AppTest

from pulmotrack.archive import ROOT, load_archive
from pulmotrack.paper import load_paper, paper_path


def test_original_paper_and_complete_page_previews():
    paper = load_paper()
    original = paper_path(paper["file"]).read_bytes()
    assert original.startswith(b"%PDF-")
    assert hashlib.sha256(original).hexdigest() == paper["sha256"]
    assert paper["sha256"] == "a819b151f81c87a21a2d7bd49a01a9cb335d3f1ec2d7fdd6a0eb407b3cd8e07e"
    assert [page["page"] for page in paper["pages"]] == list(range(1, 12))
    for page in paper["pages"]:
        path = paper_path(page["file"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page["sha256"]
        with Image.open(path) as image:
            assert image.size == (page["width"], page["height"])


def test_paper_highlights_do_not_replace_notebook_evaluations():
    paper = load_paper()
    assert paper["highlights_source_pages"] == [4, 5]
    assert paper["highlights"] == dict(ct_scans=888, segmentation_dice=0.95,
                                      detection_sensitivity=0.92, false_positives_per_scan=1.5,
                                      classification_auc=0.91)
    records = {e["id"]: e for e in load_archive()["experiments"]}
    assert records["cls_cnn_test"]["metrics"]["roc_auc"] == 0.8335
    assert records["unet_validation"]["metrics"]["dice"] == 0.9504


def test_research_paper_button_opens_the_document_viewer():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert not app.exception
    app.button(key="view_paper").click().run()
    assert not app.exception
    assert app.selectbox(key="paper_page").value == 1
    assert len(app.selectbox(key="paper_page").options) == 11
