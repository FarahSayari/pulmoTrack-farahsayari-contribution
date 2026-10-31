"""Evidence fidelity and real Streamlit interactions, without dataset/model access."""
import base64
import builtins
import csv
import hashlib
import io
import json
import subprocess
import sys

from streamlit.testing.v1 import AppTest

from pulmotrack.archive import ROOT, evidence_csv, figure_path, load_archive


def test_selected_evidence_is_exact_and_sources_are_unmodified():
    archive = load_archive()
    assert archive["schema_version"] == 2
    assert len(archive["sources"]) == 12
    assert len(archive["figures"]) == 123
    assert len({f["file"] for f in archive["figures"]}) == 123
    notebooks = {}
    for name, checksum in archive["notebooks"].items():
        raw = (ROOT / "notebooks" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == checksum
        notebooks[name] = json.loads(raw)
    assert "11_nodule_classification_updated.ipynb" not in notebooks
    expected_count = sum("image/png" in output.get("data", {}) for notebook in notebooks.values()
                         for cell in notebook["cells"] for output in cell.get("outputs", []))
    assert expected_count == len(archive["figures"])
    for figure in archive["figures"]:
        output = notebooks[figure["notebook"]]["cells"][figure["cell"] - 1]["outputs"][figure["output"] - 1]
        original = base64.b64decode("".join(output["data"]["image/png"]))
        assert figure_path(figure).read_bytes() == original
        assert hashlib.sha256(original).hexdigest() == figure["sha256"]
    animation = archive["animations"][0]
    assert animation["frames"] == 48
    assert figure_path(animation).read_bytes() == (ROOT / "notebooks" / "gradcam_scroll.gif").read_bytes()


def test_evaluations_keep_cohorts_and_checkpoints_separate():
    archive = load_archive()
    experiments = {e["id"]: e for e in archive["experiments"]}
    assert experiments["unet_validation"]["metrics"] == {"dice": 0.9504}
    assert not any(e["model"] == "tiny" for e in archive["experiments"])
    assert experiments["det_test"]["metrics"]["roc_auc"] == 0.9961
    assert experiments["det_validation"]["matrix"] == [[1608, 16], [14, 205]]
    assert experiments["det_validation"]["metrics"]["accuracy"] == 1813 / 1843
    assert experiments["det_validation"]["last_epoch_accuracy"] == 0.9788
    assert experiments["cls_resnet_validation"]["sample_count"] == 401
    assert experiments["cls_resnet_validation"]["metrics"]["accuracy"] == 289 / 401
    assert experiments["cls_resnet_validation"]["last_epoch_accuracy"] == 0.6584
    assert experiments["cls_cnn_test"]["metrics"]["roc_auc"] == 0.8335
    assert "roc_auc" not in experiments["pilot_cnn_test"]["metrics"]
    figures = {f["id"]: f for f in archive["figures"]}
    for e in archive["experiments"]:
        if "matrix" in e:
            for kind in ["matrix", "roc"]:
                figure = figures[e[kind + "_figure_id"]]
                assert figure["notebook"] == e["notebook"] and figure["cell"] == e["cell"]
                assert figure["sha256"] == e[kind + "_figure_sha256"]
    for dso, expected in [("dso1", 3), ("dso2", 4), ("dso5", 8)]:
        rows = list(csv.DictReader(io.StringIO(evidence_csv(archive, dso).decode("utf-8-sig"))))
        assert len(rows) == expected
        assert all(row["dso"] == dso for row in rows)
    runs = {r["id"]: r for r in archive["training"]}
    assert sum(len(r["points"]) for r in runs.values()) == 261
    assert runs["pilot_resnet_training"]["points"][-1]["validation_accuracy"] == 0.8621
    assert runs["cls_cnn_training"]["points"][-1]["validation_accuracy"] == 0.735
    assert runs["tiny_training"]["points"][-1]["validation_loss"] == 0.0446


def test_export_is_deterministic_and_ignores_unselected_drafts():
    manifest = ROOT / "assets/archive/manifest.json"
    previous = manifest.read_bytes()
    subprocess.run([sys.executable, "scripts/extract_archive.py"], cwd=ROOT, check=True, capture_output=True)
    assert manifest.read_bytes() == previous


def assert_no_errors(app):
    assert not app.exception, [error.message for error in app.exception]


def test_every_objective_and_archive_control_without_weights_or_notebooks(monkeypatch):
    original_open, original_io_open = builtins.open, io.open

    def guard(original):
        def guarded(file, *args, **kwargs):
            assert not str(file).endswith((".pth", ".ipynb")), "The UI must not read weights or notebooks."
            return original(file, *args, **kwargs)
        return guarded

    monkeypatch.setattr(builtins, "open", guard(original_open))
    monkeypatch.setattr(io, "open", guard(original_io_open))
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert_no_errors(app)
    app.button[0].click().run()
    assert app.sidebar.radio[0].value == "DSO1 · Detection"
    for cohort in ["det_test", "det_validation"]:
        app.radio(key="detection_evaluation").set_value(cohort).run()
        assert_no_errors(app)
    app.radio(key="cam_display").set_value("Inspect a frame").run()
    app.slider(key="cam_slice").set_value(63).run()
    assert_no_errors(app)
    app.selectbox(key="pilot_run").set_value("pilot_resnet_training").run()
    assert_no_errors(app)
    app.sidebar.radio[0].set_value("DSO2 · Classification").run()
    for model in ["cls_cnn", "cls_resnet", "cls_unet"]:
        app.selectbox(key="classification_model").set_value(model).run()
        assert_no_errors(app)
    app.sidebar.radio[0].set_value("DSO5 · Segmentation").run()
    for model in ["unet", "attention", "residual", "tiny"]:
        app.selectbox(key="seg_model").set_value(model).run()
        assert_no_errors(app)
    app.selectbox(key="segmentation_run").set_value("tiny_training").run()
    app.toggle(key="segmentation_log").set_value(True).run()
    assert_no_errors(app)
    app.sidebar.radio[0].set_value("CT gallery").run()
    for dso in ["dso1", "dso2", "dso5", "shared", "all"]:
        app.selectbox(key="gallery_dso").set_value(dso).run()
        assert_no_errors(app)
    app.radio(key="gallery_tier").set_value("All studies").run()
    for category in ["confusion_matrix", "roc", "gradcam", "prediction", "training", "preparation"]:
        app.selectbox(key="gallery_category").set_value(category).run()
        assert_no_errors(app)
    app.radio(key="gallery_tier").set_value("Earlier pilot studies").run()
    app.selectbox(key="gallery_category").set_value("roc").run()
    assert_no_errors(app)
    assert any("No saved figures" in info.value for info in app.info)
    app.sidebar.radio[0].set_value("Models & sources").run()
    for dso in ["dso1", "dso2", "dso5"]:
        app.radio(key="models_dso").set_value(dso).run()
        assert_no_errors(app)
