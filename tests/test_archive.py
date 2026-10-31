"""Check evidence fidelity and real Streamlit interaction without inference."""
import base64
import builtins
import csv
import hashlib
import io
import json

from streamlit.testing.v1 import AppTest

from pulmotrack.archive import ROOT, evidence_csv, figure_path, load_archive


def test_all_figures_are_exact_notebook_outputs():
    archive = load_archive()
    notebooks = {name: json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
                 for name in archive["notebooks"]}
    expected_count = sum("image/png" in output.get("data", {})
                         for notebook in notebooks.values() for cell in notebook["cells"]
                         for output in cell.get("outputs", []))
    assert len(archive["figures"]) == expected_count == 83
    for figure in archive["figures"]:
        output = notebooks[figure["notebook"]]["cells"][figure["cell"] - 1]["outputs"][figure["output"] - 1]
        original = base64.b64decode("".join(output["data"]["image/png"]))
        assert figure_path(figure).read_bytes() == original
        assert hashlib.sha256(original).hexdigest() == figure["sha256"]


def test_recorded_values_and_missing_metrics_survive_export():
    archive = load_archive()
    experiments = {row["id"]: row for row in archive["experiments"]}
    assert experiments["unet_validation"]["metrics"] == {"dice": 0.9504}
    assert experiments["attention_positive"]["metrics"]["dice"] == 0.1599
    assert experiments["attention_mixed"]["metrics"]["dice"] == 0.98
    assert experiments["residual_repeat"]["metrics"]["dice"] == 0.0348
    assert not any(row["model"] == "tiny" for row in archive["experiments"])
    training = {run["id"]: run for run in archive["training"]}
    assert [len(run["points"]) for run in archive["training"]] == [50, 50, 21, 17, 10]
    assert training["tiny_training"]["points"][-1]["validation_loss"] == 0.0446
    csv_rows = list(csv.DictReader(io.StringIO(evidence_csv(archive).decode("utf-8-sig"))))
    assert len(csv_rows) == 8
    validation = next(row for row in csv_rows if row["experiment"] == "Mixed validation loader")
    assert validation["precision"] == ""
    assert validation["evaluation_context"]


def assert_no_errors(app):
    assert not app.exception, [error.message for error in app.exception]


def test_all_views_and_controls_without_reading_models_or_notebooks(monkeypatch):
    original_open, original_io_open = builtins.open, io.open

    def guard(original):
        def guarded(file, *args, **kwargs):
            assert not str(file).endswith((".pth", ".ipynb")), "The UI must not read weights or notebooks."
            return original(file, *args, **kwargs)
        return guarded

    monkeypatch.setattr(builtins, "open", guard(original_open))
    monkeypatch.setattr(io, "open", guard(original_io_open))
    app = AppTest.from_file(ROOT / "app.py", default_timeout=20).run()
    assert_no_errors(app)
    app.button[0].click().run()
    assert app.sidebar.radio[0].value == "CT gallery"
    archive = load_archive()
    for model in archive["models"]:
        app.selectbox(key="gallery_model").set_value(model["id"]).run()
        groups = list(dict.fromkeys(f["group"] for f in archive["figures"]
                                   if f["model"] == model["id"] and f["category"] == "prediction"))
        for group in groups:
            app.selectbox(key=f'gallery_experiment_{model["id"]}').set_value(group).run()
            count = sum(f["group"] == group and f["category"] == "prediction" for f in archive["figures"])
            app.slider(key=f"example_{group}").set_value(count).run()
            assert_no_errors(app)
    app.radio(key="gallery_collection").set_value("CT preparation").run()
    for group in dict.fromkeys(f["group"] for f in archive["figures"] if f["category"] == "preparation"):
        app.selectbox(key="preparation_stage").set_value(group).run()
        assert_no_errors(app)
    app.radio(key="gallery_collection").set_value("Experiment charts").run()
    for group in dict.fromkeys(f["group"] for f in archive["figures"] if f["category"] in ["chart", "distribution"]):
        app.selectbox(key="saved_chart").set_value(group).run()
        assert_no_errors(app)
    app.sidebar.radio[0].set_value("Models").run()
    assert_no_errors(app)
    app.sidebar.radio[0].set_value("Results & methods").run()
    for experiment in archive["experiments"]:
        app.selectbox(key="evaluation").set_value(experiment["id"]).run()
        assert_no_errors(app)
    for run in archive["training"]:
        app.selectbox(key="training_run").set_value(run["id"]).run()
        assert_no_errors(app)
    app.toggle[0].set_value(True).run()
    assert_no_errors(app)
