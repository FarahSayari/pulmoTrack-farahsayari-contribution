"""Read exported evidence. No model or dataset dependencies."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "assets" / "archive"


def load_archive():
    return json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))


def figure_path(figure):
    path = (ARCHIVE / figure["file"]).resolve()
    if path.parent != ARCHIVE.resolve() or not path.is_file():
        raise ValueError("The requested figure is not in the saved archive.")
    return path


def evidence_csv(archive, dso=None):
    output = io.StringIO(newline="")
    fields = ["dso", "tier", "model", "experiment", "stage", "samples", "dice", "precision", "recall", "f1",
              "accuracy", "roc_auc", "accuracy_basis", "last_epoch_accuracy", "notebook", "cell", "evaluation_context"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    models = {model["id"]: model["name"] for model in archive["models"]}
    for experiment in archive["experiments"]:
        if dso and experiment["dso"] != dso:
            continue
        writer.writerow(dict(dso=experiment["dso"], tier=experiment["tier"], stage=experiment["stage"],
                             accuracy_basis=experiment["accuracy_basis"], last_epoch_accuracy=experiment.get("last_epoch_accuracy"),
                             model=models[experiment["model"]], experiment=experiment["label"],
                             samples=experiment["sample_count"], **experiment["metrics"],
                             notebook=experiment["notebook"], cell=experiment["cell"],
                             evaluation_context=experiment["note"]))
    return output.getvalue().encode("utf-8-sig")


def source_label(record):
    return f"{record['notebook']} · cell {record['cell']} (1-based)"
