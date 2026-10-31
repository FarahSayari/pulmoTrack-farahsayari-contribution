"""Export saved notebook outputs without executing notebook cells or models."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "assets" / "archive"
MODELING = "02_Modeling_Segmentation.ipynb"
ACQUISITION = "01_luna16_Data Acquisition.ipynb"

MODELS = [
    ("unet", "3D U-Net", "best_unet3d_model.pth", 1356641,
     "The baseline encoder–decoder with skip connections for 3D patch segmentation."),
    ("attention", "Attention 3D U-Net", "best_att_unet3d_model.pth", 1362103,
     "Attention gates weight the encoder features passed to the decoder."),
    ("residual", "Residual Attention 3D U-Net", "best_resattn_unet3d.pth", 1382935,
     "Residual convolution blocks combine with attention gates and skip connections."),
    ("tiny", "U-Net · tiny-nodule augmentation", "best_unet3d_tiny_augmented.pth", 1356641,
     "The baseline architecture trained with tiny-nodule oversampling and augmentation."),
]

# Cell numbers below are zero-based notebook indices; public provenance is one-based.
PREDICTIONS = {
    12: ("unet_positive_a", "unet", "Positive patches · experiment A", 5),
    19: ("unet_positive_b", "unet", "Positive patches · experiment B", 5),
    28: ("attention_positive", "attention", "Positive patches", 5),
    30: ("attention_mixed", "attention", "Mixed positive / negative patches", 10),
    36: ("residual_initial", "residual", "Positive patches · initial run", 10),
    39: ("residual_repeat", "residual", "Positive patches · later run", 10),
    46: ("tiny_examples", "tiny", "Augmented-model validation examples", 3),
}

PREPARATION = {
    6: "Voxel spacing distribution", 10: "Resampled CT and lung masks",
    14: "Lung region extraction", 18: "Axial, coronal and sagittal CT views",
    24: "Nodule center annotation", 26: "Nodule diameter annotation",
    30: "CT, spherical reference mask and overlay", 34: "Positive CT patches",
    38: "Patch extraction inspection", 41: "Reference-mask patch examples",
    42: "Reference-mask patch examples · repeat", 46: "Negative CT patches",
    51: "CT patch augmentation examples",
}

CHARTS = {14: "U-Net training and validation loss", 17: "U-Net positive-sample Dice distribution",
          34: "Residual attention · initial training loss", 38: "Residual attention · later training loss",
          44: "Augmented U-Net training and validation loss"}

EXPERIMENTS = [
    ("unet_positive_a", "unet", 12, "Positive patches · experiment A", 5,
     "Five randomly selected positive patches; not a documented held-out patient test."),
    ("unet_positive_b", "unet", 19, "Positive patches · experiment B", 5,
     "A separate random sample of five positive patches; sample identities differ from experiment A."),
    ("unet_full_positive", "unet", 16, "Full positive-patch set", None,
     "Voxel-level precision, recall and F1 over the positive-patch folder; held-out status is not established."),
    ("unet_validation", "unet", 20, "Mixed validation loader", 509,
     "Patch-level random validation split. Augmentation remains enabled; empty-mask matches receive Dice near 1 through smoothing."),
    ("attention_positive", "attention", 28, "Five positive patches", 5,
     "Random positive-patch sample; no common evaluation sample is established across models."),
    ("attention_mixed", "attention", 30, "Ten mixed patches", 10,
     "Random positives and negatives. Empty prediction/reference pairs are assigned 1 for Dice, precision, recall and F1."),
    ("residual_initial", "residual", 36, "Ten positive patches · initial run", 10,
     "Historical evaluation before the shared checkpoint path was overwritten by a later training run."),
    ("residual_repeat", "residual", 39, "Ten positive patches · later run", 10,
     "Later historical evaluation. The preceding training cell creates an oversampled dataset but splits the earlier dataset variable."),
]

TRAINING = [
    ("unet_training", "unet", 10, "Baseline training", 2036, 509),
    ("attention_training", "attention", 26, "Attention training", 2036, 509),
    ("residual_training", "residual", 34, "Residual attention · initial run", 2036, 509),
    ("residual_later_training", "residual", 38, "Residual attention · later run", 2036, 509),
    ("tiny_training", "tiny", 43, "Tiny-nodule augmentation training", 2140, 536),
]


def output_text(cell):
    return "\n".join("".join(o.get("text", [])) for o in cell.get("outputs", []))


def extract():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    notebooks = {
        p.name: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / "notebooks").glob("*.ipynb"))
    }
    figures = []
    for name, notebook in notebooks.items():
        prefix = "preparation" if name == ACQUISITION else "modeling"
        for cell_index, cell in enumerate(notebook["cells"]):
            image_index = 0
            for output_index, output in enumerate(cell.get("outputs", [])):
                encoded = output.get("data", {}).get("image/png")
                if not encoded:
                    continue
                image_index += 1
                data = base64.b64decode("".join(encoded))
                filename = f"{prefix}_{cell_index + 1:02d}_{image_index:02d}.png"
                (DESTINATION / filename).write_bytes(data)
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size
                record = dict(id=filename[:-4], file=filename, notebook=name,
                              cell=cell_index + 1, output=output_index + 1,
                              width=width, height=height,
                              sha256=hashlib.sha256(data).hexdigest())
                if name == MODELING and cell_index in PREDICTIONS:
                    group, model, title, count = PREDICTIONS[cell_index]
                    record.update(group=group, model=model, title=title,
                                  category="prediction" if image_index <= count else "distribution",
                                  example=image_index)
                elif name == ACQUISITION:
                    record.update(group=f"preparation_{cell_index}", model=None,
                                  title=PREPARATION.get(cell_index, "CT preparation"),
                                  category="preparation", example=image_index)
                else:
                    record.update(group=f"modeling_{cell_index}", model=None,
                                  title=CHARTS.get(cell_index, "Recorded experiment chart"), category="chart", example=image_index)
                figures.append(record)

    experiments = []
    for identifier, model, index, label, count, note in EXPERIMENTS:
        text = output_text(notebooks[MODELING]["cells"][index])
        metrics = {}
        for key, pattern in {
            "dice": r"(?:Average Dice Score(?: on \d+ samples)?|Final Validation Dice Score):\s*([0-9.]+)",
            "precision": r"(?:Average )?Precision:\s*([0-9.]+)",
            "recall": r"(?:Average )?Recall:\s*([0-9.]+)",
            "f1": r"(?:Average )?F1 Score:\s*([0-9.]+)",
            "accuracy": r"(?:Average )?Accuracy:\s*([0-9.]+)",
        }.items():
            match = re.search(pattern, text)
            if match:
                metrics[key] = float(match.group(1))
        experiments.append(dict(id=identifier, model=model, label=label, sample_count=count,
                                note=note, metrics=metrics, notebook=MODELING, cell=index + 1))

    training = []
    for identifier, model, index, label, train_count, validation_count in TRAINING:
        text = output_text(notebooks[MODELING]["cells"][index])
        rows = [dict(epoch=int(epoch), train_loss=float(train), validation_loss=float(val))
                for epoch, train, val in re.findall(
                    r"Epoch \[(\d+)/\d+\] - Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+)", text)]
        training.append(dict(id=identifier, model=model, label=label, points=rows,
                             training_patches=train_count, validation_patches=validation_count,
                             saved_epochs=[int(x) for x in re.findall(r"New best model saved at epoch (\d+)", text)],
                             notebook=MODELING, cell=index + 1))

    models = []
    for identifier, name, filename, parameters, description in MODELS:
        path = ROOT / "models" / filename
        models.append(dict(id=identifier, name=name, checkpoint=filename,
                           parameters=parameters, description=description,
                           size_bytes=path.stat().st_size,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    manifest = dict(schema_version=1, title="pulmoTrack-farahsayari contribution",
                    mode="saved_outputs_only", models=models, experiments=experiments,
                    training=training, figures=figures,
                    notebooks={name: hashlib.sha256((ROOT / "notebooks" / name).read_bytes()).hexdigest()
                               for name in notebooks})
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(figures)} original figures, {len(experiments)} evaluations, "
          f"{sum(len(run['points']) for run in training)} recorded epochs. No cells executed.")


if __name__ == "__main__":
    extract()
