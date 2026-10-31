# pulmoTrack-farahsayari contribution

An interactive Streamlit research archive for pulmonary nodule segmentation on **LUNA16 CT scans**, maintained by **Farah Sayari**.

The interface uses **existing notebook outputs only**. No dataset, training, inference, PyTorch installation, or model download is required to use it.

![pulmoTrack interface](docs/preview.png)

## Run the app

From the project directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally **http://localhost:8501**. On Windows, after installing dependencies, you can also double-click **run_app.bat**.

For an isolated environment, create one with `python -m venv .venv` and activate it before installing the requirements.

## Inside the interface

- **Overview:** a CT-focused introduction and evidence inventory.
- **CT gallery:** 83 original figures covering scan preparation, reference masks, saved predictions, and experiment charts. Filter by model, experiment, and saved example; download the original PNGs.
- **Models:** four stored checkpoints, architecture descriptions, parameter counts, file sizes, and checksums.
- **Results & methods:** eight recorded evaluations, interactive curves from 148 logged epochs across five runs, study limitations, and CSV/JSON exports.

The navy and teal workspace is designed around CT review. Gallery controls select saved figures; they do not reconstruct missing volumes or generate new predictions.

## Stored models

| Model | Checkpoint | Parameters |
| --- | --- | ---: |
| 3D U-Net | `best_unet3d_model.pth` | 1,356,641 |
| Attention 3D U-Net | `best_att_unet3d_model.pth` | 1,362,103 |
| Residual Attention 3D U-Net | `best_resattn_unet3d.pth` | 1,382,935 |
| U-Net with tiny-nodule oversampling and augmentation | `best_unet3d_tiny_augmented.pth` | 1,356,641 |

The notebook architectures use one-channel **48 × 48 × 48** patches. The dashboard does not read the checkpoint files at runtime.

## Interpreting the recorded results

The U-Net validation notebook reports **Dice 0.9504**, while two separate five-positive-patch samples report **0.2115** and **0.3127**. Attention U-Net reports **0.1599** on five positive patches and **0.9800** on ten mixed positive/negative patches. These are different evaluations, not a common benchmark or a fair ranking of models.

Each score in the app retains its source notebook, 1-based cell number, sample scope, and scoring context. Unrecorded metrics remain missing. The augmented U-Net has saved prediction figures and ten training epochs ending at **validation BCE loss 0.0446**, but no numerical segmentation evaluation in the saved outputs.

Relevant limitations visible in the notebook code:

- Nodule reference masks are generated spheres based on annotation centers and diameters, not manually delineated boundaries.
- Patch and mask augmentations are independently randomized; validation inherits augmentation.
- Splitting is by patch rather than patient. A later residual-training cell creates an oversampled dataset but splits an earlier dataset variable.
- Empty-mask scoring differs between evaluations. Mixed averages can be higher than positive-only segmentation scores.
- Multiple residual training runs overwrite the same checkpoint path, so historical figures cannot all be attributed to the current file.

This is a **research presentation prototype**, not a tool validated for clinical diagnosis. The original notebooks are preserved unchanged.

## Files and reproducibility

```text
app.py                       Streamlit interface
pulmotrack/archive.py         Archive reading and CSV export
assets/archive/              Original PNG outputs and provenance manifest
assets/style.css             Radiology workspace theme
scripts/extract_archive.py   Export stored outputs without executing cells
notebooks/                   Original data preparation and modeling notebooks
models/                      Four original checkpoints
tests/test_archive.py        Evidence integrity and UI interaction checks
```

All exported figures are byte-for-byte copies of embedded notebook PNGs. The manifest records figure, checkpoint, and notebook SHA-256 checksums. Curves use the values printed in saved training logs, rounded to four decimal places.

To regenerate the archive from the existing notebooks, without running their cells or models:

```powershell
python scripts/extract_archive.py
```

To run the checks:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`requirements-notebooks.txt` lists optional research dependencies. They are unnecessary for the interface; the original notebooks also require external data and edits to their historical Colab/Linux paths.
