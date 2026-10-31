"""Curated, audited study identities and source cells (all cells are 1-based)."""

DETECTION = "11_nodule_detection.ipynb"
CLASSIFICATION = "11_nodule_classification.ipynb"
PILOT_CNN = "7_train_classifier.ipynb"
PILOT_RESNET = "7b_train_classifier_resnet18.ipynb"

DSOS = [
    dict(id="dso1", number=1, name="Nodule detection", role="Main objective",
         description="Screen provided nodule candidates using 3D CT patches.",
         scope="The recorded model classifies supplied candidates as nodule or non-nodule; it is not a full-volume localization system."),
    dict(id="dso2", number=2, name="Nodule classification", role="Main objective",
         description="Explore CNN, residual and U-Net encoders for candidate classification.",
         scope="Recorded labels are true nodule and false nodule. Normal / benign / malignant classification is a project goal, not an evaluated outcome in these notebooks."),
    dict(id="dso5", number=5, name="Exploratory segmentation", role="Research challenge",
         description="Study 3D segmentation architectures and the difficulty of small nodules.",
         scope="An independent model-development exploration. The intended later pipeline uses the provided LUNA16 lung masks rather than learned nodule masks; this archive does not demonstrate that integration."),
]

EXTRA_NOTEBOOKS = {
    DETECTION: dict(prefix="detection", dso="dso1", tier="main", role="Primary detection results"),
    CLASSIFICATION: dict(prefix="classification", dso="dso2", tier="main", role="Primary binary classification results"),
    "3_resample_ct_scans (1).ipynb": dict(prefix="resampling", dso="shared", tier="preparation", role="Resampling · 888 scans"),
    "4_segment_lungs.ipynb (1).ipynb": dict(prefix="lung_demo", dso="shared", tier="preparation", role="Historical threshold-based lung masks"),
    "8_nodule_detection_preparation (1).ipynb": dict(prefix="candidates", dso="shared", tier="preparation", role="Candidate matching · 754,975 candidates"),
    "9_Detection_Dataset_Creation (2).ipynb": dict(prefix="patch_dataset", dso="shared", tier="preparation", role="Patch extraction · 1,215 positives / 8,000 negatives"),
    "5c_balance_dataset.ipynb": dict(prefix="pilot_balance", dso="shared", tier="pilot", role="Pilot balancing · 196 patches"),
    "6_create_dataset.ipynb": dict(prefix="pilot_split", dso="shared", tier="pilot", role="Pilot split · 137 train / 29 validation / 30 test"),
    PILOT_CNN: dict(prefix="pilot_cnn", dso="dso1", tier="pilot", role="Early CNN and Grad-CAM exploration"),
    PILOT_RESNET: dict(prefix="pilot_resnet", dso="dso1", tier="pilot", role="Early 3D ResNet-18 exploration"),
}

EXTRA_MODELS = [
    dict(id="det_unet", name="U-Net encoder classifier", dso="dso1", tier="main", notebook=DETECTION,
         class_name="UNet3DClassifier", description="3D encoder with global pooling and a two-class head, trained with weighted focal loss."),
    dict(id="cls_cnn", name="Improved 3D CNN", dso="dso2", tier="main", notebook=CLASSIFICATION,
         class_name="Improved3DCNN", description="Three convolution blocks and a dense classification head with dropout."),
    dict(id="cls_resnet", name="ResNet3D Lite", dso="dso2", tier="main", notebook=CLASSIFICATION,
         class_name="ResNet3DLite", description="Residual 3D blocks followed by global average pooling and a two-class head."),
    dict(id="cls_unet", name="U-Net encoder classifier", dso="dso2", tier="main", notebook=CLASSIFICATION,
         class_name="UNet3DClassifier", description="A separate U-Net encoder study with dropout, pooling and a dense classification head."),
    dict(id="pilot_cnn", name="Pilot · Simple 3D CNN", dso="dso1", tier="pilot", notebook=PILOT_CNN,
         class_name="Simple3DCNN", description="Earlier binary experiment on 196 patches. Includes a saved Grad-CAM illustration."),
    dict(id="pilot_resnet", name="Pilot · 3D ResNet-18", dso="dso1", tier="pilot", notebook=PILOT_RESNET,
         class_name="r3d_18", description="Torchvision 3D ResNet-18 adapted to one input channel and two output classes."),
]

# Matrices are transcribed from saved figures, never produced by executing a model.
# The exporter records the exact figure checksum alongside each transcription.
BINARY_EVALUATIONS = [
    dict(id="det_validation", model="det_unet", dso="dso1", notebook=DETECTION, cell=7, stage="validation",
         matrix=[[1608, 16], [14, 205]], matrix_figure=2, roc_figure=3,
         note="Best validation checkpoint selected by loss. The printed 97.88% accuracy describes the last epoch, not this matrix."),
    dict(id="det_test", model="det_unet", dso="dso1", notebook=DETECTION, cell=8, stage="test",
         matrix=[[789, 11], [9, 112]], matrix_figure=1, roc_figure=2,
         note="Separate test evaluation after loading the stored best checkpoint; 921 candidate patches."),
    dict(id="cls_cnn_validation", model="cls_cnn", dso="dso2", notebook=CLASSIFICATION, cell=10, stage="validation",
         matrix=[[35, 35], [16, 114]], matrix_figure=2, roc_figure=3,
         note="Best validation checkpoint. The printed 73.50% accuracy describes the last epoch."),
    dict(id="cls_cnn_test", model="cls_cnn", dso="dso2", notebook=CLASSIFICATION, cell=11, stage="test",
         matrix=[[43, 35], [16, 107]], matrix_figure=1, roc_figure=2,
         note="Separate 201-patch test evaluation of the best Improved 3D CNN checkpoint."),
    dict(id="cls_resnet_validation", model="cls_resnet", dso="dso2", notebook=CLASSIFICATION, cell=15, stage="validation",
         matrix=[[71, 76], [36, 218]], matrix_figure=2, roc_figure=3,
         note="The saved matrix contains 401 patches, while the notebook's earlier split log says 200 validation patches. Treat this as its own historical run. The printed 65.84% is last-epoch accuracy."),
    dict(id="cls_unet_validation", model="cls_unet", dso="dso2", notebook=CLASSIFICATION, cell=20, stage="validation",
         matrix=[[43, 27], [25, 105]], matrix_figure=2, roc_figure=3,
         note="Best validation checkpoint on 200 patches. The printed 75.00% accuracy describes the last epoch."),
]

EXTRA_TRAINING = [
    dict(id="det_training", model="det_unet", dso="dso1", notebook=DETECTION, cell=7,
         label="U-Net candidate screening", tier="main", loss="Weighted focal loss", accuracy_scale=100,
         training_patches=6451, validation_patches=1843, note="70% train / 20% validation / 10% test in the supplied code; the presentation describes a different split."),
    dict(id="cls_cnn_training", model="cls_cnn", dso="dso2", notebook=CLASSIFICATION, cell=9,
         label="Improved 3D CNN", tier="main", loss="Cross-entropy", accuracy_scale=100,
         training_patches=1603, validation_patches=200, note="17 saved epochs; early stopping. Initial split log: 1,603 / 200 / 201."),
    dict(id="cls_resnet_training", model="cls_resnet", dso="dso2", notebook=CLASSIFICATION, cell=14,
         label="ResNet3D Lite", tier="main", loss="Cross-entropy", accuracy_scale=100,
         training_patches=None, validation_patches=401, note="Six saved epochs. Validation count comes from the saved matrix, not the earlier split configuration."),
    dict(id="cls_unet_training", model="cls_unet", dso="dso2", notebook=CLASSIFICATION, cell=19,
         label="U-Net encoder classifier", tier="main", loss="Cross-entropy", accuracy_scale=100,
         training_patches=1603, validation_patches=200, note="30 saved epochs. The saved model is selected by validation loss."),
    dict(id="pilot_cnn_training", model="pilot_cnn", dso="dso1", notebook=PILOT_CNN, cell=13,
         label="Pilot · Simple 3D CNN", tier="pilot", loss="Cross-entropy", accuracy_scale=1,
         training_patches=137, validation_patches=29, note="Small 196-patch pilot, with 30 test patches; not comparable to the larger candidate study."),
    dict(id="pilot_resnet_training", model="pilot_resnet", dso="dso1", notebook=PILOT_RESNET, cell=8,
         label="Pilot · 3D ResNet-18", tier="pilot", loss="Cross-entropy", accuracy_scale=1,
         training_patches=137, validation_patches=29, note="No saved test evaluation. Source contains a later edit/typo in the loss tracking expression and a Grad-CAM error; only the existing outputs are represented."),
]

FIGURE_CELLS = {
    (DETECTION, 7): ("det_training", "det_unet", "U-Net candidate screening", ["training", "confusion_matrix", "roc"]),
    (DETECTION, 8): ("det_test", "det_unet", "U-Net · test evaluation", ["confusion_matrix", "roc"]),
    (CLASSIFICATION, 10): ("cls_cnn_training", "cls_cnn", "Improved 3D CNN", ["training", "confusion_matrix", "roc"]),
    (CLASSIFICATION, 11): ("cls_cnn_test", "cls_cnn", "Improved 3D CNN · test evaluation", ["confusion_matrix", "roc"]),
    (CLASSIFICATION, 15): ("cls_resnet_training", "cls_resnet", "ResNet3D Lite", ["training", "confusion_matrix", "roc"]),
    (CLASSIFICATION, 20): ("cls_unet_training", "cls_unet", "U-Net encoder classifier", ["training", "confusion_matrix", "roc"]),
    (PILOT_CNN, 16): ("pilot_cnn_training", "pilot_cnn", "Pilot CNN · loss curves", ["training"]),
    (PILOT_CNN, 17): ("pilot_cnn_training", "pilot_cnn", "Pilot CNN · validation accuracy", ["training"]),
    (PILOT_CNN, 24): ("pilot_cnn_cam", "pilot_cnn", "Pilot CNN · Grad-CAM example", ["gradcam"]),
    (PILOT_RESNET, 9): ("pilot_resnet_training", "pilot_resnet", "Pilot ResNet-18 · training curves", ["training", "training"]),
    (PILOT_RESNET, 15): ("pilot_resnet_cam", "pilot_resnet", "Pilot ResNet-18 · Grad-CAM example", ["gradcam"]),
}

PREPARATION_TITLES = {
    ("3_resample_ct_scans (1).ipynb", 8): "Original and resampled CT",
    ("4_segment_lungs.ipynb (1).ipynb", 10): "Historical lung-mask generation steps",
    ("4_segment_lungs.ipynb (1).ipynb", 16): "Historical lung-mask overlays",
    ("8_nodule_detection_preparation (1).ipynb", 6): "Candidate visualization · initial coordinates",
    ("8_nodule_detection_preparation (1).ipynb", 7): "Candidate visualization · initial coordinates",
    ("8_nodule_detection_preparation (1).ipynb", 9): "Strict positive candidate",
    ("8_nodule_detection_preparation (1).ipynb", 10): "Strict positive candidate · second example",
    ("8_nodule_detection_preparation (1).ipynb", 12): "Negative candidate · original notebook label",
    ("8_nodule_detection_preparation (1).ipynb", 13): "Negative candidate example",
    ("5c_balance_dataset.ipynb", 4): "Pilot class distribution · before balancing",
    ("5c_balance_dataset.ipynb", 7): "Pilot class distribution · after balancing",
}
