# Selected portfolio plots

Use the numbered images in this order. Files 01-09 are byte-for-byte copies of the original saved notebook figures. No model or dataset was rerun.

## Suggested page layout

1. **Data processing:** 01, showing original and resampled CT.
2. **DSO1 results:** 02 and 03 side by side, with the same 921-patch test-cohort caption.
3. **DSO2 results:** 04 and 05 side by side, with the same 201-patch Improved 3D CNN test-cohort caption.
4. **DSO3 results:** 06, illustrating CT, spherical reference and predicted masks.
5. **Training history:** 07, 08 and 09 in separate tabs or an expandable section. The loss functions differ; do not overlay their values as one model-ranking plot.

**Optional:** 00 is a generated concept illustration for the hero. 10 is a new descriptive class-distribution chart, supplied as PNG and SVG, using the recorded counts. Its two panels use independent y-axis scales.

## Files and ready-to-use captions

### 00_optional_ct_hero_illustration.png

Optional hero image: illustrative multiplanar CT views. Highlighted findings are not model predictions.

[Open image](00_optional_ct_hero_illustration.png)

### 01_ct_preprocessing.png

CT before and after resampling to 1 mm isotropic spacing.

Source: `3_resample_ct_scans (1).ipynb`, cell 8 (1-based).

[Open image](01_ct_preprocessing.png)

### 02_dso1_test_confusion_matrix.png

DSO1 candidate screening: best-checkpoint test evaluation, 921 patches; 789 TN, 11 FP, 9 FN and 112 TP.

Source: `11_nodule_detection.ipynb`, cell 8 (1-based).

[Open image](02_dso1_test_confusion_matrix.png)

### 03_dso1_test_roc_curve.png

DSO1 candidate screening: test ROC-AUC 0.9961 and accuracy 97.83%. The original plot rounds AUC in its legend.

Source: `11_nodule_detection.ipynb`, cell 8 (1-based).

[Open image](03_dso1_test_roc_curve.png)

### 04_dso2_test_confusion_matrix.png

DSO2 Improved 3D CNN: test evaluation, 201 patches; 43 TN, 35 FP, 16 FN and 107 TP. Labels are true nodule and false nodule.

Source: `11_nodule_classification.ipynb`, cell 11 (1-based).

[Open image](04_dso2_test_confusion_matrix.png)

### 05_dso2_test_roc_curve.png

DSO2 Improved 3D CNN: test ROC-AUC 0.8335 and accuracy 74.63%; binary true/false nodule classification.

Source: `11_nodule_classification.ipynb`, cell 11 (1-based).

[Open image](05_dso2_test_roc_curve.png)

### 06_dso3_segmentation_prediction.png

DSO3 baseline U-Net: CT slices, spherical reference masks and predicted masks. Individual displayed case Dice: 0.2556. This is from positive-patch experiment A, not the mixed validation cohort.

Source: `02_Modeling_Segmentation.ipynb`, cell 13 (1-based).

[Open image](06_dso3_segmentation_prediction.png)

### 07_dso1_training_history.png

DSO1 U-Net encoder: training/validation weighted focal loss and accuracy over 20 recorded epochs. Last-epoch accuracy is separate from the best-checkpoint evaluation.

Source: `11_nodule_detection.ipynb`, cell 7 (1-based).

[Open image](07_dso1_training_history.png)

### 08_dso2_training_history.png

DSO2 Improved 3D CNN: training/validation cross-entropy loss and accuracy over 17 recorded epochs.

Source: `11_nodule_classification.ipynb`, cell 10 (1-based).

[Open image](08_dso2_training_history.png)

### 09_dso3_training_loss.png

DSO3 baseline U-Net: training/validation binary cross-entropy loss over 50 epochs. Historical validation inherits augmentation.

Source: `02_Modeling_Segmentation.ipynb`, cell 15 (1-based).

[Open image](09_dso3_training_loss.png)

### 10_optional_class_distribution.png

Recorded patch class distributions for two separate study datasets. Counts are patches, not unique nodules, patients or CT scans. Each panel uses its own y-axis scale.

[Open image](10_optional_class_distribution.png)

## Scientific labels

- DSO1 evaluates supplied candidate patches, rather than full-volume localization.
- DSO2 evaluates true versus false nodules. Its 0.8335 test AUC is not a benign/malignant evaluation.
- DSO3 is the portfolio label for segmentation, archived as DSO5 in the notebooks.
- The segmentation example has its own case Dice (0.2556); do not label this image with the separate 0.9504 mixed-validation Dice.
- Segmentation reference masks are annotation-derived spheres.
- Original ROC legends round their AUCs; use the four-decimal values in these captions.
- `selection_manifest.json` records figure identities, source cells and original-image checksums.
