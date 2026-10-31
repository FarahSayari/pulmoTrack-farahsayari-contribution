# PulmoTrack

Deep Learning for / Pulmonary Nodule Analysis

An academic medical-imaging project using LUNA16 CT scans to study nodule candidate detection, binary classification and exploratory 3D segmentation, presented through an interactive research website.

Axial, coronal and sagittal CT concept illustration; the highlighted findings are illustrative.

## Project information

- **Category:** Research, DL, Medical Imaging, Computer Vision
- **Role:** Data Scientist, Project Manager
- **Methodology:** CRISP-DM
- **Dataset:** LUNA16
- **Status:** Completed
- **Scientific paper:** Available · Team research manuscript
- **Website:** [Open the live Streamlit website](https://pulmotrack-farahsayari-contribution-sef2wgqdo5bzl9wmgfwxry.streamlit.app/)

## Technologies

**Research:** Python, PyTorch, torchvision, NumPy, pandas, SciPy, scikit-learn, SimpleITK. / **Analysis & delivery:** Jupyter Notebook, Google Colab, Matplotlib, Plotly, Pillow, Streamlit, Git, GitHub and Streamlit Community Cloud.

MOTIVATION AND CONTRIBUTION

# Project Overview

Lung cancer is the leading cause of cancer-related death worldwide and is often diagnosed at an advanced stage, when treatment options are more limited. Earlier identification and appropriate follow-up can improve the opportunity for effective treatment.

PulmoTrack explores how deep learning can analyze pulmonary nodules in volumetric CT data. The project connects medical-image preprocessing, three research objectives, model evaluation and a deployed interface that makes the experiments understandable to a portfolio audience.

The work is an academic research study. Its recorded models evaluate nodule candidates and masks; they do not establish a clinically validated cancer diagnosis system.

Scientific context: [World Health Organization — Lung cancer](https://www.who.int/news-room/fact-sheets/detail/lung-cancer).

# My Contributions

**Research organization.** Used CRISP-DM to structure the CT workstream, organize experiments and communicate the scope of the three objectives.

**LUNA16 preprocessing.** Worked on CT resampling, annotation and candidate inspection, world-to-voxel coordinate conversion, intensity normalization and 3D patch preparation.

**Detection and classification.** Developed and studied 3D candidate-screening and binary classification approaches, comparing their saved training histories, confusion matrices and ROC evaluations.

**Exploratory segmentation.** Investigated 3D U-Net, attention and residual-attention variants, and tiny-nodule oversampling, documenting both promising summaries and difficult positive-only cases.

**Results and documentation.** Organized notebook outputs, kept evaluation cohorts and checkpoint results distinct, and contributed to the team research manuscript.

**Portfolio deployment.** Created a Streamlit research showcase with CT visualizations, model overviews, result interpretation, source resources and access to the research paper.

DATA AND RESEARCH OBJECTIVES

# LUNA16 Dataset

LUNA16 (Lung Nodule Analysis 2016) is derived from LIDC-IDRI and contains **888 volumetric chest CT scans**. It provides CT volumes, candidate locations and nodule annotations for pulmonary nodule research.

The official reference standard includes nodules at least 3 mm in diameter accepted by at least three of four radiologists. Scans with slice thickness greater than 2.5 mm were excluded. The CT collection is approximately 90 GB in scale; the deployed showcase uses saved outputs rather than distributing the dataset.

The project prepares different patch collections for different experiments: **64³ voxels** for detection/classification and **48³ voxels** for the archived segmentation studies. Candidate labels, nodule reference masks and lung-field masks serve different purposes.

Dataset source and access: [LUNA16 official data page](https://luna16.grand-challenge.org/Data/).

# Explore the Three DSOs

**DSO1 — Nodule Detection.** Screen supplied candidate locations using a U-Net-style 3D encoder with global pooling and a two-class head. The selected run uses 9,215 patches: 1,215 positives and 8,000 negatives. Weighted focal loss addresses the class imbalance. This is candidate screening, rather than full-volume localization.

**DSO2 — Nodule Classification.** Compare Improved 3D CNN, ResNet3D Lite and a separate U-Net encoder classifier. Recorded folder counts are 1,243 true-nodule and 761 false-nodule patches. The evaluated task distinguishes true nodules from false nodules; benign/malignant classification is a broader project goal.

**DSO3 — Exploratory 3D Segmentation.** Learn voxel-level nodule masks with U-Net variants and study the challenge of tiny nodules. This is a parallel research exploration, not a prerequisite for detection or classification. The intended later pipeline uses the provided LUNA16 lung masks.

Naming: exploratory segmentation is presented as DSO3 in the portfolio overview and sidebar; it is named DSO5 in the original research material.

AN ITERATIVE RESEARCH PROCESS

# CRISP-DM Methodology

CRISP-DM structures the work into six connected phases. The process is iterative: findings from data preparation and evaluation inform the next modeling decisions.

## 01 / Business understanding

Translate the early-identification motivation into research questions: can a model screen nodule candidates, distinguish true and false nodules, and delineate candidate nodule regions? Define the academic scope, separate the three objectives and set measurable evaluation criteria.

## 02 / Data understanding

Inspect 3D CT volumes, voxel spacing, scan identifiers, annotation coordinates, nodule sizes and candidate labels. Identify class imbalance, limited tiny-nodule visibility, differences between patch collections, and the distinction between lung masks and nodule masks.

## 03 / Data processing

Resample CT volumes, convert physical coordinates to voxel coordinates, normalize intensities and extract fixed-size patches. Prepare labels and spherical nodule references. Apply the documented augmentation and sampling strategies, with separate dataset splits for each study.

## 04 / Modeling

Develop 3D encoder-based candidate screening, three binary classification architectures and encoder–decoder segmentation variants. Use Adam optimization and the losses actually recorded in each study: weighted focal loss for detection, cross-entropy for classification and binary cross-entropy for segmentation.

## 05 / Evaluation

Review the appropriate metrics for each task: accuracy, ROC-AUC and confusion matrices for candidate classifiers; Dice, precision, recall and F1 for segmentation. Keep test and validation cohorts distinct, distinguish best-checkpoint results from final-epoch logs, and inspect original predictions.

## 06 / Deployment

Publish an interactive Streamlit showcase of the completed research. Package saved figures, results and source references so visitors can explore the work without downloading LUNA16 or rerunning the models. This phase communicates the experiments; it does not deploy clinical inference.

Methodology reference: [IBM CRISP-DM guide](https://www.ibm.com/docs/en/SS3RA7_18.6.0/pdf/ModelerCRISPDM.pdf).

DSO1 AND DSO2 / NOTEBOOK EVALUATIONS

# Results and Performance

The values below belong to the saved notebook evaluations. Accuracy is calculated from the corresponding original confusion matrix; ROC-AUC comes from that checkpoint’s recorded output.

## DSO1 / Nodule Detection

| Cohort | Patches | Accuracy | ROC-AUC |
| --- | --- | --- | --- |
| Validation | 1,843 | 98.37% | 0.9928 |
| Test | 921 | 97.83% | 0.9961 |

On the 921-patch test cohort, the saved model correctly identifies **112 of 121 nodule patches** (92.56% sensitivity), with **11 false positives and 9 false negatives**. These are candidate-level outcomes; they do not establish false positives per full CT scan.

## DSO2 / Nodule Classification

| Study / cohort | Patches | Accuracy | ROC-AUC |
| --- | --- | --- | --- |
| Improved 3D CNN · validation | 200 | 74.50% | 0.8269 |
| Improved 3D CNN · test | 201 | 74.63% | 0.8335 |
| ResNet3D Lite · validation | 401 | 72.07% | 0.7678 |
| U-Net encoder · validation | 200 | 74.00% | 0.8116 |

The Improved 3D CNN test evaluation identifies **107 of 123 true-nodule patches**, with 35 false positives and 16 false negatives. Its accuracy and AUC describe binary candidate labels, not malignancy diagnosis.

**How to compare these results.** Each model retains its own recorded cohort. The ResNet matrix contains 401 patches despite an earlier 200-patch validation log. Splitting is by patch, and a patient-disjoint benchmark is not established. These studies support analysis of the recorded runs, not a controlled architecture ranking.

Primary notebooks: 11_nodule_detection.ipynb and 11_nodule_classification.ipynb. Best-checkpoint evaluations are separate from last-epoch accuracies.

DSO3 / EXPLORATORY SEGMENTATION

# Results and Performance

Segmentation results depend strongly on the selected patches and the handling of empty masks. The table preserves these separate experimental contexts; all values are notebook-reported summaries.

| Model study | Evaluation context | Patches | Reported score |
| --- | --- | --- | --- |
| 3D U-Net | Five positives · A | 5 | Dice 0.2115 |
| 3D U-Net | Five positives · B | 5 | Dice 0.3127 |
| 3D U-Net | Full positive-patch folder | Not recorded | F1 0.5916 |
| 3D U-Net | Mixed validation loader | 509 | Dice 0.9504 |
| Attention U-Net | Five positive patches | 5 | Dice 0.1599 |
| Attention U-Net | Ten mixed patches | 10 | Dice 0.9800 |
| Residual attention | Ten positives · initial | 10 | Dice 0.1200 |
| Residual attention | Ten positives · later | 10 | Dice 0.0348 |

The **0.9504 validation Dice** comes from a mixed, augmented loader, where smoothing assigns empty prediction/reference pairs a Dice near 1. The attention model’s mixed-patch evaluator similarly assigns perfect scores to empty pairs. Positive-only samples show the remaining difficulty of recovering small nodule regions.

Tiny-nodule augmentation retains saved prediction examples and a final validation binary cross-entropy loss of **0.0446**. No Dice, precision, recall or F1 evaluation was recorded for that run; a lower loss is not an overlap score.

Original saved baseline prediction — positive-patch experiment A, notebook cell 13; displayed case Dice: 0.2556. Partial mask recovery illustrates the positive-case challenge, rather than the mixed validation summary.

RESEARCH DELIVERY AND SUPPORTING PAPER

# Deployment

PulmoTrack is hosted on **Streamlit Community Cloud**, with the GitHub repository as its deployment source. The interface preserves the dark navy and teal medical-imaging identity, sidebar navigation and CT-focused visual presentation.

Visitors can explore model architectures, recorded metrics, training histories, original segmentation predictions, pilot Grad-CAM, research resources and the manuscript viewer. The app presents saved evidence; it requires no CT dataset download or live model execution.

**Delivery workflow:** research notebooks → exported figures and result summaries → Streamlit interface → GitHub → Streamlit Community Cloud.

[Open the live PulmoTrack research showcase](https://pulmotrack-farahsayari-contribution-sef2wgqdo5bzl9wmgfwxry.streamlit.app/)

[https://pulmotrack-farahsayari-contribution-sef2wgqdo5bzl9wmgfwxry.streamlit.app/](https://pulmotrack-farahsayari-contribution-sef2wgqdo5bzl9wmgfwxry.streamlit.app/)

The website is a completed research showcase, rather than a patient-upload or diagnostic inference service. Repository changes can be reflected in the deployed app through Community Cloud’s GitHub integration.

Deployment reference: [Streamlit Community Cloud documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

# Research Paper

## PulmoTrack: A Deep Learning Solution for Pulmonary Nodule Segmentation, Detection, and Classification on CT and Chest X-ray Scans

**Availability:** Available through the website’s Research Paper section, with View PDF and Download PDF options. / **Document:** Team research manuscript · May 2025 · 11 pages. / **Authors:** Farah Sayari, Mahdi Landolsi, Najib Nakhla, Souha Bachtobji, Adem Tounsi, Ala Charfi.

The manuscript covers the broader team project, including CT and chest X-ray work. This portfolio document focuses on the LUNA16 CT contribution.

**Paper-reported LUNA16 highlights (pp. 4–5):** segmentation Dice 0.95; detection sensitivity 92% at 1.5 false positives per scan; classification AUC 0.91. These manuscript summaries are separate from the notebook evaluations on pages 5–6; the supplied classification notebooks evaluate true/false nodule labels.

Project source: [GitHub repository](https://github.com/FarahSayari/pulmoTrack-farahsayari-contribution). The original manuscript is stored as assets/paper/PulmoTrackPaper.pdf and is accessible through the live app.
